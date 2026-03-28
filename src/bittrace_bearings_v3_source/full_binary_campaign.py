"""Canonical source-lane bridge for the full binary V3 freeze/export path."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.v3 import (
    CampaignRequest,
    ContractValidationError,
    ExecutionAcceleration,
    ExecutionTrace,
    PipelineStageConfig,
    StageKey,
    WaveformDatasetRecord,
    WaveformPayloadRef,
    build_waveform_dataset_bundle,
    canonical_stage_sequence,
)
from bittrace.core.config import DeepTrainingConfig, LeanTrainingConfig

from .full_binary_hardmode import build_hardmode_bridge, hardmode_enabled
from .locked_frontend import (
    build_locked_frontend_stage_materialization,
    load_locked_frontend_spec,
)
from .temporal_features import build_temporal_feature_payload, load_temporal_feature_config


FILENAME_RE = re.compile(
    r"^(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing_id>[A-Za-z0-9]+)_(?P<recording>\d+)\.mat$"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "canonical_source_profile.yaml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
_ADAPTER_PROFILE_ID = "paderborn_binary_mat_v1"
_WAVEFORM_CHANNEL_NAME = "vibration"


@dataclass(frozen=True, slots=True)
class PreparedFullBinaryCampaign:
    config_path: Path
    run_root: Path
    campaign_request: CampaignRequest
    stage_configs: dict[StageKey, PipelineStageConfig]
    inventory_row_count: int
    smoke_row_count: int


def load_consumer_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    _require_mapping_key(payload, "profile_name", resolved_path)
    _require_mapping_key(payload, "data", resolved_path)
    _require_mapping_key(payload, "binary_mapping", resolved_path)
    _require_mapping_key(payload, "selection", resolved_path)
    _require_mapping_key(payload, "deploy_constraints", resolved_path)
    _require_mapping_key(payload, "ranking_intent", resolved_path)
    _require_mapping_key(payload, "splits", resolved_path)
    _require_mapping_key(payload, "backend", resolved_path)
    return payload


def build_campaign_request(
    config_path: str | Path,
    run_root: str | Path,
    *,
    campaign_seed: int = 31,
) -> CampaignRequest:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = Path(run_root).resolve()
    profile = load_consumer_config(resolved_config_path)
    inventory_rows = _resolve_inventory_rows(profile)
    smoke_rows = _select_smoke_rows(inventory_rows)
    deploy_constraints = dict(profile.get("deploy_constraints", {}))
    ranking_intent = dict(profile.get("ranking_intent", {}))
    profile_notes = tuple(str(note) for note in profile.get("notes", ()))
    hardmode = hardmode_enabled(profile)
    temporal_feature_config = load_temporal_feature_config(profile)
    locked_frontend = load_locked_frontend_spec(profile)
    if locked_frontend is not None and not temporal_feature_config.enabled:
        raise ContractValidationError(
            "Locked temporal frontend requires `enable_temporal_features: true` in the consumer profile."
        )

    return CampaignRequest(
        campaign_id=_sanitize_identifier(
            f"{profile['profile_name']}__{resolved_run_root.name}"
        ),
        campaign_seed=campaign_seed,
        output_dir=str(resolved_run_root),
        stage_sequence=canonical_stage_sequence(),
        notes=(
            profile_notes
            + tuple(str(note) for note in deploy_constraints.get("notes", ()))
            + (
                f"consumer_config={resolved_config_path}",
                "binary_objective_bias=minimize_healthy_to_unhealthy_fpr_first",
                "binary_alerting_bias=high_confidence_unhealthy_alerts",
                f"hard_mode={'true' if hardmode else 'false'}",
                "winner_selection_split=val" if hardmode else "winner_selection_split=proxy",
                "final_test_split_reserved_for_reporting_only"
                if hardmode
                else "test_metrics_may_appear_in_proxy_artifacts",
            )
            + (
                (
                    f"frontend_lock_regime={locked_frontend.regime_id}",
                    f"frontend_lock_bit_length={locked_frontend.bit_length}",
                    "frontend_sweep=false",
                    f"frontend_lock_selection_source={locked_frontend.selection_source}",
                )
                if locked_frontend is not None
                else ()
            )
        ),
        metadata={
            "consumer_project_root": str(PROJECT_ROOT),
            "config_path": str(resolved_config_path),
            "profile_name": profile["profile_name"],
            "dataset_raw_root": str(Path(profile["data"]["raw_root"]).resolve()),
            "inventory_row_count": len(inventory_rows),
            "lean_smoke_row_count": len(smoke_rows),
            "lean_main_screen_row_count": len(inventory_rows),
            "deploy_constraints": deploy_constraints,
            "ranking_intent": ranking_intent,
            "smoke_selection_policy": "first_record_per_split_class_condition",
            "hard_mode": hardmode,
            "enable_temporal_features": temporal_feature_config.enabled,
            "temporal_feature_config_fingerprint": (
                temporal_feature_config.fingerprint if temporal_feature_config.enabled else None
            ),
            "locked_frontend": locked_frontend.to_dict() if locked_frontend is not None else None,
            "frontend_sweep_enabled": locked_frontend is None,
        },
    )


def build_stage_configs(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[StageKey, PipelineStageConfig]:
    resolved_output_dir = Path(output_dir).resolve()
    resolved_config_path = Path(config_path).resolve()
    profile = load_consumer_config(resolved_config_path)
    inventory_rows = _resolve_inventory_rows(profile)
    smoke_rows = _select_smoke_rows(inventory_rows)
    temporal_feature_config = load_temporal_feature_config(profile)
    locked_frontend = load_locked_frontend_spec(profile)
    if locked_frontend is not None and not temporal_feature_config.enabled:
        raise ContractValidationError(
            "Locked temporal frontend requires `enable_temporal_features: true` in the consumer profile."
        )

    smoke_bundle = _materialize_source_bundle(
        smoke_rows,
        output_dir=resolved_output_dir / "_inputs" / "lean_smoke_source_bundle",
        profile_name=str(profile["profile_name"]),
        selection_name="lean_smoke",
        temporal_feature_config=temporal_feature_config,
    )
    full_bundle = _materialize_source_bundle(
        inventory_rows,
        output_dir=resolved_output_dir / "_inputs" / "lean_main_screen_source_bundle",
        profile_name=str(profile["profile_name"]),
        selection_name="lean_main_screen_full",
        temporal_feature_config=temporal_feature_config,
    )

    lean_training_config, deep_training_config = _load_backend_training_configs(profile)
    lean_execution_trace = _execution_trace_for_backend(lean_training_config)
    deep_execution_trace = _execution_trace_for_backend(deep_training_config)
    hardmode = hardmode_enabled(profile)
    hardmode_bridge = build_hardmode_bridge(profile) if hardmode else None
    include_test_metrics = (
        hardmode_bridge.include_test_metrics_in_frontend()
        if hardmode_bridge is not None
        else True
    )
    deploy_target = str(profile["deploy_constraints"]["target"])
    max_selected_k = _resolve_max_selected_k_per_class(profile)
    k_candidates = tuple(range(1, max_selected_k + 1))
    deploy_export = _device_agnostic_export(profile, k_candidates=k_candidates)
    objective_bias = _objective_bias_notes(profile)
    locked_frontend_notes = _locked_frontend_stage_notes(locked_frontend)
    lean_smoke_runner_kwargs: dict[str, object] = {
        "lean_training_config": lean_training_config,
    }
    lean_main_screen_runner_kwargs: dict[str, object] = {
        "lean_training_config": lean_training_config,
    }
    if locked_frontend is not None:
        locked_lean_smoke = build_locked_frontend_stage_materialization(
            stage_key=StageKey.LEAN_SMOKE,
            stage_output_dir=_canonical_stage_output_dir(
                resolved_output_dir,
                stage_key=StageKey.LEAN_SMOKE,
            ),
            source_bundle=smoke_bundle,
            include_test_metrics_in_frontend=include_test_metrics,
            locked_frontend=locked_frontend,
        )
        locked_lean_main_screen = build_locked_frontend_stage_materialization(
            stage_key=StageKey.LEAN_MAIN_SCREEN,
            stage_output_dir=_canonical_stage_output_dir(
                resolved_output_dir,
                stage_key=StageKey.LEAN_MAIN_SCREEN,
            ),
            source_bundle=full_bundle,
            include_test_metrics_in_frontend=include_test_metrics,
            locked_frontend=locked_frontend,
        )
        lean_smoke_runner_kwargs.update(
            {
                "promoted_candidate": locked_lean_smoke.promoted_candidate,
                "downstream_deep_input": locked_lean_smoke.downstream_deep_input,
                "ranking_policy": locked_lean_smoke.ranking_policy,
            }
        )
        lean_main_screen_runner_kwargs.update(
            {
                "promoted_candidate": locked_lean_main_screen.promoted_candidate,
                "downstream_deep_input": locked_lean_main_screen.downstream_deep_input,
                "ranking_policy": locked_lean_main_screen.ranking_policy,
            }
        )
    deep_smoke_runner_kwargs: dict[str, object] = {
        "deep_training_config": deep_training_config,
    }
    deep_main_screen_runner_kwargs: dict[str, object] = {
        "deep_training_config": deep_training_config,
    }
    capacity_refinement_runner_kwargs: dict[str, object] = {
        "deep_training_config": deep_training_config,
        "k_medoids_per_class_candidates": k_candidates,
        "device_agnostic_export": deploy_export,
    }
    if hardmode_bridge is not None:
        deep_smoke_runner_kwargs["evaluate_candidates"] = (
            lambda request, bridge=hardmode_bridge, config=deep_training_config: bridge.evaluate_deep_stage(
                request,
                stage_key=StageKey.DEEP_SMOKE,
                deep_training_config=config,
            )
        )
        deep_main_screen_runner_kwargs["evaluate_candidates"] = (
            lambda request, bridge=hardmode_bridge, config=deep_training_config: bridge.evaluate_deep_stage(
                request,
                stage_key=StageKey.DEEP_MAIN_SCREEN,
                deep_training_config=config,
            )
        )
        capacity_refinement_runner_kwargs["evaluate_k_candidates"] = (
            lambda request, requested_k_values, bridge=hardmode_bridge, config=deep_training_config: bridge.evaluate_capacity_stage(
                request,
                requested_k_values,
                deep_training_config=config,
            )
        )

    return {
        StageKey.LEAN_SMOKE: PipelineStageConfig(
            frontend_inputs=(smoke_bundle.to_frontend_input(include_test_metrics=include_test_metrics),),
            execution_trace=lean_execution_trace,
            notes=(
                f"lean_smoke_rows={len(smoke_rows)}",
                "lean_smoke_selection=first_record_per_split_class_condition",
                f"deploy_target={deploy_target}",
                f"include_test_metrics_in_frontend={'true' if include_test_metrics else 'false'}",
                f"enable_temporal_features={'true' if temporal_feature_config.enabled else 'false'}",
                "frontend_materialization=consumer_locked_candidate"
                if locked_frontend is not None
                else "frontend_materialization=canonical_runtime_materialization",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs=lean_smoke_runner_kwargs,
        ),
        StageKey.DEEP_SMOKE: PipelineStageConfig(
            execution_trace=deep_execution_trace,
            notes=(
                "deep_smoke_lineage=derived_from_lean_smoke_frontend_winner",
                f"deploy_target={deploy_target}",
                "deep_smoke_mode=hardmode_real_search" if hardmode else "deep_smoke_mode=materialized_proxy",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs=deep_smoke_runner_kwargs,
        ),
        StageKey.LEAN_MAIN_SCREEN: PipelineStageConfig(
            frontend_inputs=(full_bundle.to_frontend_input(include_test_metrics=include_test_metrics),),
            execution_trace=lean_execution_trace,
            notes=(
                f"lean_main_screen_rows={len(inventory_rows)}",
                "lean_main_screen_selection=full_deterministic_binary_corpus",
                f"deploy_target={deploy_target}",
                f"include_test_metrics_in_frontend={'true' if include_test_metrics else 'false'}",
                f"enable_temporal_features={'true' if temporal_feature_config.enabled else 'false'}",
                "frontend_materialization=consumer_locked_candidate"
                if locked_frontend is not None
                else "frontend_materialization=canonical_runtime_materialization",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs=lean_main_screen_runner_kwargs,
        ),
        StageKey.DEEP_MAIN_SCREEN: PipelineStageConfig(
            execution_trace=deep_execution_trace,
            notes=(
                "deep_main_screen_lineage=derived_from_lean_main_screen_frontend_winner",
                f"deploy_target={deploy_target}",
                "deep_main_screen_mode=hardmode_real_search" if hardmode else "deep_main_screen_mode=materialized_proxy",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs=deep_main_screen_runner_kwargs,
        ),
        StageKey.CAPACITY_REFINEMENT: PipelineStageConfig(
            execution_trace=deep_execution_trace,
            notes=(
                f"capacity_refinement_k_candidates={','.join(str(value) for value in k_candidates)}",
                f"max_selected_k_per_class={max_selected_k}",
                f"deploy_target={deploy_target}",
                "capacity_refinement_mode=hardmode_real_search" if hardmode else "capacity_refinement_mode=materialized_proxy",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs=capacity_refinement_runner_kwargs,
        ),
        StageKey.WINNER_DEEPEN_FREEZE_EXPORT: PipelineStageConfig(
            execution_trace=deep_execution_trace,
            notes=(
                "freeze_export_stage=stage_6_canonical_freeze_export",
                f"deploy_target={deploy_target}",
                f"max_selected_k_per_class={max_selected_k}",
            )
            + locked_frontend_notes
            + objective_bias,
            runner_kwargs={
                "device_agnostic_export": deploy_export,
                "summary": (
                    "Full Paderborn binary MCU-candidate freeze/export with deploy-constrained "
                    "k selection and healthy->unhealthy false-positive minimization prioritized."
                ),
            },
        ),
    }


def prepare_full_binary_campaign(
    config_path: str | Path,
    run_root: str | Path,
    *,
    campaign_seed: int = 31,
) -> PreparedFullBinaryCampaign:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = Path(run_root).resolve()
    profile = load_consumer_config(resolved_config_path)
    inventory_rows = _resolve_inventory_rows(profile)
    smoke_rows = _select_smoke_rows(inventory_rows)
    campaign_request = build_campaign_request(
        resolved_config_path,
        resolved_run_root,
        campaign_seed=campaign_seed,
    )
    stage_configs = build_stage_configs(
        resolved_run_root,
        config_path=resolved_config_path,
    )
    return PreparedFullBinaryCampaign(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        campaign_request=campaign_request,
        stage_configs=stage_configs,
        inventory_row_count=len(inventory_rows),
        smoke_row_count=len(smoke_rows),
    )


def write_campaign_request_json(
    campaign_request: CampaignRequest,
    path: str | Path,
) -> Path:
    resolved_path = Path(path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(campaign_request.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_path


def _require_mapping_key(payload: dict[str, Any], key: str, source_path: Path) -> None:
    if key not in payload:
        raise ContractValidationError(f"`{source_path}` is missing required top-level key `{key}`.")


def _load_backend_training_configs(
    profile: dict[str, Any],
) -> tuple[LeanTrainingConfig, DeepTrainingConfig]:
    backend = profile.get("backend", {})
    if not isinstance(backend, dict):
        raise ContractValidationError("`backend` must be a mapping.")
    lean_raw = backend.get("lean", {})
    deep_raw = backend.get("deep", {})
    if not isinstance(lean_raw, dict):
        raise ContractValidationError("`backend.lean` must be a mapping.")
    if not isinstance(deep_raw, dict):
        raise ContractValidationError("`backend.deep` must be a mapping.")
    try:
        lean_training_config = LeanTrainingConfig.from_mapping(lean_raw)
        deep_training_config = DeepTrainingConfig.from_mapping(deep_raw)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc
    return lean_training_config, deep_training_config


def _execution_trace_for_backend(
    training_config: LeanTrainingConfig | DeepTrainingConfig,
) -> ExecutionTrace:
    requested_backend = str(training_config.backend)
    if requested_backend == "cpu":
        requested_execution_acceleration = ExecutionAcceleration.CPU
    elif requested_backend == "gpu":
        requested_execution_acceleration = ExecutionAcceleration.GPU
    else:
        requested_execution_acceleration = ExecutionAcceleration.AUTO
    return ExecutionTrace(
        requested_execution_acceleration=requested_execution_acceleration,
        allow_backend_fallback=bool(training_config.allow_backend_fallback),
    )


def _build_inventory_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw_root = Path(profile["data"]["raw_root"]).resolve()
    if not raw_root.is_dir():
        raise ContractValidationError(f"`data.raw_root` must point to an existing directory: {raw_root}")

    healthy_re = re.compile(str(profile["binary_mapping"]["healthy_regex"]))
    unhealthy_re = re.compile(str(profile["binary_mapping"]["unhealthy_regex"]))

    selection = dict(profile.get("selection", {}))
    include_conditions = {str(value) for value in selection.get("operating_conditions", ())}
    include_ids = {str(value) for value in selection.get("include_bearing_ids", ())}
    exclude_ids = {str(value) for value in selection.get("exclude_bearing_ids", ())}
    include_recordings = {str(value) for value in selection.get("include_recordings", ())}

    splits = dict(profile.get("splits", {}))
    train_recordings = {str(value) for value in splits.get("train_recordings", ())}
    val_recordings = {str(value) for value in splits.get("val_recordings", ())}
    test_recordings = {str(value) for value in splits.get("test_recordings", ())}

    inventory_rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*.mat")):
        match = FILENAME_RE.match(path.name)
        if match is None:
            continue

        condition = match.group("condition")
        bearing_id = match.group("bearing_id")
        recording_text = match.group("recording")
        if include_conditions and condition not in include_conditions:
            continue
        if include_ids and bearing_id not in include_ids:
            continue
        if bearing_id in exclude_ids:
            continue
        if include_recordings and recording_text not in include_recordings:
            continue

        binary_label = _classify_binary_label(
            bearing_id,
            healthy_re=healthy_re,
            unhealthy_re=unhealthy_re,
        )
        if binary_label is None:
            continue

        split = _resolve_split(
            recording_text,
            train_recordings=train_recordings,
            val_recordings=val_recordings,
            test_recordings=test_recordings,
        )
        if split == "unused":
            continue

        inventory_rows.append(
            {
                "path": str(path.resolve()),
                "filename": path.name,
                "condition": condition,
                "bearing_id": bearing_id,
                "recording": int(recording_text),
                "binary_label": binary_label,
                "split": split,
            }
        )

    _validate_inventory_rows(inventory_rows)
    return inventory_rows


def _resolve_inventory_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    inventory_rows = _build_inventory_rows(profile)
    temporal_feature_config = load_temporal_feature_config(profile)
    if not temporal_feature_config.enabled:
        return inventory_rows
    filtered_rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        try:
            build_temporal_feature_payload(
                row["path"],
                config=temporal_feature_config,
            )
        except ContractValidationError:
            continue
        filtered_rows.append(row)
    _validate_inventory_rows(filtered_rows)
    return filtered_rows


def _classify_binary_label(
    bearing_id: str,
    *,
    healthy_re: re.Pattern[str],
    unhealthy_re: re.Pattern[str],
) -> str | None:
    if healthy_re.match(bearing_id):
        return "healthy"
    if unhealthy_re.match(bearing_id):
        return "unhealthy"
    return None


def _resolve_split(
    recording_text: str,
    *,
    train_recordings: set[str],
    val_recordings: set[str],
    test_recordings: set[str],
) -> str:
    if recording_text in train_recordings:
        return "train"
    if recording_text in val_recordings:
        return "val"
    if recording_text in test_recordings:
        return "test"
    return "unused"


def _validate_inventory_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ContractValidationError("The consumer profile did not resolve any usable `.mat` records.")

    split_class_counts: dict[tuple[str, str], int] = Counter(
        (str(row["split"]), str(row["binary_label"])) for row in rows
    )
    required_pairs = (
        ("train", "healthy"),
        ("train", "unhealthy"),
        ("val", "healthy"),
        ("val", "unhealthy"),
        ("test", "healthy"),
        ("test", "unhealthy"),
    )
    missing_pairs = [
        f"{split}:{binary_label}"
        for split, binary_label in required_pairs
        if split_class_counts.get((split, binary_label), 0) <= 0
    ]
    if missing_pairs:
        raise ContractValidationError(
            "The full binary campaign requires healthy and unhealthy examples in every canonical split; "
            f"missing {', '.join(missing_pairs)}."
        )


def _select_smoke_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["split"]),
            str(item["binary_label"]),
            str(item["condition"]),
            str(item["bearing_id"]),
            int(item["recording"]),
            str(item["filename"]),
        ),
    ):
        key = (str(row["split"]), str(row["binary_label"]), str(row["condition"]))
        selected_by_group.setdefault(key, row)

    smoke_rows = list(
        sorted(
            selected_by_group.values(),
            key=lambda item: (
                str(item["split"]),
                str(item["binary_label"]),
                str(item["condition"]),
                str(item["bearing_id"]),
                int(item["recording"]),
                str(item["filename"]),
            ),
        )
    )
    _validate_inventory_rows(smoke_rows)
    return smoke_rows


def _materialize_source_bundle(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    profile_name: str,
    selection_name: str,
    temporal_feature_config,
):
    records = tuple(
        WaveformDatasetRecord(
            source_record_id=_source_record_id(row),
            split=str(row["split"]),
            state_label=str(row["binary_label"]),
            waveforms={
                _WAVEFORM_CHANNEL_NAME: WaveformPayloadRef(
                    waveform_path=str(row["path"]),
                    metadata={
                        "source_format": "mat",
                        "filename": str(row["filename"]),
                    },
                )
            },
            label_metadata={
                "binary_label": str(row["binary_label"]),
                "class_role": str(row["binary_label"]),
            },
            operating_condition=str(row["condition"]),
            context_metadata=_context_metadata_for_row(
                row,
                temporal_feature_config=temporal_feature_config,
            ),
            lineage_metadata={
                "consumer_project": PROJECT_ROOT.name,
                "profile_name": profile_name,
                "selection_name": selection_name,
            },
        )
        for row in rows
    )
    return build_waveform_dataset_bundle(
        records,
        output_dir=output_dir,
        dataset_id=profile_name,
        adapter_profile_id=_ADAPTER_PROFILE_ID,
    )


def _source_record_id(row: dict[str, Any]) -> str:
    return (
        f"{row['condition']}__{row['bearing_id']}__r{int(row['recording']):02d}"
    )


def _context_metadata_for_row(
    row: dict[str, Any],
    *,
    temporal_feature_config,
) -> dict[str, Any]:
    context_metadata: dict[str, Any] = {
        "bearing_id": str(row["bearing_id"]),
        "recording": int(row["recording"]),
        "source_filename": str(row["filename"]),
    }
    if temporal_feature_config.enabled:
        context_metadata["temporal_features"] = build_temporal_feature_payload(
            row["path"],
            config=temporal_feature_config,
        )
    return context_metadata


def _resolve_max_selected_k_per_class(profile: dict[str, Any]) -> int:
    raw_value = profile["deploy_constraints"].get("max_selected_k_per_class")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
        raise ContractValidationError(
            "`deploy_constraints.max_selected_k_per_class` must be an integer >= 1."
        )
    return raw_value


def _device_agnostic_export(
    profile: dict[str, Any],
    *,
    k_candidates: tuple[int, ...],
) -> dict[str, object]:
    deploy_constraints = dict(profile.get("deploy_constraints", {}))
    ranking_intent = dict(profile.get("ranking_intent", {}))
    return {
        "portable": True,
        "execution_device": None,
        "hardware_binding": None,
        "consumer_profile_name": str(profile["profile_name"]),
        "target_device": str(deploy_constraints.get("target", "")),
        "max_selected_k_per_class": _resolve_max_selected_k_per_class(profile),
        "capacity_refinement_k_candidates": list(k_candidates),
        "ranking_intent_primary": str(ranking_intent.get("primary", "")),
        "ranking_intent_secondary": str(ranking_intent.get("secondary", "")),
        "ranking_intent_tertiary": str(ranking_intent.get("tertiary", "")),
        "deploy_notes": [str(note) for note in deploy_constraints.get("notes", ())],
    }


def _objective_bias_notes(profile: dict[str, Any]) -> tuple[str, ...]:
    ranking_intent = dict(profile.get("ranking_intent", {}))
    return (
        f"ranking_primary={ranking_intent.get('primary', '')}",
        f"ranking_secondary={ranking_intent.get('secondary', '')}",
        f"ranking_tertiary={ranking_intent.get('tertiary', '')}",
    )


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "campaign"


def _canonical_stage_output_dir(
    run_root: Path,
    *,
    stage_key: StageKey,
) -> Path:
    stage_sequence = canonical_stage_sequence()
    return run_root / f"{stage_sequence.index(stage_key) + 1:02d}_{stage_key.value}"


def _locked_frontend_stage_notes(
    locked_frontend,
) -> tuple[str, ...]:
    if locked_frontend is None:
        return ()
    return (
        f"frontend_lock_regime={locked_frontend.regime_id}",
        f"frontend_lock_bit_length={locked_frontend.bit_length}",
        "frontend_sweep=false",
        f"frontend_lock_selection_source={locked_frontend.selection_source}",
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RUNS_ROOT",
    "PreparedFullBinaryCampaign",
    "build_campaign_request",
    "build_stage_configs",
    "load_consumer_config",
    "prepare_full_binary_campaign",
    "write_campaign_request_json",
]
