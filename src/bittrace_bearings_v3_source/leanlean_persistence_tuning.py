"""Consumer-side persistence replay/tuning for the Lean-Lean deployment candidate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.v3 import ContractValidationError

from ._leanlean_support import _load_lean_artifact_model, _predict_lean_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "persistence_quiet_scout.yaml"
DEFAULT_WINDOW_OUTPUTS_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-window-outputs-1"
DEFAULT_SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-persistence-tuning-summary-2"

_VALID_SPLITS = ("train", "val", "test")
_LABEL_TO_INT = {"healthy": 0, "unhealthy": 1}
_INT_TO_LABEL = {value: key for key, value in _LABEL_TO_INT.items()}
_STATE_GREEN = "GREEN"
_STATE_YELLOW = "YELLOW"
_STATE_RED = "RED"


@dataclass(frozen=True, slots=True)
class FaultCounterPolicy:
    policy_id: str
    increment_on_unhealthy: int
    decrement_on_healthy: int
    yellow_threshold: int
    red_threshold: int
    optional_latch: bool
    latch_clear_threshold: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "increment_on_unhealthy": self.increment_on_unhealthy,
            "decrement_on_healthy": self.decrement_on_healthy,
            "yellow_threshold": self.yellow_threshold,
            "red_threshold": self.red_threshold,
            "optional_latch": self.optional_latch,
            "latch_clear_threshold": self.latch_clear_threshold,
        }

    @property
    def complexity_score(self) -> int:
        score = 0
        if self.optional_latch:
            score += 3
        if self.latch_clear_threshold is not None:
            score += 1
        score += max(0, self.increment_on_unhealthy - 1)
        score += max(0, self.decrement_on_healthy - 1)
        return score


@dataclass(frozen=True, slots=True)
class PersistenceSelectionPolicy:
    minimum_unhealthy_detection_rate: float
    ranking_mode: str = "quiet_scout"
    require_positive_unhealthy_detection: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_unhealthy_detection_rate": self.minimum_unhealthy_detection_rate,
            "ranking_mode": self.ranking_mode,
            "require_positive_unhealthy_detection": self.require_positive_unhealthy_detection,
        }


@dataclass(frozen=True, slots=True)
class PolicyComparisonSpec:
    comparison_label: str
    policy_role: str
    policy: FaultCounterPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_label": self.comparison_label,
            "policy_role": self.policy_role,
            "policy": self.policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PersistenceTuningConfig:
    config_path: Path
    profile_name: str
    deployment_candidate_config_path: Path
    source_deployment_run_root: Path
    window_output_template_name: str
    window_output_materialized_name: str
    split_scope: tuple[str, ...]
    output_fields: tuple[str, ...]
    policy_candidates: tuple[FaultCounterPolicy, ...]
    comparison_policies: tuple[PolicyComparisonSpec, ...]
    default_policy: FaultCounterPolicy
    selection_policy: PersistenceSelectionPolicy
    tuning_dirname: str
    summary_json_name: str
    summary_csv_name: str
    summary_md_name: str
    selected_policy_json_name: str
    example_traces_json_name: str
    per_policy_dirname: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedLeanLeanPersistenceTuning:
    config: PersistenceTuningConfig
    source_run_root: Path
    run_root: Path
    source_summary_path: Path
    source_plan_path: Path
    source_scaffold_path: Path | None
    source_template_window_outputs_path: Path | None
    source_materialized_window_outputs_path: Path


@dataclass(frozen=True, slots=True)
class WindowOutputRecord:
    source_record_id: str
    split: str
    actual_label: str
    predicted_label: str
    prediction_margin: int
    sequence_id: str
    split_sequence_id: str
    operating_condition: str
    bearing_id: str
    recording_index: int
    sequence_position: int
    frontend_input_id: str
    frontend_fingerprint: str
    semantic_bit_length: int
    packed_bit_length: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_record_id": self.source_record_id,
            "split": self.split,
            "actual_label": self.actual_label,
            "predicted_label": self.predicted_label,
            "prediction_margin": self.prediction_margin,
            "sequence_id": self.sequence_id,
            "split_sequence_id": self.split_sequence_id,
            "operating_condition": self.operating_condition,
            "bearing_id": self.bearing_id,
            "recording_index": self.recording_index,
            "sequence_position": self.sequence_position,
            "frontend_input_id": self.frontend_input_id,
            "frontend_fingerprint": self.frontend_fingerprint,
            "semantic_bit_length": self.semantic_bit_length,
            "packed_bit_length": self.packed_bit_length,
        }


def load_leanlean_persistence_tuning_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PersistenceTuningConfig:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    profile_name = _require_non_empty_string(payload.get("profile_name"), field_name="profile_name")
    deployment_candidate_config_path = _resolve_relative_path(
        config_file=resolved_path,
        raw_path=payload.get("deployment_candidate_config"),
        field_name="deployment_candidate_config",
    )
    source_deployment_run_root = _resolve_relative_path(
        config_file=resolved_path,
        raw_path=payload.get("source_deployment_run_root"),
        field_name="source_deployment_run_root",
    )
    window_outputs = _require_mapping(payload.get("window_outputs"), field_name="window_outputs")
    split_scope = _require_string_sequence(
        window_outputs.get("split_scope", _VALID_SPLITS),
        field_name="window_outputs.split_scope",
    )
    invalid_splits = [split for split in split_scope if split not in _VALID_SPLITS]
    if invalid_splits:
        raise ContractValidationError(
            "`window_outputs.split_scope` supports only train/val/test; "
            f"received {', '.join(invalid_splits)}."
        )
    output_fields = _require_string_sequence(
        window_outputs.get("fields"),
        field_name="window_outputs.fields",
    )
    if not output_fields:
        raise ContractValidationError("`window_outputs.fields` must include at least one field.")
    default_policy = _policy_from_mapping(
        _require_mapping(payload.get("fault_counter_policy"), field_name="fault_counter_policy"),
        field_name="fault_counter_policy",
        default_policy_id="default_fault_counter",
    )
    policy_candidates = _parse_policy_candidates(payload, default_policy=default_policy)
    comparison_policies = _parse_comparison_policies(payload)
    selection_policy_payload = _require_mapping(
        payload.get("selection_policy", {}),
        field_name="selection_policy",
    )
    planned_outputs = _require_mapping(payload.get("planned_outputs", {}), field_name="planned_outputs")
    return PersistenceTuningConfig(
        config_path=resolved_path,
        profile_name=profile_name,
        deployment_candidate_config_path=deployment_candidate_config_path,
        source_deployment_run_root=source_deployment_run_root,
        window_output_template_name=_require_non_empty_string(
            window_outputs.get("artifact_name", "leanlean_window_outputs_template.json"),
            field_name="window_outputs.artifact_name",
        ),
        window_output_materialized_name=_require_non_empty_string(
            window_outputs.get("materialized_artifact_name", "leanlean_window_outputs.json"),
            field_name="window_outputs.materialized_artifact_name",
        ),
        split_scope=tuple(split_scope),
        output_fields=tuple(output_fields),
        policy_candidates=tuple(policy_candidates),
        comparison_policies=tuple(comparison_policies),
        default_policy=default_policy,
        selection_policy=PersistenceSelectionPolicy(
            minimum_unhealthy_detection_rate=_require_probability(
                selection_policy_payload.get("minimum_unhealthy_detection_rate", 0.40),
                field_name="selection_policy.minimum_unhealthy_detection_rate",
            ),
            ranking_mode=_require_selection_ranking_mode(
                selection_policy_payload.get("ranking_mode", "quiet_scout"),
                field_name="selection_policy.ranking_mode",
            ),
            require_positive_unhealthy_detection=_require_bool(
                selection_policy_payload.get("require_positive_unhealthy_detection", True),
                field_name="selection_policy.require_positive_unhealthy_detection",
            ),
        ),
        tuning_dirname=_require_non_empty_string(
            planned_outputs.get("tuning_dirname", "persistence_tuning"),
            field_name="planned_outputs.tuning_dirname",
        ),
        summary_json_name=_require_non_empty_string(
            planned_outputs.get("summary_json_name", "persistence_tuning_summary.json"),
            field_name="planned_outputs.summary_json_name",
        ),
        summary_csv_name=_require_non_empty_string(
            planned_outputs.get("summary_csv_name", "persistence_tuning_summary.csv"),
            field_name="planned_outputs.summary_csv_name",
        ),
        summary_md_name=_require_non_empty_string(
            planned_outputs.get("summary_md_name", "persistence_tuning_summary.md"),
            field_name="planned_outputs.summary_md_name",
        ),
        selected_policy_json_name=_require_non_empty_string(
            planned_outputs.get("selected_policy_json_name", "selected_persistence_policy.json"),
            field_name="planned_outputs.selected_policy_json_name",
        ),
        example_traces_json_name=_require_non_empty_string(
            planned_outputs.get("example_traces_json_name", "example_replay_traces.json"),
            field_name="planned_outputs.example_traces_json_name",
        ),
        per_policy_dirname=_require_non_empty_string(
            planned_outputs.get("per_policy_dirname", "per_policy"),
            field_name="planned_outputs.per_policy_dirname",
        ),
        notes=tuple(_require_string_sequence(payload.get("notes", ()), field_name="notes")),
    )


def prepare_leanlean_persistence_tuning(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    run_id: str,
    source_run_root: str | Path | None = None,
) -> PreparedLeanLeanPersistenceTuning:
    config = load_leanlean_persistence_tuning_config(config_path)
    resolved_source_run_root = (
        Path(source_run_root).resolve()
        if source_run_root is not None
        else config.source_deployment_run_root.resolve()
    )
    if not resolved_source_run_root.is_dir():
        raise ContractValidationError(
            f"Lean-Lean persistence tuning source run root does not exist: {resolved_source_run_root}"
        )
    run_root = (resolved_source_run_root / config.tuning_dirname / _require_non_empty_string(run_id, field_name="run_id")).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractValidationError(f"Persistence tuning run root already exists and is not empty: {run_root}")
    source_summary_path = resolved_source_run_root / "leanlean_deployment_candidate_summary.json"
    source_plan_path = resolved_source_run_root / "leanlean_deployment_candidate_plan.json"
    if not source_summary_path.is_file():
        raise ContractValidationError(f"Missing deployment summary at `{source_summary_path}`.")
    if not source_plan_path.is_file():
        raise ContractValidationError(f"Missing deployment plan at `{source_plan_path}`.")
    source_scaffold_path = resolved_source_run_root / "persistence_prep" / "leanlean_persistence_tuning_prep.json"
    if not source_scaffold_path.is_file():
        source_scaffold_path = None
    source_template_window_outputs_path = (
        resolved_source_run_root / "persistence_prep" / config.window_output_template_name
    )
    if not source_template_window_outputs_path.is_file():
        source_template_window_outputs_path = None
    source_materialized_window_outputs_path = (
        resolved_source_run_root / "persistence_prep" / config.window_output_materialized_name
    )
    return PreparedLeanLeanPersistenceTuning(
        config=config,
        source_run_root=resolved_source_run_root,
        run_root=run_root,
        source_summary_path=source_summary_path,
        source_plan_path=source_plan_path,
        source_scaffold_path=source_scaffold_path,
        source_template_window_outputs_path=source_template_window_outputs_path,
        source_materialized_window_outputs_path=source_materialized_window_outputs_path,
    )


def run_leanlean_persistence_tuning(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    run_id: str,
    source_run_root: str | Path | None = None,
    force_rematerialize_window_outputs: bool = False,
) -> Path:
    prepared = prepare_leanlean_persistence_tuning(
        config_path=config_path,
        run_id=run_id,
        source_run_root=source_run_root,
    )
    return run_prepared_leanlean_persistence_tuning(
        prepared,
        force_rematerialize_window_outputs=force_rematerialize_window_outputs,
    )


def run_prepared_leanlean_persistence_tuning(
    prepared: PreparedLeanLeanPersistenceTuning,
    *,
    force_rematerialize_window_outputs: bool = False,
) -> Path:
    prepared.run_root.mkdir(parents=True, exist_ok=True)
    raw_summary = _load_json(prepared.source_summary_path)
    raw_plan = _load_json(prepared.source_plan_path)
    window_outputs_path = ensure_window_outputs_materialized(
        prepared,
        raw_summary=raw_summary,
        raw_plan=raw_plan,
        force_rematerialize=force_rematerialize_window_outputs,
    )
    window_records = _load_window_output_records(window_outputs_path)
    sequences_by_split = _group_sequences_by_split(window_records)
    selection_sequences = list(sequences_by_split["train"]) + list(sequences_by_split["val"])
    policy_results: list[dict[str, object]] = []
    comparison_results: list[dict[str, object]] = []
    per_policy_dir = prepared.run_root / prepared.config.per_policy_dirname
    per_policy_dir.mkdir(parents=True, exist_ok=True)
    for policy in prepared.config.policy_candidates:
        scope_results = _evaluate_policy_scopes(
            sequences_by_split=sequences_by_split,
            selection_sequences=selection_sequences,
            policy=policy,
        )
        policy_path = per_policy_dir / f"{policy.policy_id}.json"
        policy_results.append(
            {
                "policy": policy,
                "policy_path": policy_path,
                "metrics_by_scope": scope_results,
                "selection_decision": _evaluate_selection_policy(
                    scope_results["train_val_selection"],
                    policy,
                    prepared.config.selection_policy,
                ),
            }
        )
    for comparison_spec in prepared.config.comparison_policies:
        comparison_results.append(
            {
                "comparison_label": comparison_spec.comparison_label,
                "policy_role": comparison_spec.policy_role,
                "policy": comparison_spec.policy,
                "metrics_by_scope": _evaluate_policy_scopes(
                    sequences_by_split=sequences_by_split,
                    selection_sequences=selection_sequences,
                    policy=comparison_spec.policy,
                ),
            }
        )
    survivor_results = [
        result for result in policy_results if bool(result["selection_decision"]["passes_unhealthy_detection_floor"])
    ]
    if not survivor_results:
        best_rejected = max(
            policy_results,
            key=lambda item: float(
                item["selection_decision"]["measured_unhealthy_detection_rate"]
            ),
        )
        raise ContractValidationError(
            "No persistence policy passed the configured train/val unhealthy-detection floor "
            f"of {prepared.config.selection_policy.minimum_unhealthy_detection_rate:.6f}. "
            "Best rejected candidate was "
            f"`{best_rejected['policy'].policy_id}` at "
            f"{float(best_rejected['selection_decision']['measured_unhealthy_detection_rate']):.6f}."
        )
    rejected_results = [
        result for result in policy_results if not bool(result["selection_decision"]["passes_unhealthy_detection_floor"])
    ]
    ranked_survivors = sorted(
        survivor_results,
        key=lambda item: _selection_ranking_key(
            item["metrics_by_scope"]["train_val_selection"],
            item["policy"],
            prepared.config.selection_policy,
        ),
    )
    ranked_rejected = sorted(
        rejected_results,
        key=lambda item: _rejected_policy_sort_key(item["metrics_by_scope"]["train_val_selection"], item["policy"]),
    )
    ranked_results = ranked_survivors + ranked_rejected
    for index, item in enumerate(ranked_results, start=1):
        item["selection_rank"] = index
    for index, item in enumerate(ranked_survivors, start=1):
        item["survivor_rank"] = index
    for item in ranked_rejected:
        item["survivor_rank"] = None
    selected = ranked_survivors[0]
    for item in ranked_results:
        item["selection_decision"] = _finalize_selection_decision(
            selection_decision=_require_mapping(
                item["selection_decision"],
                field_name="selection_decision",
            ),
            selection_scope=_require_mapping(
                item["metrics_by_scope"]["train_val_selection"],
                field_name="metrics_by_scope.train_val_selection",
            ),
            policy=item["policy"],
            selection_rank=int(item["selection_rank"]),
            survivor_rank=item["survivor_rank"],
            survivor_count=len(ranked_survivors),
            selected_policy_id=selected["policy"].policy_id,
        )
        ranking_key = (
            _selection_ranking_key(
                item["metrics_by_scope"]["train_val_selection"],
                item["policy"],
                prepared.config.selection_policy,
            )
            if bool(item["selection_decision"]["passes_unhealthy_detection_floor"])
            else _rejected_policy_sort_key(item["metrics_by_scope"]["train_val_selection"], item["policy"])
        )
        item["ranking_key"] = ranking_key
        _write_json(
            Path(item["policy_path"]),
            {
                "policy": item["policy"].to_dict(),
                "selection_decision": item["selection_decision"],
                "ranking_key": list(ranking_key),
                "metrics_by_scope": item["metrics_by_scope"],
            },
        )
    selected_examples = _build_example_traces(
        sequences_by_split=sequences_by_split,
        policy=selected["policy"],
        per_split_examples=2,
    )
    example_traces_path = prepared.run_root / prepared.config.example_traces_json_name
    selected_vs_comparisons = _build_selected_policy_comparisons(
        selected_result=selected,
        comparison_results=comparison_results,
    )
    _write_json(
        example_traces_path,
        {
            "selected_aggressive_policy": {
                "policy": selected["policy"].to_dict(),
                "example_traces": selected_examples,
            },
            "comparison_policies": [
                {
                    "comparison_label": result["comparison_label"],
                    "policy_role": result["policy_role"],
                    "policy": result["policy"].to_dict(),
                    "example_traces": _build_example_traces(
                        sequences_by_split=sequences_by_split,
                        policy=result["policy"],
                        per_split_examples=2,
                    ),
                }
                for result in comparison_results
            ],
        },
    )
    selected_policy_path = prepared.run_root / prepared.config.selected_policy_json_name
    _write_json(
        selected_policy_path,
        {
            "policy": selected["policy"].to_dict(),
            "selection_rank": selected["selection_rank"],
            "survivor_rank": selected["survivor_rank"],
            "selection_scope": "train_val_selection",
            "selection_policy": prepared.config.selection_policy.to_dict(),
            "selection_decision": selected["selection_decision"],
            "selection_metrics": selected["metrics_by_scope"]["train_val_selection"],
            "test_metrics": selected["metrics_by_scope"]["test"],
            "selected_vs_comparisons": selected_vs_comparisons,
            "source_deployment_run_root": str(prepared.source_run_root),
            "source_window_outputs_path": str(window_outputs_path.resolve()),
        },
    )
    summary_json_path = prepared.run_root / prepared.config.summary_json_name
    summary_csv_path = prepared.run_root / prepared.config.summary_csv_name
    summary_md_path = prepared.run_root / prepared.config.summary_md_name
    summary_payload = _build_summary_payload(
        prepared=prepared,
        raw_summary=raw_summary,
        raw_plan=raw_plan,
        window_outputs_path=window_outputs_path,
        ranked_results=ranked_results,
        comparison_results=comparison_results,
        selected_vs_comparisons=selected_vs_comparisons,
        selected_policy_path=selected_policy_path,
        example_traces_path=example_traces_path,
    )
    _write_json(summary_json_path, summary_payload)
    _write_summary_csv(summary_csv_path, ranked_results, comparison_results)
    summary_md_path.write_text(
        _build_summary_markdown(
            prepared=prepared,
            raw_summary=raw_summary,
            ranked_results=ranked_results,
            comparison_results=comparison_results,
            selected_vs_comparisons=selected_vs_comparisons,
            window_outputs_path=window_outputs_path,
        ),
        encoding="utf-8",
    )
    return summary_json_path


def ensure_window_outputs_materialized(
    prepared: PreparedLeanLeanPersistenceTuning,
    *,
    raw_summary: Mapping[str, object],
    raw_plan: Mapping[str, object],
    force_rematerialize: bool,
) -> Path:
    existing_path = _resolve_existing_window_outputs_path(prepared)
    if existing_path is not None and not force_rematerialize:
        return existing_path
    materialized_path = prepared.source_materialized_window_outputs_path
    records = _materialize_window_output_records(prepared, raw_summary=raw_summary)
    payload = {
        "schema_version": DEFAULT_WINDOW_OUTPUTS_SCHEMA_VERSION,
        "profile_name": prepared.config.profile_name,
        "source_deployment_run_root": str(prepared.source_run_root),
        "source_deployment_summary_path": str(prepared.source_summary_path.resolve()),
        "source_deployment_plan_path": str(prepared.source_plan_path.resolve()),
        "classifier_variant": str(raw_summary.get("variant", {}).get("variant_id", "")),
        "classification_scope": "per-record Lean-Lean predictions replayable by persistence tuning",
        "split_scope": list(prepared.config.split_scope),
        "fields": list(prepared.config.output_fields),
        "record_count": len(records),
        "records": [record.to_dict() for record in records],
    }
    _write_json(materialized_path, payload)
    if prepared.source_scaffold_path is not None:
        scaffold_payload = _load_json(prepared.source_scaffold_path)
        if isinstance(scaffold_payload, Mapping):
            updated = dict(scaffold_payload)
            planned_outputs = dict(updated.get("planned_outputs", {}))
            planned_outputs["materialized_window_outputs_path"] = str(materialized_path.resolve())
            updated["planned_outputs"] = planned_outputs
            updated["materialized_window_outputs_path"] = str(materialized_path.resolve())
            updated["materialized_window_output_record_count"] = len(records)
            updated["source_deployment_plan_path"] = str(prepared.source_plan_path.resolve())
            updated["selection_intent"] = {
                "selection_scope": "train_val_selection",
                "test_scope": "report_only",
            }
            _write_json(prepared.source_scaffold_path, updated)
    return materialized_path


def _materialize_window_output_records(
    prepared: PreparedLeanLeanPersistenceTuning,
    *,
    raw_summary: Mapping[str, object],
) -> list[WindowOutputRecord]:
    variant = _require_mapping(raw_summary.get("variant"), field_name="variant")
    artifact_path = Path(
        _require_non_empty_string(variant.get("artifact_path"), field_name="variant.artifact_path")
    ).resolve()
    model = _load_lean_artifact_model(artifact_path)
    shared_bundle = prepared.source_run_root / "02_shared_backend_bundle"
    contract_payload = _load_json(shared_bundle / "contract.json")
    label_mapping = _require_mapping(contract_payload.get("label_mapping"), field_name="label_mapping")
    label_from_int = {
        int(value): key
        for key, value in ((str(key), int(value)) for key, value in label_mapping.items())
    }
    frontend_input_id = _require_non_empty_string(
        contract_payload.get("frontend_input_id"),
        field_name="frontend_input_id",
    )
    frontend_fingerprint = _require_non_empty_string(
        contract_payload.get("frontend_fingerprint"),
        field_name="frontend_fingerprint",
    )
    bundle_materialization = _require_mapping(
        contract_payload.get("bundle_materialization"),
        field_name="bundle_materialization",
    )
    semantic_bit_length = _require_int(
        bundle_materialization.get("semantic_bit_length"),
        field_name="bundle_materialization.semantic_bit_length",
        minimum=1,
    )
    packed_bit_length = _require_int(
        contract_payload.get("bit_length"),
        field_name="bit_length",
        minimum=1,
    )
    records: list[WindowOutputRecord] = []
    for split_name in prepared.config.split_scope:
        split_payload = _load_json(shared_bundle / f"{split_name}_bits.json")
        rows = tuple(int(value) for value in split_payload.get("X_packed", ()))
        actual_labels = tuple(int(value) for value in split_payload.get("y", ()))
        source_record_ids = tuple(str(value) for value in split_payload.get("source_record_ids", ()))
        if not rows or len(rows) != len(actual_labels) or len(rows) != len(source_record_ids):
            raise ContractValidationError(
                f"Split payload `{split_name}` must contain aligned X_packed/y/source_record_ids lists."
            )
        predictions, margins = _predict_lean_rows(rows, model, bit_length=packed_bit_length)
        for source_record_id, actual_int, predicted_int, margin in zip(
            source_record_ids,
            actual_labels,
            predictions,
            margins,
            strict=True,
        ):
            parsed = _parse_source_record_id(source_record_id)
            actual_label = label_from_int.get(int(actual_int))
            predicted_label = label_from_int.get(int(predicted_int))
            if actual_label is None or predicted_label is None:
                raise ContractValidationError(
                    f"Unsupported label mapping for source record `{source_record_id}`."
                )
            sequence_id = f"{parsed['operating_condition']}__{parsed['bearing_id']}"
            records.append(
                WindowOutputRecord(
                    source_record_id=source_record_id,
                    split=split_name,
                    actual_label=actual_label,
                    predicted_label=predicted_label,
                    prediction_margin=int(margin),
                    sequence_id=sequence_id,
                    split_sequence_id=f"{split_name}::{sequence_id}",
                    operating_condition=parsed["operating_condition"],
                    bearing_id=parsed["bearing_id"],
                    recording_index=int(parsed["recording_index"]),
                    sequence_position=0,
                    frontend_input_id=frontend_input_id,
                    frontend_fingerprint=frontend_fingerprint,
                    semantic_bit_length=semantic_bit_length,
                    packed_bit_length=packed_bit_length,
                )
            )
    return _assign_sequence_positions(records)


def _assign_sequence_positions(records: Sequence[WindowOutputRecord]) -> list[WindowOutputRecord]:
    grouped: dict[str, list[WindowOutputRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split_sequence_id].append(record)
    normalized: list[WindowOutputRecord] = []
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            _VALID_SPLITS.index(key.split("::", 1)[0]),
            key.split("::", 1)[1],
        ),
    )
    for split_sequence_id in ordered_keys:
        ordered = sorted(grouped[split_sequence_id], key=lambda item: (item.recording_index, item.source_record_id))
        for index, record in enumerate(ordered, start=1):
            normalized.append(
                WindowOutputRecord(
                    source_record_id=record.source_record_id,
                    split=record.split,
                    actual_label=record.actual_label,
                    predicted_label=record.predicted_label,
                    prediction_margin=record.prediction_margin,
                    sequence_id=record.sequence_id,
                    split_sequence_id=record.split_sequence_id,
                    operating_condition=record.operating_condition,
                    bearing_id=record.bearing_id,
                    recording_index=record.recording_index,
                    sequence_position=index,
                    frontend_input_id=record.frontend_input_id,
                    frontend_fingerprint=record.frontend_fingerprint,
                    semantic_bit_length=record.semantic_bit_length,
                    packed_bit_length=record.packed_bit_length,
                )
            )
    return normalized


def _resolve_existing_window_outputs_path(
    prepared: PreparedLeanLeanPersistenceTuning,
) -> Path | None:
    candidate_paths = [
        prepared.source_materialized_window_outputs_path,
        prepared.source_template_window_outputs_path,
    ]
    for candidate in candidate_paths:
        if candidate is None or not candidate.is_file():
            continue
        payload = _load_json(candidate)
        records = payload.get("records")
        if isinstance(records, list) and records:
            return candidate
    return None


def _load_window_output_records(path: Path) -> list[WindowOutputRecord]:
    payload = _load_json(path)
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ContractValidationError(f"`{path}` must contain non-empty `records`.")
    records: list[WindowOutputRecord] = []
    for index, raw_record in enumerate(raw_records):
        record = _require_mapping(raw_record, field_name=f"records[{index}]")
        records.append(
            WindowOutputRecord(
                source_record_id=_require_non_empty_string(
                    record.get("source_record_id"),
                    field_name=f"records[{index}].source_record_id",
                ),
                split=_require_split_name(record.get("split"), field_name=f"records[{index}].split"),
                actual_label=_require_label(record.get("actual_label"), field_name=f"records[{index}].actual_label"),
                predicted_label=_require_label(
                    record.get("predicted_label"),
                    field_name=f"records[{index}].predicted_label",
                ),
                prediction_margin=_require_int(
                    record.get("prediction_margin"),
                    field_name=f"records[{index}].prediction_margin",
                    minimum=0,
                ),
                sequence_id=_require_non_empty_string(
                    record.get("sequence_id"),
                    field_name=f"records[{index}].sequence_id",
                ),
                split_sequence_id=_require_non_empty_string(
                    record.get("split_sequence_id"),
                    field_name=f"records[{index}].split_sequence_id",
                ),
                operating_condition=_require_non_empty_string(
                    record.get("operating_condition"),
                    field_name=f"records[{index}].operating_condition",
                ),
                bearing_id=_require_non_empty_string(
                    record.get("bearing_id"),
                    field_name=f"records[{index}].bearing_id",
                ),
                recording_index=_require_int(
                    record.get("recording_index"),
                    field_name=f"records[{index}].recording_index",
                    minimum=1,
                ),
                sequence_position=_require_int(
                    record.get("sequence_position"),
                    field_name=f"records[{index}].sequence_position",
                    minimum=1,
                ),
                frontend_input_id=_require_non_empty_string(
                    record.get("frontend_input_id"),
                    field_name=f"records[{index}].frontend_input_id",
                ),
                frontend_fingerprint=_require_non_empty_string(
                    record.get("frontend_fingerprint"),
                    field_name=f"records[{index}].frontend_fingerprint",
                ),
                semantic_bit_length=_require_int(
                    record.get("semantic_bit_length"),
                    field_name=f"records[{index}].semantic_bit_length",
                    minimum=1,
                ),
                packed_bit_length=_require_int(
                    record.get("packed_bit_length"),
                    field_name=f"records[{index}].packed_bit_length",
                    minimum=1,
                ),
            )
        )
    return records


def _group_sequences_by_split(
    records: Sequence[WindowOutputRecord],
) -> dict[str, list[list[WindowOutputRecord]]]:
    grouped: dict[str, dict[str, list[WindowOutputRecord]]] = {
        split_name: defaultdict(list)
        for split_name in _VALID_SPLITS
    }
    for record in records:
        grouped[record.split][record.split_sequence_id].append(record)
    return {
        split_name: [
            sorted(sequence_records, key=lambda item: (item.sequence_position, item.recording_index, item.source_record_id))
            for _, sequence_records in sorted(split_group.items())
        ]
        for split_name, split_group in grouped.items()
    }


def _evaluate_policy_scopes(
    *,
    sequences_by_split: Mapping[str, Sequence[Sequence[WindowOutputRecord]]],
    selection_sequences: Sequence[Sequence[WindowOutputRecord]],
    policy: FaultCounterPolicy,
) -> dict[str, object]:
    return {
        "train": _evaluate_scope("train", sequences_by_split["train"], policy),
        "val": _evaluate_scope("val", sequences_by_split["val"], policy),
        "train_val_selection": _evaluate_scope("train_val_selection", selection_sequences, policy),
        "test": _evaluate_scope("test", sequences_by_split["test"], policy),
    }


def _evaluate_scope(
    scope_name: str,
    sequences: Sequence[Sequence[WindowOutputRecord]],
    policy: FaultCounterPolicy,
) -> dict[str, object]:
    sequence_summaries: list[dict[str, object]] = []
    healthy_sequences: list[dict[str, object]] = []
    unhealthy_sequences: list[dict[str, object]] = []
    for sequence in sequences:
        summary = _replay_sequence(sequence, policy)
        sequence_summaries.append(summary)
        if summary["actual_label"] == "healthy":
            healthy_sequences.append(summary)
        else:
            unhealthy_sequences.append(summary)
    return {
        "scope": scope_name,
        "policy": policy.to_dict(),
        "sequence_count": len(sequence_summaries),
        "healthy_sequence_count": len(healthy_sequences),
        "unhealthy_sequence_count": len(unhealthy_sequences),
        "healthy_metrics": _aggregate_class_metrics(healthy_sequences, class_label="healthy"),
        "unhealthy_metrics": _aggregate_class_metrics(unhealthy_sequences, class_label="unhealthy"),
        "all_sequences_metrics": _aggregate_all_sequence_metrics(sequence_summaries),
        "sequence_summaries": sequence_summaries,
    }


def _replay_sequence(
    sequence: Sequence[WindowOutputRecord],
    policy: FaultCounterPolicy,
) -> dict[str, object]:
    if not sequence:
        raise ContractValidationError("Persistence replay requires non-empty sequences.")
    actual_label = sequence[0].actual_label
    if any(record.actual_label != actual_label for record in sequence):
        raise ContractValidationError(
            f"Sequence `{sequence[0].split_sequence_id}` contains mixed actual labels."
        )
    counter = 0
    red_latched = False
    steps: list[dict[str, object]] = []
    state_counts = {_STATE_GREEN: 0, _STATE_YELLOW: 0, _STATE_RED: 0}
    alarm_bursts = 0
    red_bursts = 0
    latch_events = 0
    clear_events = 0
    red_clear_events = 0
    windows_to_clear_after_alarm: list[int] = []
    windows_to_clear_after_red: list[int] = []
    current_alarm_burst_start: int | None = None
    current_red_burst_start: int | None = None
    time_to_first_yellow: int | None = None
    time_to_first_red: int | None = None
    previous_state = _STATE_GREEN
    for index, record in enumerate(sequence, start=1):
        if record.predicted_label == "unhealthy":
            counter += policy.increment_on_unhealthy
        else:
            counter = max(0, counter - policy.decrement_on_healthy)
        latch_engaged_this_step = False
        if red_latched and policy.latch_clear_threshold is not None and counter <= policy.latch_clear_threshold:
            red_latched = False
        if red_latched:
            state = _STATE_RED
        elif counter >= policy.red_threshold:
            state = _STATE_RED
            if policy.optional_latch:
                red_latched = True
                latch_events += 1
                latch_engaged_this_step = True
        elif counter >= policy.yellow_threshold:
            state = _STATE_YELLOW
        else:
            state = _STATE_GREEN
        if state != _STATE_GREEN and previous_state == _STATE_GREEN:
            alarm_bursts += 1
            current_alarm_burst_start = index
        if state == _STATE_RED and previous_state != _STATE_RED:
            red_bursts += 1
            current_red_burst_start = index
        if state == _STATE_GREEN and previous_state != _STATE_GREEN:
            clear_events += 1
            if current_alarm_burst_start is not None:
                windows_to_clear_after_alarm.append(index - current_alarm_burst_start)
                current_alarm_burst_start = None
        if state != _STATE_RED and previous_state == _STATE_RED:
            red_clear_events += 1
            if current_red_burst_start is not None:
                windows_to_clear_after_red.append(index - current_red_burst_start)
                current_red_burst_start = None
        if time_to_first_yellow is None and state in {_STATE_YELLOW, _STATE_RED}:
            time_to_first_yellow = index
        if time_to_first_red is None and state == _STATE_RED:
            time_to_first_red = index
        state_counts[state] += 1
        steps.append(
            {
                "sequence_position": record.sequence_position,
                "recording_index": record.recording_index,
                "source_record_id": record.source_record_id,
                "predicted_label": record.predicted_label,
                "prediction_margin": record.prediction_margin,
                "counter_after_update": counter,
                "alarm_state": state,
                "red_latched": red_latched,
                "latch_engaged_this_step": latch_engaged_this_step,
            }
        )
        previous_state = state
    unresolved_alarm_bursts = 1 if current_alarm_burst_start is not None else 0
    unresolved_red_bursts = 1 if current_red_burst_start is not None else 0
    return {
        "split": sequence[0].split,
        "sequence_id": sequence[0].sequence_id,
        "split_sequence_id": sequence[0].split_sequence_id,
        "operating_condition": sequence[0].operating_condition,
        "bearing_id": sequence[0].bearing_id,
        "actual_label": actual_label,
        "n_windows": len(sequence),
        "alarm_bursts": alarm_bursts,
        "red_bursts": red_bursts,
        "non_green_window_count": state_counts[_STATE_YELLOW] + state_counts[_STATE_RED],
        "yellow_window_count": state_counts[_STATE_YELLOW],
        "red_window_count": state_counts[_STATE_RED],
        "state_counts": dict(state_counts),
        "detected": time_to_first_yellow is not None,
        "reached_red": time_to_first_red is not None,
        "time_to_first_yellow_windows": time_to_first_yellow,
        "time_to_first_red_windows": time_to_first_red,
        "latch_events": latch_events,
        "latched_red": latch_events > 0,
        "clear_events_total": clear_events,
        "red_clear_events_total": red_clear_events,
        "mean_windows_to_clear_after_alarm": _mean_or_none(windows_to_clear_after_alarm),
        "mean_windows_to_clear_after_red": _mean_or_none(windows_to_clear_after_red),
        "unresolved_alarm_bursts": unresolved_alarm_bursts,
        "unresolved_red_bursts": unresolved_red_bursts,
        "steps": steps,
    }


def _aggregate_class_metrics(
    sequences: Sequence[Mapping[str, object]],
    *,
    class_label: str,
) -> dict[str, object]:
    total_sequences = len(sequences)
    total_windows = sum(int(sequence["n_windows"]) for sequence in sequences)
    non_green_windows = sum(int(sequence["non_green_window_count"]) for sequence in sequences)
    red_windows = sum(int(sequence["red_window_count"]) for sequence in sequences)
    alarm_bursts = sum(int(sequence["alarm_bursts"]) for sequence in sequences)
    red_bursts = sum(int(sequence["red_bursts"]) for sequence in sequences)
    clear_events = sum(int(sequence["clear_events_total"]) for sequence in sequences)
    unresolved_alarm_bursts = sum(int(sequence["unresolved_alarm_bursts"]) for sequence in sequences)
    red_clear_events = sum(int(sequence["red_clear_events_total"]) for sequence in sequences)
    unresolved_red_bursts = sum(int(sequence["unresolved_red_bursts"]) for sequence in sequences)
    latch_sequences = sum(1 for sequence in sequences if bool(sequence["latched_red"]))
    detection_sequences = sum(1 for sequence in sequences if bool(sequence["detected"]))
    red_sequences = sum(1 for sequence in sequences if bool(sequence["reached_red"]))
    detection_times = [
        int(sequence["time_to_first_yellow_windows"])
        for sequence in sequences
        if sequence["time_to_first_yellow_windows"] is not None
    ]
    red_times = [
        int(sequence["time_to_first_red_windows"])
        for sequence in sequences
        if sequence["time_to_first_red_windows"] is not None
    ]
    clear_durations = [
        float(sequence["mean_windows_to_clear_after_alarm"])
        for sequence in sequences
        if sequence["mean_windows_to_clear_after_alarm"] is not None
    ]
    red_clear_durations = [
        float(sequence["mean_windows_to_clear_after_red"])
        for sequence in sequences
        if sequence["mean_windows_to_clear_after_red"] is not None
    ]
    payload = {
        "class_label": class_label,
        "sequence_count": total_sequences,
        "window_count": total_windows,
        "false_alarm_bursts_total": alarm_bursts if class_label == "healthy" else None,
        "healthy_red_bursts_total": red_bursts if class_label == "healthy" else None,
        "healthy_yellow_or_red_window_fraction": _safe_fraction(non_green_windows, total_windows)
        if class_label == "healthy"
        else None,
        "healthy_red_window_fraction": _safe_fraction(red_windows, total_windows)
        if class_label == "healthy"
        else None,
        "unhealthy_detection_rate": _safe_fraction(detection_sequences, total_sequences)
        if class_label == "unhealthy"
        else None,
        "unhealthy_red_detection_rate": _safe_fraction(red_sequences, total_sequences)
        if class_label == "unhealthy"
        else None,
        "time_to_first_yellow_mean_windows": _mean_or_none(detection_times),
        "time_to_first_yellow_median_windows": _median_or_none(detection_times),
        "time_to_first_red_mean_windows": _mean_or_none(red_times),
        "time_to_first_red_median_windows": _median_or_none(red_times),
        "red_latch_frequency": _safe_fraction(latch_sequences, total_sequences),
        "clear_behavior_summary": {
            "clear_events_total": clear_events,
            "mean_windows_to_clear_after_alarm": _mean_or_none(clear_durations),
            "unresolved_alarm_bursts_total": unresolved_alarm_bursts,
            "red_clear_events_total": red_clear_events,
            "mean_windows_to_clear_after_red": _mean_or_none(red_clear_durations),
            "unresolved_red_bursts_total": unresolved_red_bursts,
        },
    }
    if class_label == "healthy":
        payload["healthy_sequences_with_alarm_rate"] = _safe_fraction(detection_sequences, total_sequences)
        payload["healthy_sequences_with_red_rate"] = _safe_fraction(red_sequences, total_sequences)
    return payload


def _aggregate_all_sequence_metrics(
    sequences: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    total_sequences = len(sequences)
    latch_sequences = sum(1 for sequence in sequences if bool(sequence["latched_red"]))
    return {
        "sequence_count": total_sequences,
        "red_latch_frequency_all_sequences": _safe_fraction(latch_sequences, total_sequences),
    }


def _selection_ranking_key(
    selection_scope: Mapping[str, object],
    policy: FaultCounterPolicy,
    selection_policy: PersistenceSelectionPolicy,
) -> tuple[float | int, ...]:
    healthy = _require_mapping(selection_scope.get("healthy_metrics"), field_name="healthy_metrics")
    unhealthy = _require_mapping(selection_scope.get("unhealthy_metrics"), field_name="unhealthy_metrics")
    healthy_red_bursts = int(healthy.get("healthy_red_bursts_total") or 0)
    healthy_red_fraction = float(healthy.get("healthy_red_window_fraction") or 0.0)
    healthy_alarm_bursts = int(healthy.get("false_alarm_bursts_total") or 0)
    healthy_non_green_fraction = float(healthy.get("healthy_yellow_or_red_window_fraction") or 0.0)
    unhealthy_detection_rate = float(unhealthy.get("unhealthy_detection_rate") or 0.0)
    unhealthy_red_rate = float(unhealthy.get("unhealthy_red_detection_rate") or 0.0)
    time_to_first_yellow = _time_for_sort(unhealthy.get("time_to_first_yellow_mean_windows"))
    time_to_first_red = _time_for_sort(unhealthy.get("time_to_first_red_mean_windows"))
    if selection_policy.ranking_mode == "aggressive_alarm":
        return (
            healthy_red_bursts,
            round(healthy_red_fraction, 9),
            -round(unhealthy_detection_rate, 9),
            -round(unhealthy_red_rate, 9),
            round(time_to_first_yellow, 9),
            round(time_to_first_red, 9),
            healthy_alarm_bursts,
            round(healthy_non_green_fraction, 9),
            policy.complexity_score,
            1 if policy.optional_latch else 0,
            policy.increment_on_unhealthy,
            policy.decrement_on_healthy,
            policy.yellow_threshold,
            policy.red_threshold,
            policy.policy_id,
        )
    return (
        healthy_red_bursts,
        round(healthy_red_fraction, 9),
        healthy_alarm_bursts,
        round(healthy_non_green_fraction, 9),
        -round(unhealthy_detection_rate, 9),
        -round(unhealthy_red_rate, 9),
        round(time_to_first_yellow, 9),
        round(time_to_first_red, 9),
        policy.complexity_score,
        1 if policy.optional_latch else 0,
        policy.increment_on_unhealthy,
        policy.decrement_on_healthy,
        policy.yellow_threshold,
        policy.red_threshold,
        policy.policy_id,
    )


def _rejected_policy_sort_key(
    selection_scope: Mapping[str, object],
    policy: FaultCounterPolicy,
) -> tuple[float | int, ...]:
    healthy = _require_mapping(selection_scope.get("healthy_metrics"), field_name="healthy_metrics")
    unhealthy = _require_mapping(selection_scope.get("unhealthy_metrics"), field_name="unhealthy_metrics")
    return (
        -round(float(unhealthy.get("unhealthy_detection_rate") or 0.0), 9),
        int(healthy.get("healthy_red_bursts_total") or 0),
        round(float(healthy.get("healthy_red_window_fraction") or 0.0), 9),
        int(healthy.get("false_alarm_bursts_total") or 0),
        round(float(healthy.get("healthy_yellow_or_red_window_fraction") or 0.0), 9),
        policy.complexity_score,
        1 if policy.optional_latch else 0,
        policy.increment_on_unhealthy,
        policy.decrement_on_healthy,
        policy.yellow_threshold,
        policy.red_threshold,
        policy.policy_id,
    )


def _evaluate_selection_policy(
    selection_scope: Mapping[str, object],
    policy: FaultCounterPolicy,
    selection_policy: PersistenceSelectionPolicy,
) -> dict[str, object]:
    del policy
    unhealthy = _require_mapping(selection_scope.get("unhealthy_metrics"), field_name="unhealthy_metrics")
    measured_unhealthy_detection_rate = float(unhealthy.get("unhealthy_detection_rate") or 0.0)
    minimum_unhealthy_detection_rate = selection_policy.minimum_unhealthy_detection_rate
    has_positive_unhealthy_detection = measured_unhealthy_detection_rate > 0.0
    passes_floor = measured_unhealthy_detection_rate >= minimum_unhealthy_detection_rate
    if selection_policy.require_positive_unhealthy_detection and not has_positive_unhealthy_detection:
        passes_floor = False
    if selection_policy.require_positive_unhealthy_detection and not has_positive_unhealthy_detection:
        selection_status = "rejected_zero_unhealthy_detection"
    elif passes_floor:
        selection_status = "survivor"
    else:
        selection_status = "rejected_insufficient_unhealthy_detection"
    return {
        "passes_unhealthy_detection_floor": passes_floor,
        "selection_status": selection_status,
        "minimum_unhealthy_detection_rate": minimum_unhealthy_detection_rate,
        "measured_unhealthy_detection_rate": round(measured_unhealthy_detection_rate, 6),
        "has_positive_unhealthy_detection": has_positive_unhealthy_detection,
        "require_positive_unhealthy_detection": selection_policy.require_positive_unhealthy_detection,
        "ranking_mode": selection_policy.ranking_mode,
        "unhealthy_detection_margin": round(
            measured_unhealthy_detection_rate - minimum_unhealthy_detection_rate,
            6,
        ),
    }


def _finalize_selection_decision(
    *,
    selection_decision: Mapping[str, object],
    selection_scope: Mapping[str, object],
    policy: FaultCounterPolicy,
    selection_rank: int,
    survivor_rank: int | None,
    survivor_count: int,
    selected_policy_id: str,
) -> dict[str, object]:
    finalized = dict(selection_decision)
    finalized["selection_rank"] = selection_rank
    finalized["survivor_rank"] = survivor_rank
    finalized["selected"] = policy.policy_id == selected_policy_id
    finalized["selection_reason"] = _build_selection_reason(
        selection_decision=selection_decision,
        selection_scope=selection_scope,
        policy=policy,
        selection_rank=selection_rank,
        survivor_rank=survivor_rank,
        survivor_count=survivor_count,
        selected_policy_id=selected_policy_id,
    )
    return finalized


def _build_selection_reason(
    *,
    selection_decision: Mapping[str, object],
    selection_scope: Mapping[str, object],
    policy: FaultCounterPolicy,
    selection_rank: int,
    survivor_rank: int | None,
    survivor_count: int,
    selected_policy_id: str,
) -> str:
    healthy = _require_mapping(selection_scope.get("healthy_metrics"), field_name="healthy_metrics")
    unhealthy = _require_mapping(selection_scope.get("unhealthy_metrics"), field_name="unhealthy_metrics")
    measured = float(selection_decision["measured_unhealthy_detection_rate"])
    minimum = float(selection_decision["minimum_unhealthy_detection_rate"])
    if str(selection_decision["selection_status"]) == "rejected_zero_unhealthy_detection":
        return "Rejected because unhealthy detection remained at 0.0; zero-detection policies cannot win."
    if not bool(selection_decision["passes_unhealthy_detection_floor"]):
        return (
            "Rejected for insufficient train/val unhealthy detection: "
            f"{measured:.6f} < floor {minimum:.6f}."
        )
    ranking_priority = _selection_priority_text(
        _require_non_empty_string(
            selection_decision.get("ranking_mode", "quiet_scout"),
            field_name="selection_decision.ranking_mode",
        )
    )
    if policy.policy_id == selected_policy_id:
        return (
            "Selected after passing the train/val unhealthy-detection floor "
            f"({measured:.6f} >= {minimum:.6f}) and ranking best among {survivor_count} survivors "
            f"{ranking_priority}. "
            f"Selection healthy RED bursts={int(healthy.get('healthy_red_bursts_total') or 0)}, "
            f"healthy RED frac={float(healthy.get('healthy_red_window_fraction') or 0.0):.6f}, "
            f"healthy YELLOW/RED frac={float(healthy.get('healthy_yellow_or_red_window_fraction') or 0.0):.6f}, "
            f"unhealthy detect={float(unhealthy.get('unhealthy_detection_rate') or 0.0):.6f}."
        )
    return (
        "Survived the train/val unhealthy-detection floor "
        f"({measured:.6f} >= {minimum:.6f}) and ranked "
        f"#{survivor_rank} of {survivor_count} survivors "
        f"(overall rank #{selection_rank})."
    )


def _build_example_traces(
    *,
    sequences_by_split: Mapping[str, Sequence[Sequence[WindowOutputRecord]]],
    policy: FaultCounterPolicy,
    per_split_examples: int,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for split_name in _VALID_SPLITS:
        examples: dict[str, list[dict[str, object]]] = {"healthy": [], "unhealthy": []}
        for sequence in sequences_by_split[split_name]:
            replay = _replay_sequence(sequence, policy)
            class_label = str(replay["actual_label"])
            if len(examples[class_label]) >= per_split_examples:
                continue
            examples[class_label].append(
                {
                    key: replay[key]
                    for key in (
                        "split",
                        "sequence_id",
                        "split_sequence_id",
                        "operating_condition",
                        "bearing_id",
                        "actual_label",
                        "n_windows",
                        "alarm_bursts",
                        "red_bursts",
                        "time_to_first_yellow_windows",
                        "time_to_first_red_windows",
                        "latched_red",
                        "clear_events_total",
                        "unresolved_alarm_bursts",
                        "steps",
                    )
                }
            )
        payload[split_name] = examples
    return payload


def _selection_priority_text(ranking_mode: str) -> str:
    if ranking_mode == "aggressive_alarm":
        return (
            "by healthy RED bursts/fraction first, then unhealthy detection/timely escalation, "
            "then healthy YELLOW chatter, then simpler behavior"
        )
    return (
        "by healthy RED bursts/fraction first, then healthy YELLOW chatter, then unhealthy "
        "detection/timely escalation, then simpler behavior"
    )


def _build_latch_behavior_summary(scope_result: Mapping[str, object]) -> str:
    all_sequences = _require_mapping(
        scope_result.get("all_sequences_metrics"),
        field_name="all_sequences_metrics",
    )
    healthy = _require_mapping(scope_result.get("healthy_metrics"), field_name="healthy_metrics")
    unhealthy = _require_mapping(scope_result.get("unhealthy_metrics"), field_name="unhealthy_metrics")
    overall_frequency = float(all_sequences.get("red_latch_frequency_all_sequences") or 0.0)
    sequence_count = int(all_sequences.get("sequence_count") or 0)
    healthy_frequency = float(healthy.get("red_latch_frequency") or 0.0)
    unhealthy_frequency = float(unhealthy.get("red_latch_frequency") or 0.0)
    if overall_frequency <= 0.0:
        return "No RED latch events observed."
    return (
        f"RED latch engaged on {overall_frequency:.6f} of {sequence_count} sequences overall; "
        f"healthy={healthy_frequency:.6f}, unhealthy={unhealthy_frequency:.6f}."
    )


def _scope_metric_deltas(
    candidate_scope: Mapping[str, object],
    baseline_scope: Mapping[str, object],
) -> dict[str, object]:
    candidate_healthy = _require_mapping(candidate_scope.get("healthy_metrics"), field_name="candidate.healthy_metrics")
    candidate_unhealthy = _require_mapping(
        candidate_scope.get("unhealthy_metrics"),
        field_name="candidate.unhealthy_metrics",
    )
    candidate_all = _require_mapping(
        candidate_scope.get("all_sequences_metrics"),
        field_name="candidate.all_sequences_metrics",
    )
    baseline_healthy = _require_mapping(baseline_scope.get("healthy_metrics"), field_name="baseline.healthy_metrics")
    baseline_unhealthy = _require_mapping(
        baseline_scope.get("unhealthy_metrics"),
        field_name="baseline.unhealthy_metrics",
    )
    baseline_all = _require_mapping(
        baseline_scope.get("all_sequences_metrics"),
        field_name="baseline.all_sequences_metrics",
    )
    return {
        "healthy_red_bursts_total_delta": int(candidate_healthy.get("healthy_red_bursts_total") or 0)
        - int(baseline_healthy.get("healthy_red_bursts_total") or 0),
        "healthy_red_window_fraction_delta": round(
            float(candidate_healthy.get("healthy_red_window_fraction") or 0.0)
            - float(baseline_healthy.get("healthy_red_window_fraction") or 0.0),
            6,
        ),
        "healthy_yellow_or_red_window_fraction_delta": round(
            float(candidate_healthy.get("healthy_yellow_or_red_window_fraction") or 0.0)
            - float(baseline_healthy.get("healthy_yellow_or_red_window_fraction") or 0.0),
            6,
        ),
        "false_alarm_bursts_total_delta": int(candidate_healthy.get("false_alarm_bursts_total") or 0)
        - int(baseline_healthy.get("false_alarm_bursts_total") or 0),
        "unhealthy_detection_rate_delta": round(
            float(candidate_unhealthy.get("unhealthy_detection_rate") or 0.0)
            - float(baseline_unhealthy.get("unhealthy_detection_rate") or 0.0),
            6,
        ),
        "unhealthy_red_detection_rate_delta": round(
            float(candidate_unhealthy.get("unhealthy_red_detection_rate") or 0.0)
            - float(baseline_unhealthy.get("unhealthy_red_detection_rate") or 0.0),
            6,
        ),
        "time_to_first_yellow_mean_windows_delta": _optional_delta(
            candidate_unhealthy.get("time_to_first_yellow_mean_windows"),
            baseline_unhealthy.get("time_to_first_yellow_mean_windows"),
        ),
        "time_to_first_red_mean_windows_delta": _optional_delta(
            candidate_unhealthy.get("time_to_first_red_mean_windows"),
            baseline_unhealthy.get("time_to_first_red_mean_windows"),
        ),
        "red_latch_frequency_delta": round(
            float(candidate_all.get("red_latch_frequency_all_sequences") or 0.0)
            - float(baseline_all.get("red_latch_frequency_all_sequences") or 0.0),
            6,
        ),
    }


def _build_selected_policy_comparisons(
    *,
    selected_result: Mapping[str, object],
    comparison_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for result in comparison_results:
        comparison_label = _require_non_empty_string(
            result.get("comparison_label"),
            field_name="comparison_label",
        )
        payload[comparison_label] = {
            "comparison_label": comparison_label,
            "policy_role": _require_non_empty_string(result.get("policy_role"), field_name="policy_role"),
            "policy": result["policy"].to_dict(),
            "selection_scope_delta": _scope_metric_deltas(
                selected_result["metrics_by_scope"]["train_val_selection"],
                result["metrics_by_scope"]["train_val_selection"],
            ),
            "test_scope_delta": _scope_metric_deltas(
                selected_result["metrics_by_scope"]["test"],
                result["metrics_by_scope"]["test"],
            ),
            "assessment": _build_second_profile_assessment(
                selected_policy=selected_result["policy"],
                comparison_policy=result["policy"],
                selection_scope_delta=_scope_metric_deltas(
                    selected_result["metrics_by_scope"]["train_val_selection"],
                    result["metrics_by_scope"]["train_val_selection"],
                ),
                test_scope_delta=_scope_metric_deltas(
                    selected_result["metrics_by_scope"]["test"],
                    result["metrics_by_scope"]["test"],
                ),
            ),
        }
    return payload


def _build_second_profile_assessment(
    *,
    selected_policy: FaultCounterPolicy,
    comparison_policy: FaultCounterPolicy,
    selection_scope_delta: Mapping[str, object],
    test_scope_delta: Mapping[str, object],
) -> dict[str, object]:
    same_policy = selected_policy.policy_id == comparison_policy.policy_id
    selection_detection_delta = float(selection_scope_delta.get("unhealthy_detection_rate_delta") or 0.0)
    selection_red_detection_delta = float(
        selection_scope_delta.get("unhealthy_red_detection_rate_delta") or 0.0
    )
    selection_ttf_y_delta = selection_scope_delta.get("time_to_first_yellow_mean_windows_delta")
    selection_ttf_r_delta = selection_scope_delta.get("time_to_first_red_mean_windows_delta")
    materially_improves = (
        not same_policy
        and (
            selection_detection_delta >= 0.03
            or selection_red_detection_delta >= 0.03
            or (selection_ttf_y_delta is not None and float(selection_ttf_y_delta) <= -0.5)
            or (selection_ttf_r_delta is not None and float(selection_ttf_r_delta) <= -0.5)
        )
    )
    healthy_cost_bounded = (
        int(selection_scope_delta.get("healthy_red_bursts_total_delta") or 0) <= 1
        and float(selection_scope_delta.get("healthy_red_window_fraction_delta") or 0.0) <= 0.01
        and float(selection_scope_delta.get("healthy_yellow_or_red_window_fraction_delta") or 0.0) <= 0.05
        and int(test_scope_delta.get("healthy_red_bursts_total_delta") or 0) <= 1
        and float(test_scope_delta.get("healthy_red_window_fraction_delta") or 0.0) <= 0.01
        and float(test_scope_delta.get("healthy_yellow_or_red_window_fraction_delta") or 0.0) <= 0.05
    )
    worth_keeping = materially_improves and healthy_cost_bounded
    if same_policy:
        improvement_answer = (
            "No. The aggressive pass selected the same policy as the conservative default, "
            "so there is no separate escalation gain to keep."
        )
        cost_answer = "None beyond the conservative default because the selected policy is identical."
        worth_answer = "No. Keep only the conservative default until a distinct aggressive policy wins."
    else:
        improvement_answer = (
            "Yes."
            if materially_improves
            else "No."
        )
        improvement_answer += (
            f" Train+val unhealthy detection delta={selection_detection_delta:+.6f}, "
            f"unhealthy RED detection delta={selection_red_detection_delta:+.6f}, "
            f"TTF-Y delta={_format_delta(selection_ttf_y_delta)} windows, "
            f"TTF-R delta={_format_delta(selection_ttf_r_delta)} windows."
        )
        cost_answer = (
            "Healthy-side cost on train+val: "
            f"RED bursts delta={int(selection_scope_delta.get('healthy_red_bursts_total_delta') or 0):+d}, "
            f"RED fraction delta={float(selection_scope_delta.get('healthy_red_window_fraction_delta') or 0.0):+.6f}, "
            "YELLOW/RED fraction delta="
            f"{float(selection_scope_delta.get('healthy_yellow_or_red_window_fraction_delta') or 0.0):+.6f}."
        )
        worth_answer = (
            "Yes. Keep it as a second profile alongside the conservative default."
            if worth_keeping
            else "No. The escalation gain does not justify the added healthy-side chatter."
        )
    return {
        "materially_improves_unhealthy_detection_or_escalation": materially_improves,
        "healthy_side_cost_bounded": healthy_cost_bounded,
        "worth_keeping_as_second_policy_profile": worth_keeping,
        "answers": {
            "does_the_aggressive_profile_materially_improve_unhealthy_detection_or_escalation": improvement_answer,
            "what_healthy_side_chatter_cost_does_it_pay": cost_answer,
            "is_it_worth_keeping_as_a_second_policy_profile_alongside_the_conservative_default": worth_answer,
        },
    }


def _build_summary_payload(
    *,
    prepared: PreparedLeanLeanPersistenceTuning,
    raw_summary: Mapping[str, object],
    raw_plan: Mapping[str, object],
    window_outputs_path: Path,
    ranked_results: Sequence[Mapping[str, object]],
    comparison_results: Sequence[Mapping[str, object]],
    selected_vs_comparisons: Mapping[str, object],
    selected_policy_path: Path,
    example_traces_path: Path,
) -> dict[str, object]:
    survivor_policy_ids: list[str] = []
    rejected_policy_ids: list[str] = []
    policies_payload: list[dict[str, object]] = []
    for result in ranked_results:
        policy = result["policy"]
        selection_decision = _require_mapping(
            result["selection_decision"],
            field_name="selection_decision",
        )
        scope_results = {
            scope_name: _compact_scope_metrics(scope_result)
            for scope_name, scope_result in result["metrics_by_scope"].items()
        }
        if bool(selection_decision["passes_unhealthy_detection_floor"]):
            survivor_policy_ids.append(policy.policy_id)
        else:
            rejected_policy_ids.append(policy.policy_id)
        policies_payload.append(
            {
                "selection_rank": int(result["selection_rank"]),
                "survivor_rank": result["survivor_rank"],
                "selected": bool(result is ranked_results[0]),
                "policy": policy.to_dict(),
                "policy_path": str(Path(result["policy_path"]).resolve()),
                "selection_decision": selection_decision,
                "ranking_key": list(result["ranking_key"]),
                "metrics_by_scope": scope_results,
            }
        )
    comparison_payload = [
        {
            "comparison_label": _require_non_empty_string(
                result.get("comparison_label"),
                field_name="comparison_label",
            ),
            "policy_role": _require_non_empty_string(result.get("policy_role"), field_name="policy_role"),
            "policy": result["policy"].to_dict(),
            "metrics_by_scope": {
                scope_name: _compact_scope_metrics(scope_result)
                for scope_name, scope_result in result["metrics_by_scope"].items()
            },
        }
        for result in comparison_results
    ]
    return {
        "schema_version": DEFAULT_SUMMARY_SCHEMA_VERSION,
        "profile_name": prepared.config.profile_name,
        "config_path": str(prepared.config.config_path),
        "run_root": str(prepared.run_root),
        "source_deployment_run_root": str(prepared.source_run_root),
        "source_deployment_summary_path": str(prepared.source_summary_path.resolve()),
        "source_deployment_plan_path": str(prepared.source_plan_path.resolve()),
        "source_window_outputs_path": str(window_outputs_path.resolve()),
        "source_scaffold_path": (
            str(prepared.source_scaffold_path.resolve())
            if prepared.source_scaffold_path is not None
            else None
        ),
        "selection_scope": "train_val_selection",
        "test_scope": "report_only",
        "selection_policy": prepared.config.selection_policy.to_dict(),
        "ranking_intent": {
            "step_1": "reject policies below the train_val_selection unhealthy-detection floor",
            "step_1b": "reject zero-detection policies when positive unhealthy detection is required",
            "step_2": "minimize healthy RED bursts and healthy RED window fraction",
            "step_3": (
                "maximize unhealthy detection and timely escalation"
                if prepared.config.selection_policy.ranking_mode == "aggressive_alarm"
                else "minimize healthy YELLOW chatter"
            ),
            "step_4": (
                "minimize healthy YELLOW chatter"
                if prepared.config.selection_policy.ranking_mode == "aggressive_alarm"
                else "maximize unhealthy detection and timely escalation"
            ),
            "step_5": "prefer simpler policies when metrics are effectively tied",
        },
        "raw_classifier_metrics": {
            "deployment_summary_path": str(prepared.source_summary_path.resolve()),
            "summary_row": raw_summary.get("summary_row"),
            "variant": raw_summary.get("variant"),
            "plan_persistence_prep": raw_plan.get("persistence_prep"),
        },
        "policy_candidates": [policy.to_dict() for policy in prepared.config.policy_candidates],
        "comparison_policies": [spec.to_dict() for spec in prepared.config.comparison_policies],
        "survivor_policy_ids": survivor_policy_ids,
        "rejected_policy_ids": rejected_policy_ids,
        "policies": policies_payload,
        "comparison_results": comparison_payload,
        "selected_vs_comparisons": selected_vs_comparisons,
        "selected_policy_path": str(selected_policy_path.resolve()),
        "example_traces_path": str(example_traces_path.resolve()),
        "notes": list(prepared.config.notes),
    }


def _compact_scope_metrics(scope_result: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in scope_result.items()
        if key != "sequence_summaries"
    }


def _write_summary_csv(
    path: Path,
    ranked_results: Sequence[Mapping[str, object]],
    comparison_results: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = [
        "policy_role",
        "comparison_label",
        "selection_rank",
        "survivor_rank",
        "selected",
        "selection_status",
        "passes_unhealthy_detection_floor",
        "minimum_unhealthy_detection_rate",
        "measured_unhealthy_detection_rate",
        "unhealthy_detection_margin",
        "policy_id",
        "increment_on_unhealthy",
        "decrement_on_healthy",
        "yellow_threshold",
        "red_threshold",
        "optional_latch",
        "latch_clear_threshold",
        "selection_reason",
        "selection_healthy_red_bursts_total",
        "selection_healthy_red_window_fraction",
        "selection_false_alarm_bursts_total",
        "selection_healthy_yellow_or_red_window_fraction",
        "selection_unhealthy_detection_rate",
        "selection_unhealthy_red_detection_rate",
        "selection_time_to_first_yellow_mean_windows",
        "selection_time_to_first_red_mean_windows",
        "selection_red_latch_frequency",
        "selection_latch_behavior_summary",
        "test_healthy_red_bursts_total",
        "test_healthy_red_window_fraction",
        "test_false_alarm_bursts_total",
        "test_healthy_yellow_or_red_window_fraction",
        "test_unhealthy_detection_rate",
        "test_unhealthy_red_detection_rate",
        "test_time_to_first_yellow_mean_windows",
        "test_time_to_first_red_mean_windows",
        "test_red_latch_frequency",
        "test_latch_behavior_summary",
        "policy_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in comparison_results:
            selection_scope = result["metrics_by_scope"]["train_val_selection"]
            test_scope = result["metrics_by_scope"]["test"]
            selection_healthy = selection_scope["healthy_metrics"]
            selection_unhealthy = selection_scope["unhealthy_metrics"]
            test_healthy = test_scope["healthy_metrics"]
            test_unhealthy = test_scope["unhealthy_metrics"]
            writer.writerow(
                {
                    "policy_role": result["policy_role"],
                    "comparison_label": result["comparison_label"],
                    "selection_rank": None,
                    "survivor_rank": None,
                    "selected": False,
                    "selection_status": "comparison_only",
                    "passes_unhealthy_detection_floor": None,
                    "minimum_unhealthy_detection_rate": None,
                    "measured_unhealthy_detection_rate": None,
                    "unhealthy_detection_margin": None,
                    "policy_id": result["policy"].policy_id,
                    "increment_on_unhealthy": result["policy"].increment_on_unhealthy,
                    "decrement_on_healthy": result["policy"].decrement_on_healthy,
                    "yellow_threshold": result["policy"].yellow_threshold,
                    "red_threshold": result["policy"].red_threshold,
                    "optional_latch": result["policy"].optional_latch,
                    "latch_clear_threshold": result["policy"].latch_clear_threshold,
                    "selection_reason": "Explicit comparison policy; not part of aggressive candidate ranking.",
                    "selection_healthy_red_bursts_total": selection_healthy["healthy_red_bursts_total"],
                    "selection_healthy_red_window_fraction": selection_healthy["healthy_red_window_fraction"],
                    "selection_false_alarm_bursts_total": selection_healthy["false_alarm_bursts_total"],
                    "selection_healthy_yellow_or_red_window_fraction": selection_healthy["healthy_yellow_or_red_window_fraction"],
                    "selection_unhealthy_detection_rate": selection_unhealthy["unhealthy_detection_rate"],
                    "selection_unhealthy_red_detection_rate": selection_unhealthy["unhealthy_red_detection_rate"],
                    "selection_time_to_first_yellow_mean_windows": selection_unhealthy["time_to_first_yellow_mean_windows"],
                    "selection_time_to_first_red_mean_windows": selection_unhealthy["time_to_first_red_mean_windows"],
                    "selection_red_latch_frequency": selection_scope["all_sequences_metrics"]["red_latch_frequency_all_sequences"],
                    "selection_latch_behavior_summary": _build_latch_behavior_summary(selection_scope),
                    "test_healthy_red_bursts_total": test_healthy["healthy_red_bursts_total"],
                    "test_healthy_red_window_fraction": test_healthy["healthy_red_window_fraction"],
                    "test_false_alarm_bursts_total": test_healthy["false_alarm_bursts_total"],
                    "test_healthy_yellow_or_red_window_fraction": test_healthy["healthy_yellow_or_red_window_fraction"],
                    "test_unhealthy_detection_rate": test_unhealthy["unhealthy_detection_rate"],
                    "test_unhealthy_red_detection_rate": test_unhealthy["unhealthy_red_detection_rate"],
                    "test_time_to_first_yellow_mean_windows": test_unhealthy["time_to_first_yellow_mean_windows"],
                    "test_time_to_first_red_mean_windows": test_unhealthy["time_to_first_red_mean_windows"],
                    "test_red_latch_frequency": test_scope["all_sequences_metrics"]["red_latch_frequency_all_sequences"],
                    "test_latch_behavior_summary": _build_latch_behavior_summary(test_scope),
                    "policy_path": None,
                }
            )
        for result in ranked_results:
            policy: FaultCounterPolicy = result["policy"]
            selection_decision = _require_mapping(
                result["selection_decision"],
                field_name="selection_decision",
            )
            selection_scope = result["metrics_by_scope"]["train_val_selection"]
            test_scope = result["metrics_by_scope"]["test"]
            selection_healthy = selection_scope["healthy_metrics"]
            selection_unhealthy = selection_scope["unhealthy_metrics"]
            test_healthy = test_scope["healthy_metrics"]
            test_unhealthy = test_scope["unhealthy_metrics"]
            writer.writerow(
                {
                    "policy_role": "aggressive_candidate",
                    "comparison_label": None,
                    "selection_rank": result["selection_rank"],
                    "survivor_rank": result["survivor_rank"],
                    "selected": result is ranked_results[0],
                    "selection_status": selection_decision["selection_status"],
                    "passes_unhealthy_detection_floor": selection_decision["passes_unhealthy_detection_floor"],
                    "minimum_unhealthy_detection_rate": selection_decision["minimum_unhealthy_detection_rate"],
                    "measured_unhealthy_detection_rate": selection_decision["measured_unhealthy_detection_rate"],
                    "unhealthy_detection_margin": selection_decision["unhealthy_detection_margin"],
                    "policy_id": policy.policy_id,
                    "increment_on_unhealthy": policy.increment_on_unhealthy,
                    "decrement_on_healthy": policy.decrement_on_healthy,
                    "yellow_threshold": policy.yellow_threshold,
                    "red_threshold": policy.red_threshold,
                    "optional_latch": policy.optional_latch,
                    "latch_clear_threshold": policy.latch_clear_threshold,
                    "selection_reason": selection_decision["selection_reason"],
                    "selection_healthy_red_bursts_total": selection_healthy["healthy_red_bursts_total"],
                    "selection_healthy_red_window_fraction": selection_healthy["healthy_red_window_fraction"],
                    "selection_false_alarm_bursts_total": selection_healthy["false_alarm_bursts_total"],
                    "selection_healthy_yellow_or_red_window_fraction": selection_healthy["healthy_yellow_or_red_window_fraction"],
                    "selection_unhealthy_detection_rate": selection_unhealthy["unhealthy_detection_rate"],
                    "selection_unhealthy_red_detection_rate": selection_unhealthy["unhealthy_red_detection_rate"],
                    "selection_time_to_first_yellow_mean_windows": selection_unhealthy["time_to_first_yellow_mean_windows"],
                    "selection_time_to_first_red_mean_windows": selection_unhealthy["time_to_first_red_mean_windows"],
                    "selection_red_latch_frequency": selection_scope["all_sequences_metrics"]["red_latch_frequency_all_sequences"],
                    "selection_latch_behavior_summary": _build_latch_behavior_summary(selection_scope),
                    "test_healthy_red_bursts_total": test_healthy["healthy_red_bursts_total"],
                    "test_healthy_red_window_fraction": test_healthy["healthy_red_window_fraction"],
                    "test_false_alarm_bursts_total": test_healthy["false_alarm_bursts_total"],
                    "test_healthy_yellow_or_red_window_fraction": test_healthy["healthy_yellow_or_red_window_fraction"],
                    "test_unhealthy_detection_rate": test_unhealthy["unhealthy_detection_rate"],
                    "test_unhealthy_red_detection_rate": test_unhealthy["unhealthy_red_detection_rate"],
                    "test_time_to_first_yellow_mean_windows": test_unhealthy["time_to_first_yellow_mean_windows"],
                    "test_time_to_first_red_mean_windows": test_unhealthy["time_to_first_red_mean_windows"],
                    "test_red_latch_frequency": test_scope["all_sequences_metrics"]["red_latch_frequency_all_sequences"],
                    "test_latch_behavior_summary": _build_latch_behavior_summary(test_scope),
                    "policy_path": str(Path(result["policy_path"]).resolve()),
                }
            )


def _build_summary_markdown(
    *,
    prepared: PreparedLeanLeanPersistenceTuning,
    raw_summary: Mapping[str, object],
    ranked_results: Sequence[Mapping[str, object]],
    comparison_results: Sequence[Mapping[str, object]],
    selected_vs_comparisons: Mapping[str, object],
    window_outputs_path: Path,
) -> str:
    variant = _require_mapping(raw_summary.get("variant"), field_name="variant")
    metrics = _require_mapping(variant.get("metrics"), field_name="variant.metrics")
    survivor_results = [
        result for result in ranked_results if bool(result["selection_decision"]["passes_unhealthy_detection_floor"])
    ]
    rejected_results = [
        result for result in ranked_results if not bool(result["selection_decision"]["passes_unhealthy_detection_floor"])
    ]
    lines = [
        "# Lean-Lean Persistence Tuning",
        "",
        f"Source deployment candidate run root: `{prepared.source_run_root}`",
        f"Materialized window outputs: `{window_outputs_path}`",
        "",
        "## Raw Classifier Metrics",
        "",
        "Raw classifier metrics remain authoritative and are reported separately from persistence replay metrics.",
        "",
        "| Split | Accuracy | Healthy->Unhealthy FPR | Unhealthy Precision | Unhealthy Recall | Unhealthy F1 | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in _VALID_SPLITS:
        split_metrics = _require_mapping(metrics.get(split_name), field_name=f"variant.metrics.{split_name}")
        lines.append(
            "| {split} | {accuracy:.6f} | {fpr:.6f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {macro:.6f} |".format(
                split=split_name,
                accuracy=float(split_metrics["accuracy"]),
                fpr=float(split_metrics["healthy_to_unhealthy_fpr"]),
                precision=float(split_metrics["unhealthy_precision"]),
                recall=float(split_metrics["unhealthy_recall"]),
                f1=float(split_metrics["unhealthy_f1"]),
                macro=float(split_metrics["macro_f1"]),
            )
        )
    lines.extend(
        [
            "",
            "## Persistence Ranking",
            "",
            "Ranking uses `train_val_selection` only. `test` is report-only.",
            (
                "Policies must first pass the configured train/val unhealthy-detection floor: "
                f"`{prepared.config.selection_policy.minimum_unhealthy_detection_rate:.6f}`."
            ),
            (
                "Selection priority: "
                f"{_selection_priority_text(prepared.config.selection_policy.ranking_mode)}."
            ),
            "",
            "### Survivors",
            "",
            "| Rank | Policy | Passed floor | Healthy RED bursts | Healthy RED frac | Healthy YELLOW/RED frac | Unhealthy detect | Unhealthy RED detect | TTF Y mean | TTF R mean | Latch |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in survivor_results:
        policy: FaultCounterPolicy = result["policy"]
        selection_decision = _require_mapping(result["selection_decision"], field_name="selection_decision")
        selection_scope = result["metrics_by_scope"]["train_val_selection"]
        selection_healthy = selection_scope["healthy_metrics"]
        selection_unhealthy = selection_scope["unhealthy_metrics"]
        lines.append(
            "| {rank} | `{policy}` | {passed} | {sel_red} | {sel_red_frac:.6f} | {sel_ng:.6f} | {sel_det:.6f} | {sel_red_rate:.6f} | {ttf_y} | {ttf_r} | {latch} |".format(
                rank=int(result["survivor_rank"]),
                policy=policy.policy_id,
                passed="yes" if bool(selection_decision["passes_unhealthy_detection_floor"]) else "no",
                sel_red=int(selection_healthy["healthy_red_bursts_total"] or 0),
                sel_red_frac=float(selection_healthy["healthy_red_window_fraction"] or 0.0),
                sel_ng=float(selection_healthy["healthy_yellow_or_red_window_fraction"] or 0.0),
                sel_det=float(selection_unhealthy["unhealthy_detection_rate"] or 0.0),
                sel_red_rate=float(selection_unhealthy["unhealthy_red_detection_rate"] or 0.0),
                ttf_y=_format_optional_number(selection_unhealthy["time_to_first_yellow_mean_windows"]),
                ttf_r=_format_optional_number(selection_unhealthy["time_to_first_red_mean_windows"]),
                latch=_build_latch_behavior_summary(selection_scope),
            )
        )
    if rejected_results:
        lines.extend(
            [
                "",
                "### Rejected For Insufficient Unhealthy Detection",
                "",
                "| Overall rank | Policy | Train/val unhealthy detect | Floor | Healthy RED bursts | Healthy YELLOW/RED frac | TTF Y mean | TTF R mean | Latch |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in rejected_results:
            policy = result["policy"]
            selection_decision = _require_mapping(result["selection_decision"], field_name="selection_decision")
            selection_scope = result["metrics_by_scope"]["train_val_selection"]
            selection_healthy = selection_scope["healthy_metrics"]
            selection_unhealthy = selection_scope["unhealthy_metrics"]
            lines.append(
                "| {rank} | `{policy}` | {sel_det:.6f} | {floor:.6f} | {sel_red} | {sel_ng:.6f} | {ttf_y} | {ttf_r} | {latch} |".format(
                    rank=int(result["selection_rank"]),
                    policy=policy.policy_id,
                    sel_det=float(selection_unhealthy["unhealthy_detection_rate"] or 0.0),
                    floor=float(selection_decision["minimum_unhealthy_detection_rate"]),
                    sel_red=int(selection_healthy["healthy_red_bursts_total"] or 0),
                    sel_ng=float(selection_healthy["healthy_yellow_or_red_window_fraction"] or 0.0),
                    ttf_y=_format_optional_number(selection_unhealthy["time_to_first_yellow_mean_windows"]),
                    ttf_r=_format_optional_number(selection_unhealthy["time_to_first_red_mean_windows"]),
                    latch=_build_latch_behavior_summary(selection_scope),
                )
            )
    if comparison_results:
        lines.extend(
            [
                "",
                "## Conservative Default Comparison",
                "",
                "| Role | Policy | Healthy RED bursts | Healthy RED frac | Healthy YELLOW/RED frac | Unhealthy detect | Unhealthy RED detect | TTF Y mean | TTF R mean | Latch |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in comparison_results:
            selection_scope = result["metrics_by_scope"]["train_val_selection"]
            selection_healthy = selection_scope["healthy_metrics"]
            selection_unhealthy = selection_scope["unhealthy_metrics"]
            lines.append(
                "| {role} | `{policy}` | {sel_red} | {sel_red_frac:.6f} | {sel_ng:.6f} | {sel_det:.6f} | {sel_red_rate:.6f} | {ttf_y} | {ttf_r} | {latch} |".format(
                    role=_require_non_empty_string(result["comparison_label"], field_name="comparison_label"),
                    policy=result["policy"].policy_id,
                    sel_red=int(selection_healthy["healthy_red_bursts_total"] or 0),
                    sel_red_frac=float(selection_healthy["healthy_red_window_fraction"] or 0.0),
                    sel_ng=float(selection_healthy["healthy_yellow_or_red_window_fraction"] or 0.0),
                    sel_det=float(selection_unhealthy["unhealthy_detection_rate"] or 0.0),
                    sel_red_rate=float(selection_unhealthy["unhealthy_red_detection_rate"] or 0.0),
                    ttf_y=_format_optional_number(selection_unhealthy["time_to_first_yellow_mean_windows"]),
                    ttf_r=_format_optional_number(selection_unhealthy["time_to_first_red_mean_windows"]),
                    latch=_build_latch_behavior_summary(selection_scope),
                )
            )
    selected = ranked_results[0]
    selected_policy: FaultCounterPolicy = selected["policy"]
    selected_decision = _require_mapping(selected["selection_decision"], field_name="selection_decision")
    selected_selection = selected["metrics_by_scope"]["train_val_selection"]
    selected_test = selected["metrics_by_scope"]["test"]
    lines.extend(
        [
            "",
            "## Selected Policy",
            "",
            f"- Policy: `{selected_policy.policy_id}`",
            f"- Parameters: increment `{selected_policy.increment_on_unhealthy}`, decrement `{selected_policy.decrement_on_healthy}`, yellow `{selected_policy.yellow_threshold}`, red `{selected_policy.red_threshold}`, latch `{selected_policy.optional_latch}`, latch_clear_threshold `{selected_policy.latch_clear_threshold}`",
            f"- Passed unhealthy-detection floor: `{selected_decision['passes_unhealthy_detection_floor']}` at `{float(selected_decision['measured_unhealthy_detection_rate']):.6f}` vs floor `{float(selected_decision['minimum_unhealthy_detection_rate']):.6f}`",
            f"- Selection reason: {selected_decision['selection_reason']}",
            f"- Selection healthy RED bursts: `{int(selected_selection['healthy_metrics']['healthy_red_bursts_total'] or 0)}`",
            f"- Selection healthy RED window fraction: `{float(selected_selection['healthy_metrics']['healthy_red_window_fraction'] or 0.0):.6f}`",
            f"- Selection healthy YELLOW/RED window fraction: `{float(selected_selection['healthy_metrics']['healthy_yellow_or_red_window_fraction'] or 0.0):.6f}`",
            f"- Selection unhealthy detection rate: `{float(selected_selection['unhealthy_metrics']['unhealthy_detection_rate'] or 0.0):.6f}`",
            f"- Selection unhealthy RED detection rate: `{float(selected_selection['unhealthy_metrics']['unhealthy_red_detection_rate'] or 0.0):.6f}`",
            f"- Selection time-to-first-YELLOW mean: `{_format_optional_number(selected_selection['unhealthy_metrics']['time_to_first_yellow_mean_windows'])}` windows",
            f"- Selection time-to-first-RED mean: `{_format_optional_number(selected_selection['unhealthy_metrics']['time_to_first_red_mean_windows'])}` windows",
            f"- Selection latch behavior: `{_build_latch_behavior_summary(selected_selection)}`",
            f"- Test healthy RED bursts: `{int(selected_test['healthy_metrics']['healthy_red_bursts_total'] or 0)}`",
            f"- Test unhealthy detection rate: `{float(selected_test['unhealthy_metrics']['unhealthy_detection_rate'] or 0.0):.6f}`",
            f"- Test unhealthy RED detection rate: `{float(selected_test['unhealthy_metrics']['unhealthy_red_detection_rate'] or 0.0):.6f}`",
            f"- Test latch behavior: `{_build_latch_behavior_summary(selected_test)}`",
        ]
    )
    if selected_vs_comparisons:
        for comparison_label, comparison_payload in selected_vs_comparisons.items():
            assessment = _require_mapping(
                _require_mapping(comparison_payload, field_name=f"selected_vs_comparisons.{comparison_label}").get("assessment"),
                field_name=f"selected_vs_comparisons.{comparison_label}.assessment",
            )
            answers = _require_mapping(
                assessment.get("answers"),
                field_name=f"selected_vs_comparisons.{comparison_label}.assessment.answers",
            )
            selection_delta = _require_mapping(
                comparison_payload.get("selection_scope_delta"),
                field_name=f"selected_vs_comparisons.{comparison_label}.selection_scope_delta",
            )
            lines.extend(
                [
                    "",
                    f"## Comparison Verdict: {comparison_label}",
                    "",
                    f"- Train/val unhealthy detection delta: `{float(selection_delta.get('unhealthy_detection_rate_delta') or 0.0):+.6f}`",
                    f"- Train/val unhealthy RED detection delta: `{float(selection_delta.get('unhealthy_red_detection_rate_delta') or 0.0):+.6f}`",
                    f"- Train/val time-to-first-YELLOW delta: `{_format_delta(selection_delta.get('time_to_first_yellow_mean_windows_delta'))}` windows",
                    f"- Train/val time-to-first-RED delta: `{_format_delta(selection_delta.get('time_to_first_red_mean_windows_delta'))}` windows",
                    f"- Train/val healthy RED bursts delta: `{int(selection_delta.get('healthy_red_bursts_total_delta') or 0):+d}`",
                    f"- Train/val healthy RED fraction delta: `{float(selection_delta.get('healthy_red_window_fraction_delta') or 0.0):+.6f}`",
                    f"- Train/val healthy YELLOW/RED fraction delta: `{float(selection_delta.get('healthy_yellow_or_red_window_fraction_delta') or 0.0):+.6f}`",
                    f"- Does the aggressive profile materially improve unhealthy detection or escalation? {answers['does_the_aggressive_profile_materially_improve_unhealthy_detection_or_escalation']}",
                    f"- What healthy-side chatter cost does it pay? {answers['what_healthy_side_chatter_cost_does_it_pay']}",
                    f"- Is it worth keeping as a second policy profile alongside the conservative default? {answers['is_it_worth_keeping_as_a_second_policy_profile_alongside_the_conservative_default']}",
                ]
            )
    return "\n".join(lines) + "\n"


def _parse_policy_candidates(
    payload: Mapping[str, object],
    *,
    default_policy: FaultCounterPolicy,
) -> list[FaultCounterPolicy]:
    explicit = payload.get("policy_candidates")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes, bytearray)):
            raise ContractValidationError("`policy_candidates` must be a sequence of mappings.")
        candidates = [
            _policy_from_mapping(
                _require_mapping(candidate, field_name=f"policy_candidates[{index}]"),
                field_name=f"policy_candidates[{index}]",
                default_policy_id=f"policy_{index + 1}",
            )
            for index, candidate in enumerate(explicit)
        ]
        if not candidates:
            raise ContractValidationError("`policy_candidates` must not be empty.")
        return _dedupe_policies(candidates)
    raw_grid = payload.get("policy_grid")
    if raw_grid is not None:
        grid = _require_mapping(raw_grid, field_name="policy_grid")
        generated: list[FaultCounterPolicy] = []
        product_values = product(
            _require_int_sequence(
                grid.get("increment_on_unhealthy"),
                field_name="policy_grid.increment_on_unhealthy",
                minimum=1,
            ),
            _require_int_sequence(
                grid.get("decrement_on_healthy"),
                field_name="policy_grid.decrement_on_healthy",
                minimum=0,
            ),
            _require_int_sequence(
                grid.get("yellow_threshold"),
                field_name="policy_grid.yellow_threshold",
                minimum=1,
            ),
            _require_int_sequence(
                grid.get("red_threshold"),
                field_name="policy_grid.red_threshold",
                minimum=1,
            ),
            _require_bool_sequence(
                grid.get("optional_latch"),
                field_name="policy_grid.optional_latch",
            ),
        )
        for increment, decrement, yellow_threshold, red_threshold, optional_latch in product_values:
            if red_threshold < yellow_threshold:
                continue
            policy_id = (
                f"fc_i{increment}_d{decrement}_y{yellow_threshold}_r{red_threshold}_"
                f"{'latch' if optional_latch else 'nolatch'}"
            )
            generated.append(
                FaultCounterPolicy(
                    policy_id=policy_id,
                    increment_on_unhealthy=increment,
                    decrement_on_healthy=decrement,
                    yellow_threshold=yellow_threshold,
                    red_threshold=red_threshold,
                    optional_latch=optional_latch,
                )
            )
        if not generated:
            raise ContractValidationError("`policy_grid` did not generate any valid candidates.")
        return _dedupe_policies(generated)
    return [default_policy]


def _parse_comparison_policies(payload: Mapping[str, object]) -> list[PolicyComparisonSpec]:
    raw_comparisons = payload.get("comparison_policies")
    if raw_comparisons is None:
        return []
    if not isinstance(raw_comparisons, Sequence) or isinstance(raw_comparisons, (str, bytes, bytearray)):
        raise ContractValidationError("`comparison_policies` must be a sequence of mappings.")
    comparisons: list[PolicyComparisonSpec] = []
    for index, raw_comparison in enumerate(raw_comparisons):
        mapping = _require_mapping(raw_comparison, field_name=f"comparison_policies[{index}]")
        comparisons.append(
            PolicyComparisonSpec(
                comparison_label=_require_non_empty_string(
                    mapping.get("comparison_label", f"comparison_{index + 1}"),
                    field_name=f"comparison_policies[{index}].comparison_label",
                ),
                policy_role=_require_non_empty_string(
                    mapping.get("policy_role", "comparison_policy"),
                    field_name=f"comparison_policies[{index}].policy_role",
                ),
                policy=_policy_from_mapping(
                    mapping,
                    field_name=f"comparison_policies[{index}]",
                    default_policy_id=f"comparison_policy_{index + 1}",
                ),
            )
        )
    return comparisons


def _policy_from_mapping(
    raw: Mapping[str, object],
    *,
    field_name: str,
    default_policy_id: str,
) -> FaultCounterPolicy:
    policy_id = str(raw.get("policy_id", default_policy_id)).strip()
    if not policy_id:
        raise ContractValidationError(f"`{field_name}.policy_id` must be non-empty when provided.")
    increment_on_unhealthy = _require_int(
        raw.get("increment_on_unhealthy"),
        field_name=f"{field_name}.increment_on_unhealthy",
        minimum=1,
    )
    decrement_on_healthy = _require_int(
        raw.get("decrement_on_healthy"),
        field_name=f"{field_name}.decrement_on_healthy",
        minimum=0,
    )
    yellow_threshold = _require_int(
        raw.get("yellow_threshold"),
        field_name=f"{field_name}.yellow_threshold",
        minimum=1,
    )
    red_threshold = _require_int(
        raw.get("red_threshold"),
        field_name=f"{field_name}.red_threshold",
        minimum=1,
    )
    if red_threshold < yellow_threshold:
        raise ContractValidationError(
            f"`{field_name}.red_threshold` must be >= `{field_name}.yellow_threshold`."
        )
    optional_latch = _require_bool(
        raw.get("optional_latch", False),
        field_name=f"{field_name}.optional_latch",
    )
    latch_clear_threshold = raw.get("latch_clear_threshold")
    normalized_latch_clear: int | None
    if latch_clear_threshold is None:
        normalized_latch_clear = None
    else:
        normalized_latch_clear = _require_int(
            latch_clear_threshold,
            field_name=f"{field_name}.latch_clear_threshold",
            minimum=0,
        )
        if normalized_latch_clear >= yellow_threshold:
            raise ContractValidationError(
                f"`{field_name}.latch_clear_threshold` must be < `{field_name}.yellow_threshold`."
            )
    return FaultCounterPolicy(
        policy_id=policy_id,
        increment_on_unhealthy=increment_on_unhealthy,
        decrement_on_healthy=decrement_on_healthy,
        yellow_threshold=yellow_threshold,
        red_threshold=red_threshold,
        optional_latch=optional_latch,
        latch_clear_threshold=normalized_latch_clear,
    )


def _dedupe_policies(policies: Sequence[FaultCounterPolicy]) -> list[FaultCounterPolicy]:
    deduped: list[FaultCounterPolicy] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.policy_id in seen:
            raise ContractValidationError(f"Duplicate persistence policy id `{policy.policy_id}`.")
        seen.add(policy.policy_id)
        deduped.append(policy)
    return deduped


def _require_selection_ranking_mode(value: object, *, field_name: str) -> str:
    ranking_mode = _require_non_empty_string(value, field_name=field_name)
    if ranking_mode not in {"quiet_scout", "aggressive_alarm"}:
        raise ContractValidationError(
            f"`{field_name}` must be `quiet_scout` or `aggressive_alarm`."
        )
    return ranking_mode


def _parse_source_record_id(source_record_id: str) -> dict[str, object]:
    parts = source_record_id.split("__")
    if len(parts) != 3 or not parts[2].startswith("r"):
        raise ContractValidationError(
            f"Unsupported source_record_id format for persistence replay: `{source_record_id}`."
        )
    return {
        "operating_condition": parts[0],
        "bearing_id": parts[1],
        "recording_index": int(parts[2][1:]),
    }


def _resolve_relative_path(
    *,
    config_file: Path,
    raw_path: object,
    field_name: str,
) -> Path:
    normalized = _require_non_empty_string(raw_path, field_name=field_name)
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (config_file.parent / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON mapping.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _time_for_sort(value: object) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _optional_delta(candidate: object, baseline: object) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(float(candidate) - float(baseline), 6)


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _mean_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / float(len(values)), 6)


def _median_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 6)


def _format_optional_number(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _format_delta(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _require_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of strings.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_require_non_empty_string(item, field_name=f"{field_name}[{index}]"))
    return tuple(normalized)


def _require_int_sequence(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of integers.")
    normalized: list[int] = []
    for index, item in enumerate(value):
        normalized.append(_require_int(item, field_name=f"{field_name}[{index}]", minimum=minimum))
    if not normalized:
        raise ContractValidationError(f"`{field_name}` must not be empty.")
    return tuple(normalized)


def _require_bool_sequence(value: object, *, field_name: str) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of booleans.")
    normalized: list[bool] = []
    for index, item in enumerate(value):
        normalized.append(_require_bool(item, field_name=f"{field_name}[{index}]"))
    if not normalized:
        raise ContractValidationError(f"`{field_name}` must not be empty.")
    return tuple(normalized)


def _require_probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"`{field_name}` must be a number between 0.0 and 1.0.")
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ContractValidationError(f"`{field_name}` must be between 0.0 and 1.0.")
    return round(normalized, 6)


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return int(value)


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a boolean.")
    return value


def _require_split_name(value: object, *, field_name: str) -> str:
    split_name = _require_non_empty_string(value, field_name=field_name)
    if split_name not in _VALID_SPLITS:
        raise ContractValidationError(f"`{field_name}` must be one of {', '.join(_VALID_SPLITS)}.")
    return split_name


def _require_label(value: object, *, field_name: str) -> str:
    label = _require_non_empty_string(value, field_name=field_name)
    if label not in _LABEL_TO_INT:
        raise ContractValidationError(f"`{field_name}` must be `healthy` or `unhealthy`.")
    return label


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PersistenceTuningConfig",
    "PersistenceSelectionPolicy",
    "PolicyComparisonSpec",
    "PreparedLeanLeanPersistenceTuning",
    "FaultCounterPolicy",
    "WindowOutputRecord",
    "ensure_window_outputs_materialized",
    "load_leanlean_persistence_tuning_config",
    "prepare_leanlean_persistence_tuning",
    "run_leanlean_persistence_tuning",
    "run_prepared_leanlean_persistence_tuning",
]
