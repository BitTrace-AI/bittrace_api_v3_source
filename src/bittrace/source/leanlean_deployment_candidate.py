"""Lean-Lean deployment-candidate workflow over the locked temporal frontend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.core.config import EvolutionConfig
from bittrace.v3 import ContractValidationError, StageKey

from ._leanlean_support import (
    DEFAULT_RUNS_ROOT as BACKEND_DEFAULT_RUNS_ROOT,
    PreparedBackendArchitectureComparison,
    SearchVariantConfig,
    _build_summary_row,
    _evolution_config_to_dict,
    _materialize_shared_backend_bundle,
    _parse_evaluation_config,
    _parse_search_config,
    _resolve_relative_path,
    _run_lean_lean_variant,
    _selection_spec_from_mapping,
    _write_json,
    _write_summary_csv,
)
from .full_binary_campaign import (
    _device_agnostic_export,
    _load_backend_training_configs,
    _materialize_source_bundle,
    _resolve_inventory_rows,
    _resolve_max_selected_k_per_class,
    load_consumer_config,
)
from .locked_frontend import (
    LockedFrontendSpec,
    build_locked_frontend_stage_materialization,
    load_locked_frontend_spec,
)
from .temporal_features import load_temporal_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "canonical_deployment_candidate.yaml"
DEFAULT_PERSISTENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "persistence_quiet_scout.yaml"
DEFAULT_RUNS_ROOT = BACKEND_DEFAULT_RUNS_ROOT
SUMMARY_ARTIFACT_NAME = "leanlean_deployment_candidate_summary.json"
PLAN_ARTIFACT_NAME = "leanlean_deployment_candidate_plan.json"
SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-deployment-candidate-summary-1"
PLAN_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-deployment-candidate-plan-1"
PERSISTENCE_PREP_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-persistence-prep-1"
WINDOW_OUTPUT_TEMPLATE_SCHEMA_VERSION = "bittrace-bearings-v3-source-leanlean-window-outputs-template-1"
_SUMMARY_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class PersistencePrepConfig:
    config_path: Path
    profile_name: str
    window_output_artifact_name: str
    scaffold_json_name: str
    split_scope: tuple[str, ...]
    fields: tuple[str, ...]
    fault_counter_policy: Mapping[str, object]
    planned_outputs: Mapping[str, object]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "profile_name": self.profile_name,
            "window_output_artifact_name": self.window_output_artifact_name,
            "scaffold_json_name": self.scaffold_json_name,
            "split_scope": list(self.split_scope),
            "fields": list(self.fields),
            "fault_counter_policy": dict(self.fault_counter_policy),
            "planned_outputs": dict(self.planned_outputs),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PreparedLeanLeanDeploymentCandidate:
    config_path: Path
    run_root: Path
    profile_name: str
    source_profile_path: Path
    source_profile_name: str
    locked_frontend: LockedFrontendSpec
    comparison_prepared: PreparedBackendArchitectureComparison
    deploy_export: Mapping[str, object]
    deploy_constraints: Mapping[str, object]
    ranking_intent: Mapping[str, object]
    persistence_config: PersistencePrepConfig | None


def load_leanlean_deployment_candidate_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    for key in ("profile_name", "source_profile", "leanlean_deployment_candidate"):
        if key not in payload:
            raise ContractValidationError(f"`{resolved_path}` is missing required top-level key `{key}`.")
    return payload


def load_persistence_prep_config(config_path: str | Path) -> PersistencePrepConfig:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    profile_name = _require_non_empty_string(
        payload.get("profile_name"),
        field_name="profile_name",
    )
    window_outputs = _require_mapping(
        payload.get("window_outputs"),
        field_name="window_outputs",
    )
    output_fields = _require_string_sequence(
        window_outputs.get("fields"),
        field_name="window_outputs.fields",
    )
    if not output_fields:
        raise ContractValidationError("`window_outputs.fields` must include at least one field name.")
    split_scope = _require_string_sequence(
        window_outputs.get("split_scope", ("train", "val", "test")),
        field_name="window_outputs.split_scope",
    )
    invalid_splits = [split for split in split_scope if split not in _SUMMARY_SPLITS]
    if invalid_splits:
        raise ContractValidationError(
            "`window_outputs.split_scope` supports only train/val/test; "
            f"received {', '.join(invalid_splits)}."
        )
    fault_counter_policy = _require_mapping(
        payload.get("fault_counter_policy"),
        field_name="fault_counter_policy",
    )
    planned_outputs = _require_mapping(
        payload.get("planned_outputs"),
        field_name="planned_outputs",
    )
    return PersistencePrepConfig(
        config_path=resolved_path,
        profile_name=profile_name,
        window_output_artifact_name=_require_non_empty_string(
            window_outputs.get("artifact_name", "leanlean_window_outputs_template.json"),
            field_name="window_outputs.artifact_name",
        ),
        scaffold_json_name=_require_non_empty_string(
            planned_outputs.get("scaffold_json_name", "leanlean_persistence_tuning_prep.json"),
            field_name="planned_outputs.scaffold_json_name",
        ),
        split_scope=tuple(split_scope),
        fields=tuple(output_fields),
        fault_counter_policy=_normalize_fault_counter_policy(fault_counter_policy),
        planned_outputs={
            "summary_csv_name": _require_non_empty_string(
                planned_outputs.get("summary_csv_name", "persistence_tuning_summary.csv"),
                field_name="planned_outputs.summary_csv_name",
            ),
            "summary_md_name": _require_non_empty_string(
                planned_outputs.get("summary_md_name", "persistence_tuning_summary.md"),
                field_name="planned_outputs.summary_md_name",
            ),
        },
        notes=tuple(
            _require_string_sequence(payload.get("notes", ()), field_name="notes")
        ),
    )


def prepare_leanlean_deployment_candidate(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
    *,
    search_seed: int | None = None,
) -> PreparedLeanLeanDeploymentCandidate:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = (
        Path(run_root).resolve()
        if run_root is not None
        else (DEFAULT_RUNS_ROOT / resolved_config_path.stem / "manual_run").resolve()
    )
    payload = load_leanlean_deployment_candidate_config(resolved_config_path)
    candidate = _require_mapping(
        payload["leanlean_deployment_candidate"],
        field_name="leanlean_deployment_candidate",
    )

    source_profile_path = _resolve_relative_path(
        config_file=resolved_config_path,
        raw_path=payload["source_profile"],
        field_name="source_profile",
    )
    source_profile = load_consumer_config(source_profile_path)
    locked_frontend = load_locked_frontend_spec(source_profile)
    if locked_frontend is None:
        raise ContractValidationError(
            "Lean-Lean deployment candidate requires `locked_frontend.enabled: true` in the source profile."
        )
    if not locked_frontend.temporal_features_enabled:
        raise ContractValidationError(
            "Lean-Lean deployment candidate requires temporal features on the locked frontend."
        )
    hard_mode = source_profile.get("hard_mode", {})
    include_test_metrics_in_frontend = (
        bool(hard_mode.get("include_test_metrics_in_frontend", False))
        if isinstance(hard_mode, Mapping)
        else False
    )
    if include_test_metrics_in_frontend:
        raise ContractValidationError(
            "Lean-Lean deployment candidate requires `hard_mode.include_test_metrics_in_frontend: false`."
        )

    evaluation = _parse_evaluation_config(candidate.get("evaluation"))
    selection_spec = _selection_spec_from_mapping(
        candidate.get("selection_spec"),
        path="leanlean_deployment_candidate.selection_spec",
    )
    search_config = _parse_search_config(
        candidate.get("search"),
        path="leanlean_deployment_candidate.search",
    )
    if search_seed is not None:
        search_config = _with_search_seed(search_config, search_seed=search_seed)

    inventory_rows = _resolve_inventory_rows(source_profile)
    temporal_feature_config = load_temporal_feature_config(source_profile)
    source_bundle = _materialize_source_bundle(
        inventory_rows,
        output_dir=resolved_run_root / "_inputs" / "shared_source_bundle",
        profile_name=str(payload["profile_name"]),
        selection_name="leanlean_deployment_candidate_shared",
        temporal_feature_config=temporal_feature_config,
    )
    shared_frontend_dir = resolved_run_root / "01_shared_frontend"
    shared_frontend = build_locked_frontend_stage_materialization(
        stage_key=StageKey.LEAN_MAIN_SCREEN,
        stage_output_dir=shared_frontend_dir,
        source_bundle=source_bundle,
        include_test_metrics_in_frontend=False,
        locked_frontend=locked_frontend,
    )
    shared_bundle = _materialize_shared_backend_bundle(
        bundle_dir=resolved_run_root / "02_shared_backend_bundle",
        source_bundle=source_bundle,
        deep_input=shared_frontend.downstream_deep_input,
    )
    lean_training_config, deep_training_config = _load_backend_training_configs(source_profile)
    split_counts = Counter(str(row["split"]) for row in inventory_rows)
    state_counts = Counter(str(row["binary_label"]) for row in inventory_rows)
    comparison_prepared = PreparedBackendArchitectureComparison(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        source_profile_path=source_profile_path,
        source_profile_name=str(source_profile["profile_name"]),
        locked_frontend=locked_frontend,
        source_bundle=source_bundle,
        shared_bundle=shared_bundle,
        shared_frontend_dir=shared_frontend_dir,
        shared_row_count=len(inventory_rows),
        split_counts=dict(sorted(split_counts.items())),
        state_counts=dict(sorted(state_counts.items())),
        evaluation=evaluation,
        selection_spec=selection_spec,
        search_config=search_config,
        lean_training_config=lean_training_config,
        deep_training_config=deep_training_config,
        lean_deep_config=SearchVariantConfig(seed_offset=0),
        lean_lean_config=SearchVariantConfig(seed_offset=0),
    )

    max_selected_k = _resolve_max_selected_k_per_class(source_profile)
    persistence_config_path = candidate.get("persistence_tuning_config", payload.get("persistence_tuning_config"))
    persistence_config = (
        load_persistence_prep_config(
            _resolve_relative_path(
                config_file=resolved_config_path,
                raw_path=persistence_config_path,
                field_name="persistence_tuning_config",
            )
        )
        if persistence_config_path is not None
        else (
            load_persistence_prep_config(DEFAULT_PERSISTENCE_CONFIG_PATH)
            if DEFAULT_PERSISTENCE_CONFIG_PATH.exists()
            else None
        )
    )
    deploy_constraints = dict(source_profile.get("deploy_constraints", {}))
    ranking_intent = dict(source_profile.get("ranking_intent", {}))
    deploy_export = _device_agnostic_export(
        source_profile,
        k_candidates=tuple(range(1, max_selected_k + 1)),
    )
    return PreparedLeanLeanDeploymentCandidate(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        source_profile_path=source_profile_path,
        source_profile_name=str(source_profile["profile_name"]),
        locked_frontend=locked_frontend,
        comparison_prepared=comparison_prepared,
        deploy_export=deploy_export,
        deploy_constraints=deploy_constraints,
        ranking_intent=ranking_intent,
        persistence_config=persistence_config,
    )


def write_leanlean_deployment_candidate_plan(
    prepared: PreparedLeanLeanDeploymentCandidate,
) -> Path:
    comparison = prepared.comparison_prepared
    plan_path = prepared.run_root / PLAN_ARTIFACT_NAME
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "profile_name": prepared.profile_name,
        "config_path": str(prepared.config_path),
        "run_root": str(prepared.run_root),
        "source_profile_path": str(prepared.source_profile_path),
        "source_profile_name": prepared.source_profile_name,
        "locked_frontend": prepared.locked_frontend.to_dict(),
        "shared_dataset": {
            "row_count": comparison.shared_row_count,
            "split_counts": dict(comparison.split_counts),
            "state_counts": dict(comparison.state_counts),
        },
        "shared_bundle": {
            "bundle_dir": str(comparison.shared_bundle.bundle_dir.resolve()),
            "bundle_contract_path": str(comparison.shared_bundle.contract_path.resolve()),
            "frontend_input_id": comparison.shared_bundle.frontend_input_id,
            "frontend_fingerprint": comparison.shared_bundle.frontend_fingerprint,
            "semantic_bit_length": comparison.shared_bundle.semantic_bit_length,
            "packed_bit_length": comparison.shared_bundle.packed_bit_length,
            "packed64_compatibility": True,
            "preserves_temporal_threshold_36_identity": True,
        },
        "deployment_candidate": {
            "architecture_locked": "LEAN_LEAN",
            "frontend_regime_locked": prepared.locked_frontend.regime_id,
            "strict_split_discipline": True,
            "persistence_runtime_enabled": False,
            "consumer_side_first": True,
        },
        "deploy_constraints": dict(prepared.deploy_constraints),
        "deploy_export": dict(prepared.deploy_export),
        "ranking_intent": {
            "primary": str(prepared.ranking_intent.get("primary", "")),
            "secondary": str(prepared.ranking_intent.get("secondary", "")),
            "tertiary": str(prepared.ranking_intent.get("tertiary", "")),
            "quaternary": str(prepared.ranking_intent.get("quaternary", "")),
            "quinary": str(prepared.ranking_intent.get("quinary", "")),
            "notes": [str(note) for note in prepared.ranking_intent.get("notes", ())],
        },
        "search": {
            "selection_spec": {
                "primary_metric": comparison.selection_spec.primary_metric,
                "tiebreak_metrics": list(comparison.selection_spec.tiebreak_metrics),
            },
            "leanlean_search": _evolution_config_to_dict(comparison.search_config),
        },
        "evaluation": {
            "summary_metric_split": comparison.evaluation.summary_metric_split,
            "separability_split": comparison.evaluation.separability_split,
            "latency_split": comparison.evaluation.latency_split,
            "latency_warmup_passes": comparison.evaluation.latency_warmup_passes,
            "latency_timed_passes": comparison.evaluation.latency_timed_passes,
        },
        "backend_request": {
            "backend": comparison.lean_training_config.backend,
            "allow_backend_fallback": comparison.lean_training_config.allow_backend_fallback,
        },
        "persistence_prep": (
            prepared.persistence_config.to_dict()
            if prepared.persistence_config is not None
            else None
        ),
    }
    _write_json(plan_path, payload)
    return plan_path


def write_persistence_prep_artifacts(
    prepared: PreparedLeanLeanDeploymentCandidate,
    *,
    deployment_summary_path: Path | None = None,
    variant_result: Mapping[str, object] | None = None,
) -> dict[str, Path] | None:
    persistence = prepared.persistence_config
    if persistence is None:
        return None
    prep_dir = prepared.run_root / "persistence_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    window_output_template_path = prep_dir / persistence.window_output_artifact_name
    scaffold_path = prep_dir / persistence.scaffold_json_name
    planned_summary_csv_path = prep_dir / str(persistence.planned_outputs["summary_csv_name"])
    planned_summary_md_path = prep_dir / str(persistence.planned_outputs["summary_md_name"])
    variant_paths = {}
    if variant_result is not None:
        variant_paths = {
            "variant_summary_path": str(variant_result.get("variant_summary_path", "")),
            "artifact_path": str(variant_result.get("artifact_path", "")),
            "metrics_summary_path": str(
                variant_result.get("native_paths", {}).get("metrics_summary_path", "")
            ),
        }
    _write_json(
        window_output_template_path,
        {
            "schema_version": WINDOW_OUTPUT_TEMPLATE_SCHEMA_VERSION,
            "profile_name": persistence.profile_name,
            "deployment_candidate_run_root": str(prepared.run_root),
            "source_profile_path": str(prepared.source_profile_path),
            "classification_scope": "per-window healthy/unhealthy outputs from Lean-Lean only",
            "split_scope": list(persistence.split_scope),
            "fields": list(persistence.fields),
            "records": [],
            "notes": [
                "Template only: populate after classifier inference export is added or replayed.",
                "Keep persistence tuning separate from raw classifier evaluation artifacts.",
            ],
        },
    )
    _write_json(
        scaffold_path,
        {
            "schema_version": PERSISTENCE_PREP_SCHEMA_VERSION,
            "profile_name": persistence.profile_name,
            "config_path": str(persistence.config_path),
            "deployment_candidate_run_root": str(prepared.run_root),
            "deployment_candidate_summary_path": (
                str(deployment_summary_path.resolve()) if deployment_summary_path is not None else None
            ),
            "source_profile_path": str(prepared.source_profile_path),
            "locked_frontend": prepared.locked_frontend.to_dict(),
            "input_contract": {
                "window_output_template_path": str(window_output_template_path.resolve()),
                "classifier_variant": "LEAN_LEAN",
                "expected_classifier_fields": list(persistence.fields),
                "source_artifacts": {
                    "shared_bundle_contract_path": str(
                        prepared.comparison_prepared.shared_bundle.contract_path.resolve()
                    ),
                    "shared_frontend_dir": str(
                        prepared.comparison_prepared.shared_frontend_dir.resolve()
                    ),
                    **variant_paths,
                },
            },
            "fault_counter_policy": dict(persistence.fault_counter_policy),
            "planned_runtime_logic": {
                "increment_on_unhealthy": int(persistence.fault_counter_policy["increment_on_unhealthy"]),
                "decrement_on_healthy": int(persistence.fault_counter_policy["decrement_on_healthy"]),
                "yellow_threshold": int(persistence.fault_counter_policy["yellow_threshold"]),
                "red_threshold": int(persistence.fault_counter_policy["red_threshold"]),
                "optional_latch": bool(persistence.fault_counter_policy["optional_latch"]),
            },
            "planned_outputs": {
                "summary_csv_path": str(planned_summary_csv_path.resolve()),
                "summary_md_path": str(planned_summary_md_path.resolve()),
            },
            "separation_from_classifier_eval": {
                "raw_classifier_metrics_remain_authoritative": True,
                "persistence_changes_alerting_only": True,
                "tuning_split_discipline": "fit on train/val sequences, keep test alarm reporting separate",
            },
            "notes": list(persistence.notes),
        },
    )
    return {
        "window_output_template_path": window_output_template_path,
        "scaffold_path": scaffold_path,
    }


def run_prepared_leanlean_deployment_candidate(
    prepared: PreparedLeanLeanDeploymentCandidate,
) -> Path:
    plan_path = write_leanlean_deployment_candidate_plan(prepared)
    variant_result = _run_lean_lean_variant(prepared.comparison_prepared)
    summary_row = _build_summary_row(
        variant_result,
        summary_metric_split=prepared.comparison_prepared.evaluation.summary_metric_split,
    )
    summary_csv_path = prepared.run_root / "summary.csv"
    _write_summary_csv(summary_csv_path, (summary_row,))
    summary_md_path = prepared.run_root / "summary.md"
    summary_md_path.write_text(
        _build_candidate_markdown(prepared, variant_result),
        encoding="utf-8",
    )
    summary_json_path = prepared.run_root / SUMMARY_ARTIFACT_NAME
    persistence_refs = write_persistence_prep_artifacts(
        prepared,
        deployment_summary_path=summary_json_path,
        variant_result=variant_result,
    )
    _write_json(
        summary_json_path,
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "profile_name": prepared.profile_name,
            "config_path": str(prepared.config_path),
            "run_root": str(prepared.run_root),
            "source_profile_path": str(prepared.source_profile_path),
            "source_profile_name": prepared.source_profile_name,
            "plan_path": str(plan_path.resolve()),
            "summary_csv_path": str(summary_csv_path.resolve()),
            "summary_md_path": str(summary_md_path.resolve()),
            "variant": variant_result,
            "summary_row": summary_row,
            "deploy_constraints": dict(prepared.deploy_constraints),
            "deploy_export": dict(prepared.deploy_export),
            "ranking_intent": {
                key: prepared.ranking_intent.get(key)
                for key in ("primary", "secondary", "tertiary", "quaternary", "quinary")
            },
            "persistence_prep": (
                {
                    "config": prepared.persistence_config.to_dict(),
                    "artifacts": {
                        key: str(path.resolve())
                        for key, path in persistence_refs.items()
                    },
                }
                if prepared.persistence_config is not None and persistence_refs is not None
                else None
            ),
        },
    )
    return summary_json_path


def run_leanlean_deployment_candidate(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
    *,
    search_seed: int | None = None,
) -> Path:
    prepared = prepare_leanlean_deployment_candidate(
        config_path=config_path,
        run_root=run_root,
        search_seed=search_seed,
    )
    return run_prepared_leanlean_deployment_candidate(prepared)


def _build_candidate_markdown(
    prepared: PreparedLeanLeanDeploymentCandidate,
    variant_result: Mapping[str, object],
) -> str:
    metrics = variant_result["metrics"]
    latency = variant_result["latency"]
    separability = variant_result["separability"]
    persistence_line = (
        f"- Persistence prep scaffold: `persistence_prep/{prepared.persistence_config.scaffold_json_name}`"
        if prepared.persistence_config is not None
        else "- Persistence prep scaffold: not configured"
    )
    lines = [
        "# Lean-Lean Deployment Candidate",
        "",
        f"Source profile: `{prepared.source_profile_path}`",
        f"Locked frontend: `{prepared.locked_frontend.regime_id}` ({prepared.locked_frontend.bit_length} semantic bits, packed to 64 only for backend/GPU compatibility)",
        f"Deployment target: `{prepared.deploy_constraints.get('target', '')}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Accuracy | Healthy->Unhealthy FPR | Unhealthy Precision | Unhealthy Recall | Unhealthy F1 | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in _SUMMARY_SPLITS:
        split_metrics = metrics[split_name]
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
            "## Notes",
            "",
            "- Architecture is locked to Lean-Lean only; no Lean-Deep, front-only promotion, multiclass, or ensemble logic is included.",
            "- Train drives search, val remains the winner-selection split, and test remains final reporting only.",
            "- Conservative unhealthy-alert ranking stays ordered as FPR first, then unhealthy precision, recall, F1, and macro F1.",
            f"- Model size: `{int(variant_result['model_size_bytes'])}` bytes; latency on `{latency['split']}`: `{float(latency['per_sample_ms']):.6f}` ms/sample.",
            f"- Train-space separation margin on `{separability['split']}`: `{float(separability['separation_margin']):.6f}`.",
            persistence_line,
        ]
    )
    return "\n".join(lines) + "\n"


def _normalize_fault_counter_policy(raw: Mapping[str, object]) -> dict[str, object]:
    mode = _require_non_empty_string(raw.get("mode", "fault_counter"), field_name="fault_counter_policy.mode")
    if mode != "fault_counter":
        raise ContractValidationError("`fault_counter_policy.mode` must be `fault_counter`.")
    return {
        "mode": mode,
        "increment_on_unhealthy": _require_int(
            raw.get("increment_on_unhealthy"),
            field_name="fault_counter_policy.increment_on_unhealthy",
            minimum=1,
        ),
        "decrement_on_healthy": _require_int(
            raw.get("decrement_on_healthy"),
            field_name="fault_counter_policy.decrement_on_healthy",
            minimum=0,
        ),
        "yellow_threshold": _require_int(
            raw.get("yellow_threshold"),
            field_name="fault_counter_policy.yellow_threshold",
            minimum=1,
        ),
        "red_threshold": _require_int(
            raw.get("red_threshold"),
            field_name="fault_counter_policy.red_threshold",
            minimum=1,
        ),
        "optional_latch": _require_bool(
            raw.get("optional_latch", False),
            field_name="fault_counter_policy.optional_latch",
        ),
    }


def _with_search_seed(config: EvolutionConfig, *, search_seed: int) -> EvolutionConfig:
    if isinstance(search_seed, bool) or not isinstance(search_seed, int) or search_seed < 0:
        raise ContractValidationError("`search_seed` must be a non-negative integer.")
    return EvolutionConfig(
        seed=int(search_seed),
        generations=config.generations,
        population_size=config.population_size,
        mu=config.mu,
        lam=config.lam,
        elite_count=config.elite_count,
        min_layers=config.min_layers,
        max_layers=config.max_layers,
        mutation_rate=config.mutation_rate,
        mutation_rate_schedule=config.mutation_rate_schedule,
        selection_mode=config.selection_mode,
        tournament_k=config.tournament_k,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint=config.checkpoint,
    )


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
        if not isinstance(item, str) or item.strip() == "":
            raise ContractValidationError(f"`{field_name}[{index}]` must be a non-empty string.")
        normalized.append(item.strip())
    return tuple(normalized)


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return int(value)


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a boolean.")
    return value


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PERSISTENCE_CONFIG_PATH",
    "DEFAULT_RUNS_ROOT",
    "PLAN_ARTIFACT_NAME",
    "SUMMARY_ARTIFACT_NAME",
    "PersistencePrepConfig",
    "PreparedLeanLeanDeploymentCandidate",
    "load_leanlean_deployment_candidate_config",
    "load_persistence_prep_config",
    "prepare_leanlean_deployment_candidate",
    "run_leanlean_deployment_candidate",
    "run_prepared_leanlean_deployment_candidate",
    "write_leanlean_deployment_candidate_plan",
    "write_persistence_prep_artifacts",
]
