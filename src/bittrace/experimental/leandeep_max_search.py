"""Separate Lean-Deep max-search workflow over the locked temporal frontend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.core.config import EvolutionConfig
from bittrace.core.deep.engine import _apply_layers_as_embedding, run_deep_evolution
from bittrace.v3 import ContractValidationError, StageKey

from .backend_architecture_comparison import (
    DEFAULT_RUNS_ROOT as BACKEND_DEFAULT_RUNS_ROOT,
    PreparedBackendArchitectureComparison,
    _build_summary_row,
    _compute_separability,
    _evaluate_deep_split,
    _evolution_config_to_dict,
    _load_deep_artifact_model,
    _materialize_shared_backend_bundle,
    _measure_latency,
    _parse_evaluation_config,
    _predict_deep_rows,
    _resolve_relative_path,
    _selection_spec_from_mapping,
    _write_json,
    _write_summary_csv,
)
from bittrace.source.full_binary_campaign import (
    _device_agnostic_export,
    _load_backend_training_configs,
    _materialize_source_bundle,
    _resolve_inventory_rows,
    _resolve_max_selected_k_per_class,
    load_consumer_config,
)
from .leanlean_max_search import (
    MaxSearchSpec,
    _comparison_against_reference,
    _default_report_title,
    _exploration_policy,
    _extract_reference_layer_cap,
    _load_json_mapping,
    _meaningful_metric_gain_from_deltas,
    _plan_artifact_name,
    _plan_schema_version,
    _ranking_metrics_from_split_metrics,
    _ranking_policy_payload,
    _ranking_sort_key,
    _require_int,
    _require_mapping,
    _require_probability,
    _require_string_sequence,
    _resolve_fractional_branch_count,
    _resolve_optional_relative_path,
    _resolve_required_relative_path,
    _resolved_trial_seed,
    _search_budget,
    _search_provenance_breakdown,
    _summary_artifact_name,
    _summary_schema_version,
    _survivor_rank,
    _validation_alertability,
    _with_search_seed,
)
from bittrace.source.locked_frontend import (
    LockedFrontendSpec,
    build_locked_frontend_stage_materialization,
    load_locked_frontend_spec,
)
from bittrace.source.temporal_features import load_temporal_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experimental" / "leandeep_max_search.yaml"
DEFAULT_RUNS_ROOT = BACKEND_DEFAULT_RUNS_ROOT
_SUMMARY_SPLITS = ("train", "val", "test")
_CANDIDATE_SUMMARY_NAME = "candidate_summary.json"
_RESEARCH_MAX_SEARCH_LAYER_CEILING = 8
_DECISION_PATH = "shared_frontend_plus_deep_all_layer_residue_readout"
_CANDIDATE_LABEL = "Lean-Deep max-search candidate"


@dataclass(frozen=True, slots=True)
class PreparedLeanDeepMaxSearch:
    config_path: Path
    run_root: Path
    profile_name: str
    report_title: str
    source_profile_path: Path
    source_profile_name: str
    locked_frontend: LockedFrontendSpec
    comparison_prepared: PreparedBackendArchitectureComparison
    deploy_export: Mapping[str, object]
    deploy_constraints: Mapping[str, object]
    ranking_intent: Mapping[str, object]
    notes: tuple[str, ...]
    max_search_spec: MaxSearchSpec
    deployment_candidate_summary_path: Path
    deployment_candidate_summary: Mapping[str, object]
    current_max_search_summary_path: Path | None
    current_max_search_summary: Mapping[str, object] | None
    deep_layer_max_search_summary_path: Path | None
    deep_layer_max_search_summary: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _CandidatePlan:
    candidate_order: int
    candidate_id: str
    family_branch_index: int
    provenance: Mapping[str, object]
    evolution_config: EvolutionConfig
    output_dir: Path


@dataclass(frozen=True, slots=True)
class _TrialResult:
    trial_index: int
    seed: int
    run_dir: Path
    artifact_path: Path
    metrics_summary_path: Path
    history_json_path: Path
    history_csv_path: Path
    split_metrics: Mapping[str, Mapping[str, object]]
    latency: Mapping[str, object]
    separability: Mapping[str, object]
    layer_count: int
    model_size_bytes: int
    embedding_bit_length: int
    completed_generations: int
    stopped_early: bool
    engine_best_candidate: Mapping[str, object]
    search_budget: Mapping[str, object]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    candidate_order: int
    candidate_id: str
    provenance: Mapping[str, object]
    chosen_trial: _TrialResult
    trials: tuple[_TrialResult, ...]
    ranking_metrics: Mapping[str, float]
    search_ranking_eligible: bool
    validation_selection_eligibility: Mapping[str, object]
    summary_path: Path


def load_leandeep_max_search_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    for key in (
        "profile_name",
        "source_profile",
        "deployment_candidate_summary_path",
        "leandeep_max_search",
    ):
        if key not in payload:
            raise ContractValidationError(f"`{resolved_path}` is missing required top-level key `{key}`.")
    return payload


def prepare_leandeep_max_search(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
) -> PreparedLeanDeepMaxSearch:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = (
        Path(run_root).resolve()
        if run_root is not None
        else (DEFAULT_RUNS_ROOT / resolved_config_path.stem / "manual_run").resolve()
    )
    payload = load_leandeep_max_search_config(resolved_config_path)
    max_search = _require_mapping(
        payload["leandeep_max_search"],
        field_name="leandeep_max_search",
    )
    notes = _require_string_sequence(
        max_search.get("notes", ()),
        field_name="leandeep_max_search.notes",
    )
    report_title = str(max_search.get("report_title", _default_report_title(str(payload["profile_name"])))).strip()
    if report_title == "":
        raise ContractValidationError("`leandeep_max_search.report_title` must be a non-empty string when provided.")

    source_profile_path = _resolve_relative_path(
        config_file=resolved_config_path,
        raw_path=payload["source_profile"],
        field_name="source_profile",
    )
    source_profile = load_consumer_config(source_profile_path)
    locked_frontend = load_locked_frontend_spec(source_profile)
    if locked_frontend is None:
        raise ContractValidationError(
            "Lean-Deep max search requires `locked_frontend.enabled: true` in the source profile."
        )
    if not locked_frontend.temporal_features_enabled:
        raise ContractValidationError(
            "Lean-Deep max search requires temporal features on the locked frontend."
        )
    hard_mode = source_profile.get("hard_mode", {})
    include_test_metrics_in_frontend = (
        bool(hard_mode.get("include_test_metrics_in_frontend", False))
        if isinstance(hard_mode, Mapping)
        else False
    )
    if include_test_metrics_in_frontend:
        raise ContractValidationError(
            "Lean-Deep max search requires `hard_mode.include_test_metrics_in_frontend: false`."
        )

    evaluation = _parse_evaluation_config(max_search.get("evaluation"))
    search_spec = _parse_max_search_spec(
        max_search.get("search"),
        path="leandeep_max_search.search",
    )

    inventory_rows = _resolve_inventory_rows(source_profile)
    temporal_feature_config = load_temporal_feature_config(source_profile)
    source_bundle = _materialize_source_bundle(
        inventory_rows,
        output_dir=resolved_run_root / "_inputs" / "shared_source_bundle",
        profile_name=str(payload["profile_name"]),
        selection_name="leandeep_max_search_shared",
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
        selection_spec=search_spec.selection_spec,
        search_config=search_spec.local_refine_evolution,
        lean_training_config=lean_training_config,
        deep_training_config=deep_training_config,
        lean_deep_config=None,  # type: ignore[arg-type]
        lean_lean_config=None,  # type: ignore[arg-type]
    )
    max_selected_k = _resolve_max_selected_k_per_class(source_profile)
    deploy_constraints = dict(source_profile.get("deploy_constraints", {}))
    ranking_intent = dict(source_profile.get("ranking_intent", {}))
    deploy_export = _device_agnostic_export(
        source_profile,
        k_candidates=tuple(range(1, max_selected_k + 1)),
    )

    deployment_candidate_summary_path = _resolve_required_relative_path(
        config_file=resolved_config_path,
        raw_path=payload.get("deployment_candidate_summary_path"),
        field_name="deployment_candidate_summary_path",
    )
    current_max_search_summary_path = _resolve_optional_relative_path(
        config_file=resolved_config_path,
        raw_path=payload.get("current_max_search_summary_path"),
        field_name="current_max_search_summary_path",
    )
    deep_layer_max_search_summary_path = _resolve_optional_relative_path(
        config_file=resolved_config_path,
        raw_path=payload.get("deep_layer_max_search_summary_path"),
        field_name="deep_layer_max_search_summary_path",
    )
    deployment_candidate_summary = _load_json_mapping(deployment_candidate_summary_path)
    current_max_search_summary = (
        _load_json_mapping(current_max_search_summary_path)
        if current_max_search_summary_path is not None
        else None
    )
    deep_layer_max_search_summary = (
        _load_json_mapping(deep_layer_max_search_summary_path)
        if deep_layer_max_search_summary_path is not None
        else None
    )

    return PreparedLeanDeepMaxSearch(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        report_title=report_title,
        source_profile_path=source_profile_path,
        source_profile_name=str(source_profile["profile_name"]),
        locked_frontend=locked_frontend,
        comparison_prepared=comparison_prepared,
        deploy_export=deploy_export,
        deploy_constraints=deploy_constraints,
        ranking_intent=ranking_intent,
        notes=notes,
        max_search_spec=search_spec,
        deployment_candidate_summary_path=deployment_candidate_summary_path,
        deployment_candidate_summary=deployment_candidate_summary,
        current_max_search_summary_path=current_max_search_summary_path,
        current_max_search_summary=current_max_search_summary,
        deep_layer_max_search_summary_path=deep_layer_max_search_summary_path,
        deep_layer_max_search_summary=deep_layer_max_search_summary,
    )


def write_leandeep_max_search_plan(
    prepared: PreparedLeanDeepMaxSearch,
) -> Path:
    comparison = prepared.comparison_prepared
    plan_path = prepared.run_root / _plan_artifact_name(prepared.profile_name)
    payload = {
        "schema_version": _plan_schema_version(prepared.profile_name),
        "profile_name": prepared.profile_name,
        "report_title": prepared.report_title,
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
        "experiment_track": {
            "kind": prepared.profile_name,
            "separate_from_deployment_candidate": True,
            "deployment_candidate_replaced": False,
            "frontend_regime_locked": prepared.locked_frontend.regime_id,
            "backend_architecture_locked": "LEAN_DEEP",
            "persistence_runtime_enabled": False,
            "binary_only": True,
        },
        "deploy_constraints": dict(prepared.deploy_constraints),
        "deploy_export": dict(prepared.deploy_export),
        "ranking_policy": _ranking_policy_payload(),
        "ranking_intent": {
            "primary": str(prepared.ranking_intent.get("primary", "")),
            "secondary": str(prepared.ranking_intent.get("secondary", "")),
            "tertiary": str(prepared.ranking_intent.get("tertiary", "")),
            "quaternary": str(prepared.ranking_intent.get("quaternary", "")),
            "quinary": str(prepared.ranking_intent.get("quinary", "")),
            "notes": [str(note) for note in prepared.ranking_intent.get("notes", ())],
        },
        "evaluation": {
            "summary_metric_split": comparison.evaluation.summary_metric_split,
            "separability_split": comparison.evaluation.separability_split,
            "latency_split": comparison.evaluation.latency_split,
            "latency_warmup_passes": comparison.evaluation.latency_warmup_passes,
            "latency_timed_passes": comparison.evaluation.latency_timed_passes,
        },
        "backend_request": {
            "backend": comparison.deep_training_config.backend,
            "allow_backend_fallback": comparison.deep_training_config.allow_backend_fallback,
            "k_medoids_per_class": comparison.deep_training_config.k_medoids_per_class,
            "adaptive_k": comparison.deep_training_config.adaptive_k,
            "adaptive_k_candidates": list(comparison.deep_training_config.adaptive_k_candidates),
        },
        "search": {
            "selection_spec": {
                "primary_metric": comparison.selection_spec.primary_metric,
                "tiebreak_metrics": list(comparison.selection_spec.tiebreak_metrics),
            },
            "exploration_policy": _exploration_policy(prepared.max_search_spec),
            "local_refine_evolution": _evolution_config_to_dict(prepared.max_search_spec.local_refine_evolution),
            "bounded_random_evolution": (
                _evolution_config_to_dict(prepared.max_search_spec.bounded_random_evolution)
                if prepared.max_search_spec.bounded_random_evolution is not None
                else None
            ),
            "winner_replay_evolution": (
                _evolution_config_to_dict(prepared.max_search_spec.winner_replay_evolution)
                if prepared.max_search_spec.winner_replay_evolution is not None
                else None
            ),
            "winner_mutation_evolution": (
                _evolution_config_to_dict(prepared.max_search_spec.winner_mutation_evolution)
                if prepared.max_search_spec.winner_mutation_evolution is not None
                else None
            ),
        },
        "comparison_targets": {
            "deployment_candidate_summary_path": str(prepared.deployment_candidate_summary_path.resolve()),
            "current_max_search_summary_path": (
                str(prepared.current_max_search_summary_path.resolve())
                if prepared.current_max_search_summary_path is not None
                else None
            ),
            "deep_layer_max_search_summary_path": (
                str(prepared.deep_layer_max_search_summary_path.resolve())
                if prepared.deep_layer_max_search_summary_path is not None
                else None
            ),
        },
        "notes": list(prepared.notes),
    }
    _write_json(plan_path, payload)
    return plan_path


def run_prepared_leandeep_max_search(
    prepared: PreparedLeanDeepMaxSearch,
) -> Path:
    plan_path = write_leandeep_max_search_plan(prepared)
    initial_plans = _initial_candidate_plans(prepared, next_candidate_order=1)
    candidate_results: list[_CandidateResult] = [
        _run_candidate_plan(prepared, plan)
        for plan in initial_plans
    ]
    provisional_winner = _select_best_candidate(candidate_results)
    followup_plans = _followup_candidate_plans(
        prepared,
        next_candidate_order=len(candidate_results) + 1,
        winner_candidate=provisional_winner,
    )
    candidate_results.extend(_run_candidate_plan(prepared, plan) for plan in followup_plans)
    final_winner = _select_best_candidate(candidate_results)
    ranked_results = _rank_candidate_results(candidate_results)

    summary_rows = [
        _candidate_summary_row(
            result,
            overall_rank=rank_index,
            survivor_rank=_survivor_rank(ranked_results, result.candidate_id),
        )
        for rank_index, result in enumerate(ranked_results, start=1)
    ]
    summary_csv_path = prepared.run_root / "summary.csv"
    _write_summary_csv(summary_csv_path, summary_rows)
    provenance_breakdown = _search_provenance_breakdown(candidate_results)
    comparisons: dict[str, dict[str, object]] = {
        "deployment_candidate": _comparison_against_reference(
            final_winner,
            reference_summary=prepared.deployment_candidate_summary,
            reference_label="deployment_candidate",
        ),
    }
    if prepared.current_max_search_summary is not None:
        comparisons["leanlean_max_search_winner"] = _comparison_against_reference(
            final_winner,
            reference_summary=prepared.current_max_search_summary,
            reference_label="leanlean_max_search_winner",
        )
    if prepared.deep_layer_max_search_summary is not None:
        comparisons["leanlean_deep_max_search_winner"] = _comparison_against_reference(
            final_winner,
            reference_summary=prepared.deep_layer_max_search_summary,
            reference_label="leanlean_deep_max_search_winner",
        )
    search_scale_comparison = _search_scale_comparison(prepared)
    conservative_operating_point_assessment = _lean_deep_assessment(
        prepared,
        final_winner=final_winner,
        comparisons=comparisons,
    )
    summary_md_path = prepared.run_root / "summary.md"
    summary_md_path.write_text(
        _build_summary_markdown(
            prepared,
            ranked_results=ranked_results,
            final_winner=final_winner,
            provenance_breakdown=provenance_breakdown,
            comparisons=comparisons,
            search_scale_comparison=search_scale_comparison,
            conservative_operating_point_assessment=conservative_operating_point_assessment,
        ),
        encoding="utf-8",
    )
    summary_json_path = prepared.run_root / _summary_artifact_name(prepared.profile_name)
    _write_json(
        summary_json_path,
        {
            "schema_version": _summary_schema_version(prepared.profile_name),
            "profile_name": prepared.profile_name,
            "report_title": prepared.report_title,
            "config_path": str(prepared.config_path),
            "run_root": str(prepared.run_root),
            "source_profile_path": str(prepared.source_profile_path),
            "source_profile_name": prepared.source_profile_name,
            "plan_path": str(plan_path.resolve()),
            "summary_csv_path": str(summary_csv_path.resolve()),
            "summary_md_path": str(summary_md_path.resolve()),
            "separate_track": True,
            "deployment_candidate_replaced": False,
            "locked_frontend": prepared.locked_frontend.to_dict(),
            "ranking_policy": _ranking_policy_payload(),
            "deploy_constraints": dict(prepared.deploy_constraints),
            "deploy_export": dict(prepared.deploy_export),
            "ranking_intent": {
                key: prepared.ranking_intent.get(key)
                for key in ("primary", "secondary", "tertiary", "quaternary", "quinary")
            },
            "notes": list(prepared.notes),
            "provisional_winner_candidate_id": provisional_winner.candidate_id,
            "final_winner_candidate_id": final_winner.candidate_id,
            "search_provenance_breakdown": provenance_breakdown,
            "search_scale_comparison": search_scale_comparison,
            "conservative_operating_point_assessment": conservative_operating_point_assessment,
            "deployment_candidate_summary_path": str(prepared.deployment_candidate_summary_path.resolve()),
            "current_max_search_summary_path": (
                str(prepared.current_max_search_summary_path.resolve())
                if prepared.current_max_search_summary_path is not None
                else None
            ),
            "deep_layer_max_search_summary_path": (
                str(prepared.deep_layer_max_search_summary_path.resolve())
                if prepared.deep_layer_max_search_summary_path is not None
                else None
            ),
            "comparisons": comparisons,
            "summary_row": _candidate_summary_row(final_winner, overall_rank=1, survivor_rank=1),
            "winner": _candidate_result_to_dict(final_winner),
            "candidates": [
                _candidate_result_to_dict(result)
                for result in ranked_results
            ],
        },
    )
    return summary_json_path


def run_leandeep_max_search(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
) -> Path:
    prepared = prepare_leandeep_max_search(
        config_path=config_path,
        run_root=run_root,
    )
    return run_prepared_leandeep_max_search(prepared)


def _parse_max_search_spec(raw: object, *, path: str) -> MaxSearchSpec:
    raw_mapping = _require_mapping(raw, field_name=path)
    trials_per_candidate = _require_int(
        raw_mapping.get("trials_per_candidate"),
        field_name=f"{path}.trials_per_candidate",
        minimum=1,
    )
    initial_search_branches = _require_int(
        raw_mapping.get("search_branches", 1),
        field_name=f"{path}.search_branches",
        minimum=1,
    )
    bounded_random_fraction = _require_probability(
        raw_mapping.get("bounded_random_fraction", 0.0),
        field_name=f"{path}.bounded_random_fraction",
    )
    bounded_random_branches = _resolve_fractional_branch_count(
        initial_search_branches=initial_search_branches,
        fraction=bounded_random_fraction,
        field_name=f"{path}.bounded_random_fraction",
    )
    winner_replay_branches = _require_int(
        raw_mapping.get("winner_replay_branches", 0),
        field_name=f"{path}.winner_replay_branches",
        minimum=0,
    )
    winner_mutation_branches = _require_int(
        raw_mapping.get("winner_mutation_branches", 0),
        field_name=f"{path}.winner_mutation_branches",
        minimum=0,
    )
    seed_stride = _require_int(
        raw_mapping.get("seed_stride", 101),
        field_name=f"{path}.seed_stride",
        minimum=1,
    )
    selection_spec = _selection_spec_from_mapping(
        raw_mapping.get("selection_spec"),
        path=f"{path}.selection_spec",
    )
    local_refine_evolution = _parse_research_search_config(
        raw_mapping.get("evolution"),
        path=f"{path}.evolution",
    )
    return MaxSearchSpec(
        trials_per_candidate=trials_per_candidate,
        seed_stride=seed_stride,
        initial_search_branches=initial_search_branches,
        bounded_random_fraction=bounded_random_fraction,
        bounded_random_branches=bounded_random_branches,
        winner_replay_branches=winner_replay_branches,
        winner_mutation_branches=winner_mutation_branches,
        selection_spec=selection_spec,
        local_refine_evolution=local_refine_evolution,
        bounded_random_evolution=_merge_optional_evolution(
            base_path=path,
            key="bounded_random_evolution",
            raw_override=raw_mapping.get("bounded_random_evolution"),
            required=bounded_random_branches > 0,
        ),
        winner_replay_evolution=_merge_optional_evolution(
            base_path=path,
            key="winner_replay_evolution",
            raw_override=raw_mapping.get("winner_replay_evolution"),
            required=winner_replay_branches > 0,
        ),
        winner_mutation_evolution=_merge_optional_evolution(
            base_path=path,
            key="winner_mutation_evolution",
            raw_override=raw_mapping.get("winner_mutation_evolution"),
            required=winner_mutation_branches > 0,
        ),
    )


def _merge_optional_evolution(
    *,
    base_path: str,
    key: str,
    raw_override: object,
    required: bool,
) -> EvolutionConfig | None:
    if raw_override in ({}, None):
        if required:
            raise ContractValidationError(f"`{base_path}.{key}` is required for the configured branches.")
        return None
    return _parse_research_search_config(raw_override, path=f"{base_path}.{key}")


def _parse_research_search_config(raw: object, *, path: str) -> EvolutionConfig:
    if not isinstance(raw, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    try:
        config = EvolutionConfig.from_mapping(raw)
    except Exception as exc:  # pragma: no cover - config validation path
        raise ContractValidationError(f"`{path}` is invalid: {exc}") from exc
    if config.checkpoint.save_path is not None or config.checkpoint.resume_from is not None:
        raise ContractValidationError(
            "This research-only max-search path forbids persistence; leave the evolution checkpoint empty."
        )
    if config.max_layers > _RESEARCH_MAX_SEARCH_LAYER_CEILING:
        raise ContractValidationError(
            "This research-only Lean-Deep max-search path caps `search.max_layers` at 8."
        )
    return config


def _initial_candidate_plans(
    prepared: PreparedLeanDeepMaxSearch,
    *,
    next_candidate_order: int,
) -> tuple[_CandidatePlan, ...]:
    spec = prepared.max_search_spec
    plans: list[_CandidatePlan] = []
    candidate_order = next_candidate_order
    local_refine_branches = spec.initial_search_branches - spec.bounded_random_branches
    for family_branch_index in range(1, local_refine_branches + 1):
        plans.append(
            _candidate_plan(
                prepared,
                candidate_order=candidate_order,
                family_branch_index=family_branch_index,
                source="local_refine",
                lineage_basis="frontend_input",
                evolution_config=spec.local_refine_evolution,
            )
        )
        candidate_order += 1
    if spec.bounded_random_branches > 0:
        if spec.bounded_random_evolution is None:  # pragma: no cover - config guard
            raise ContractValidationError("Bounded-random max search requires `bounded_random_evolution`.")
        for family_branch_index in range(1, spec.bounded_random_branches + 1):
            plans.append(
                _candidate_plan(
                    prepared,
                    candidate_order=candidate_order,
                    family_branch_index=family_branch_index,
                    source="bounded_random",
                    lineage_basis="frontend_input",
                    evolution_config=spec.bounded_random_evolution,
                )
            )
            candidate_order += 1
    return tuple(plans)


def _followup_candidate_plans(
    prepared: PreparedLeanDeepMaxSearch,
    *,
    next_candidate_order: int,
    winner_candidate: _CandidateResult,
) -> tuple[_CandidatePlan, ...]:
    spec = prepared.max_search_spec
    plans: list[_CandidatePlan] = []
    candidate_order = next_candidate_order
    winner_source = {
        "winner_source_candidate_id": winner_candidate.candidate_id,
        "winner_source_provenance": winner_candidate.provenance.get("source"),
        "winner_source_ranking_metrics": dict(winner_candidate.ranking_metrics),
        "warm_start_used": False,
        "winner_followup_mode": "search_family_only_no_artifact_resume",
    }
    if spec.winner_replay_branches > 0:
        if spec.winner_replay_evolution is None:  # pragma: no cover - config guard
            raise ContractValidationError("Winner replay requires `winner_replay_evolution`.")
        for family_branch_index in range(1, spec.winner_replay_branches + 1):
            plans.append(
                _candidate_plan(
                    prepared,
                    candidate_order=candidate_order,
                    family_branch_index=family_branch_index,
                    source="winner_replay",
                    lineage_basis="provisional_validation_winner",
                    evolution_config=spec.winner_replay_evolution,
                    extra_provenance=winner_source,
                )
            )
            candidate_order += 1
    if spec.winner_mutation_branches > 0:
        if spec.winner_mutation_evolution is None:  # pragma: no cover - config guard
            raise ContractValidationError("Winner mutation requires `winner_mutation_evolution`.")
        for family_branch_index in range(1, spec.winner_mutation_branches + 1):
            plans.append(
                _candidate_plan(
                    prepared,
                    candidate_order=candidate_order,
                    family_branch_index=family_branch_index,
                    source="winner_mutation",
                    lineage_basis="provisional_validation_winner",
                    evolution_config=spec.winner_mutation_evolution,
                    extra_provenance=winner_source,
                )
            )
            candidate_order += 1
    return tuple(plans)


def _candidate_plan(
    prepared: PreparedLeanDeepMaxSearch,
    *,
    candidate_order: int,
    family_branch_index: int,
    source: str,
    lineage_basis: str,
    evolution_config: EvolutionConfig,
    extra_provenance: Mapping[str, object] | None = None,
) -> _CandidatePlan:
    frontend_input_id = prepared.comparison_prepared.shared_bundle.frontend_input_id
    candidate_id = f"leandeep-max-{candidate_order:02d}-{source}-{frontend_input_id}"
    provenance = {
        "source": source,
        "lineage_basis": lineage_basis,
        "family_branch_index": family_branch_index,
        "base_seed": int(evolution_config.seed),
        "warm_start_used": False,
    }
    if extra_provenance is not None:
        provenance.update(dict(extra_provenance))
    return _CandidatePlan(
        candidate_order=candidate_order,
        candidate_id=candidate_id,
        family_branch_index=family_branch_index,
        provenance=provenance,
        evolution_config=evolution_config,
        output_dir=prepared.run_root / "candidates" / candidate_id,
    )


def _run_candidate_plan(
    prepared: PreparedLeanDeepMaxSearch,
    candidate_plan: _CandidatePlan,
) -> _CandidateResult:
    candidate_plan.output_dir.mkdir(parents=True, exist_ok=True)
    trials = tuple(
        _run_candidate_trial(
            prepared,
            candidate_plan,
            trial_index=trial_index,
        )
        for trial_index in range(1, prepared.max_search_spec.trials_per_candidate + 1)
    )
    chosen_trial = _select_best_trial(trials)
    validation_selection_eligibility = _validation_alertability(chosen_trial.split_metrics["val"])
    result = _CandidateResult(
        candidate_order=candidate_plan.candidate_order,
        candidate_id=candidate_plan.candidate_id,
        provenance=candidate_plan.provenance,
        chosen_trial=chosen_trial,
        trials=trials,
        ranking_metrics=_ranking_metrics_from_split_metrics(chosen_trial.split_metrics["val"]),
        search_ranking_eligible=bool(validation_selection_eligibility["ranking_eligible"]),
        validation_selection_eligibility=validation_selection_eligibility,
        summary_path=candidate_plan.output_dir / _CANDIDATE_SUMMARY_NAME,
    )
    _write_json(
        result.summary_path,
        {
            "schema_version": "bittrace-bearings-v3-1-leandeep-max-search-candidate-summary-1",
            "candidate_id": result.candidate_id,
            "candidate_order": result.candidate_order,
            "label": _CANDIDATE_LABEL,
            "decision_path": _DECISION_PATH,
            "provenance": dict(result.provenance),
            "search_budget": dict(chosen_trial.search_budget),
            "validation_selection_eligibility": dict(result.validation_selection_eligibility),
            "ranking_metrics": dict(result.ranking_metrics),
            "search_ranking_eligible": result.search_ranking_eligible,
            "chosen_trial": _trial_result_to_dict(chosen_trial),
            "trials": [_trial_result_to_dict(trial) for trial in result.trials],
            "winner_variant_summary": _candidate_variant_result(result),
        },
    )
    return result


def _run_candidate_trial(
    prepared: PreparedLeanDeepMaxSearch,
    candidate_plan: _CandidatePlan,
    *,
    trial_index: int,
) -> _TrialResult:
    run_dir = candidate_plan.output_dir / f"trial_{trial_index:02d}"
    resolved_seed = _resolved_trial_seed(
        base_seed=int(candidate_plan.evolution_config.seed),
        candidate_order=candidate_plan.candidate_order,
        trial_index=trial_index,
        seed_stride=prepared.max_search_spec.seed_stride,
    )
    evolution_config = _with_search_seed(candidate_plan.evolution_config, search_seed=resolved_seed)
    run_result = run_deep_evolution(
        prepared.comparison_prepared.shared_bundle.bundle_dir,
        run_dir,
        evolution_config=evolution_config,
        deep_config=prepared.comparison_prepared.deep_training_config,
        backend=prepared.comparison_prepared.deep_training_config.backend,
        allow_backend_fallback=prepared.comparison_prepared.deep_training_config.allow_backend_fallback,
        selection_spec=prepared.max_search_spec.selection_spec,
        include_test_metrics=False,
    )
    model = _load_deep_artifact_model(run_result.artifact_path)
    split_metrics = {
        split_name: _evaluate_deep_split(
            prepared.comparison_prepared.shared_bundle.splits[split_name],
            model,
        )
        for split_name in _SUMMARY_SPLITS
    }
    separability_rows = _apply_layers_as_embedding(
        prepared.comparison_prepared.shared_bundle.splits[
            prepared.comparison_prepared.evaluation.separability_split
        ].rows,
        model.layers,
        bit_length=prepared.comparison_prepared.shared_bundle.packed_bit_length,
    )
    separability = _compute_separability(
        split_name=prepared.comparison_prepared.evaluation.separability_split,
        rows=separability_rows,
        labels=prepared.comparison_prepared.shared_bundle.splits[
            prepared.comparison_prepared.evaluation.separability_split
        ].labels,
        bit_length=model.embedding_bit_length,
    )
    latency = _measure_latency(
        rows=prepared.comparison_prepared.shared_bundle.splits[
            prepared.comparison_prepared.evaluation.latency_split
        ].rows,
        predict_many=lambda rows: _predict_deep_rows(rows, model),
        split_name=prepared.comparison_prepared.evaluation.latency_split,
        warmup_passes=prepared.comparison_prepared.evaluation.latency_warmup_passes,
        timed_passes=prepared.comparison_prepared.evaluation.latency_timed_passes,
    )
    metrics_summary = _load_json_mapping(run_result.metrics_summary_path)
    return _TrialResult(
        trial_index=trial_index,
        seed=resolved_seed,
        run_dir=run_dir,
        artifact_path=run_result.artifact_path,
        metrics_summary_path=run_result.metrics_summary_path,
        history_json_path=run_result.evolution_result.history_json_path,
        history_csv_path=run_result.evolution_result.history_csv_path,
        split_metrics=split_metrics,
        latency=latency,
        separability=separability,
        layer_count=len(model.layers),
        model_size_bytes=run_result.artifact_path.stat().st_size,
        embedding_bit_length=model.embedding_bit_length,
        completed_generations=run_result.evolution_result.completed_generations,
        stopped_early=run_result.evolution_result.stopped_early,
        engine_best_candidate=dict(_require_mapping(metrics_summary.get("best_candidate", {}), field_name="best_candidate")),
        search_budget={
            **_search_budget(evolution_config),
            "trials_per_candidate": prepared.max_search_spec.trials_per_candidate,
        },
        provenance={
            **dict(candidate_plan.provenance),
            "resolved_seed": resolved_seed,
            "trial_index": trial_index,
            "seed_stride": prepared.max_search_spec.seed_stride,
        },
    )


def _select_best_trial(trials: tuple[_TrialResult, ...]) -> _TrialResult:
    eligible = tuple(
        trial for trial in trials
        if bool(_validation_alertability(trial.split_metrics["val"])["ranking_eligible"])
    )
    pool = eligible if eligible else tuple(trials)
    return min(
        pool,
        key=lambda trial: _ranking_sort_key(
            _ranking_metrics_from_split_metrics(trial.split_metrics["val"]),
            candidate_order=trial.trial_index,
            model_size_bytes=trial.model_size_bytes,
            latency_ms=float(trial.latency["per_sample_ms"]),
        ),
    )


def _select_best_candidate(
    candidates: list[_CandidateResult] | tuple[_CandidateResult, ...],
) -> _CandidateResult:
    eligible = tuple(candidate for candidate in candidates if candidate.search_ranking_eligible)
    pool = eligible if eligible else tuple(candidates)
    return min(
        pool,
        key=lambda candidate: _ranking_sort_key(
            candidate.ranking_metrics,
            candidate_order=candidate.candidate_order,
            model_size_bytes=candidate.chosen_trial.model_size_bytes,
            latency_ms=float(candidate.chosen_trial.latency["per_sample_ms"]),
        ),
    )


def _rank_candidate_results(
    candidates: list[_CandidateResult] | tuple[_CandidateResult, ...],
) -> list[_CandidateResult]:
    return sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.search_ranking_eligible else 1,
            *_ranking_sort_key(
                candidate.ranking_metrics,
                candidate_order=candidate.candidate_order,
                model_size_bytes=candidate.chosen_trial.model_size_bytes,
                latency_ms=float(candidate.chosen_trial.latency["per_sample_ms"]),
            ),
        ),
    )


def _candidate_variant_result(result: _CandidateResult) -> dict[str, object]:
    return {
        "variant_id": result.candidate_id,
        "label": _CANDIDATE_LABEL,
        "decision_path": _DECISION_PATH,
        "artifact_path": str(result.chosen_trial.artifact_path.resolve()),
        "layer_count": result.chosen_trial.layer_count,
        "embedding_bit_length": result.chosen_trial.embedding_bit_length,
        "model_size_bytes": result.chosen_trial.model_size_bytes,
        "metrics": {
            split_name: dict(result.chosen_trial.split_metrics[split_name])
            for split_name in _SUMMARY_SPLITS
        },
        "latency": dict(result.chosen_trial.latency),
        "separability": dict(result.chosen_trial.separability),
    }


def _candidate_summary_row(
    result: _CandidateResult,
    *,
    overall_rank: int,
    survivor_rank: int | None,
) -> dict[str, object]:
    row = _build_summary_row(
        _candidate_variant_result(result),
        summary_metric_split="test",
    )
    return {
        "overall_rank": overall_rank,
        "survivor_rank": survivor_rank,
        "candidate_id": result.candidate_id,
        "candidate_order": result.candidate_order,
        "provenance_source": result.provenance.get("source"),
        "family_branch_index": result.provenance.get("family_branch_index"),
        "search_ranking_eligible": result.search_ranking_eligible,
        "chosen_trial_index": result.chosen_trial.trial_index,
        "resolved_seed": result.chosen_trial.seed,
        "candidate_summary_path": str(result.summary_path.resolve()),
        "layer_count": result.chosen_trial.layer_count,
        "embedding_bit_length": result.chosen_trial.embedding_bit_length,
        **row,
    }


def _trial_result_to_dict(trial: _TrialResult) -> dict[str, object]:
    return {
        "trial_index": trial.trial_index,
        "seed": trial.seed,
        "run_dir": str(trial.run_dir.resolve()),
        "artifact_path": str(trial.artifact_path.resolve()),
        "metrics_summary_path": str(trial.metrics_summary_path.resolve()),
        "history_json_path": str(trial.history_json_path.resolve()),
        "history_csv_path": str(trial.history_csv_path.resolve()),
        "metrics": {
            split_name: dict(trial.split_metrics[split_name])
            for split_name in _SUMMARY_SPLITS
        },
        "latency": dict(trial.latency),
        "separability": dict(trial.separability),
        "layer_count": trial.layer_count,
        "embedding_bit_length": trial.embedding_bit_length,
        "model_size_bytes": trial.model_size_bytes,
        "completed_generations": trial.completed_generations,
        "stopped_early": trial.stopped_early,
        "engine_best_candidate": dict(trial.engine_best_candidate),
        "search_budget": dict(trial.search_budget),
        "provenance": dict(trial.provenance),
    }


def _candidate_result_to_dict(result: _CandidateResult) -> dict[str, object]:
    return {
        "candidate_id": result.candidate_id,
        "candidate_order": result.candidate_order,
        "provenance": dict(result.provenance),
        "search_ranking_eligible": result.search_ranking_eligible,
        "validation_selection_eligibility": dict(result.validation_selection_eligibility),
        "ranking_metrics": dict(result.ranking_metrics),
        "candidate_summary_path": str(result.summary_path.resolve()),
        "winner_variant_summary": _candidate_variant_result(result),
        "chosen_trial": _trial_result_to_dict(result.chosen_trial),
        "trials": [_trial_result_to_dict(trial) for trial in result.trials],
    }


def _search_scale_comparison(prepared: PreparedLeanDeepMaxSearch) -> dict[str, object]:
    planned_candidate_count = (
        prepared.max_search_spec.initial_search_branches
        + prepared.max_search_spec.winner_replay_branches
        + prepared.max_search_spec.winner_mutation_branches
    )
    planned_total_trial_count = planned_candidate_count * prepared.max_search_spec.trials_per_candidate
    comparison: dict[str, object] = {
        "planned_profile": {
            "initial_search_branches": prepared.max_search_spec.initial_search_branches,
            "local_refine_initial_branches": (
                prepared.max_search_spec.initial_search_branches - prepared.max_search_spec.bounded_random_branches
            ),
            "bounded_random_initial_branches": prepared.max_search_spec.bounded_random_branches,
            "bounded_random_fraction": prepared.max_search_spec.bounded_random_fraction,
            "winner_replay_branches": prepared.max_search_spec.winner_replay_branches,
            "winner_mutation_branches": prepared.max_search_spec.winner_mutation_branches,
            "trials_per_candidate": prepared.max_search_spec.trials_per_candidate,
            "planned_candidate_count": planned_candidate_count,
            "planned_total_trial_count": planned_total_trial_count,
            "max_layers_cap": prepared.max_search_spec.local_refine_evolution.max_layers,
        }
    }
    reference_key = None
    reference_summary = None
    if prepared.current_max_search_summary is not None:
        reference_key = "leanlean_max_search_winner"
        reference_summary = prepared.current_max_search_summary
    elif prepared.deep_layer_max_search_summary is not None:
        reference_key = "leanlean_deep_max_search_winner"
        reference_summary = prepared.deep_layer_max_search_summary
    if not isinstance(reference_summary, Mapping):
        return comparison

    comparison["reference_key"] = reference_key
    comparison["reference_profile_name"] = reference_summary.get("profile_name")
    reference_breakdown = reference_summary.get("search_provenance_breakdown")
    reference_total_candidates = None
    reference_total_trials = None
    if isinstance(reference_breakdown, Mapping):
        reference_total_candidates = int(reference_breakdown.get("total_candidate_count", 0))
        reference_total_trials = int(reference_breakdown.get("total_trial_count", 0))
    reference_plan = None
    plan_path = reference_summary.get("plan_path")
    if isinstance(plan_path, str) and plan_path.strip() != "":
        resolved_plan_path = Path(plan_path)
        if resolved_plan_path.exists():
            reference_plan = _load_json_mapping(resolved_plan_path)
    if isinstance(reference_plan, Mapping):
        search = reference_plan.get("search")
        if isinstance(search, Mapping):
            exploration = search.get("exploration_policy")
            if isinstance(exploration, Mapping):
                reference_initial_branches = int(exploration.get("initial_search_branches", 0))
                comparison["reference_profile"] = {
                    "initial_search_branches": reference_initial_branches,
                    "bounded_random_fraction": float(exploration.get("bounded_random_fraction", 0.0)),
                    "winner_replay_branches": int(exploration.get("winner_replay_branches", 0)),
                    "winner_mutation_branches": int(exploration.get("winner_mutation_branches", 0)),
                }
                if reference_initial_branches > 0:
                    comparison["reference_profile"]["initial_branch_multiplier"] = round(
                        prepared.max_search_spec.initial_search_branches / reference_initial_branches,
                        6,
                    )
            local_refine_evolution = search.get("local_refine_evolution")
            if isinstance(local_refine_evolution, Mapping):
                comparison.setdefault("reference_profile", {})
                comparison["reference_profile"]["max_layers_cap"] = int(local_refine_evolution.get("max_layers", 0))
    if reference_total_candidates is not None and reference_total_candidates > 0:
        comparison.setdefault("reference_profile", {})
        comparison["reference_profile"]["candidate_count"] = reference_total_candidates
        comparison["reference_profile"]["candidate_count_multiplier"] = round(
            planned_candidate_count / reference_total_candidates,
            6,
        )
    if reference_total_trials is not None and reference_total_trials > 0:
        comparison.setdefault("reference_profile", {})
        comparison["reference_profile"]["total_trial_count"] = reference_total_trials
        comparison["reference_profile"]["total_trial_count_multiplier"] = round(
            planned_total_trial_count / reference_total_trials,
            6,
        )
        comparison["reference_profile"]["trials_per_candidate"] = round(
            reference_total_trials / max(1, reference_total_candidates or 1),
            6,
        )
        comparison["reference_profile"]["trials_per_candidate_multiplier"] = round(
            prepared.max_search_spec.trials_per_candidate
            / (reference_total_trials / max(1, reference_total_candidates or 1)),
            6,
        )
    return comparison


def _build_summary_markdown(
    prepared: PreparedLeanDeepMaxSearch,
    *,
    ranked_results: list[_CandidateResult] | tuple[_CandidateResult, ...],
    final_winner: _CandidateResult,
    provenance_breakdown: Mapping[str, object],
    comparisons: Mapping[str, Mapping[str, object]],
    search_scale_comparison: Mapping[str, object],
    conservative_operating_point_assessment: Mapping[str, object],
) -> str:
    winner_metrics = final_winner.chosen_trial.split_metrics
    lines = [
        f"# {prepared.report_title}",
        "",
        f"Source profile: `{prepared.source_profile_path}`",
        "Separate track: `true` (deployment candidate unchanged)",
        f"Locked frontend: `{prepared.locked_frontend.regime_id}` ({prepared.locked_frontend.bit_length} semantic bits, packed to 64 only for backend compatibility)",
        f"Chosen winner layer count: `{final_winner.chosen_trial.layer_count}`",
        f"Search layer ceiling: `{prepared.max_search_spec.local_refine_evolution.max_layers}`",
        "",
        "## Winner Split Metrics",
        "",
        "| Split | Accuracy | Healthy->Unhealthy FPR | Unhealthy Precision | Unhealthy Recall | Unhealthy F1 | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in _SUMMARY_SPLITS:
        split_metrics = winner_metrics[split_name]
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
    planned_profile = _require_mapping(search_scale_comparison["planned_profile"], field_name="planned_profile")
    lines.extend(
        [
            "",
            "## Search Breakdown",
            "",
            f"- Candidate branches: `{provenance_breakdown['total_candidate_count']}`",
            f"- Total trial runs: `{provenance_breakdown['total_trial_count']}`",
            f"- Planned initial branches: `{planned_profile['initial_search_branches']}`",
            f"- Planned trials per candidate: `{planned_profile['trials_per_candidate']}`",
            f"- Planned bounded-random fraction: `{float(planned_profile['bounded_random_fraction']):.2f}`",
            f"- Winner replay branches: `{planned_profile['winner_replay_branches']}`",
            f"- Winner mutation branches: `{planned_profile['winner_mutation_branches']}`",
            f"- Winner candidate: `{final_winner.candidate_id}` from `{final_winner.provenance['source']}`",
            f"- Winner layer count: `{final_winner.chosen_trial.layer_count}`",
            f"- Embedding bit length: `{final_winner.chosen_trial.embedding_bit_length}`",
            f"- Model size: `{final_winner.chosen_trial.model_size_bytes}` bytes",
            f"- Latency on `{final_winner.chosen_trial.latency['split']}`: `{float(final_winner.chosen_trial.latency['per_sample_ms']):.9f}` ms/sample",
        ]
    )
    reference_profile = search_scale_comparison.get("reference_profile")
    reference_key = search_scale_comparison.get("reference_key")
    if isinstance(reference_profile, Mapping) and isinstance(reference_key, str):
        lines.extend(
            [
                "",
                f"## Relative To `{reference_key}` Search Budget",
                "",
                f"- Initial branches: `{planned_profile['initial_search_branches']}` vs `{reference_profile.get('initial_search_branches')}`.",
                f"- Trials per candidate: `{planned_profile['trials_per_candidate']}` vs `{reference_profile.get('trials_per_candidate')}`.",
                f"- Planned total trial count: `{planned_profile['planned_total_trial_count']}` vs `{reference_profile.get('total_trial_count')}`.",
                f"- Layer ceiling: `{planned_profile['max_layers_cap']}` vs `{reference_profile.get('max_layers_cap')}`.",
                f"- Initial-branch multiplier: `{float(reference_profile.get('initial_branch_multiplier', 0.0)):.3f}`.",
                f"- Trial-count multiplier: `{float(reference_profile.get('total_trial_count_multiplier', 0.0)):.3f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Verdict: `{conservative_operating_point_assessment.get('verdict')}`.",
            f"- Answer: {conservative_operating_point_assessment.get('answer')}",
        ]
    )
    lines.extend(
        [
            "",
            "## Candidate Leaderboard",
            "",
            "| Rank | Source | Candidate | Layers | Val FPR | Val Unhealthy Precision | Val Unhealthy Recall | Test FPR | Test Unhealthy F1 | Test Macro F1 | Size (bytes) | Latency (ms/sample) |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank_index, result in enumerate(ranked_results, start=1):
        lines.append(
            "| {rank} | {source} | {candidate} | {layers} | {val_fpr:.6f} | {val_precision:.6f} | {val_recall:.6f} | {test_fpr:.6f} | {test_f1:.6f} | {test_macro:.6f} | {size_bytes} | {latency:.9f} |".format(
                rank=rank_index,
                source=result.provenance["source"],
                candidate=result.candidate_id,
                layers=result.chosen_trial.layer_count,
                val_fpr=float(result.chosen_trial.split_metrics["val"]["healthy_to_unhealthy_fpr"]),
                val_precision=float(result.chosen_trial.split_metrics["val"]["unhealthy_precision"]),
                val_recall=float(result.chosen_trial.split_metrics["val"]["unhealthy_recall"]),
                test_fpr=float(result.chosen_trial.split_metrics["test"]["healthy_to_unhealthy_fpr"]),
                test_f1=float(result.chosen_trial.split_metrics["test"]["unhealthy_f1"]),
                test_macro=float(result.chosen_trial.split_metrics["test"]["macro_f1"]),
                size_bytes=int(result.chosen_trial.model_size_bytes),
                latency=float(result.chosen_trial.latency["per_sample_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reference Comparisons",
            "",
        ]
    )
    for key in ("deployment_candidate", "leanlean_max_search_winner", "leanlean_deep_max_search_winner"):
        comparison = comparisons.get(key)
        if not isinstance(comparison, Mapping):
            continue
        lines.append(f"- `{key}` run root: `{comparison.get('reference_run_root')}`")
        for split_name in ("val", "test"):
            split_delta = comparison.get("deltas", {}).get(split_name)
            if not isinstance(split_delta, Mapping):
                continue
            lines.append(
                "- {label} {split}: FPR delta `{fpr:+.6f}`, unhealthy_precision delta `{precision:+.6f}`, unhealthy_recall delta `{recall:+.6f}`, unhealthy_f1 delta `{f1:+.6f}`, macro_f1 delta `{macro:+.6f}`.".format(
                    label=key,
                    split=split_name,
                    fpr=float(split_delta["healthy_to_unhealthy_fpr_delta"]),
                    precision=float(split_delta["unhealthy_precision_delta"]),
                    recall=float(split_delta["unhealthy_recall_delta"]),
                    f1=float(split_delta["unhealthy_f1_delta"]),
                    macro=float(split_delta["macro_f1_delta"]),
                )
            )
        if "latency_delta_ms_per_sample" in comparison:
            lines.append(
                f"- {key} latency delta: `{float(comparison['latency_delta_ms_per_sample']):+.9f}` ms/sample."
            )
        if "layer_count_delta" in comparison:
            lines.append(
                f"- {key} layer-count delta: `{int(comparison['layer_count_delta']):+d}`."
            )
        if "model_size_delta_bytes" in comparison:
            lines.append(
                f"- {key} model size delta: `{int(comparison['model_size_delta_bytes']):+d}` bytes."
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a separate Lean-Deep max-search workflow and does not overwrite or redefine the deployment-candidate track.",
            "- Ranking stays conservative: validation healthy->unhealthy FPR first, then unhealthy precision, recall, unhealthy F1, and macro F1.",
            "- Search provenance remains explicit across `local_refine`, `bounded_random`, `winner_replay`, and `winner_mutation`.",
            "- The frontend semantics, binary target, split discipline, and deploy-constrained k limits stay fixed; this track only tests whether a materially larger Lean-Deep search can beat Lean-Lean on the conservative operating point rather than merely buying recall with noisier alerts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _lean_deep_assessment(
    prepared: PreparedLeanDeepMaxSearch,
    *,
    final_winner: _CandidateResult,
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    reference_priority = (
        "leanlean_deep_max_search_winner",
        "leanlean_max_search_winner",
        "deployment_candidate",
    )
    reference_key = next((key for key in reference_priority if key in comparisons), "deployment_candidate")
    reference_summary = {
        "leanlean_deep_max_search_winner": prepared.deep_layer_max_search_summary,
        "leanlean_max_search_winner": prepared.current_max_search_summary,
        "deployment_candidate": prepared.deployment_candidate_summary,
    }.get(reference_key)
    comparison = comparisons.get(reference_key)
    if not isinstance(reference_summary, Mapping) or not isinstance(comparison, Mapping):
        return {
            "reference_key": reference_key,
            "verdict": "insufficient_reference",
            "materially_improves_conservative_operating_point": False,
            "answer": "No reliable conclusion was emitted because the Lean-Lean comparison reference could not be resolved.",
        }
    reference_layer_cap = _extract_reference_layer_cap(reference_summary)
    winner_layer_count = int(final_winner.chosen_trial.layer_count)
    val_deltas = comparison.get("deltas", {}).get("val")
    if not isinstance(val_deltas, Mapping):
        return {
            "reference_key": reference_key,
            "reference_layer_cap": reference_layer_cap,
            "winner_layer_count": winner_layer_count,
            "verdict": "insufficient_reference",
            "materially_improves_conservative_operating_point": False,
            "answer": "No reliable conclusion was emitted because the validation deltas were unavailable.",
        }
    lexicographically_better = _lexicographically_better_from_deltas(val_deltas)
    meaningful_metric_gain = _meaningful_metric_gain_from_deltas(val_deltas)
    fpr_delta = float(val_deltas.get("healthy_to_unhealthy_fpr_delta", 0.0))
    precision_delta = float(val_deltas.get("unhealthy_precision_delta", 0.0))
    recall_delta = float(val_deltas.get("unhealthy_recall_delta", 0.0))
    unhealthy_f1_delta = float(val_deltas.get("unhealthy_f1_delta", 0.0))
    beats_all_references = True
    for key in reference_priority:
        if key not in comparisons:
            continue
        split_deltas = comparisons.get(key, {}).get("deltas", {}).get("val")
        if not isinstance(split_deltas, Mapping) or not _lexicographically_better_from_deltas(split_deltas):
            beats_all_references = False
            break
    if lexicographically_better and meaningful_metric_gain and beats_all_references:
        verdict = "material_improvement"
        answer = (
            "Yes: Lean-Deep materially beat the Lean-Lean references on the conservative validation operating "
            "point, not just on recall."
        )
    elif fpr_delta > 0.0 and (recall_delta > 0.0 or unhealthy_f1_delta > 0.0) and precision_delta <= 0.0:
        verdict = "recall_for_noise_tradeoff"
        answer = (
            "No: Lean-Deep mainly bought recall at the cost of a higher healthy->unhealthy false-positive rate, "
            "so the alert stream got noisier instead of improving the conservative operating point."
        )
    elif not lexicographically_better:
        verdict = "no_conservative_improvement"
        answer = (
            "No: the Lean-Deep winner did not beat the Lean-Lean conservative validation operating point."
        )
    else:
        verdict = "mixed_or_marginal"
        answer = (
            "Not materially: Lean-Deep showed a mixed or only marginal shift relative to Lean-Lean, without a clear "
            "conservative-operating-point win."
        )
    return {
        "reference_key": reference_key,
        "reference_layer_cap": reference_layer_cap,
        "winner_layer_count": winner_layer_count,
        "lexicographically_better_than_primary_reference": lexicographically_better,
        "meaningful_improvement_vs_primary_reference": meaningful_metric_gain,
        "beats_all_leanlean_references_lexicographically": beats_all_references,
        "materially_improves_conservative_operating_point": verdict == "material_improvement",
        "recall_for_noise_tradeoff": verdict == "recall_for_noise_tradeoff",
        "verdict": verdict,
        "answer": answer,
    }


def _lexicographically_better_from_deltas(deltas: Mapping[str, object]) -> bool:
    tolerance = 1e-9
    metric_directions = (
        ("healthy_to_unhealthy_fpr_delta", "asc"),
        ("unhealthy_precision_delta", "desc"),
        ("unhealthy_recall_delta", "desc"),
        ("unhealthy_f1_delta", "desc"),
        ("macro_f1_delta", "desc"),
    )
    for metric_name, direction in metric_directions:
        raw_value = deltas.get(metric_name)
        if not isinstance(raw_value, (int, float)):
            return False
        value = float(raw_value)
        if abs(value) <= tolerance:
            continue
        if direction == "asc":
            return value < 0.0
        return value > 0.0
    return False


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RUNS_ROOT",
    "PreparedLeanDeepMaxSearch",
    "load_leandeep_max_search_config",
    "prepare_leandeep_max_search",
    "run_leandeep_max_search",
    "run_prepared_leandeep_max_search",
    "write_leandeep_max_search_plan",
]
