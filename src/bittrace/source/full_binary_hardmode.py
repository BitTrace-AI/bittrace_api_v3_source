"""Hard-mode deep-search bridge for the full binary Paderborn V3 campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from bittrace.core.config import DeepTrainingConfig, EvolutionConfig
from bittrace.core.deep.engine import DeepLayer, _apply_layers_as_embedding, _predict_row, run_deep_evolution
from bittrace.core.evolution.loop import SelectionSpec
from bittrace.v3 import (
    ArtifactRef,
    ContractValidationError,
    DeepInputRef,
    DeepStageCandidateInput,
    ScoutAlertabilityStatus,
    StageKey,
    StageRequest,
    WaveformDatasetRecord,
)
from bittrace.v3.artifacts import compute_file_sha256
from bittrace.v3.frontend_encoding import encode_frontend_record


_PACKED_BIT_LENGTH = 64
_PACKED_ROW_FORMAT = "packed_int_lsb0"
_BUNDLE_SCHEMA_VERSION = "bittrace-bearings-v3-hardmode-packed-bundle-1"
_SEARCH_SUMMARY_ARTIFACT_NAME = "bt3.hardmode_search_summary.json"
_VALIDATION_PASSES_ARTIFACT_NAME = "bt3.hardmode_validation_passes.json"
_VALIDATION_AGGREGATE_ARTIFACT_NAME = "bt3.hardmode_validation_aggregate.json"
_VALIDATION_SELECTION_ARTIFACT_NAME = "bt3.hardmode_validation_selection.json"
_FINAL_TEST_EVALUATION_ARTIFACT_NAME = "bt3.hardmode_final_test_evaluation.json"
_DEEP_CANDIDATE_REPORT_ARTIFACT_NAME = "bt3.deep_candidate_report.json"
_DEEP_METRICS_SUMMARY_ARTIFACT_NAME = "bt3.deep_metrics_summary.json"
_SEARCH_SUMMARY_KIND = "bittrace_bearings_v3_source_hardmode_search_summary"
_VALIDATION_PASSES_KIND = "bittrace_bearings_v3_source_hardmode_validation_passes"
_VALIDATION_AGGREGATE_KIND = "bittrace_bearings_v3_source_hardmode_validation_aggregate"
_VALIDATION_SELECTION_KIND = "bittrace_bearings_v3_source_hardmode_validation_selection"
_FINAL_TEST_EVALUATION_KIND = "bittrace_bearings_v3_source_hardmode_final_test_evaluation"
_DEEP_CANDIDATE_REPORT_KIND = "bittrace_bearings_v3_source_hardmode_deep_candidate_report"
_DEEP_METRICS_SUMMARY_KIND = "bittrace_bearings_v3_source_hardmode_deep_metrics_summary"
_LABEL_TO_INT = {"healthy": 0, "unhealthy": 1}
_INT_TO_LABEL = {value: key for key, value in _LABEL_TO_INT.items()}
_DEFAULT_VALIDATION_FPR_GATE = 0.42
_DEFAULT_VALIDATION_PASS_COUNT = 4
_VALIDATION_PASS_STRATEGY_AUTO = "auto"
_VALIDATION_PASS_STRATEGY_BY_RECORDING = "by_val_recording"
_VALIDATION_PASS_STRATEGY_STRATIFIED_SHARDS = "stratified_row_shards"
_VALIDATION_AGGREGATION_MODE = "conservative_pass_extrema"


@dataclass(frozen=True, slots=True)
class _BundleSplit:
    split_name: str
    rows: tuple[int, ...]
    labels: tuple[int, ...]
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PackedBundle:
    bundle_dir: Path
    dataset_id: str | None
    adapter_profile_id: str | None
    frontend_input_id: str
    frontend_fingerprint: str
    splits: Mapping[str, _BundleSplit]


@dataclass(frozen=True, slots=True)
class _ValidationPassSpec:
    pass_index: int
    pass_id: str
    row_indices: tuple[int, ...]
    recording_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactModel:
    layers: tuple[DeepLayer, ...]
    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
    class_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _HardModeStageSpec:
    trials: int
    seed_stride: int
    evolution_template: Mapping[str, object]
    initial_search_branches: int
    bounded_random_fraction: float
    bounded_random_branches: int
    winner_replay_branches: int
    winner_mutation_branches: int
    bounded_random_evolution_template: Mapping[str, object] | None
    winner_replay_evolution_template: Mapping[str, object] | None
    winner_mutation_evolution_template: Mapping[str, object] | None
    selection_spec: SelectionSpec


@dataclass(frozen=True, slots=True)
class _CandidatePlan:
    candidate_order: int
    family_branch_index: int
    provenance: Mapping[str, object]
    evolution_template: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _TrialResult:
    trial_index: int
    seed: int
    run_dir: Path
    artifact_ref: ArtifactRef
    engine_metrics_summary_ref: ArtifactRef
    engine_proxy_metrics: Mapping[str, object]
    train_metrics: Mapping[str, object]
    val_metrics: Mapping[str, object]
    chosen_ranking_metrics: Mapping[str, float]
    completed_generations: int
    stopped_early: bool
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CandidateWorkup:
    candidate_input: DeepStageCandidateInput
    summary: Mapping[str, object]
    report_ref: ArtifactRef
    metrics_summary_ref: ArtifactRef
    chosen_trial: _TrialResult
    search_ranking_eligible: bool


class HardModeBinaryCampaignBridge:
    """Consumer-side hard-mode deep-search orchestration for the binary campaign."""

    def __init__(self, profile: Mapping[str, Any]) -> None:
        hard_mode = profile.get("hard_mode", {})
        if not isinstance(hard_mode, Mapping) or not hard_mode.get("enabled", False):
            raise ContractValidationError("Hard-mode bridge requires `hard_mode.enabled: true`.")
        self._hard_mode = dict(hard_mode)
        self._stage_specs = {
            StageKey.DEEP_SMOKE: _parse_stage_spec(
                self._hard_mode.get("deep_smoke"),
                trials_key="trials",
                path="hard_mode.deep_smoke",
            ),
            StageKey.DEEP_MAIN_SCREEN: _parse_stage_spec(
                self._hard_mode.get("deep_main_screen"),
                trials_key="trials",
                path="hard_mode.deep_main_screen",
            ),
            StageKey.CAPACITY_REFINEMENT: _parse_stage_spec(
                self._hard_mode.get("capacity_refinement"),
                trials_key="trials_per_k",
                path="hard_mode.capacity_refinement",
            ),
        }
        self._bundle_cache: dict[tuple[str, str], _PackedBundle] = {}

    def include_test_metrics_in_frontend(self) -> bool:
        return bool(self._hard_mode.get("include_test_metrics_in_frontend", False))

    def evaluate_deep_stage(
        self,
        request: StageRequest,
        *,
        stage_key: StageKey,
        deep_training_config: DeepTrainingConfig,
    ) -> tuple[DeepStageCandidateInput, ...]:
        deep_input = _require_single_deep_input(request, stage_key=stage_key)
        workups = self._evaluate_stage_search(
            request=request,
            stage_key=stage_key,
            deep_input=deep_input,
            deep_training_config=deep_training_config,
            selected_k_per_class=1,
            k_medoids_search_values=(1,),
            candidate_order_start=1,
        )
        stable_selection_policy = dict(workups[0].summary["stable_selection_policy"])
        ranking_policy = _hardmode_ranking_policy(
            split_name="val",
            stable_selection_policy=stable_selection_policy,
        )
        selected_workup = _select_best_workup(workups, selection_mode="final")
        stage_search_ref = _write_stage_search_summary(
            Path(request.output_dir) / _SEARCH_SUMMARY_ARTIFACT_NAME,
            stage_key=stage_key,
            stage_spec=self._stage_specs[stage_key],
            stable_selection_policy=stable_selection_policy,
            candidate_summaries=tuple(workup.summary for workup in workups),
        )
        _write_validation_selection(
            Path(request.output_dir) / _VALIDATION_SELECTION_ARTIFACT_NAME,
            stage_key=stage_key,
            ranking_policy=ranking_policy,
            stable_selection_policy=stable_selection_policy,
            candidate_summaries=tuple(workup.summary for workup in workups),
            selected_candidate_id=(
                selected_workup.candidate_input.candidate_id
                if selected_workup.candidate_input.ranking_eligible
                else None
            ),
            stage_search_summary_ref=stage_search_ref,
        )
        return (selected_workup.candidate_input,)

    def evaluate_capacity_stage(
        self,
        request: StageRequest,
        k_candidates: Sequence[int],
        *,
        deep_training_config: DeepTrainingConfig,
    ) -> tuple[DeepStageCandidateInput, ...]:
        deep_input = _require_single_deep_input(request, stage_key=StageKey.CAPACITY_REFINEMENT)
        stage_spec = self._stage_specs[StageKey.CAPACITY_REFINEMENT]
        candidate_order_stride = _total_candidate_branches(stage_spec)
        grouped_workups = tuple(
            self._evaluate_stage_search(
                request=request,
                stage_key=StageKey.CAPACITY_REFINEMENT,
                deep_input=deep_input,
                deep_training_config=deep_training_config,
                selected_k_per_class=k_value,
                k_medoids_search_values=tuple(int(value) for value in k_candidates),
                candidate_order_start=1 + ((index - 1) * candidate_order_stride),
            )
            for index, k_value in enumerate(k_candidates, start=1)
        )
        selected_workups = tuple(
            _select_best_workup(workups, selection_mode="final")
            for workups in grouped_workups
        )
        all_workups = tuple(
            workup
            for workups in grouped_workups
            for workup in workups
        )
        stable_selection_policy = dict(all_workups[0].summary["stable_selection_policy"])
        ranking_policy = _hardmode_ranking_policy(
            split_name="val",
            stable_selection_policy=stable_selection_policy,
        )
        stage_search_ref = _write_stage_search_summary(
            Path(request.output_dir) / _SEARCH_SUMMARY_ARTIFACT_NAME,
            stage_key=StageKey.CAPACITY_REFINEMENT,
            stage_spec=self._stage_specs[StageKey.CAPACITY_REFINEMENT],
            stable_selection_policy=stable_selection_policy,
            candidate_summaries=tuple(workup.summary for workup in all_workups),
        )
        selected_workup = _select_best_workup(selected_workups, selection_mode="final")
        _write_validation_selection(
            Path(request.output_dir) / _VALIDATION_SELECTION_ARTIFACT_NAME,
            stage_key=StageKey.CAPACITY_REFINEMENT,
            ranking_policy=ranking_policy,
            stable_selection_policy=stable_selection_policy,
            candidate_summaries=tuple(workup.summary for workup in all_workups),
            selected_candidate_id=(
                selected_workup.candidate_input.candidate_id
                if selected_workup.candidate_input.ranking_eligible
                else None
            ),
            stage_search_summary_ref=stage_search_ref,
        )
        bundle = self._packed_bundle(request, deep_input=deep_input)
        if selected_workup.candidate_input.ranking_eligible:
            _write_final_test_evaluation(
                Path(request.output_dir) / _FINAL_TEST_EVALUATION_ARTIFACT_NAME,
                stage_key=StageKey.CAPACITY_REFINEMENT,
                ranking_policy=ranking_policy,
                candidate_summary=selected_workup.summary,
                test_metrics=_evaluate_artifact_split(
                    bundle,
                    selected_workup.chosen_trial.artifact_ref,
                    split_name="test",
                ),
            )
        return tuple(
            replace(workup.candidate_input, candidate_order=candidate_order)
            for candidate_order, workup in enumerate(selected_workups, start=1)
        )

    def _evaluate_stage_search(
        self,
        *,
        request: StageRequest,
        stage_key: StageKey,
        deep_input: DeepInputRef,
        deep_training_config: DeepTrainingConfig,
        selected_k_per_class: int,
        k_medoids_search_values: tuple[int, ...],
        candidate_order_start: int,
    ) -> tuple[_CandidateWorkup, ...]:
        stage_spec = self._stage_specs[stage_key]
        next_candidate_order = candidate_order_start
        initial_plans = tuple(
            self._initial_candidate_plans(
                stage_spec=stage_spec,
                next_candidate_order=next_candidate_order,
            )
        )
        initial_workups = tuple(
            self._evaluate_candidate_group(
                request=request,
                stage_key=stage_key,
                deep_input=deep_input,
                deep_training_config=deep_training_config,
                selected_k_per_class=selected_k_per_class,
                k_medoids_search_values=k_medoids_search_values,
                candidate_plan=plan,
            )
            for plan in initial_plans
        )
        if not initial_workups:
            raise ContractValidationError(
                f"Hard-mode `{stage_key.value}` resolved to zero candidate branches."
            )
        provisional_winner = _select_best_workup(initial_workups, selection_mode="search")
        next_candidate_order += len(initial_plans)
        followup_plans = tuple(
            self._followup_candidate_plans(
                stage_spec=stage_spec,
                next_candidate_order=next_candidate_order,
                winner_workup=provisional_winner,
            )
        )
        followup_workups = tuple(
            self._evaluate_candidate_group(
                request=request,
                stage_key=stage_key,
                deep_input=deep_input,
                deep_training_config=deep_training_config,
                selected_k_per_class=selected_k_per_class,
                k_medoids_search_values=k_medoids_search_values,
                candidate_plan=plan,
            )
            for plan in followup_plans
        )
        return initial_workups + followup_workups

    def _evaluate_candidate_group(
        self,
        *,
        request: StageRequest,
        stage_key: StageKey,
        deep_input: DeepInputRef,
        deep_training_config: DeepTrainingConfig,
        selected_k_per_class: int,
        k_medoids_search_values: tuple[int, ...],
        candidate_plan: _CandidatePlan,
    ) -> _CandidateWorkup:
        stage_spec = self._stage_specs[stage_key]
        bundle = self._packed_bundle(request, deep_input=deep_input)
        candidate_id = _candidate_id(
            stage_key=stage_key,
            frontend_input_id=deep_input.frontend_input_id,
            frontend_fingerprint=deep_input.frontend_fingerprint,
            candidate_order=candidate_plan.candidate_order,
            provenance_label=str(candidate_plan.provenance["source"]),
            selected_k_per_class=selected_k_per_class,
        )
        candidate_dir = Path(request.output_dir) / "_hardmode" / "candidates" / candidate_id
        stable_selection_policy = _stable_selection_policy()
        trial_results = tuple(
            self._run_trial(
                bundle=bundle,
                stage_spec=stage_spec,
                candidate_dir=candidate_dir,
                deep_training_config=deep_training_config,
                selected_k_per_class=selected_k_per_class,
                candidate_plan=candidate_plan,
                trial_index=trial_index,
            )
            for trial_index in range(1, stage_spec.trials + 1)
        )
        chosen_trial = min(
            trial_results,
            key=lambda trial: _ranking_sort_key(
                trial.chosen_ranking_metrics,
                candidate_order=trial.trial_index,
            ),
        )
        validation_metrics = dict(chosen_trial.val_metrics)
        alertability = _validation_alertability(validation_metrics)
        final_ranking_eligible = bool(alertability["ranking_eligible"])
        guardrail_triggered = bool(alertability["guardrail_triggered"])
        selection_reason = _combine_validation_selection_reason(
            alertability=alertability,
        )
        selection_eligibility = {
            "search_ranking_eligible": bool(alertability["ranking_eligible"]),
            "final_ranking_eligible": final_ranking_eligible,
            "scout_alertability_status": _enum_value(alertability["status"]),
            "scout_alertability_guardrail_triggered": guardrail_triggered,
            "scout_alertability_reason": selection_reason,
        }
        candidate_report_ref = _write_json_artifact(
            candidate_dir / _DEEP_CANDIDATE_REPORT_ARTIFACT_NAME,
            kind=_DEEP_CANDIDATE_REPORT_KIND,
            schema_version="bittrace-bearings-v3-hardmode-deep-candidate-report-3",
            payload={
                "stage_key": stage_key.value,
                "candidate_id": candidate_id,
                "candidate_order": candidate_plan.candidate_order,
                "selected_k_per_class": selected_k_per_class,
                "ranking_split": "val",
                "validation_reference_split": "val",
                "train_search_split": "train",
                "final_test_split_reserved": stage_key == StageKey.CAPACITY_REFINEMENT,
                "frontend_input_id": deep_input.frontend_input_id,
                "frontend_fingerprint": deep_input.frontend_fingerprint,
                "bundle_dir": str(bundle.bundle_dir.resolve()),
                "provenance": dict(candidate_plan.provenance),
                "search_budget": _search_budget(
                    stage_spec,
                    candidate_plan.evolution_template,
                ),
                "exploration_policy": _exploration_policy(stage_spec),
                "stable_selection_policy": stable_selection_policy,
                "chosen_trial": _trial_result_to_dict(chosen_trial),
                "all_trials": [_trial_result_to_dict(trial) for trial in trial_results],
                "validation_reference_metrics": validation_metrics,
                "validation_selection_eligibility": selection_eligibility,
            },
        )
        metrics_summary_ref = _write_json_artifact(
            candidate_dir / _DEEP_METRICS_SUMMARY_ARTIFACT_NAME,
            kind=_DEEP_METRICS_SUMMARY_KIND,
            schema_version="bittrace-bearings-v3-hardmode-deep-metrics-summary-3",
            payload={
                "stage_key": stage_key.value,
                "candidate_id": candidate_id,
                "metric_split": "val",
                "deploy_style_metrics": dict(chosen_trial.chosen_ranking_metrics),
                "validation_reference_metrics": validation_metrics,
                "proxy_metrics_label": (
                    "engine_proxy_metrics are retained as search telemetry only; they are not "
                    "final deploy-style quality metrics."
                ),
                "engine_proxy_metrics": dict(chosen_trial.engine_proxy_metrics),
                "provenance": dict(candidate_plan.provenance),
                "stable_selection_policy": stable_selection_policy,
            },
        )
        effective_engine_deep_config = {
            "materialization_mode": "consumer_hardmode_real_deep_search",
            "promotion_stage": request.promotion_stage.value if request.promotion_stage is not None else None,
            "k_medoids_per_class": selected_k_per_class,
            "k_medoids_search_values": list(k_medoids_search_values),
            "backend": deep_training_config.backend,
            "allow_backend_fallback": deep_training_config.allow_backend_fallback,
            "hard_mode_trials": stage_spec.trials,
            "validation_ranking_only": True,
            "final_test_reserved": stage_key == StageKey.CAPACITY_REFINEMENT,
            "search_provenance": dict(candidate_plan.provenance),
            "search_budget": _search_budget(stage_spec, candidate_plan.evolution_template),
            "exploration_policy": _exploration_policy(stage_spec),
            "stable_selection_policy": stable_selection_policy,
        }
        candidate_input = DeepStageCandidateInput(
            candidate_id=candidate_id,
            candidate_order=candidate_plan.candidate_order,
            branch_mode="consumer_hardmode_real_deep_search",
            ranking_metrics=dict(chosen_trial.chosen_ranking_metrics),
            scout_alertability_status=alertability["status"],
            ranking_eligible=final_ranking_eligible,
            scout_alertability_guardrail_triggered=guardrail_triggered,
            scout_alertability_reason=selection_reason,
            effective_engine_deep_config=effective_engine_deep_config,
            best_deep_artifact_ref=chosen_trial.artifact_ref,
            metrics_summary_ref=metrics_summary_ref,
            candidate_report_ref=candidate_report_ref,
            frontend_input_id=deep_input.frontend_input_id,
            frontend_fingerprint=deep_input.frontend_fingerprint,
            selected_k_per_class=selected_k_per_class,
            k_medoids_search_values=k_medoids_search_values,
        )
        summary = {
            "candidate_id": candidate_id,
            "candidate_order": candidate_plan.candidate_order,
            "selected_k_per_class": selected_k_per_class,
            "frontend_input_id": deep_input.frontend_input_id,
            "frontend_fingerprint": deep_input.frontend_fingerprint,
            "provenance": dict(candidate_plan.provenance),
            "search_budget": _search_budget(stage_spec, candidate_plan.evolution_template),
            "stable_selection_policy": stable_selection_policy,
            "ranking_metrics": dict(candidate_input.ranking_metrics),
            "validation_metrics": validation_metrics,
            "validation_reference_metrics": validation_metrics,
            "validation_selection_eligibility": selection_eligibility,
            "chosen_trial": _trial_result_to_dict(chosen_trial),
            "all_trials": [_trial_result_to_dict(trial) for trial in trial_results],
            "candidate_report_ref": candidate_report_ref.to_dict(),
            "metrics_summary_ref": metrics_summary_ref.to_dict(),
        }
        return _CandidateWorkup(
            candidate_input=candidate_input,
            summary=summary,
            report_ref=candidate_report_ref,
            metrics_summary_ref=metrics_summary_ref,
            chosen_trial=chosen_trial,
            search_ranking_eligible=bool(alertability["ranking_eligible"]),
        )

    def _run_trial(
        self,
        *,
        bundle: _PackedBundle,
        stage_spec: _HardModeStageSpec,
        candidate_dir: Path,
        deep_training_config: DeepTrainingConfig,
        selected_k_per_class: int,
        candidate_plan: _CandidatePlan,
        trial_index: int,
    ) -> _TrialResult:
        seed = (
            int(candidate_plan.evolution_template["seed"])
            + ((candidate_plan.family_branch_index - 1) * stage_spec.seed_stride)
            + (trial_index - 1)
        )
        evolution_config = EvolutionConfig.from_mapping(
            {
                **dict(candidate_plan.evolution_template),
                "seed": seed,
            }
        )
        effective_deep_config = DeepTrainingConfig(
            k_medoids_per_class=selected_k_per_class,
            adaptive_k=False,
            adaptive_k_candidates=(),
            backend=deep_training_config.backend,
            allow_backend_fallback=deep_training_config.allow_backend_fallback,
        )
        run_dir = candidate_dir / f"trial_{trial_index:02d}"
        result = run_deep_evolution(
            bundle.bundle_dir,
            run_dir,
            evolution_config=evolution_config,
            deep_config=effective_deep_config,
            backend=deep_training_config.backend,
            allow_backend_fallback=deep_training_config.allow_backend_fallback,
            selection_spec=stage_spec.selection_spec,
            include_test_metrics=False,
        )
        engine_metrics_summary = _load_json_mapping(result.metrics_summary_path)
        artifact_ref = _artifact_ref(
            result.artifact_path,
            kind="bittrace_core_deep_best_artifact",
            schema_version="bittrace-core-deep-best-artifact-1",
        )
        engine_metrics_summary_ref = _artifact_ref(
            result.metrics_summary_path,
            kind="bittrace_core_deep_metrics_summary",
            schema_version="bittrace-core-deep-metrics-summary-1",
        )
        artifact_model = _load_artifact_model(artifact_ref)
        train_metrics = _evaluate_artifact_split(
            bundle,
            artifact_ref,
            split_name="train",
            artifact_model=artifact_model,
        )
        val_predictions, val_margins = _predict_split(
            bundle.splits["val"],
            artifact_model,
        )
        val_metrics = _metrics_from_predictions(
            bundle.splits["val"],
            val_predictions,
            val_margins,
            split_name="val",
        )
        return _TrialResult(
            trial_index=trial_index,
            seed=seed,
            run_dir=run_dir,
            artifact_ref=artifact_ref,
            engine_metrics_summary_ref=engine_metrics_summary_ref,
            engine_proxy_metrics=dict(engine_metrics_summary.get("splits", {})),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            chosen_ranking_metrics=_ranking_metrics_from_split_metrics(val_metrics),
            completed_generations=result.evolution_result.completed_generations,
            stopped_early=result.evolution_result.stopped_early,
            provenance={
                **dict(candidate_plan.provenance),
                "resolved_seed": seed,
                "trial_index": trial_index,
                "seed_stride": stage_spec.seed_stride,
            },
        )

    def _initial_candidate_plans(
        self,
        *,
        stage_spec: _HardModeStageSpec,
        next_candidate_order: int,
    ) -> tuple[_CandidatePlan, ...]:
        plans: list[_CandidatePlan] = []
        local_refine_branches = stage_spec.initial_search_branches - stage_spec.bounded_random_branches
        candidate_order = next_candidate_order
        for family_branch_index in range(1, local_refine_branches + 1):
            plans.append(
                _CandidatePlan(
                    candidate_order=candidate_order,
                    family_branch_index=family_branch_index,
                    provenance=_candidate_provenance(
                        source="local_refine",
                        lineage_basis="frontend_input",
                        family_branch_index=family_branch_index,
                        evolution_template=stage_spec.evolution_template,
                    ),
                    evolution_template=stage_spec.evolution_template,
                )
            )
            candidate_order += 1
        if stage_spec.bounded_random_branches > 0:
            if stage_spec.bounded_random_evolution_template is None:
                raise ContractValidationError(
                    "Hard-mode bounded-random search requires `bounded_random_evolution`."
                )
            for family_branch_index in range(1, stage_spec.bounded_random_branches + 1):
                plans.append(
                    _CandidatePlan(
                        candidate_order=candidate_order,
                        family_branch_index=family_branch_index,
                        provenance=_candidate_provenance(
                            source="bounded_random",
                            lineage_basis="frontend_input",
                            family_branch_index=family_branch_index,
                            evolution_template=stage_spec.bounded_random_evolution_template,
                        ),
                        evolution_template=stage_spec.bounded_random_evolution_template,
                    )
                )
                candidate_order += 1
        return tuple(plans)

    def _followup_candidate_plans(
        self,
        *,
        stage_spec: _HardModeStageSpec,
        next_candidate_order: int,
        winner_workup: _CandidateWorkup,
    ) -> tuple[_CandidatePlan, ...]:
        plans: list[_CandidatePlan] = []
        candidate_order = next_candidate_order
        winner_provenance = winner_workup.summary.get("provenance", {})
        winner_source = {
            "winner_source_candidate_id": winner_workup.candidate_input.candidate_id,
            "winner_source_provenance": winner_provenance.get("source"),
            "winner_source_ranking_metrics": dict(winner_workup.candidate_input.ranking_metrics),
            "warm_start_used": False,
            "winner_followup_mode": (
                "search_family_only_no_artifact_resume"
            ),
        }
        if stage_spec.winner_replay_branches > 0:
            if stage_spec.winner_replay_evolution_template is None:
                raise ContractValidationError(
                    "Hard-mode winner replay requires `winner_replay_evolution`."
                )
            for family_branch_index in range(1, stage_spec.winner_replay_branches + 1):
                plans.append(
                    _CandidatePlan(
                        candidate_order=candidate_order,
                        family_branch_index=family_branch_index,
                        provenance=_candidate_provenance(
                            source="winner_replay",
                            lineage_basis="provisional_validation_winner",
                            family_branch_index=family_branch_index,
                            evolution_template=stage_spec.winner_replay_evolution_template,
                            extra=winner_source,
                        ),
                        evolution_template=stage_spec.winner_replay_evolution_template,
                    )
                )
                candidate_order += 1
        if stage_spec.winner_mutation_branches > 0:
            if stage_spec.winner_mutation_evolution_template is None:
                raise ContractValidationError(
                    "Hard-mode winner mutation requires `winner_mutation_evolution`."
                )
            for family_branch_index in range(1, stage_spec.winner_mutation_branches + 1):
                plans.append(
                    _CandidatePlan(
                        candidate_order=candidate_order,
                        family_branch_index=family_branch_index,
                        provenance=_candidate_provenance(
                            source="winner_mutation",
                            lineage_basis="provisional_validation_winner",
                            family_branch_index=family_branch_index,
                            evolution_template=stage_spec.winner_mutation_evolution_template,
                            extra=winner_source,
                        ),
                        evolution_template=stage_spec.winner_mutation_evolution_template,
                    )
                )
                candidate_order += 1
        return tuple(plans)

    def _packed_bundle(
        self,
        request: StageRequest,
        *,
        deep_input: DeepInputRef,
    ) -> _PackedBundle:
        cache_key = (request.output_dir, deep_input.bundle_fingerprint)
        cached = self._bundle_cache.get(cache_key)
        if cached is not None:
            return cached
        frontend_input_id = deep_input.frontend_input_id
        frontend_fingerprint = deep_input.frontend_fingerprint
        if not frontend_input_id or not frontend_fingerprint:
            raise ContractValidationError(
                "Hard-mode deep search requires `frontend_input_id` and `frontend_fingerprint` lineage."
            )
        contract_payload = _load_json_mapping(Path(deep_input.bundle_contract_path))
        handoff_payload = _load_json_mapping(Path(deep_input.handoff_manifest_path))
        raw_records = handoff_payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ContractValidationError(
                "Hard-mode deep search requires non-empty canonical `records` in the deep-input handoff."
            )
        bundle_dir = (
            Path(request.output_dir)
            / "_hardmode"
            / f"packed_bundle_{deep_input.bundle_fingerprint[:12]}"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)
        split_payloads: dict[str, dict[str, list[object]]] = {
            "train": {"X_packed": [], "y": [], "source_record_ids": []},
            "val": {"X_packed": [], "y": [], "source_record_ids": []},
            "test": {"X_packed": [], "y": [], "source_record_ids": []},
        }
        bundle_feature_names: tuple[str, ...] | None = None
        bundle_bit_length: int | None = None
        locked_frontend = contract_payload.get("locked_frontend")
        original_frontend_bit_length = _read_positive_int(
            _mapping_get_optional_int(contract_payload, "frontend_encoding", "bit_length"),
            default=_read_positive_int(
                _mapping_get_optional_int(locked_frontend, "bit_length"),
                default=None,
            ),
        )
        for raw_record in raw_records:
            record = _coerce_hardmode_waveform_record(raw_record)
            state_label = str(record.state_label)
            if state_label not in _LABEL_TO_INT:
                raise ContractValidationError(
                    "Hard-mode binary bridge only supports `healthy` and `unhealthy` state labels."
                )
            split_name = str(record.split).lower()
            if split_name not in split_payloads:
                raise ContractValidationError(
                    f"Hard-mode packed bundle encountered unsupported split `{record.split}`."
                )
            encoded = encode_frontend_record(
                record,
                dataset_id=handoff_payload.get("dataset_id"),
                adapter_profile_id=handoff_payload.get("adapter_profile_id"),
                frontend_input_id=frontend_input_id,
                frontend_fingerprint=frontend_fingerprint,
                contract_payload=contract_payload,
            )
            if bundle_feature_names is None:
                bundle_feature_names = encoded.bit_feature_names
                bundle_bit_length = encoded.bit_length
            split_payloads[split_name]["X_packed"].append(encoded.packed_row_int)
            split_payloads[split_name]["y"].append(_LABEL_TO_INT[state_label])
            split_payloads[split_name]["source_record_ids"].append(record.source_record_id)
        for split_name, payload in split_payloads.items():
            if not payload["X_packed"]:
                raise ContractValidationError(
                    f"Hard-mode packed bundle is missing required split rows for `{split_name}`."
                )
            _write_json(
                bundle_dir / f"{split_name}_bits.json",
                payload,
            )
        semantic_bit_length = _resolve_semantic_bit_length(
            original_frontend_bit_length,
            bundle_bit_length,
        )
        _write_json(
            bundle_dir / "contract.json",
            {
                "schema_version": _BUNDLE_SCHEMA_VERSION,
                "row_format": _PACKED_ROW_FORMAT,
                "bit_length": _PACKED_BIT_LENGTH,
                "feature_names": list(
                    _materialize_packed64_feature_names(
                        bundle_feature_names,
                        semantic_bit_length=bundle_bit_length,
                    )
                ),
                "dataset_id": handoff_payload.get("dataset_id"),
                "adapter_profile_id": handoff_payload.get("adapter_profile_id"),
                "frontend_input_id": frontend_input_id,
                "frontend_fingerprint": frontend_fingerprint,
                "label_mapping": dict(_LABEL_TO_INT),
                "bundle_materialization": {
                    "consumer_bundle_bit_length": _PACKED_BIT_LENGTH,
                    "frontend_semantic_bit_length": semantic_bit_length,
                    "frontend_semantic_bits_lsb0": _semantic_bit_range(semantic_bit_length),
                    "padding_bits_lsb0": _padding_bit_range(semantic_bit_length),
                    "padding_rule": "zero_fill_high_bits",
                    "materialization_reason": "deep_gpu_compatibility_packed64",
                    "preserves_locked_frontend_regime": True,
                },
                "locked_frontend": (
                    dict(locked_frontend)
                    if isinstance(locked_frontend, Mapping)
                    else None
                ),
                "split_semantics": {
                    "train": "candidate generation only",
                    "val": "winner selection only",
                    "test": "reserved for final reporting only",
                },
            },
        )
        bundle = _PackedBundle(
            bundle_dir=bundle_dir,
            dataset_id=handoff_payload.get("dataset_id") if isinstance(handoff_payload.get("dataset_id"), str) else None,
            adapter_profile_id=(
                handoff_payload.get("adapter_profile_id")
                if isinstance(handoff_payload.get("adapter_profile_id"), str)
                else None
            ),
            frontend_input_id=frontend_input_id,
            frontend_fingerprint=frontend_fingerprint,
            splits={
                split_name: _BundleSplit(
                    split_name=split_name,
                    rows=tuple(int(value) for value in payload["X_packed"]),
                    labels=tuple(int(value) for value in payload["y"]),
                    source_record_ids=tuple(str(value) for value in payload["source_record_ids"]),
                )
                for split_name, payload in split_payloads.items()
            },
        )
        self._bundle_cache[cache_key] = bundle
        return bundle


def _materialize_packed64_feature_names(
    feature_names: tuple[str, ...] | None,
    *,
    semantic_bit_length: int | None,
) -> tuple[str, ...]:
    semantic_length = semantic_bit_length if semantic_bit_length is not None else 0
    if semantic_length < 0 or semantic_length > _PACKED_BIT_LENGTH:
        raise ContractValidationError(
            f"Hard-mode packed-64 materialization requires 0 <= semantic bits <= {_PACKED_BIT_LENGTH}; "
            f"received {semantic_length}."
        )
    base_names = (
        tuple(feature_names)
        if feature_names is not None
        else tuple(f"bit_{index}" for index in range(semantic_length))
    )
    if len(base_names) != semantic_length:
        raise ContractValidationError(
            "Hard-mode packed-64 materialization requires one feature name per semantic frontend bit."
        )
    return base_names + tuple(
        f"padding_zero_bit_{index}"
        for index in range(semantic_length, _PACKED_BIT_LENGTH)
    )


def _resolve_semantic_bit_length(
    original_frontend_bit_length: int | None,
    encoded_bit_length: int | None,
) -> int:
    semantic_bit_length = (
        original_frontend_bit_length
        if original_frontend_bit_length is not None
        else encoded_bit_length
    )
    if semantic_bit_length is None:
        raise ContractValidationError(
            "Hard-mode packed-64 materialization could not determine the frontend semantic bit length."
        )
    if semantic_bit_length > _PACKED_BIT_LENGTH:
        raise ContractValidationError(
            f"Hard-mode packed-64 materialization received {semantic_bit_length} semantic bits; "
            f"maximum supported is {_PACKED_BIT_LENGTH}."
        )
    return semantic_bit_length


def _semantic_bit_range(semantic_bit_length: int) -> list[int] | None:
    if semantic_bit_length <= 0:
        return None
    return [0, semantic_bit_length - 1]


def _padding_bit_range(semantic_bit_length: int) -> list[int] | None:
    if semantic_bit_length >= _PACKED_BIT_LENGTH:
        return None
    return [semantic_bit_length, _PACKED_BIT_LENGTH - 1]


def _mapping_get_optional_int(
    payload: object,
    *path: str,
) -> int | None:
    current = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _read_positive_int(current, default=None)


def _read_positive_int(value: object, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError("Hard-mode packed-64 materialization requires positive integer bit lengths.")
    return int(value)


def _coerce_hardmode_waveform_record(
    record: WaveformDatasetRecord | Mapping[str, object],
) -> WaveformDatasetRecord:
    if isinstance(record, WaveformDatasetRecord):
        return record
    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Hard-mode record coercion requires a `WaveformDatasetRecord` or JSON-object mapping."
        )
    if "context" in record:
        context = record.get("context", {})
        if context is None:
            context = {}
        if not isinstance(context, Mapping):
            raise ContractValidationError("Hard-mode waveform record `context` must be a mapping.")
        label_metadata = record.get("label_metadata", {})
        if label_metadata is None:
            label_metadata = {}
        lineage_metadata = record.get("lineage_metadata", {})
        if lineage_metadata is None:
            lineage_metadata = {}
        return WaveformDatasetRecord(
            source_record_id=record.get("source_record_id"),
            split=record.get("split"),
            state_label=record.get("state_label"),
            waveforms=record.get("waveforms", {}),
            label_metadata=label_metadata,
            sampling_hz=context.get("sampling_hz"),
            rpm=context.get("rpm"),
            operating_condition=context.get("operating_condition"),
            context_metadata=context.get("metadata", {}),
            lineage_metadata=lineage_metadata,
        )
    return WaveformDatasetRecord.from_dict(record)


def hardmode_enabled(profile: Mapping[str, Any]) -> bool:
    hard_mode = profile.get("hard_mode", {})
    return isinstance(hard_mode, Mapping) and bool(hard_mode.get("enabled", False))


def build_hardmode_bridge(profile: Mapping[str, Any]) -> HardModeBinaryCampaignBridge:
    return HardModeBinaryCampaignBridge(profile)


def _parse_stage_spec(
    raw: object,
    *,
    trials_key: str,
    path: str,
) -> _HardModeStageSpec:
    if not isinstance(raw, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    trials_value = raw.get(trials_key)
    if isinstance(trials_value, bool) or not isinstance(trials_value, int) or trials_value < 1:
        raise ContractValidationError(f"`{path}.{trials_key}` must be an integer >= 1.")
    seed_stride = raw.get("seed_stride", 100)
    if isinstance(seed_stride, bool) or not isinstance(seed_stride, int) or seed_stride < 1:
        raise ContractValidationError(f"`{path}.seed_stride` must be an integer >= 1.")
    evolution_raw = raw.get("evolution")
    if not isinstance(evolution_raw, Mapping):
        raise ContractValidationError(f"`{path}.evolution` must be a mapping.")
    try:
        base_evolution = dict(evolution_raw)
        EvolutionConfig.from_mapping(base_evolution)
    except Exception as exc:  # pragma: no cover - config validation path
        raise ContractValidationError(f"`{path}.evolution` is invalid: {exc}") from exc
    initial_search_branches = _require_int_field(
        raw.get("search_branches", 1),
        path=f"{path}.search_branches",
        minimum=1,
    )
    bounded_random_fraction = _require_probability(
        raw.get("bounded_random_fraction", 0.0),
        path=f"{path}.bounded_random_fraction",
    )
    bounded_random_branches = _resolve_fractional_branch_count(
        initial_search_branches=initial_search_branches,
        fraction=bounded_random_fraction,
        path=f"{path}.bounded_random_fraction",
    )
    winner_replay_branches = _require_int_field(
        raw.get("winner_replay_branches", 0),
        path=f"{path}.winner_replay_branches",
        minimum=0,
    )
    winner_mutation_branches = _require_int_field(
        raw.get("winner_mutation_branches", 0),
        path=f"{path}.winner_mutation_branches",
        minimum=0,
    )
    selection_raw = raw.get("selection_spec", {})
    selection_spec = _selection_spec_from_mapping(selection_raw, path=f"{path}.selection_spec")
    return _HardModeStageSpec(
        trials=trials_value,
        seed_stride=seed_stride,
        evolution_template=base_evolution,
        initial_search_branches=initial_search_branches,
        bounded_random_fraction=bounded_random_fraction,
        bounded_random_branches=bounded_random_branches,
        winner_replay_branches=winner_replay_branches,
        winner_mutation_branches=winner_mutation_branches,
        bounded_random_evolution_template=_merge_optional_evolution_template(
            base_evolution,
            raw.get("bounded_random_evolution"),
            path=f"{path}.bounded_random_evolution",
            required=bounded_random_branches > 0,
        ),
        winner_replay_evolution_template=_merge_optional_evolution_template(
            base_evolution,
            raw.get("winner_replay_evolution"),
            path=f"{path}.winner_replay_evolution",
            required=winner_replay_branches > 0,
        ),
        winner_mutation_evolution_template=_merge_optional_evolution_template(
            base_evolution,
            raw.get("winner_mutation_evolution"),
            path=f"{path}.winner_mutation_evolution",
            required=winner_mutation_branches > 0,
        ),
        selection_spec=selection_spec,
    )


def _selection_spec_from_mapping(raw: object, *, path: str) -> SelectionSpec:
    if raw in ({}, None):
        return SelectionSpec(primary_metric="fitness", tiebreak_metrics=("accuracy", "mean_margin"))
    if not isinstance(raw, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    primary_metric = raw.get("primary_metric", "fitness")
    tiebreak_metrics = raw.get("tiebreak_metrics", ())
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ContractValidationError(f"`{path}.primary_metric` must be a non-empty string.")
    if not isinstance(tiebreak_metrics, Sequence) or isinstance(
        tiebreak_metrics,
        (str, bytes, bytearray),
    ):
        raise ContractValidationError(f"`{path}.tiebreak_metrics` must be a sequence of strings.")
    normalized_tiebreak_metrics = []
    for index, metric_name in enumerate(tiebreak_metrics):
        if not isinstance(metric_name, str) or not metric_name:
            raise ContractValidationError(
                f"`{path}.tiebreak_metrics[{index}]` must be a non-empty string."
            )
        normalized_tiebreak_metrics.append(metric_name)
    try:
        return SelectionSpec(
            primary_metric=primary_metric,
            tiebreak_metrics=tuple(normalized_tiebreak_metrics),
        )
    except ValueError as exc:
        raise ContractValidationError(f"`{path}` is invalid: {exc}") from exc


def _require_validation_pass_strategy(raw: object, *, path: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ContractValidationError(f"`{path}` must be a non-empty string.")
    normalized = raw.strip().lower()
    allowed = {
        _VALIDATION_PASS_STRATEGY_AUTO,
        _VALIDATION_PASS_STRATEGY_BY_RECORDING,
        _VALIDATION_PASS_STRATEGY_STRATIFIED_SHARDS,
    }
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ContractValidationError(
            f"`{path}` must be one of: {allowed_text}."
        )
    return normalized


def _require_single_deep_input(request: StageRequest, *, stage_key: StageKey) -> DeepInputRef:
    deep_inputs = tuple(request.deep_inputs)
    if len(deep_inputs) != 1:
        raise ContractValidationError(
            f"Hard-mode `{stage_key.value}` expects exactly one resolved `deep_input`."
        )
    return deep_inputs[0]


def _candidate_id(
    *,
    stage_key: StageKey,
    frontend_input_id: str | None,
    frontend_fingerprint: str | None,
    candidate_order: int,
    provenance_label: str,
    selected_k_per_class: int,
) -> str:
    lineage_token = frontend_input_id or (frontend_fingerprint[:12] if frontend_fingerprint else "lineage")
    return (
        f"hardmode-{stage_key.value}-{candidate_order:02d}-{provenance_label}-"
        f"{lineage_token}-k{selected_k_per_class}"
    )


def _artifact_ref(path: Path, *, kind: str, schema_version: str) -> ArtifactRef:
    resolved_path = path.resolve()
    return ArtifactRef(
        kind=kind,
        schema_version=schema_version,
        path=str(resolved_path),
        sha256=compute_file_sha256(resolved_path),
    )


def _load_json_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_artifact(
    path: Path,
    *,
    kind: str,
    schema_version: str,
    payload: Mapping[str, object],
) -> ArtifactRef:
    resolved_path = path.resolve()
    _write_json(
        resolved_path,
        {
            "kind": kind,
            "schema_version": schema_version,
            **dict(payload),
        },
    )
    return _artifact_ref(resolved_path, kind=kind, schema_version=schema_version)


def _write_stage_search_summary(
    path: Path,
    *,
    stage_key: StageKey,
    stage_spec: _HardModeStageSpec,
    stable_selection_policy: Mapping[str, object],
    candidate_summaries: Sequence[Mapping[str, object]],
) -> ArtifactRef:
    return _write_json_artifact(
        path,
        kind=_SEARCH_SUMMARY_KIND,
        schema_version="bittrace-bearings-v3-hardmode-search-summary-2",
        payload={
            "stage_key": stage_key.value,
            "split_discipline": {
                "train": "candidate generation and search only",
                "val": "winner selection only",
                "test": "reserved for final reporting only",
            },
            "proxy_metrics_label": (
                "engine_proxy_metrics remain search telemetry only; deploy-style metrics "
                "below are the authoritative hard-mode metrics."
            ),
            "exploration_policy": _exploration_policy(stage_spec),
            "stable_selection_policy": dict(stable_selection_policy),
            "candidate_summaries": list(candidate_summaries),
        },
    )


def _write_validation_passes(
    path: Path,
    *,
    stage_key: StageKey,
    stable_selection_policy: Mapping[str, object],
    candidate_summaries: Sequence[Mapping[str, object]],
    stage_search_summary_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    return _write_json_artifact(
        path,
        kind=_VALIDATION_PASSES_KIND,
        schema_version="bittrace-bearings-v3-hardmode-validation-passes-1",
        payload={
            "stage_key": stage_key.value,
            "selection_split": "val",
            "stable_selection_policy": dict(stable_selection_policy),
            "candidate_validation_passes": [
                _candidate_validation_passes_entry(summary)
                for summary in candidate_summaries
            ],
            "stage_search_summary_ref": (
                stage_search_summary_ref.to_dict()
                if stage_search_summary_ref is not None
                else None
            ),
        },
    )


def _write_validation_aggregate(
    path: Path,
    *,
    stage_key: StageKey,
    ranking_policy: Mapping[str, object],
    stable_selection_policy: Mapping[str, object],
    candidate_summaries: Sequence[Mapping[str, object]],
    stage_search_summary_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    ranked_summaries = sorted(
        candidate_summaries,
        key=lambda summary: _ranking_sort_key(
            summary["ranking_metrics"],
            candidate_order=int(summary["candidate_order"]),
        ),
    )
    survivor_ranks = {
        str(summary["candidate_id"]): index
        for index, summary in enumerate(
            (
                summary
                for summary in ranked_summaries
                if bool(summary["validation_selection_eligibility"]["final_ranking_eligible"])
            ),
            start=1,
        )
    }
    overall_ranks = {
        str(summary["candidate_id"]): index
        for index, summary in enumerate(ranked_summaries, start=1)
    }
    return _write_json_artifact(
        path,
        kind=_VALIDATION_AGGREGATE_KIND,
        schema_version="bittrace-bearings-v3-hardmode-validation-aggregate-1",
        payload={
            "stage_key": stage_key.value,
            "selection_split": "val",
            "ranking_policy": dict(ranking_policy),
            "stable_selection_policy": dict(stable_selection_policy),
            "candidate_validation_aggregates": [
                _candidate_validation_aggregate_entry(
                    summary,
                    overall_rank=overall_ranks[str(summary["candidate_id"])],
                    survivor_rank=survivor_ranks.get(str(summary["candidate_id"])),
                )
                for summary in ranked_summaries
            ],
            "stage_search_summary_ref": (
                stage_search_summary_ref.to_dict()
                if stage_search_summary_ref is not None
                else None
            ),
        },
    )


def _write_validation_selection(
    path: Path,
    *,
    stage_key: StageKey,
    ranking_policy: Mapping[str, object],
    stable_selection_policy: Mapping[str, object],
    candidate_summaries: Sequence[Mapping[str, object]],
    selected_candidate_id: str | None,
    stage_search_summary_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    survivor_candidate_ids = [
        str(summary["candidate_id"])
        for summary in sorted(
            candidate_summaries,
            key=lambda summary: _ranking_sort_key(
                summary["ranking_metrics"],
                candidate_order=int(summary["candidate_order"]),
            ),
        )
        if bool(summary["validation_selection_eligibility"]["final_ranking_eligible"])
    ]
    return _write_json_artifact(
        path,
        kind=_VALIDATION_SELECTION_KIND,
        schema_version="bittrace-bearings-v3-hardmode-validation-selection-3",
        payload={
            "stage_key": stage_key.value,
            "selection_split": "val",
            "ranking_policy": dict(ranking_policy),
            "stable_selection_policy": dict(stable_selection_policy),
            "selection_status": (
                "selected"
                if selected_candidate_id is not None
                else "no_validation_survivor"
            ),
            "selected_candidate_id": selected_candidate_id,
            "survivor_candidate_ids": survivor_candidate_ids,
            "rejected_candidate_ids": [
                str(summary["candidate_id"])
                for summary in candidate_summaries
                if str(summary["candidate_id"]) not in set(survivor_candidate_ids)
            ],
            "candidate_summaries": list(candidate_summaries),
            "stage_search_summary_ref": (
                stage_search_summary_ref.to_dict()
                if stage_search_summary_ref is not None
                else None
            ),
        },
    )


def _write_final_test_evaluation(
    path: Path,
    *,
    stage_key: StageKey,
    ranking_policy: Mapping[str, object],
    candidate_summary: Mapping[str, object],
    test_metrics: Mapping[str, object],
) -> ArtifactRef:
    return _write_json_artifact(
        path,
        kind=_FINAL_TEST_EVALUATION_KIND,
        schema_version="bittrace-bearings-v3-hardmode-final-test-evaluation-2",
        payload={
            "stage_key": stage_key.value,
            "selection_split": "val",
            "final_evaluation_split": "test",
            "validation_ranking_policy": dict(ranking_policy),
            "selected_candidate_summary": dict(candidate_summary),
            "test_only_final_metrics": dict(test_metrics),
            "warning": "Test metrics are final-evaluation only and were not used for winner selection.",
        },
    )


def _evaluate_artifact_split(
    bundle: _PackedBundle,
    artifact_ref: ArtifactRef,
    *,
    split_name: str,
    artifact_model: _ArtifactModel | None = None,
    row_indices: Sequence[int] | None = None,
    split_label: str | None = None,
) -> dict[str, object]:
    split = bundle.splits[split_name]
    effective_artifact_model = artifact_model or _load_artifact_model(artifact_ref)
    predictions, margins = _predict_split(split, effective_artifact_model)
    return _metrics_from_predictions(
        split,
        predictions,
        margins,
        split_name=split_label or split_name,
        row_indices=row_indices,
    )


def _deep_layer_from_payload(payload: object) -> DeepLayer:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Deep artifact `model.layers` entries must be mappings.")
    op = payload.get("op")
    shift = payload.get("shift", 0)
    mask_bits = payload.get("mask_bits")
    rule = payload.get("rule")
    if not isinstance(op, str) or not op:
        raise ContractValidationError("Deep artifact layers require non-empty `op` values.")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ContractValidationError("Deep artifact layer `shift` values must be integers.")
    mask = None if mask_bits is None else _lsb0_bitstring_to_int(str(mask_bits))
    if rule is not None and (isinstance(rule, bool) or not isinstance(rule, int)):
        raise ContractValidationError("Deep artifact layer `rule` values must be integers.")
    return DeepLayer(op=op, shift=shift, mask=mask, rule=rule)


def _lsb0_bitstring_to_int(bitstring: str) -> int:
    value = 0
    for index, bit in enumerate(bitstring):
        if bit == "1":
            value |= 1 << index
        elif bit != "0":
            raise ContractValidationError("Bitstrings must contain only `0` and `1`.")
    return value


def _load_artifact_model(artifact_ref: ArtifactRef) -> _ArtifactModel:
    artifact_payload = _load_json_mapping(Path(artifact_ref.path))
    return _ArtifactModel(
        layers=tuple(
            _deep_layer_from_payload(payload)
            for payload in artifact_payload["model"]["layers"]
        ),
        prototypes=tuple(
            _lsb0_bitstring_to_int(str(payload))
            for payload in artifact_payload["model"]["prototypes"]
        ),
        prototype_labels=tuple(
            int(value) for value in artifact_payload["model"]["prototype_labels"]
        ),
        class_labels=tuple(int(value) for value in artifact_payload["class_labels"]),
    )


def _predict_split(
    split: _BundleSplit,
    artifact_model: _ArtifactModel,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    embeddings = _apply_layers_as_embedding(
        split.rows,
        artifact_model.layers,
        bit_length=_PACKED_BIT_LENGTH,
    )
    predictions: list[int] = []
    margins: list[int] = []
    for embedding in embeddings:
        predicted_class, margin = _predict_row(
            embedding,
            prototypes=artifact_model.prototypes,
            prototype_labels=artifact_model.prototype_labels,
            class_labels=artifact_model.class_labels,
        )
        predictions.append(int(predicted_class))
        margins.append(int(margin))
    return tuple(predictions), tuple(margins)


def _metrics_from_predictions(
    split: _BundleSplit,
    predictions: Sequence[int],
    margins: Sequence[int],
    *,
    split_name: str,
    row_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    if row_indices is None:
        labels = split.labels
        effective_predictions = predictions
        effective_margins = margins
    else:
        labels = tuple(split.labels[index] for index in row_indices)
        effective_predictions = tuple(predictions[index] for index in row_indices)
        effective_margins = tuple(margins[index] for index in row_indices)
    return _binary_metrics(
        split_name=split_name,
        labels=labels,
        predictions=effective_predictions,
        margins=effective_margins,
    )


def _binary_metrics(
    *,
    split_name: str,
    labels: Sequence[int],
    predictions: Sequence[int],
    margins: Sequence[int],
) -> dict[str, object]:
    unhealthy_label = _LABEL_TO_INT["unhealthy"]
    healthy_label = _LABEL_TO_INT["healthy"]
    tp = sum(1 for actual, predicted in zip(labels, predictions, strict=True) if actual == unhealthy_label and predicted == unhealthy_label)
    fp = sum(1 for actual, predicted in zip(labels, predictions, strict=True) if actual == healthy_label and predicted == unhealthy_label)
    tn = sum(1 for actual, predicted in zip(labels, predictions, strict=True) if actual == healthy_label and predicted == healthy_label)
    fn = sum(1 for actual, predicted in zip(labels, predictions, strict=True) if actual == unhealthy_label and predicted == healthy_label)
    total = len(labels)
    unhealthy_precision = tp / max(1, tp + fp)
    unhealthy_recall = tp / max(1, tp + fn)
    unhealthy_f1 = _f1(unhealthy_precision, unhealthy_recall)
    healthy_precision = tn / max(1, tn + fn)
    healthy_recall = tn / max(1, tn + fp)
    healthy_f1 = _f1(healthy_precision, healthy_recall)
    macro_f1 = (healthy_f1 + unhealthy_f1) / 2.0
    accuracy = (tp + tn) / max(1, total)
    mean_margin = sum(margins) / max(1, len(margins))
    healthy_to_unhealthy_fpr = fp / max(1, fp + tn)
    return {
        "split": split_name,
        "n_rows": total,
        "accuracy": round(accuracy, 6),
        "mean_margin": round(mean_margin, 6),
        "confusion_matrix": {
            "labels": ["healthy", "unhealthy"],
            "matrix": [[tn, fp], [fn, tp]],
            "counts": {
                "healthy": {"healthy": tn, "unhealthy": fp},
                "unhealthy": {"healthy": fn, "unhealthy": tp},
            },
        },
        "healthy_to_unhealthy_fpr": round(healthy_to_unhealthy_fpr, 6),
        "unhealthy_precision": round(unhealthy_precision, 6),
        "unhealthy_recall": round(unhealthy_recall, 6),
        "unhealthy_f1": round(unhealthy_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _resolve_validation_pass_plan(
    split: _BundleSplit,
    *,
    stage_spec: _HardModeStageSpec,
) -> tuple[str, tuple[_ValidationPassSpec, ...]]:
    recording_specs = _validation_pass_specs_by_recording(split)
    if stage_spec.validation_pass_strategy == _VALIDATION_PASS_STRATEGY_AUTO:
        if recording_specs:
            return _VALIDATION_PASS_STRATEGY_BY_RECORDING, recording_specs
        return (
            _VALIDATION_PASS_STRATEGY_STRATIFIED_SHARDS,
            _validation_pass_specs_stratified(split, pass_count=stage_spec.validation_passes),
        )
    if stage_spec.validation_pass_strategy == _VALIDATION_PASS_STRATEGY_BY_RECORDING:
        if not recording_specs:
            raise ContractValidationError(
                "Hard-mode repeated validation `by_val_recording` requires at least two "
                "validation recording groups with both healthy and unhealthy rows."
            )
        return _VALIDATION_PASS_STRATEGY_BY_RECORDING, recording_specs
    return (
        _VALIDATION_PASS_STRATEGY_STRATIFIED_SHARDS,
        _validation_pass_specs_stratified(split, pass_count=stage_spec.validation_passes),
    )


def _validation_pass_specs_by_recording(split: _BundleSplit) -> tuple[_ValidationPassSpec, ...]:
    grouped_indices: dict[str, list[int]] = {}
    for row_index, source_record_id in enumerate(split.source_record_ids):
        recording_id = _validation_recording_id(source_record_id)
        grouped_indices.setdefault(recording_id, []).append(row_index)
    if len(grouped_indices) < 2:
        return ()
    specs: list[_ValidationPassSpec] = []
    for pass_index, recording_id in enumerate(
        sorted(grouped_indices, key=_sortable_recording_id),
        start=1,
    ):
        row_indices = tuple(grouped_indices[recording_id])
        if not _indices_have_binary_coverage(split, row_indices):
            return ()
        specs.append(
            _ValidationPassSpec(
                pass_index=pass_index,
                pass_id=f"val_recording_r{recording_id}",
                row_indices=row_indices,
                recording_ids=(recording_id,),
            )
        )
    return tuple(specs)


def _validation_pass_specs_stratified(
    split: _BundleSplit,
    *,
    pass_count: int,
) -> tuple[_ValidationPassSpec, ...]:
    label_to_indices = {
        label: [
            row_index
            for row_index, value in enumerate(split.labels)
            if value == label
        ]
        for label in sorted(_INT_TO_LABEL)
    }
    for label, row_indices in label_to_indices.items():
        if len(row_indices) < pass_count:
            raise ContractValidationError(
                "Hard-mode repeated validation stratified shards require at least "
                f"{pass_count} `{_INT_TO_LABEL[label]}` rows in validation."
            )
    shard_buckets: list[list[int]] = [[] for _ in range(pass_count)]
    for label in sorted(label_to_indices):
        ordered_indices = sorted(
            label_to_indices[label],
            key=lambda row_index: (split.source_record_ids[row_index], row_index),
        )
        for position, row_index in enumerate(ordered_indices):
            shard_buckets[position % pass_count].append(row_index)
    specs: list[_ValidationPassSpec] = []
    for pass_index, bucket in enumerate(shard_buckets, start=1):
        row_indices = tuple(
            sorted(bucket, key=lambda row_index: (split.source_record_ids[row_index], row_index))
        )
        if not _indices_have_binary_coverage(split, row_indices):
            raise ContractValidationError(
                "Hard-mode repeated validation stratified shards must keep both healthy and "
                f"unhealthy rows in every shard; failed shard {pass_index}."
            )
        recording_ids = tuple(
            sorted(
                {
                    _validation_recording_id(split.source_record_ids[row_index])
                    for row_index in row_indices
                },
                key=_sortable_recording_id,
            )
        )
        specs.append(
            _ValidationPassSpec(
                pass_index=pass_index,
                pass_id=f"val_shard_{pass_index:02d}",
                row_indices=row_indices,
                recording_ids=recording_ids,
            )
        )
    return tuple(specs)


def _indices_have_binary_coverage(
    split: _BundleSplit,
    row_indices: Sequence[int],
) -> bool:
    labels = {split.labels[row_index] for row_index in row_indices}
    return _LABEL_TO_INT["healthy"] in labels and _LABEL_TO_INT["unhealthy"] in labels


def _validation_recording_id(source_record_id: str) -> str:
    if "__r" in source_record_id:
        return source_record_id.rsplit("__r", 1)[-1]
    return source_record_id


def _sortable_recording_id(value: str) -> tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _evaluate_validation_passes(
    split: _BundleSplit,
    predictions: Sequence[int],
    margins: Sequence[int],
    *,
    validation_pass_specs: Sequence[_ValidationPassSpec],
    resolved_validation_pass_strategy: str,
) -> tuple[Mapping[str, object], ...]:
    payloads: list[Mapping[str, object]] = []
    for spec in validation_pass_specs:
        metrics = _metrics_from_predictions(
            split,
            predictions,
            margins,
            split_name=spec.pass_id,
            row_indices=spec.row_indices,
        )
        payloads.append(
            {
                "pass_index": spec.pass_index,
                "pass_id": spec.pass_id,
                "resolved_strategy": resolved_validation_pass_strategy,
                "recording_ids": list(spec.recording_ids),
                "n_rows": len(spec.row_indices),
                "n_healthy": sum(
                    1
                    for row_index in spec.row_indices
                    if split.labels[row_index] == _LABEL_TO_INT["healthy"]
                ),
                "n_unhealthy": sum(
                    1
                    for row_index in spec.row_indices
                    if split.labels[row_index] == _LABEL_TO_INT["unhealthy"]
                ),
                "metrics": metrics,
            }
        )
    return tuple(payloads)


def _aggregate_validation_passes(
    validation_passes: Sequence[Mapping[str, object]],
    *,
    stage_spec: _HardModeStageSpec,
    resolved_validation_pass_strategy: str,
) -> dict[str, object]:
    if not validation_passes:
        raise ContractValidationError("Hard-mode repeated validation resolved to zero passes.")
    metric_extrema = {}
    for metric_name in (
        "healthy_to_unhealthy_fpr",
        "unhealthy_precision",
        "unhealthy_recall",
        "unhealthy_f1",
        "macro_f1",
    ):
        metric_values = [
            float(pass_payload["metrics"][metric_name])
            for pass_payload in validation_passes
        ]
        metric_extrema[metric_name] = {
            "min": round(min(metric_values), 6),
            "max": round(max(metric_values), 6),
        }
    ranking_metrics = {
        "healthy_to_unhealthy_fpr": metric_extrema["healthy_to_unhealthy_fpr"]["max"],
        "unhealthy_precision": metric_extrema["unhealthy_precision"]["min"],
        "unhealthy_recall": metric_extrema["unhealthy_recall"]["min"],
        "unhealthy_f1": metric_extrema["unhealthy_f1"]["min"],
        "macro_f1": metric_extrema["macro_f1"]["min"],
    }
    observed_fpr = float(ranking_metrics["healthy_to_unhealthy_fpr"])
    return {
        "selection_basis": "repeated_validation_aggregate",
        "requested_pass_strategy": stage_spec.validation_pass_strategy,
        "resolved_pass_strategy": resolved_validation_pass_strategy,
        "requested_pass_count": stage_spec.validation_passes,
        "resolved_pass_count": len(validation_passes),
        "aggregation_mode": _VALIDATION_AGGREGATION_MODE,
        "pass_ids": [str(pass_payload["pass_id"]) for pass_payload in validation_passes],
        "metric_extrema": metric_extrema,
        "ranking_metrics": ranking_metrics,
        "fpr_gate": {
            "metric": "healthy_to_unhealthy_fpr",
            "threshold": round(stage_spec.validation_fpr_gate, 6),
            "observed": round(observed_fpr, 6),
            "passed": observed_fpr <= float(stage_spec.validation_fpr_gate),
        },
    }


def _stable_selection_policy() -> dict[str, object]:
    return {
        "selection_basis": "validation_metrics",
        "selection_split": "val",
        "selection_mode": "single_validation_pass",
        "selection_notes": [
            "validation metrics are authoritative for winner selection",
            "explicit FPR gate and repeated validation passes are disabled in this rollback",
        ],
    }


def _validation_alertability(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    unhealthy_recall = float(metrics["unhealthy_recall"])
    tp = int(metrics["tp"])
    if unhealthy_recall <= 0.0 or tp <= 0:
        return {
            "status": ScoutAlertabilityStatus.DEAD_DETECTOR,
            "ranking_eligible": False,
            "guardrail_triggered": True,
            "reason": "validation_unhealthy_recall<=0.0_or_tp<=0",
        }
    return {
        "status": ScoutAlertabilityStatus.ELIGIBLE,
        "ranking_eligible": True,
        "guardrail_triggered": False,
        "reason": "validation_unhealthy_recall>0.0_and_tp>0",
    }


def _ranking_metrics_from_split_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        "healthy_to_unhealthy_fpr": float(metrics["healthy_to_unhealthy_fpr"]),
        "unhealthy_precision": float(metrics["unhealthy_precision"]),
        "unhealthy_recall": float(metrics["unhealthy_recall"]),
        "unhealthy_f1": float(metrics["unhealthy_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _ranking_sort_key(
    metrics: Mapping[str, float],
    *,
    candidate_order: int,
) -> tuple[float, float, float, float, float, int]:
    return (
        float(metrics["healthy_to_unhealthy_fpr"]),
        -float(metrics["unhealthy_precision"]),
        -float(metrics["unhealthy_recall"]),
        -float(metrics["unhealthy_f1"]),
        -float(metrics["macro_f1"]),
        candidate_order,
    )


def _hardmode_ranking_policy(
    *,
    split_name: str,
    stable_selection_policy: Mapping[str, object],
) -> dict[str, object]:
    return {
        "comparison_type": "lexicographic_mixed_direction",
        "metric_order": [
            {"metric": "healthy_to_unhealthy_fpr", "priority": "primary", "direction": "asc", "split": split_name},
            {"metric": "unhealthy_precision", "priority": "secondary", "direction": "desc", "split": split_name},
            {"metric": "unhealthy_recall", "priority": "tertiary", "direction": "desc", "split": split_name},
            {"metric": "unhealthy_f1", "priority": "quaternary", "direction": "desc", "split": split_name},
            {"metric": "macro_f1", "priority": "quinary", "direction": "desc", "split": split_name},
        ],
        "ranking_mode": "scout_unhealthy_alert",
        "selection_split": "val",
        "selection_basis": "validation_metrics",
        "stable_selection_policy": dict(stable_selection_policy),
    }


def _trial_result_to_dict(trial: _TrialResult) -> dict[str, object]:
    return {
        "trial_index": trial.trial_index,
        "seed": trial.seed,
        "run_dir": str(trial.run_dir.resolve()),
        "artifact_ref": trial.artifact_ref.to_dict(),
        "engine_metrics_summary_ref": trial.engine_metrics_summary_ref.to_dict(),
        "engine_proxy_metrics": dict(trial.engine_proxy_metrics),
        "train_metrics": dict(trial.train_metrics),
        "val_metrics": dict(trial.val_metrics),
        "ranking_metrics": dict(trial.chosen_ranking_metrics),
        "completed_generations": trial.completed_generations,
        "stopped_early": trial.stopped_early,
        "provenance": dict(trial.provenance),
    }


def _select_best_workup(
    workups: Sequence[_CandidateWorkup],
    *,
    selection_mode: str,
) -> _CandidateWorkup:
    if selection_mode == "final":
        eligible_workups = tuple(
            workup for workup in workups if workup.candidate_input.ranking_eligible
        )
        if eligible_workups:
            workups = eligible_workups
        else:
            search_eligible_workups = tuple(
                workup for workup in workups if workup.search_ranking_eligible
            )
            if search_eligible_workups:
                workups = search_eligible_workups
    elif selection_mode == "search":
        eligible_workups = tuple(
            workup for workup in workups if workup.search_ranking_eligible
        )
        if eligible_workups:
            workups = eligible_workups
    else:  # pragma: no cover - internal caller contract
        raise ContractValidationError(f"Unsupported hard-mode selection mode `{selection_mode}`.")
    return min(
        workups,
        key=lambda workup: _ranking_sort_key(
            workup.candidate_input.ranking_metrics,
            candidate_order=workup.candidate_input.candidate_order,
        ),
    )


def _combine_validation_selection_reason(
    *,
    alertability: Mapping[str, object],
) -> str | None:
    reason = alertability.get("reason")
    return None if reason is None else str(reason)


def _enum_value(value: object) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _candidate_validation_passes_entry(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_id": str(summary["candidate_id"]),
        "candidate_order": int(summary["candidate_order"]),
        "selected_k_per_class": int(summary["selected_k_per_class"]),
        "provenance": dict(summary["provenance"]),
        "chosen_trial_index": int(summary["chosen_trial"]["trial_index"]),
        "validation_passes": list(summary["validation_passes"]),
    }


def _candidate_validation_aggregate_entry(
    summary: Mapping[str, object],
    *,
    overall_rank: int,
    survivor_rank: int | None,
) -> dict[str, object]:
    return {
        "candidate_id": str(summary["candidate_id"]),
        "candidate_order": int(summary["candidate_order"]),
        "selected_k_per_class": int(summary["selected_k_per_class"]),
        "provenance": dict(summary["provenance"]),
        "overall_rank": overall_rank,
        "survivor_rank": survivor_rank,
        "ranking_metrics": dict(summary["ranking_metrics"]),
        "validation_reference_metrics": dict(summary["validation_reference_metrics"]),
        "validation_aggregate": dict(summary["validation_aggregate"]),
        "validation_selection_eligibility": dict(summary["validation_selection_eligibility"]),
    }


def _candidate_provenance(
    *,
    source: str,
    lineage_basis: str,
    family_branch_index: int,
    evolution_template: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "lineage_basis": lineage_basis,
        "family_branch_index": family_branch_index,
        "base_seed": int(evolution_template["seed"]),
        "warm_start_used": False,
    }
    if extra is not None:
        payload.update(dict(extra))
    return payload


def _search_budget(
    stage_spec: _HardModeStageSpec,
    evolution_template: Mapping[str, object],
) -> dict[str, object]:
    return {
        "trials_per_candidate": stage_spec.trials,
        "generations": int(evolution_template["generations"]),
        "population_size": int(evolution_template["population_size"]),
        "mu": int(evolution_template["mu"]),
        "lam": int(evolution_template["lam"]),
        "elite_count": int(evolution_template["elite_count"]),
        "min_layers": int(evolution_template["min_layers"]),
        "max_layers": int(evolution_template["max_layers"]),
        "mutation_rate": float(evolution_template["mutation_rate"]),
        "mutation_rate_schedule": str(evolution_template["mutation_rate_schedule"]),
        "selection_mode": str(evolution_template["selection_mode"]),
        "tournament_k": int(evolution_template["tournament_k"]),
        "early_stopping_patience": int(evolution_template["early_stopping_patience"]),
    }


def _exploration_policy(stage_spec: _HardModeStageSpec) -> dict[str, object]:
    return {
        "initial_search_branches": stage_spec.initial_search_branches,
        "local_refine_initial_branches": (
            stage_spec.initial_search_branches - stage_spec.bounded_random_branches
        ),
        "bounded_random_initial_branches": stage_spec.bounded_random_branches,
        "bounded_random_fraction": stage_spec.bounded_random_fraction,
        "winner_replay_branches": stage_spec.winner_replay_branches,
        "winner_mutation_branches": stage_spec.winner_mutation_branches,
        "seed_stride": stage_spec.seed_stride,
        "warm_start_used": False,
        "winner_followup_note": (
            "winner_replay and winner_mutation replay the winning search family with fresh "
            "seeded runs; artifact resume is not used."
        ),
    }


def _total_candidate_branches(stage_spec: _HardModeStageSpec) -> int:
    return (
        stage_spec.initial_search_branches
        + stage_spec.winner_replay_branches
        + stage_spec.winner_mutation_branches
    )


def _require_int_field(raw_value: object, *, path: str, minimum: int) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < minimum:
        raise ContractValidationError(f"`{path}` must be an integer >= {minimum}.")
    return raw_value


def _require_probability(raw_value: object, *, path: str) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ContractValidationError(f"`{path}` must be a float in [0.0, 1.0].")
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        raise ContractValidationError(f"`{path}` must be a float in [0.0, 1.0].")
    return value


def _resolve_fractional_branch_count(
    *,
    initial_search_branches: int,
    fraction: float,
    path: str,
) -> int:
    exact_count = initial_search_branches * fraction
    rounded_count = round(exact_count)
    if not math.isclose(exact_count, rounded_count, rel_tol=0.0, abs_tol=1e-9):
        raise ContractValidationError(
            f"`{path}` must map exactly onto an integer number of branches for the configured "
            "`search_branches`."
        )
    bounded_random_branches = int(rounded_count)
    if bounded_random_branches >= initial_search_branches and fraction > 0.0:
        raise ContractValidationError(
            f"`{path}` must leave at least one non-random local-refine branch."
        )
    return bounded_random_branches


def _merge_optional_evolution_template(
    base_template: Mapping[str, object],
    raw_override: object,
    *,
    path: str,
    required: bool,
) -> dict[str, object] | None:
    if raw_override in ({}, None):
        if required:
            raise ContractValidationError(f"`{path}` must be a mapping when enabled.")
        return None
    if not isinstance(raw_override, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    merged_template = {
        **dict(base_template),
        **dict(raw_override),
    }
    try:
        EvolutionConfig.from_mapping(merged_template)
    except Exception as exc:  # pragma: no cover - config validation path
        raise ContractValidationError(f"`{path}` is invalid: {exc}") from exc
    return merged_template


__all__ = [
    "HardModeBinaryCampaignBridge",
    "build_hardmode_bridge",
    "hardmode_enabled",
]
