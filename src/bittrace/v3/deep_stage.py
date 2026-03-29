"""Canonical V3 deep-stage artifact emission for Deep stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bittrace.core.config import DeepTrainingConfig
from bittrace.v3.artifacts import write_json_artifact
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepInputRef,
    DeepPromotionCandidate,
    DeepRankingMode,
    ExecutionAcceleration,
    ExecutionTrace,
    PassFail,
    PromotedDeepResult,
    PromotionStage,
    RetainedAlternateKind,
    RetainedAlternateRef,
    ScoutAlertabilityStatus,
    StageKey,
    StageRequest,
    StageResult,
)
from bittrace.v3.deep_materialization import (
    materialize_waveform_capacity_refinement_candidates,
    materialize_waveform_deep_candidates,
)


DEEP_PROMOTION_ARTIFACT_NAME = "bt3.deep_promotion.json"
DEEP_STAGE_RESULT_ARTIFACT_NAME = "bt3.stage_result.json"
_CANONICAL_DEEP_STAGE_KEYS = frozenset(
    {StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN, StageKey.CAPACITY_REFINEMENT}
)
_CANONICAL_MAIN_SCREEN_K = 1
_DEFAULT_CAPACITY_REFINEMENT_K_CANDIDATES = (1, 3, 5)
_RUNTIME_STATUS_ALIASES = {
    ScoutAlertabilityStatus.ELIGIBLE.value: ScoutAlertabilityStatus.ELIGIBLE,
    ScoutAlertabilityStatus.DEAD_DETECTOR.value: ScoutAlertabilityStatus.DEAD_DETECTOR,
    ScoutAlertabilityStatus.BLOCKED.value: ScoutAlertabilityStatus.BLOCKED,
    "rejected_dead_detector": ScoutAlertabilityStatus.DEAD_DETECTOR,
    "not_applicable": ScoutAlertabilityStatus.BLOCKED,
    "not_evaluated": ScoutAlertabilityStatus.BLOCKED,
}
_CANONICAL_DEEP_RANKING_POLICY = {
    "comparison_type": "lexicographic_mixed_direction",
    "metric_order": [
        {
            "metric": "healthy_to_unhealthy_fpr",
            "priority": "primary",
            "direction": "asc",
            "split": "val",
        },
        {
            "metric": "unhealthy_precision",
            "priority": "secondary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "unhealthy_recall",
            "priority": "tertiary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "unhealthy_f1",
            "priority": "quaternary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "macro_f1",
            "priority": "quinary",
            "direction": "desc",
            "split": "val",
        },
    ],
    "ranking_mode": DeepRankingMode.SCOUT_UNHEALTHY_ALERT.value,
    "eligibility_guardrail": {
        "ranking_eligible_field": "ranking_eligible",
        "guardrail_trigger_field": "scout_alertability_guardrail_triggered",
        "guardrail_status_field": "scout_alertability_status",
        "guardrail_reason_field": "scout_alertability_reason",
        "eligible_if": {
            "unhealthy_recall": ">0.0",
            "tp": ">0",
        },
        "reject_when": [
            "unhealthy_recall <= 0.0",
            "tp <= 0",
        ],
    },
    "tie_policy": (
        "If healthy_to_unhealthy_fpr, unhealthy_precision, unhealthy_recall, "
        "unhealthy_f1, and macro_f1 tie exactly, retain the earliest candidate "
        "in the sweep spec order."
    ),
}
_REQUIRED_RANKING_METRICS = (
    "healthy_to_unhealthy_fpr",
    "unhealthy_precision",
    "unhealthy_recall",
    "unhealthy_f1",
    "macro_f1",
)

DeepStageCandidatesEvaluator = Callable[
    [StageRequest],
    Sequence[object],
]
CapacityRefinementCandidatesEvaluator = Callable[
    [StageRequest, Sequence[int]],
    Sequence[object],
]


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a bool.")
    return value


def _require_int(value: object, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"`{field_name}` must be an int.")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"`{field_name}` must be >= {minimum}.")
    return value


def _require_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"`{field_name}` must be numeric.")
    return float(value)


def _require_str(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"`{field_name}` must be a string.")
    if not allow_empty and value == "":
        raise ContractValidationError(f"`{field_name}` cannot be empty.")
    return value


def _require_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name=field_name)


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _copy_ranking_metrics(
    value: object,
    *,
    field_name: str,
) -> dict[str, float]:
    metrics = {
        _require_str(metric_name, field_name=f"{field_name}.<key>"): _require_float(
            metric_value,
            field_name=f"{field_name}.{metric_name}",
        )
        for metric_name, metric_value in _require_mapping(value, field_name=field_name).items()
    }
    missing = [metric_name for metric_name in _REQUIRED_RANKING_METRICS if metric_name not in metrics]
    if missing:
        raise ContractValidationError(
            f"`{field_name}` is missing required canonical Scout metrics: {', '.join(missing)}."
        )
    return metrics


def _coerce_optional_artifact_ref(value: object, *, field_name: str) -> ArtifactRef | None:
    if value is None:
        return None
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, Mapping):
        return ArtifactRef.from_dict(value)
    raise ContractValidationError(f"`{field_name}` must be an `ArtifactRef` or mapping.")


def _coerce_alertability_status(
    value: ScoutAlertabilityStatus | str,
    *,
    field_name: str,
) -> ScoutAlertabilityStatus:
    if isinstance(value, ScoutAlertabilityStatus):
        return value
    if not isinstance(value, str):
        raise ContractValidationError(f"`{field_name}` must be a `ScoutAlertabilityStatus` value.")
    try:
        return _RUNTIME_STATUS_ALIASES[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(_RUNTIME_STATUS_ALIASES))
        raise ContractValidationError(
            f"`{field_name}` must be one of: {allowed}."
        ) from exc


def _coerce_ranking_mode(value: DeepRankingMode | str) -> DeepRankingMode:
    try:
        ranking_mode = value if isinstance(value, DeepRankingMode) else DeepRankingMode(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in DeepRankingMode)
        raise ContractValidationError(
            f"`ranking_mode` must be one of: {allowed}."
        ) from exc
    if ranking_mode != DeepRankingMode.SCOUT_UNHEALTHY_ALERT:
        raise ContractValidationError("Deep ranking mode must be `scout_unhealthy_alert`.")
    return ranking_mode


def _coerce_optional_pass_fail(value: PassFail | str | None) -> PassFail | None:
    if value is None:
        return None
    return value if isinstance(value, PassFail) else PassFail(value)


def _expected_promotion_stage(stage_key: StageKey) -> PromotionStage:
    if stage_key == StageKey.CAPACITY_REFINEMENT:
        return PromotionStage.CAPACITY_REFINEMENT
    return PromotionStage.MAIN_SCREEN


def _resolve_promotion_stage(request: StageRequest) -> PromotionStage:
    expected_stage = _expected_promotion_stage(request.stage_key)
    if request.promotion_stage is None:
        return expected_stage
    promotion_stage = (
        request.promotion_stage
        if isinstance(request.promotion_stage, PromotionStage)
        else PromotionStage(request.promotion_stage)
    )
    if promotion_stage != expected_stage:
        raise ContractValidationError(
            f"Canonical V3 `{request.stage_key.value}` only supports "
            f"`promotion_stage={expected_stage.value}`."
        )
    return promotion_stage


def _resolve_selected_k_per_class(
    *,
    explicit_value: object,
    legacy_value: object,
    effective_engine_deep_config: Mapping[str, object],
    field_name_prefix: str,
) -> int | None:
    candidate_k_values: list[int] = []
    if explicit_value is not None:
        candidate_k_values.append(
            _require_int(
                explicit_value,
                field_name=f"{field_name_prefix}.selected_k_per_class",
                minimum=1,
            )
        )
    if legacy_value is not None:
        candidate_k_values.append(
            _require_int(
                legacy_value,
                field_name=f"{field_name_prefix}.k_medoids_per_class",
                minimum=1,
            )
        )
    config_k = effective_engine_deep_config.get("k_medoids_per_class")
    if config_k is not None:
        candidate_k_values.append(
            _require_int(
                config_k,
                field_name=f"{field_name_prefix}.effective_engine_deep_config.k_medoids_per_class",
                minimum=1,
            )
        )
    if not candidate_k_values:
        return None
    if any(k_value != candidate_k_values[0] for k_value in candidate_k_values[1:]):
        raise ContractValidationError(
            f"`{field_name_prefix}` must keep `selected_k_per_class` and `k_medoids_per_class` consistent."
        )
    return candidate_k_values[0]


def _require_stage_input_selected_k(
    candidate: "DeepStageCandidateInput",
    *,
    field_name: str,
    required: bool,
) -> int | None:
    selected_k = candidate.selected_k_per_class
    if selected_k is None:
        selected_k = _resolve_selected_k_per_class(
            explicit_value=None,
            legacy_value=candidate.k_medoids_per_class,
            effective_engine_deep_config=candidate.effective_engine_deep_config,
            field_name_prefix=field_name,
        )
    if required and selected_k is None:
        raise ContractValidationError(
            f"`{field_name}` must carry exactly one deployed `selected_k_per_class`."
        )
    return selected_k


def _validate_unique_k_values(values: Sequence[int], *, field_name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        duplicate_text = ", ".join(str(value) for value in duplicates)
        raise ContractValidationError(
            f"`{field_name}` must not repeat `k` values; duplicated: {duplicate_text}."
        )


@dataclass(frozen=True, slots=True)
class DeepStageCandidateInput:
    """Validated Deep runtime output translated into V3 promotion inputs."""

    candidate_id: str
    candidate_order: int
    branch_mode: str
    ranking_metrics: Mapping[str, object]
    scout_alertability_status: ScoutAlertabilityStatus | str
    ranking_eligible: bool = True
    scout_alertability_guardrail_triggered: bool = False
    scout_alertability_reason: str | None = None
    effective_engine_deep_config: Mapping[str, object] = field(default_factory=dict)
    best_deep_artifact_ref: ArtifactRef | Mapping[str, object] | None = None
    metrics_summary_ref: ArtifactRef | Mapping[str, object] | None = None
    candidate_report_ref: ArtifactRef | Mapping[str, object] | None = None
    checkpoint_ref: ArtifactRef | Mapping[str, object] | None = None
    frontend_input_id: str | None = None
    frontend_fingerprint: str | None = None
    parent_anchor_fingerprint: str | None = None
    k_medoids_per_class: int | None = None
    selected_k_per_class: int | None = None
    k_medoids_search_values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(self.candidate_id, field_name="DeepStageCandidateInput.candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_order",
            _require_int(
                self.candidate_order,
                field_name="DeepStageCandidateInput.candidate_order",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "branch_mode",
            _require_str(self.branch_mode, field_name="DeepStageCandidateInput.branch_mode"),
        )
        object.__setattr__(
            self,
            "ranking_metrics",
            _copy_ranking_metrics(
                self.ranking_metrics,
                field_name="DeepStageCandidateInput.ranking_metrics",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_status",
            _coerce_alertability_status(
                self.scout_alertability_status,
                field_name="DeepStageCandidateInput.scout_alertability_status",
            ),
        )
        object.__setattr__(
            self,
            "ranking_eligible",
            _require_bool(
                self.ranking_eligible,
                field_name="DeepStageCandidateInput.ranking_eligible",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_guardrail_triggered",
            _require_bool(
                self.scout_alertability_guardrail_triggered,
                field_name="DeepStageCandidateInput.scout_alertability_guardrail_triggered",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_reason",
            _require_optional_str(
                self.scout_alertability_reason,
                field_name="DeepStageCandidateInput.scout_alertability_reason",
            ),
        )
        object.__setattr__(
            self,
            "effective_engine_deep_config",
            dict(_require_mapping(
                self.effective_engine_deep_config,
                field_name="DeepStageCandidateInput.effective_engine_deep_config",
            )),
        )
        for field_name in (
            "best_deep_artifact_ref",
            "metrics_summary_ref",
            "candidate_report_ref",
            "checkpoint_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_optional_artifact_ref(
                    getattr(self, field_name),
                    field_name=f"DeepStageCandidateInput.{field_name}",
                ),
            )
        for field_name in (
            "frontend_input_id",
            "frontend_fingerprint",
            "parent_anchor_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_optional_str(
                    getattr(self, field_name),
                    field_name=f"DeepStageCandidateInput.{field_name}",
                ),
            )
        if self.k_medoids_per_class is not None:
            object.__setattr__(
                self,
                "k_medoids_per_class",
                _require_int(
                    self.k_medoids_per_class,
                    field_name="DeepStageCandidateInput.k_medoids_per_class",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "selected_k_per_class",
            _resolve_selected_k_per_class(
                explicit_value=self.selected_k_per_class,
                legacy_value=self.k_medoids_per_class,
                effective_engine_deep_config=self.effective_engine_deep_config,
                field_name_prefix="DeepStageCandidateInput",
            ),
        )
        object.__setattr__(
            self,
            "k_medoids_search_values",
            tuple(
                _require_int(
                    k_value,
                    field_name=f"DeepStageCandidateInput.k_medoids_search_values[{index}]",
                    minimum=1,
                )
                for index, k_value in enumerate(tuple(self.k_medoids_search_values))
            ),
        )

    @classmethod
    def from_runtime_output(cls, payload: Mapping[str, object]) -> DeepStageCandidateInput:
        """Translate a validated Deep runtime payload into the V3 stage input model."""

        return cls(
            candidate_id=payload.get("candidate_id"),
            candidate_order=payload.get("candidate_order"),
            branch_mode=payload.get("branch_mode"),
            ranking_metrics=payload.get("ranking_metrics", {}),
            scout_alertability_status=payload.get("scout_alertability_status", ScoutAlertabilityStatus.BLOCKED.value),
            ranking_eligible=payload.get("ranking_eligible", True),
            scout_alertability_guardrail_triggered=payload.get(
                "scout_alertability_guardrail_triggered",
                False,
            ),
            scout_alertability_reason=payload.get("scout_alertability_reason"),
            effective_engine_deep_config=_require_mapping(
                payload.get("effective_engine_deep_config", {}),
                field_name="runtime_output.effective_engine_deep_config",
            ),
            best_deep_artifact_ref=payload.get("best_deep_artifact_ref"),
            metrics_summary_ref=payload.get("metrics_summary_ref"),
            candidate_report_ref=payload.get("candidate_report_ref"),
            checkpoint_ref=payload.get("checkpoint_ref"),
            frontend_input_id=payload.get("frontend_input_id"),
            frontend_fingerprint=payload.get("frontend_fingerprint"),
            parent_anchor_fingerprint=payload.get("parent_anchor_fingerprint"),
            k_medoids_per_class=payload.get("k_medoids_per_class"),
            selected_k_per_class=payload.get("selected_k_per_class"),
            k_medoids_search_values=tuple(payload.get("k_medoids_search_values", ())),
        )

    def to_contract_candidate(self) -> DeepPromotionCandidate:
        ranking_eligible = (
            self.ranking_eligible
            and self.scout_alertability_status == ScoutAlertabilityStatus.ELIGIBLE
            and not self.scout_alertability_guardrail_triggered
        )
        guardrail_triggered = (
            self.scout_alertability_guardrail_triggered
            or self.scout_alertability_status == ScoutAlertabilityStatus.DEAD_DETECTOR
        )
        return DeepPromotionCandidate(
            candidate_id=self.candidate_id,
            candidate_order=self.candidate_order,
            branch_mode=self.branch_mode,
            ranking_eligible=ranking_eligible,
            scout_alertability_status=self.scout_alertability_status,
            scout_alertability_guardrail_triggered=guardrail_triggered,
            scout_alertability_reason=self.scout_alertability_reason,
            effective_engine_deep_config=self.effective_engine_deep_config,
            best_deep_artifact_ref=self.best_deep_artifact_ref,
            metrics_summary_ref=self.metrics_summary_ref,
            candidate_report_ref=self.candidate_report_ref,
            checkpoint_ref=self.checkpoint_ref,
            frontend_input_id=self.frontend_input_id,
            frontend_fingerprint=self.frontend_fingerprint,
            parent_anchor_fingerprint=self.parent_anchor_fingerprint,
            k_medoids_per_class=self.k_medoids_per_class,
            selected_k_per_class=self.selected_k_per_class,
            k_medoids_search_values=self.k_medoids_search_values,
        )


@dataclass(frozen=True, slots=True)
class DeepRetainedAlternateSpec:
    """Metadata-only retained Deep alternate validated against the stage lineage."""

    candidate_id: str
    artifact_ref: ArtifactRef | Mapping[str, object]
    reason: str
    frontend_input_id: str
    frontend_fingerprint: str
    rank: int | None = None
    candidate_kind: RetainedAlternateKind = RetainedAlternateKind.DEEP_CANDIDATE
    ranking_mode: DeepRankingMode | str = DeepRankingMode.SCOUT_UNHEALTHY_ALERT
    promotion_stage: PromotionStage | str = PromotionStage.MAIN_SCREEN
    parent_anchor_fingerprint: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(self.candidate_id, field_name="DeepRetainedAlternateSpec.candidate_id"),
        )
        object.__setattr__(
            self,
            "artifact_ref",
            _coerce_optional_artifact_ref(
                self.artifact_ref,
                field_name="DeepRetainedAlternateSpec.artifact_ref",
            ),
        )
        if self.artifact_ref is None:
            raise ContractValidationError("`DeepRetainedAlternateSpec.artifact_ref` is required.")
        object.__setattr__(
            self,
            "reason",
            _require_str(self.reason, field_name="DeepRetainedAlternateSpec.reason"),
        )
        object.__setattr__(
            self,
            "frontend_input_id",
            _require_str(
                self.frontend_input_id,
                field_name="DeepRetainedAlternateSpec.frontend_input_id",
            ),
        )
        object.__setattr__(
            self,
            "frontend_fingerprint",
            _require_str(
                self.frontend_fingerprint,
                field_name="DeepRetainedAlternateSpec.frontend_fingerprint",
            ),
        )
        if self.rank is not None:
            object.__setattr__(
                self,
                "rank",
                _require_int(
                    self.rank,
                    field_name="DeepRetainedAlternateSpec.rank",
                    minimum=1,
                ),
            )
        if not isinstance(self.candidate_kind, RetainedAlternateKind):
            object.__setattr__(
                self,
                "candidate_kind",
                RetainedAlternateKind(self.candidate_kind),
            )
        object.__setattr__(
            self,
            "ranking_mode",
            _coerce_ranking_mode(self.ranking_mode),
        )
        promotion_stage = (
            self.promotion_stage
            if isinstance(self.promotion_stage, PromotionStage)
            else PromotionStage(self.promotion_stage)
        )
        object.__setattr__(self, "promotion_stage", promotion_stage)
        object.__setattr__(
            self,
            "parent_anchor_fingerprint",
            _require_optional_str(
                self.parent_anchor_fingerprint,
                field_name="DeepRetainedAlternateSpec.parent_anchor_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(_require_mapping(self.metadata, field_name="DeepRetainedAlternateSpec.metadata")),
        )

    def to_contract(self) -> RetainedAlternateRef:
        metadata = dict(self.metadata)
        metadata["ranking_mode"] = self.ranking_mode.value
        metadata["promotion_stage"] = self.promotion_stage.value
        metadata["frontend_input_id"] = self.frontend_input_id
        metadata["frontend_fingerprint"] = self.frontend_fingerprint
        if self.parent_anchor_fingerprint is not None:
            metadata["parent_anchor_fingerprint"] = self.parent_anchor_fingerprint
        return RetainedAlternateRef(
            candidate_kind=self.candidate_kind,
            candidate_id=self.candidate_id,
            artifact_ref=self.artifact_ref,
            reason=self.reason,
            rank=self.rank,
            metadata_only=True,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class DeepStageRunResult:
    """Written deep-stage artifacts and the normalized candidate ordering."""

    deep_promotion: PromotedDeepResult
    deep_promotion_ref: ArtifactRef
    stage_result: StageResult
    stage_result_ref: ArtifactRef
    evaluated_candidates: tuple[DeepPromotionCandidate, ...]
    ranked_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RankedDeepCandidate:
    stage_input: DeepStageCandidateInput
    contract_candidate: DeepPromotionCandidate


CapacityRefinementCandidateEvaluator = Callable[
    [int, int],
    DeepStageCandidateInput | Mapping[str, object],
]


def deep_stage_candidate_input_from_runtime_output(
    payload: Mapping[str, object],
) -> DeepStageCandidateInput:
    """Translate a validated Deep runtime payload into the canonical V3 stage input."""

    return DeepStageCandidateInput.from_runtime_output(payload)


def run_capacity_refinement_stage(
    request: StageRequest,
    *,
    k_medoids_per_class_candidates: Sequence[int] | None = None,
    evaluate_k_candidate: CapacityRefinementCandidateEvaluator | None = None,
    evaluate_k_candidates: CapacityRefinementCandidatesEvaluator | None = None,
    pass_fail: PassFail | str | None = None,
    exact_blocker: str | None = None,
    ranking_mode: DeepRankingMode | str = DeepRankingMode.SCOUT_UNHEALTHY_ALERT,
    execution_trace: ExecutionTrace | None = None,
    deep_training_config: DeepTrainingConfig | Mapping[str, object] | None = None,
    retained_alternates: Sequence[DeepRetainedAlternateSpec] = (),
    canonical_downstream_reference: ArtifactRef | None = None,
    device_agnostic_export: Mapping[str, object] | None = None,
) -> DeepStageRunResult:
    """Execute canonical S5 multi-`k` evaluation across explicit `k` candidates."""

    _validate_deep_stage_request(request)
    if _resolve_promotion_stage(request) != PromotionStage.CAPACITY_REFINEMENT:
        raise ContractValidationError(
            "Canonical V3 capacity-refinement orchestration requires "
            "`promotion_stage=capacity_refinement`."
        )
    requested_k_candidates = (
        tuple(k_medoids_per_class_candidates)
        if k_medoids_per_class_candidates is not None
        else _DEFAULT_CAPACITY_REFINEMENT_K_CANDIDATES
    )
    normalized_k_candidates = _normalize_k_candidates_tested(
        requested_k_candidates,
        field_name="k_medoids_per_class_candidates",
    )
    if evaluate_k_candidate is not None and evaluate_k_candidates is not None:
        raise ContractValidationError(
            "Canonical V3 capacity-refinement accepts at most one of "
            "`evaluate_k_candidate` or `evaluate_k_candidates`."
        )
    effective_deep_training_config = _coerce_deep_training_config(deep_training_config)
    evaluated_candidates: list[DeepStageCandidateInput] = []
    if evaluate_k_candidates is not None:
        evaluated_candidates.extend(
            _coerce_candidate_input(candidate)
            for candidate in tuple(evaluate_k_candidates(request, normalized_k_candidates))
        )
        if len(evaluated_candidates) != len(normalized_k_candidates):
            raise ContractValidationError(
                "Capacity-refinement multi-candidate evaluation must return exactly one "
                "candidate per explicit `k_medoids_per_class_candidates` value."
            )
        for candidate_order, (candidate_input, k_value) in enumerate(
            zip(evaluated_candidates, normalized_k_candidates, strict=True),
            start=1,
        ):
            if candidate_input.candidate_order != candidate_order:
                raise ContractValidationError(
                    "Capacity-refinement multi-candidate evaluation must preserve the explicit "
                    "`k_medoids_per_class_candidates` order in `candidate_order`."
                )
            selected_k = _require_stage_input_selected_k(
                candidate_input,
                field_name=f"capacity_refinement.candidate[{candidate_order - 1}]",
                required=True,
            )
            if selected_k != k_value:
                raise ContractValidationError(
                    "Capacity-refinement multi-candidate evaluation must return candidates whose "
                    "`selected_k_per_class` values match the requested explicit `k` ordering."
                )
    elif evaluate_k_candidate is None:
        if not request.deep_inputs:
            raise ContractValidationError(
                "Canonical V3 capacity-refinement requires explicit `evaluate_k_candidate` "
                "or valid replay-ready `deep_inputs`."
            )
        evaluated_candidates.extend(
            _coerce_candidate_input(candidate)
            for candidate in materialize_waveform_capacity_refinement_candidates(
                request,
                deep_inputs=request.deep_inputs,
                k_medoids_per_class_candidates=normalized_k_candidates,
                deep_training_config=effective_deep_training_config,
            )
        )
    else:
        for candidate_order, k_value in enumerate(normalized_k_candidates, start=1):
            candidate_input = _coerce_candidate_input(evaluate_k_candidate(k_value, candidate_order))
            if candidate_input.candidate_order != candidate_order:
                raise ContractValidationError(
                    "Capacity-refinement candidate evaluation must preserve the explicit "
                    "`k_medoids_per_class_candidates` order in `candidate_order`."
                )
            selected_k = _require_stage_input_selected_k(
                candidate_input,
                field_name=f"capacity_refinement.candidate[{candidate_order - 1}]",
                required=True,
            )
            if selected_k != k_value:
                raise ContractValidationError(
                    "Capacity-refinement candidate evaluation must return a candidate whose "
                    "`selected_k_per_class` matches the requested explicit `k`."
                )
            evaluated_candidates.append(candidate_input)

    return run_deep_stage(
        request,
        candidates=tuple(evaluated_candidates),
        k_candidates_tested=normalized_k_candidates,
        pass_fail=pass_fail,
        exact_blocker=exact_blocker,
        ranking_mode=ranking_mode,
        execution_trace=execution_trace,
        deep_training_config=effective_deep_training_config,
        retained_alternates=retained_alternates,
        canonical_downstream_reference=canonical_downstream_reference,
        device_agnostic_export=device_agnostic_export,
    )


def run_deep_stage(
    request: StageRequest,
    *,
    candidates: Sequence[DeepStageCandidateInput | Mapping[str, object]] | None = None,
    evaluate_candidates: DeepStageCandidatesEvaluator | None = None,
    k_candidates_tested: Sequence[int] | None = None,
    pass_fail: PassFail | str | None = None,
    exact_blocker: str | None = None,
    ranking_mode: DeepRankingMode | str = DeepRankingMode.SCOUT_UNHEALTHY_ALERT,
    execution_trace: ExecutionTrace | None = None,
    deep_training_config: DeepTrainingConfig | Mapping[str, object] | None = None,
    retained_alternates: Sequence[DeepRetainedAlternateSpec] = (),
    canonical_downstream_reference: ArtifactRef | None = None,
    device_agnostic_export: Mapping[str, object] | None = None,
) -> DeepStageRunResult:
    """Emit the canonical V3 deep-promotion and stage-result artifacts."""

    _validate_deep_stage_request(request)
    effective_ranking_mode = _coerce_ranking_mode(ranking_mode)
    effective_promotion_stage = _resolve_promotion_stage(request)
    requested_pass_fail = _coerce_optional_pass_fail(pass_fail)
    effective_deep_training_config = _coerce_deep_training_config(deep_training_config)
    effective_execution_trace = _resolve_execution_trace(
        execution_trace if execution_trace is not None else request.execution_trace,
        deep_training_config=effective_deep_training_config,
    )
    candidate_inputs = _resolve_candidate_inputs(
        request,
        candidates=candidates,
        evaluate_candidates=evaluate_candidates,
        requested_pass_fail=requested_pass_fail,
        exact_blocker=exact_blocker,
        deep_training_config=effective_deep_training_config,
    )

    normalized_candidates = tuple(
        _normalize_candidate(
            candidate,
            request=request,
            promotion_stage=effective_promotion_stage,
        )
        for candidate in candidate_inputs
    )
    ranked_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in normalized_candidates
                if candidate.contract_candidate.ranking_eligible
            ),
            key=_deep_ranking_sort_key,
        )
    )
    effective_pass_fail, effective_exact_blocker = _resolve_stage_outcome(
        requested_pass_fail=requested_pass_fail,
        exact_blocker=exact_blocker,
        ranked_candidates=ranked_candidates,
    )
    effective_deep_inputs = _resolve_deep_inputs(
        request,
        pass_fail=effective_pass_fail,
    )
    effective_k_candidates_tested = _resolve_k_candidates_tested(
        promotion_stage=effective_promotion_stage,
        explicit_k_candidates_tested=k_candidates_tested,
        normalized_candidates=normalized_candidates,
    )
    alternate_refs = tuple(
        _validate_retained_alternate(
            spec,
            request=request,
            ranking_mode=effective_ranking_mode,
            promotion_stage=effective_promotion_stage,
        ).to_contract()
        for spec in retained_alternates
    )
    promoted_candidate = ranked_candidates[0].contract_candidate if effective_pass_fail == PassFail.PASS else None
    deep_promotion = PromotedDeepResult(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=effective_pass_fail,
        promotion_stage=effective_promotion_stage,
        ranking_mode=effective_ranking_mode,
        promoted_candidate=promoted_candidate,
        frontend_inputs=effective_deep_inputs,
        parent_anchor_ref=request.parent_anchor_ref,
        tested_k_candidates=(
            tuple(candidate.contract_candidate for candidate in normalized_candidates)
            if effective_promotion_stage == PromotionStage.CAPACITY_REFINEMENT
            else ()
        ),
        k_candidates_tested=effective_k_candidates_tested,
        retained_alternates=alternate_refs,
        canonical_downstream_reference=canonical_downstream_reference,
        exact_blocker=effective_exact_blocker,
        device_agnostic_export=dict(device_agnostic_export or {}),
        ranking_policy=dict(_CANONICAL_DEEP_RANKING_POLICY),
        execution_trace=effective_execution_trace,
        compliance_checks=_deep_promotion_compliance_checks(
            pass_fail=effective_pass_fail,
            promoted_candidate=promoted_candidate,
            retained_alternates=alternate_refs,
            execution_trace=effective_execution_trace,
            promotion_stage=effective_promotion_stage,
            k_candidates_tested=effective_k_candidates_tested,
        ),
    )
    promotion_ref = write_json_artifact(
        Path(request.output_dir) / DEEP_PROMOTION_ARTIFACT_NAME,
        deep_promotion,
    )
    stage_result = StageResult(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=effective_pass_fail,
        exact_blocker=effective_exact_blocker,
        primary_artifact_ref=promotion_ref,
        artifact_refs=(promotion_ref,),
        retained_alternates=alternate_refs,
        execution_trace=effective_execution_trace,
        compliance_checks=_stage_result_compliance_checks(
            pass_fail=effective_pass_fail,
            promoted_candidate=promoted_candidate,
            execution_trace=effective_execution_trace,
        ),
    )
    stage_result_ref = write_json_artifact(
        Path(request.output_dir) / DEEP_STAGE_RESULT_ARTIFACT_NAME,
        stage_result,
    )

    return DeepStageRunResult(
        deep_promotion=deep_promotion,
        deep_promotion_ref=promotion_ref,
        stage_result=stage_result,
        stage_result_ref=stage_result_ref,
        evaluated_candidates=tuple(candidate.contract_candidate for candidate in normalized_candidates),
        ranked_candidate_ids=tuple(candidate.contract_candidate.candidate_id for candidate in ranked_candidates),
    )


def _validate_deep_stage_request(request: StageRequest) -> None:
    if request.stage_key not in _CANONICAL_DEEP_STAGE_KEYS:
        allowed = ", ".join(stage_key.value for stage_key in sorted(_CANONICAL_DEEP_STAGE_KEYS, key=str))
        raise ContractValidationError(
            f"Canonical V3 deep stage only supports `{allowed}`."
        )


def _resolve_candidate_inputs(
    request: StageRequest,
    *,
    candidates: Sequence[DeepStageCandidateInput | Mapping[str, object]] | None,
    evaluate_candidates: DeepStageCandidatesEvaluator | None,
    requested_pass_fail: PassFail | None,
    exact_blocker: str | None,
    deep_training_config: DeepTrainingConfig | None,
) -> tuple[DeepStageCandidateInput | Mapping[str, object], ...]:
    explicit_candidates = tuple(candidates or ())
    if explicit_candidates:
        if evaluate_candidates is not None:
            raise ContractValidationError(
                "Canonical V3 deep stages accept at most one of `candidates` or "
                "`evaluate_candidates`."
            )
        return explicit_candidates
    if evaluate_candidates is not None:
        return tuple(evaluate_candidates(request))
    if request.deep_inputs:
        return materialize_waveform_deep_candidates(
            request,
            deep_inputs=request.deep_inputs,
            deep_training_config=deep_training_config,
        )
    if requested_pass_fail == PassFail.FAIL or exact_blocker is not None:
        return ()
    raise ContractValidationError(
        "Canonical V3 deep stages require explicit `candidates` or valid replay-ready `deep_inputs`."
    )


def _coerce_deep_training_config(
    value: DeepTrainingConfig | Mapping[str, object] | None,
) -> DeepTrainingConfig | None:
    if value is None:
        return None
    if isinstance(value, DeepTrainingConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return DeepTrainingConfig.from_mapping(value)
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc
    raise ContractValidationError(
        "`deep_training_config` must be a `DeepTrainingConfig` or mapping."
    )


def _resolve_execution_trace(
    execution_trace: ExecutionTrace | None,
    *,
    deep_training_config: DeepTrainingConfig | None,
) -> ExecutionTrace | None:
    if deep_training_config is None:
        return execution_trace
    requested_acceleration = _requested_execution_acceleration(deep_training_config.backend)
    if execution_trace is None:
        return ExecutionTrace(
            requested_execution_acceleration=requested_acceleration,
            allow_backend_fallback=deep_training_config.allow_backend_fallback,
        )
    if execution_trace.requested_execution_acceleration != requested_acceleration:
        raise ContractValidationError(
            "`execution_trace.requested_execution_acceleration` must match `deep_training_config.backend`."
        )
    if (
        execution_trace.allow_backend_fallback is not None
        and execution_trace.allow_backend_fallback != deep_training_config.allow_backend_fallback
    ):
        raise ContractValidationError(
            "`execution_trace.allow_backend_fallback` must match `deep_training_config.allow_backend_fallback`."
        )
    if execution_trace.allow_backend_fallback is None:
        return ExecutionTrace(
            requested_execution_acceleration=execution_trace.requested_execution_acceleration,
            resolved_execution_acceleration=execution_trace.resolved_execution_acceleration,
            backend_actual=execution_trace.backend_actual,
            execution_device=execution_trace.execution_device,
            fallback_reason=execution_trace.fallback_reason,
            blocked_reason=execution_trace.blocked_reason,
            allow_backend_fallback=deep_training_config.allow_backend_fallback,
        )
    return execution_trace


def _requested_execution_acceleration(backend: str) -> ExecutionAcceleration:
    if backend == "cpu":
        return ExecutionAcceleration.CPU
    if backend == "gpu":
        return ExecutionAcceleration.GPU
    return ExecutionAcceleration.AUTO


def _coerce_candidate_input(
    candidate: DeepStageCandidateInput | Mapping[str, object],
) -> DeepStageCandidateInput:
    if isinstance(candidate, DeepStageCandidateInput):
        return candidate
    if isinstance(candidate, Mapping):
        return DeepStageCandidateInput.from_runtime_output(candidate)
    raise ContractValidationError(
        "`candidates` entries must be `DeepStageCandidateInput` values or runtime-output mappings."
    )


def _normalize_candidate(
    candidate: DeepStageCandidateInput | Mapping[str, object],
    *,
    request: StageRequest,
    promotion_stage: PromotionStage,
) -> _RankedDeepCandidate:
    stage_input = _coerce_candidate_input(candidate)
    _validate_candidate_lineage(stage_input, request=request)
    _validate_candidate_for_promotion_stage(stage_input, promotion_stage=promotion_stage)
    return _RankedDeepCandidate(
        stage_input=stage_input,
        contract_candidate=stage_input.to_contract_candidate(),
    )


def _validate_candidate_lineage(candidate: DeepStageCandidateInput, *, request: StageRequest) -> None:
    lineage_pairs, lineage_ids, lineage_fingerprints = _request_lineage_index(request)
    if (
        candidate.frontend_input_id is not None
        and lineage_ids
        and candidate.frontend_input_id not in lineage_ids
    ):
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` has `frontend_input_id` not present in the stage request."
        )
    if (
        candidate.frontend_fingerprint is not None
        and lineage_fingerprints
        and candidate.frontend_fingerprint not in lineage_fingerprints
    ):
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` has `frontend_fingerprint` not present in the stage request."
        )
    if (
        candidate.frontend_input_id is not None
        and candidate.frontend_fingerprint is not None
        and lineage_pairs
        and (candidate.frontend_input_id, candidate.frontend_fingerprint) not in lineage_pairs
    ):
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` has frontend lineage that does not match any request `deep_input`."
        )
    if request.parent_anchor_ref is None:
        if candidate.parent_anchor_fingerprint is not None:
            raise ContractValidationError(
                f"Deep candidate `{candidate.candidate_id}` cannot declare a parent anchor when the stage request has none."
            )
        return
    if (
        candidate.parent_anchor_fingerprint is not None
        and candidate.parent_anchor_fingerprint != request.parent_anchor_ref.parent_anchor_fingerprint
    ):
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` has parent-anchor lineage inconsistent with the stage request."
        )


def _validate_candidate_for_promotion_stage(
    candidate: DeepStageCandidateInput,
    *,
    promotion_stage: PromotionStage,
) -> None:
    if promotion_stage == PromotionStage.CAPACITY_REFINEMENT:
        _validate_capacity_refinement_candidate(candidate, promotion_stage=promotion_stage)
        return
    _validate_main_screen_candidate(candidate, promotion_stage=promotion_stage)


def _validate_main_screen_candidate(
    candidate: DeepStageCandidateInput,
    *,
    promotion_stage: PromotionStage,
) -> None:
    config = candidate.effective_engine_deep_config
    config_stage = config.get("promotion_stage")
    if config_stage is not None and config_stage != promotion_stage.value:
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` must carry `promotion_stage={promotion_stage.value}`."
        )
    if candidate.k_medoids_per_class is not None and candidate.k_medoids_per_class != _CANONICAL_MAIN_SCREEN_K:
        raise ContractValidationError(
            "Canonical V3 deep main-screen stage only supports `k_medoids_per_class=1` in this slice."
        )
    if any(k_value != _CANONICAL_MAIN_SCREEN_K for k_value in candidate.k_medoids_search_values):
        raise ContractValidationError(
            "Canonical V3 deep main-screen stage cannot carry multi-k search values in this slice."
        )
    config_k = config.get("k_medoids_per_class")
    if config_k is not None and _require_int(
        config_k,
        field_name=f"{candidate.candidate_id}.effective_engine_deep_config.k_medoids_per_class",
        minimum=1,
    ) != _CANONICAL_MAIN_SCREEN_K:
        raise ContractValidationError(
            "Canonical V3 deep main-screen stage only supports `effective_engine_deep_config.k_medoids_per_class=1`."
        )
    config_search_values = config.get("k_medoids_search_values")
    if config_search_values is not None:
        if not isinstance(config_search_values, Sequence) or isinstance(
            config_search_values,
            (str, bytes, bytearray),
        ):
            raise ContractValidationError(
                f"`{candidate.candidate_id}.effective_engine_deep_config.k_medoids_search_values` must be a sequence."
            )
        if any(
            _require_int(
                value,
                field_name=f"{candidate.candidate_id}.effective_engine_deep_config.k_medoids_search_values[{index}]",
                minimum=1,
            ) != _CANONICAL_MAIN_SCREEN_K
            for index, value in enumerate(config_search_values)
        ):
            raise ContractValidationError(
                "Canonical V3 deep main-screen stage cannot carry multi-k effective-engine search values."
            )


def _validate_capacity_refinement_candidate(
    candidate: DeepStageCandidateInput,
    *,
    promotion_stage: PromotionStage,
) -> None:
    config = candidate.effective_engine_deep_config
    config_stage = config.get("promotion_stage")
    if config_stage is not None and config_stage != promotion_stage.value:
        raise ContractValidationError(
            f"Deep candidate `{candidate.candidate_id}` must carry `promotion_stage={promotion_stage.value}`."
        )
    _require_stage_input_selected_k(
        candidate,
        field_name=f"Deep candidate `{candidate.candidate_id}`",
        required=True,
    )


def _deep_ranking_sort_key(candidate: _RankedDeepCandidate) -> tuple[float, float, float, float, float, int]:
    metrics = candidate.stage_input.ranking_metrics
    return (
        metrics["healthy_to_unhealthy_fpr"],
        -metrics["unhealthy_precision"],
        -metrics["unhealthy_recall"],
        -metrics["unhealthy_f1"],
        -metrics["macro_f1"],
        candidate.contract_candidate.candidate_order,
    )


def _resolve_stage_outcome(
    *,
    requested_pass_fail: PassFail | None,
    exact_blocker: str | None,
    ranked_candidates: Sequence[_RankedDeepCandidate],
) -> tuple[PassFail, str | None]:
    if requested_pass_fail == PassFail.FAIL:
        return PassFail.FAIL, exact_blocker
    if ranked_candidates:
        return PassFail.PASS, exact_blocker
    return (
        PassFail.FAIL,
        exact_blocker
        or "No ranking-eligible Deep candidate remained after canonical Scout alertability screening.",
    )


def _resolve_deep_inputs(
    request: StageRequest,
    *,
    pass_fail: PassFail,
) -> tuple[DeepInputRef, ...]:
    deep_inputs = tuple(request.deep_inputs)
    if pass_fail == PassFail.PASS and not deep_inputs and request.parent_anchor_ref is None:
        raise ContractValidationError(
            "Passing canonical V3 deep stages require at least one `deep_input` or a "
            "`parent_anchor_ref` in the stage request."
        )
    return deep_inputs


def _normalize_k_candidates_tested(
    values: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, ...]:
    normalized_values = tuple(
        _require_int(value, field_name=f"{field_name}[{index}]", minimum=1)
        for index, value in enumerate(values)
    )
    if not normalized_values:
        raise ContractValidationError(f"`{field_name}` must contain at least one `k` candidate.")
    _validate_unique_k_values(normalized_values, field_name=field_name)
    return normalized_values


def _resolve_k_candidates_tested(
    *,
    promotion_stage: PromotionStage,
    explicit_k_candidates_tested: Sequence[int] | None,
    normalized_candidates: Sequence[_RankedDeepCandidate],
) -> tuple[int, ...]:
    if promotion_stage != PromotionStage.CAPACITY_REFINEMENT:
        if explicit_k_candidates_tested is not None:
            raise ContractValidationError(
                "`k_candidates_tested` is only valid for `promotion_stage=capacity_refinement`."
            )
        return ()

    candidate_k_values = tuple(
        _require_stage_input_selected_k(
            candidate.stage_input,
            field_name=f"normalized_candidates[{index}]",
            required=True,
        )
        for index, candidate in enumerate(normalized_candidates)
    )
    normalized_candidate_k_values = tuple(
        value for value in candidate_k_values if value is not None
    )
    if explicit_k_candidates_tested is None:
        return normalized_candidate_k_values

    normalized_explicit_values = _normalize_k_candidates_tested(
        explicit_k_candidates_tested,
        field_name="k_candidates_tested",
    )
    if normalized_explicit_values != normalized_candidate_k_values:
        raise ContractValidationError(
            "`k_candidates_tested` must match the executed per-`k` candidate summaries exactly."
        )
    return normalized_explicit_values


def _request_lineage_index(
    request: StageRequest,
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    pairs: set[tuple[str, str]] = set()
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for deep_input in request.deep_inputs:
        if deep_input.frontend_input_id is not None:
            ids.add(deep_input.frontend_input_id)
        if deep_input.frontend_fingerprint is not None:
            fingerprints.add(deep_input.frontend_fingerprint)
        if deep_input.frontend_input_id is not None and deep_input.frontend_fingerprint is not None:
            pairs.add((deep_input.frontend_input_id, deep_input.frontend_fingerprint))
    return pairs, ids, fingerprints


def _validate_retained_alternate(
    spec: DeepRetainedAlternateSpec,
    *,
    request: StageRequest,
    ranking_mode: DeepRankingMode,
    promotion_stage: PromotionStage,
) -> DeepRetainedAlternateSpec:
    if spec.ranking_mode != ranking_mode:
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` must remain lineage-consistent with `ranking_mode={ranking_mode.value}`."
        )
    if spec.promotion_stage != promotion_stage:
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` must remain lineage-consistent with `promotion_stage={promotion_stage.value}`."
        )

    lineage_pairs, lineage_ids, lineage_fingerprints = _request_lineage_index(request)
    if lineage_ids and spec.frontend_input_id not in lineage_ids:
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` has `frontend_input_id` inconsistent with the stage request lineage."
        )
    if lineage_fingerprints and spec.frontend_fingerprint not in lineage_fingerprints:
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` has `frontend_fingerprint` inconsistent with the stage request lineage."
        )
    if lineage_pairs and (spec.frontend_input_id, spec.frontend_fingerprint) not in lineage_pairs:
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` must be lineage-consistent with one request `deep_input`."
        )
    if request.parent_anchor_ref is None:
        if spec.parent_anchor_fingerprint is not None:
            raise ContractValidationError(
                f"Retained Deep alternate `{spec.candidate_id}` cannot declare a parent anchor outside the stage request lineage."
            )
        return spec
    if (
        spec.parent_anchor_fingerprint is not None
        and spec.parent_anchor_fingerprint != request.parent_anchor_ref.parent_anchor_fingerprint
    ):
        raise ContractValidationError(
            f"Retained Deep alternate `{spec.candidate_id}` has parent-anchor lineage inconsistent with the stage request."
        )
    return spec


def _deep_promotion_compliance_checks(
    *,
    pass_fail: PassFail,
    promoted_candidate: DeepPromotionCandidate | None,
    retained_alternates: Sequence[RetainedAlternateRef],
    execution_trace: ExecutionTrace | None,
    promotion_stage: PromotionStage,
    k_candidates_tested: Sequence[int],
) -> dict[str, bool]:
    return {
        "winner_present_when_promoted": pass_fail == PassFail.PASS and promoted_candidate is not None,
        "winner_ranking_eligible_when_promoted": (
            pass_fail != PassFail.PASS
            or (promoted_candidate is not None and promoted_candidate.ranking_eligible)
        ),
        "winner_scout_alertability_recorded": (
            pass_fail != PassFail.PASS
            or (
                promoted_candidate is not None
                and promoted_candidate.scout_alertability_status is not None
            )
        ),
        "canonical_ranking_mode_enforced": True,
        "canonical_promotion_stage_enforced": True,
        "main_screen_promotion_stage_enforced": promotion_stage == PromotionStage.MAIN_SCREEN,
        "capacity_refinement_promotion_stage_enforced": (
            promotion_stage == PromotionStage.CAPACITY_REFINEMENT
        ),
        "k_candidates_tested_recorded": (
            promotion_stage != PromotionStage.CAPACITY_REFINEMENT or bool(k_candidates_tested)
        ),
        "selected_k_per_class_locked": (
            promotion_stage != PromotionStage.CAPACITY_REFINEMENT
            or (
                pass_fail != PassFail.PASS
                or (
                    promoted_candidate is not None
                    and promoted_candidate.selected_k_per_class is not None
                )
            )
        ),
        "retained_alternates_lineage_consistent": True,
        "execution_trace_present": execution_trace is not None,
        "requested_backend_recorded": execution_trace is not None,
        "actual_backend_recorded": execution_trace is not None and execution_trace.backend_actual is not None,
        "retained_alternates_metadata_only": all(alternate.metadata_only for alternate in retained_alternates),
    }


def _stage_result_compliance_checks(
    *,
    pass_fail: PassFail,
    promoted_candidate: DeepPromotionCandidate | None,
    execution_trace: ExecutionTrace | None,
) -> dict[str, bool]:
    return {
        "winner_present_when_promoted": pass_fail == PassFail.PASS and promoted_candidate is not None,
        "exactly_one_promoted_deep_winner": True,
        "execution_trace_present": execution_trace is not None,
        "requested_backend_recorded": execution_trace is not None,
        "actual_backend_recorded": execution_trace is not None and execution_trace.backend_actual is not None,
    }


__all__ = [
    "CapacityRefinementCandidatesEvaluator",
    "CapacityRefinementCandidateEvaluator",
    "DEEP_PROMOTION_ARTIFACT_NAME",
    "DEEP_STAGE_RESULT_ARTIFACT_NAME",
    "DeepStageCandidatesEvaluator",
    "DeepRetainedAlternateSpec",
    "DeepStageCandidateInput",
    "DeepStageRunResult",
    "deep_stage_candidate_input_from_runtime_output",
    "run_capacity_refinement_stage",
    "run_deep_stage",
]
