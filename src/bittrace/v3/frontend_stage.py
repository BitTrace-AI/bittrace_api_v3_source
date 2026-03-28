"""Canonical V3 frontend-stage artifact emission for lean stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bittrace.core.config import LeanTrainingConfig
from bittrace.v3.artifacts import write_json_artifact
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepInputRef,
    ExecutionAcceleration,
    ExecutionTrace,
    FrontendInput,
    FrontendPromotionCandidate,
    FrontendRankingMode,
    PassFail,
    PromotedFrontendWinner,
    RetainedAlternateKind,
    RetainedAlternateRef,
    StageKey,
    StageRequest,
    StageResult,
)
from bittrace.v3.frontend_materialization import materialize_waveform_frontend


FRONTEND_PROMOTION_ARTIFACT_NAME = "bt3.frontend_promotion.json"
STAGE_RESULT_ARTIFACT_NAME = "bt3.stage_result.json"
_CANONICAL_FRONTEND_STAGE_KEYS = frozenset({StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN})
_RETENTION_MODE_METADATA_KEY = "retained_alternate_mode"
_REPLAY_READY_DEEP_INPUT_METADATA_KEY = "effective_deep_input"
_MISSING_PROMOTION_BLOCKER = "frontend_promotion_not_materialized_from_canonical_waveform_inputs"
_CANONICAL_FRONTEND_METRICS = (
    "healthy_unhealthy_margin",
    "inter_class_separation",
    "intra_class_compactness",
    "bit_balance",
    "bit_stability",
)
_CANONICAL_FRONTEND_RANKING_POLICY = {
    "comparison_type": "guardrailed_lexicographic_descending_maximize_encoder_proxy",
    "encoder_proxy_metrics_used": list(_CANONICAL_FRONTEND_METRICS),
    "metric_order": [
        {
            "metric": "healthy_unhealthy_margin",
            "priority": "primary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "inter_class_separation",
            "priority": "secondary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "bit_balance",
            "priority": "tertiary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "intra_class_compactness",
            "priority": "quaternary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "bit_stability",
            "priority": "quinary",
            "direction": "desc",
            "split": "val",
        },
    ],
    "ranking_mode": FrontendRankingMode.ENCODER_PROXY_QUALITY.value,
    "tie_policy": (
        "Among candidates that tie exactly on the canonical encoder proxy metrics, "
        "retain the earliest candidate in sweep-spec order."
    ),
}


@dataclass(frozen=True, slots=True)
class FrontendRetainedAlternateSpec:
    """Retained frontend alternate emitted as governance metadata in V3."""

    candidate_id: str
    artifact_ref: ArtifactRef
    reason: str
    rank: int | None = None
    candidate_kind: RetainedAlternateKind = RetainedAlternateKind.FRONTEND_CANDIDATE
    retention_mode: str = "metadata_only"
    metadata: Mapping[str, object] = field(default_factory=dict)
    downstream_deep_input: DeepInputRef | None = None

    def __post_init__(self) -> None:
        mode = _normalize_retention_mode(self.retention_mode)
        object.__setattr__(self, "retention_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))

        if mode == "replay_ready" and self.downstream_deep_input is None:
            raise ContractValidationError(
                "Retained alternates with `retention_mode=replay_ready` require a valid effective Deep handoff ref."
            )
        if mode == "metadata_only" and self.downstream_deep_input is not None:
            raise ContractValidationError(
                "Retained alternates with `retention_mode=metadata_only` cannot carry an effective Deep handoff ref."
            )

    def to_contract(self) -> RetainedAlternateRef:
        metadata = dict(self.metadata)
        metadata[_RETENTION_MODE_METADATA_KEY] = self.retention_mode
        if self.downstream_deep_input is not None:
            metadata[_REPLAY_READY_DEEP_INPUT_METADATA_KEY] = self.downstream_deep_input.to_dict()

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
class FrontendStageRunResult:
    """Written frontend-stage artifacts and their resolved references."""

    frontend_promotion: PromotedFrontendWinner
    frontend_promotion_ref: ArtifactRef
    stage_result: StageResult
    stage_result_ref: ArtifactRef


def run_frontend_stage(
    request: StageRequest,
    *,
    promoted_candidate: FrontendPromotionCandidate | None = None,
    downstream_deep_input: DeepInputRef | None = None,
    frontend_input: FrontendInput | None = None,
    pass_fail: PassFail = PassFail.PASS,
    exact_blocker: str | None = None,
    ranking_mode: FrontendRankingMode | str = FrontendRankingMode.ENCODER_PROXY_QUALITY,
    ranking_policy: Mapping[str, object] | None = None,
    execution_trace: ExecutionTrace | None = None,
    lean_training_config: LeanTrainingConfig | Mapping[str, object] | None = None,
    retained_alternates: Sequence[FrontendRetainedAlternateSpec] = (),
    canonical_frontend_path: str = "encoder_only",
    lean_side_prototype_classification: bool = False,
    lean_side_medoid_judging: bool = False,
) -> FrontendStageRunResult:
    """Emit the canonical V3 frontend-promotion and stage-result artifacts."""

    _validate_frontend_stage_request(request)
    _validate_canonical_frontend_behavior(
        canonical_frontend_path=canonical_frontend_path,
        lean_side_prototype_classification=lean_side_prototype_classification,
        lean_side_medoid_judging=lean_side_medoid_judging,
    )

    effective_pass_fail = pass_fail if isinstance(pass_fail, PassFail) else PassFail(pass_fail)
    effective_exact_blocker = exact_blocker
    effective_frontend_input = frontend_input if frontend_input is not None else _resolve_frontend_input(request)
    effective_lean_training_config = _coerce_lean_training_config(lean_training_config)
    if promoted_candidate is None:
        if downstream_deep_input is not None:
            raise ContractValidationError(
                "Canonical V3 frontend stage cannot carry `downstream_deep_input` without `promoted_candidate`."
            )
        if effective_pass_fail == PassFail.PASS and effective_exact_blocker is None:
            if effective_frontend_input is None:
                raise ContractValidationError(
                    "Canonical V3 frontend materialization requires exactly one `frontend_input`."
                )
            materialized_frontend = materialize_waveform_frontend(
                request,
                frontend_input=effective_frontend_input,
                promotion_artifact_path=Path(request.output_dir) / FRONTEND_PROMOTION_ARTIFACT_NAME,
                canonical_frontend_path=canonical_frontend_path,
                lean_training_config=effective_lean_training_config,
            )
            promoted_candidate = materialized_frontend.promoted_candidate
            downstream_deep_input = materialized_frontend.downstream_deep_input
        elif effective_pass_fail == PassFail.PASS:
            effective_pass_fail = PassFail.FAIL
            if effective_exact_blocker is None:
                effective_exact_blocker = _MISSING_PROMOTION_BLOCKER
    effective_execution_trace = _resolve_execution_trace(
        execution_trace if execution_trace is not None else request.execution_trace,
        lean_training_config=effective_lean_training_config,
    )
    alternate_refs = tuple(spec.to_contract() for spec in retained_alternates)
    effective_ranking_policy = dict(ranking_policy) if ranking_policy is not None else dict(_CANONICAL_FRONTEND_RANKING_POLICY)

    frontend_promotion = PromotedFrontendWinner(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=effective_pass_fail,
        frontend_input_ref=effective_frontend_input,
        promoted_candidate=promoted_candidate,
        downstream_deep_input=downstream_deep_input,
        exact_blocker=effective_exact_blocker,
        ranking_mode=ranking_mode,
        ranking_policy=effective_ranking_policy,
        retained_alternates=alternate_refs,
        execution_trace=effective_execution_trace,
        compliance_checks=_frontend_promotion_compliance_checks(
            pass_fail=effective_pass_fail,
            promoted_candidate=promoted_candidate,
            downstream_deep_input=downstream_deep_input,
            execution_trace=effective_execution_trace,
        ),
    )

    promotion_ref = write_json_artifact(
        Path(request.output_dir) / FRONTEND_PROMOTION_ARTIFACT_NAME,
        frontend_promotion,
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
        Path(request.output_dir) / STAGE_RESULT_ARTIFACT_NAME,
        stage_result,
    )

    return FrontendStageRunResult(
        frontend_promotion=frontend_promotion,
        frontend_promotion_ref=promotion_ref,
        stage_result=stage_result,
        stage_result_ref=stage_result_ref,
    )


def _validate_frontend_stage_request(request: StageRequest) -> None:
    if request.stage_key not in _CANONICAL_FRONTEND_STAGE_KEYS:
        allowed = ", ".join(stage_key.value for stage_key in sorted(_CANONICAL_FRONTEND_STAGE_KEYS, key=str))
        raise ContractValidationError(
            f"Canonical V3 frontend stage only supports `{allowed}`."
        )


def _validate_canonical_frontend_behavior(
    *,
    canonical_frontend_path: str,
    lean_side_prototype_classification: bool,
    lean_side_medoid_judging: bool,
) -> None:
    if canonical_frontend_path != "encoder_only":
        raise ContractValidationError("Canonical V3 frontend path must remain encoder-only.")
    if lean_side_prototype_classification:
        raise ContractValidationError(
            "Canonical V3 frontend stage cannot enable Lean-side prototype classification."
        )
    if lean_side_medoid_judging:
        raise ContractValidationError(
            "Canonical V3 frontend stage cannot enable Lean-side medoid judging."
        )


def _resolve_frontend_input(request: StageRequest) -> FrontendInput | None:
    if not request.frontend_inputs:
        return None
    if len(request.frontend_inputs) != 1:
        raise ContractValidationError(
            "Canonical V3 frontend stage expects exactly one `frontend_input`."
        )
    return request.frontend_inputs[0]


def _coerce_lean_training_config(
    value: LeanTrainingConfig | Mapping[str, object] | None,
) -> LeanTrainingConfig | None:
    if value is None:
        return None
    if isinstance(value, LeanTrainingConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return LeanTrainingConfig.from_mapping(value)
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc
    raise ContractValidationError(
        "`lean_training_config` must be a `LeanTrainingConfig` or mapping."
    )


def _resolve_execution_trace(
    execution_trace: ExecutionTrace | None,
    *,
    lean_training_config: LeanTrainingConfig | None,
) -> ExecutionTrace | None:
    if lean_training_config is None:
        return execution_trace
    requested_acceleration = _requested_execution_acceleration(lean_training_config.backend)
    if execution_trace is None:
        return ExecutionTrace(
            requested_execution_acceleration=requested_acceleration,
            allow_backend_fallback=lean_training_config.allow_backend_fallback,
        )
    if execution_trace.requested_execution_acceleration != requested_acceleration:
        raise ContractValidationError(
            "`execution_trace.requested_execution_acceleration` must match `lean_training_config.backend`."
        )
    if (
        execution_trace.allow_backend_fallback is not None
        and execution_trace.allow_backend_fallback != lean_training_config.allow_backend_fallback
    ):
        raise ContractValidationError(
            "`execution_trace.allow_backend_fallback` must match `lean_training_config.allow_backend_fallback`."
        )
    if execution_trace.allow_backend_fallback is None:
        return ExecutionTrace(
            requested_execution_acceleration=execution_trace.requested_execution_acceleration,
            resolved_execution_acceleration=execution_trace.resolved_execution_acceleration,
            backend_actual=execution_trace.backend_actual,
            execution_device=execution_trace.execution_device,
            fallback_reason=execution_trace.fallback_reason,
            blocked_reason=execution_trace.blocked_reason,
            allow_backend_fallback=lean_training_config.allow_backend_fallback,
        )
    return execution_trace


def _requested_execution_acceleration(backend: str) -> ExecutionAcceleration:
    if backend == "cpu":
        return ExecutionAcceleration.CPU
    if backend == "gpu":
        return ExecutionAcceleration.GPU
    return ExecutionAcceleration.AUTO


def _normalize_retention_mode(value: str) -> str:
    if value not in {"metadata_only", "replay_ready"}:
        raise ContractValidationError(
            "Retained alternate `retention_mode` must be `metadata_only` or `replay_ready`."
        )
    return value


def _frontend_promotion_compliance_checks(
    *,
    pass_fail: PassFail,
    promoted_candidate: FrontendPromotionCandidate | None,
    downstream_deep_input: DeepInputRef | None,
    execution_trace: ExecutionTrace | None,
) -> dict[str, bool]:
    return {
        "winner_present_when_promoted": pass_fail == PassFail.PASS and promoted_candidate is not None,
        "downstream_effective_deep_input_present": pass_fail != PassFail.PASS or downstream_deep_input is not None,
        "execution_trace_present": execution_trace is not None,
        "requested_backend_recorded": execution_trace is not None,
        "actual_backend_recorded": execution_trace is not None and execution_trace.backend_actual is not None,
    }


def _stage_result_compliance_checks(
    *,
    pass_fail: PassFail,
    promoted_candidate: FrontendPromotionCandidate | None,
    execution_trace: ExecutionTrace | None,
) -> dict[str, bool]:
    return {
        "winner_present_when_promoted": pass_fail == PassFail.PASS and promoted_candidate is not None,
        "exactly_one_promoted_frontend_winner": True,
        "execution_trace_present": execution_trace is not None,
        "requested_backend_recorded": execution_trace is not None,
        "actual_backend_recorded": execution_trace is not None and execution_trace.backend_actual is not None,
    }


__all__ = [
    "FRONTEND_PROMOTION_ARTIFACT_NAME",
    "FrontendRetainedAlternateSpec",
    "FrontendStageRunResult",
    "STAGE_RESULT_ARTIFACT_NAME",
    "run_frontend_stage",
]
