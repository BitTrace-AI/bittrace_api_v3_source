"""Canonical V3 campaign/orchestrator contracts and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import random
from types import MappingProxyType

from bittrace.v3.artifacts import load_json_artifact_ref, write_json_artifact
from bittrace.v3.contracts import (
    ArtifactRef,
    CandidateProvenanceOrigin,
    CampaignManifest,
    CampaignRequest,
    CampaignResult,
    CampaignStageLineage,
    ContractValidationError,
    DeepInputRef,
    ExecutionTrace,
    FrontendInput,
    FrontendPromotionCandidate,
    ParentAnchorRef,
    PassFail,
    PromotedDeepResult,
    PromotedFrontendWinner,
    PromotionStage,
    SearchCandidateProvenance,
    StageKey,
    StageSearchPolicy,
    StageRequest,
    StageResult,
)


CAMPAIGN_MANIFEST_ARTIFACT_NAME = "bt3.campaign_manifest.json"
CAMPAIGN_RESULT_ARTIFACT_NAME = "bt3.campaign_result.json"
STAGE_REQUEST_ARTIFACT_NAME = "bt3.stage_request.json"
CANONICAL_STAGE_SEQUENCE: tuple[StageKey, ...] = (
    StageKey.LEAN_SMOKE,
    StageKey.DEEP_SMOKE,
    StageKey.LEAN_MAIN_SCREEN,
    StageKey.DEEP_MAIN_SCREEN,
    StageKey.CAPACITY_REFINEMENT,
    StageKey.WINNER_DEEPEN_FREEZE_EXPORT,
)
CANONICAL_STAGE_NAMES = MappingProxyType(
    {
        StageKey.LEAN_SMOKE: "Lean Smoke",
        StageKey.DEEP_SMOKE: "Deep Smoke",
        StageKey.LEAN_MAIN_SCREEN: "Lean Main Screen",
        StageKey.DEEP_MAIN_SCREEN: "Deep Main Screen",
        StageKey.CAPACITY_REFINEMENT: "Capacity Refinement",
        StageKey.WINNER_DEEPEN_FREEZE_EXPORT: "Winner Deepen Freeze Export",
    }
)
SEARCH_POLICY_STAGE_KEYS = frozenset(
    {
        StageKey.LEAN_SMOKE,
        StageKey.DEEP_SMOKE,
        StageKey.LEAN_MAIN_SCREEN,
        StageKey.DEEP_MAIN_SCREEN,
    }
)


@dataclass(frozen=True, slots=True)
class PipelineStageConfig:
    """Minimal canonical stage-request inputs plus wrapped stage-runner kwargs."""

    frontend_inputs: tuple[FrontendInput, ...] = ()
    deep_inputs: tuple[DeepInputRef, ...] = ()
    input_artifacts: tuple[ArtifactRef, ...] = ()
    parent_anchor_ref: ParentAnchorRef | None = None
    promotion_stage: PromotionStage | None = None
    execution_trace: ExecutionTrace | None = None
    notes: tuple[str, ...] = ()
    search_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    search_config_ref: str | None = None
    search_bounds_ref: str | None = None
    runner_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frontend_inputs", tuple(self.frontend_inputs))
        object.__setattr__(self, "deep_inputs", tuple(self.deep_inputs))
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "search_bounds", dict(self.search_bounds))
        object.__setattr__(self, "runner_kwargs", dict(self.runner_kwargs))


@dataclass(frozen=True, slots=True)
class CampaignStageExecution:
    """Resolved request/result record for one executed canonical stage."""

    stage_key: StageKey
    stage_request: StageRequest
    stage_request_ref: ArtifactRef
    stage_result: StageResult
    stage_result_ref: ArtifactRef
    emitted_artifact_refs: tuple[ArtifactRef, ...]
    promoted_winner_refs: tuple[ArtifactRef, ...]
    lineage: CampaignStageLineage
    runner_output: object


@dataclass(frozen=True, slots=True)
class CampaignPipelineRunResult:
    """Canonical campaign execution outputs and final campaign-result artifact."""

    campaign_request: CampaignRequest
    campaign_manifest: CampaignManifest | None
    campaign_manifest_ref: ArtifactRef | None
    completed_stage_executions: tuple[CampaignStageExecution, ...]
    failed_stage_execution: CampaignStageExecution | None
    campaign_result: CampaignResult
    campaign_result_ref: ArtifactRef


def _coerce_stage_key(value: StageKey | str, *, field_name: str) -> StageKey:
    if isinstance(value, StageKey):
        return value
    if not isinstance(value, str):
        raise ContractValidationError(f"`{field_name}` must be a `StageKey` value.")
    try:
        return StageKey(value)
    except ValueError as exc:
        allowed = ", ".join(stage_key.value for stage_key in StageKey)
        raise ContractValidationError(f"`{field_name}` must be one of: {allowed}.") from exc


def _artifact_ref_set(refs: Sequence[ArtifactRef]) -> set[ArtifactRef]:
    return set(refs)


def _dedupe_artifact_refs(*groups: Sequence[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    ordered_refs: list[ArtifactRef] = []
    seen: set[ArtifactRef] = set()
    for group in groups:
        for ref in group:
            if ref in seen:
                continue
            ordered_refs.append(ref)
            seen.add(ref)
    return tuple(ordered_refs)


def _coerce_stage_configs(
    stage_configs: Mapping[StageKey | str, PipelineStageConfig],
) -> dict[StageKey, PipelineStageConfig]:
    return {
        _coerce_stage_key(stage_key, field_name="stage_configs.<key>"): stage_config
        for stage_key, stage_config in stage_configs.items()
    }


def _validate_stage_configs(
    stage_sequence: Sequence[StageKey],
    *,
    stage_configs: Mapping[StageKey, PipelineStageConfig],
) -> None:
    expected = tuple(stage_sequence)
    missing = [stage_key.value for stage_key in expected if stage_key not in stage_configs]
    unexpected = [stage_key.value for stage_key in stage_configs if stage_key not in expected]
    if missing:
        raise ContractValidationError(
            "`stage_configs` must provide an explicit config for each requested canonical stage: "
            + ", ".join(missing)
            + "."
        )
    if unexpected:
        raise ContractValidationError(
            "`stage_configs` contains stages outside the requested canonical sequence: "
            + ", ".join(unexpected)
            + "."
        )


def resolve_stage_search_policy(
    campaign_request: CampaignRequest,
    *,
    stage_key: StageKey,
) -> StageSearchPolicy | None:
    stage_policy = campaign_request.stage_search_policies.get(stage_key)
    if stage_policy is not None:
        return stage_policy
    if campaign_request.search_policy is None:
        return None
    if stage_key not in SEARCH_POLICY_STAGE_KEYS:
        return None
    return campaign_request.search_policy


def _normalized_stage_search_seed(
    *,
    campaign_seed: int,
    stage_key: StageKey,
    random_seed: int,
) -> int:
    digest = hashlib.sha256(
        f"{campaign_seed}:{stage_key.value}:{random_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _normalize_search_bound(
    value: object,
    *,
    field_name: str,
) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a 2-item numeric range.")
    items = tuple(value)
    if len(items) != 2:
        raise ContractValidationError(f"`{field_name}` must contain exactly two bound values.")
    lower, upper = items
    if isinstance(lower, bool) or not isinstance(lower, (int, float)):
        raise ContractValidationError(f"`{field_name}[0]` must be numeric.")
    if isinstance(upper, bool) or not isinstance(upper, (int, float)):
        raise ContractValidationError(f"`{field_name}[1]` must be numeric.")
    normalized = (float(lower), float(upper))
    if normalized[0] > normalized[1]:
        raise ContractValidationError(f"`{field_name}` must be ordered as [min, max].")
    return normalized


def sample_bounded_search_values(
    *,
    campaign_seed: int,
    stage_key: StageKey,
    search_policy: StageSearchPolicy,
    search_bounds: Mapping[str, object],
) -> dict[str, float | int]:
    if not search_policy.random_exploration_enabled:
        return {}
    if stage_key not in SEARCH_POLICY_STAGE_KEYS:
        raise ContractValidationError(
            "Bounded random search is only valid for canonical search stages."
        )
    if not search_bounds:
        raise ContractValidationError(
            "Bounded random search requires explicit per-stage search bounds."
        )
    rng = random.Random(
        _normalized_stage_search_seed(
            campaign_seed=campaign_seed,
            stage_key=stage_key,
            random_seed=search_policy.random_seed or 0,
        )
    )
    sampled_values: dict[str, float | int] = {}
    for field_name in sorted(search_bounds):
        lower, upper = _normalize_search_bound(
            search_bounds[field_name],
            field_name=f"search_bounds.{field_name}",
        )
        if lower.is_integer() and upper.is_integer():
            sampled_values[field_name] = rng.randint(int(lower), int(upper))
        else:
            sampled_values[field_name] = round(rng.uniform(lower, upper), 12)
    return sampled_values


def _apply_frontend_candidate_provenance(
    candidate: FrontendPromotionCandidate,
    *,
    stage_key: StageKey,
    search_policy: StageSearchPolicy | None,
    campaign_seed: int,
    stage_config: PipelineStageConfig,
) -> FrontendPromotionCandidate:
    provenance_origin = CandidateProvenanceOrigin.LOCAL_REFINE
    sampled_values: Mapping[str, object] = {}
    random_seed: int | None = None
    if search_policy is not None and search_policy.random_exploration_enabled:
        provenance_origin = CandidateProvenanceOrigin.BOUNDED_RANDOM
        sampled_values = sample_bounded_search_values(
            campaign_seed=campaign_seed,
            stage_key=stage_key,
            search_policy=search_policy,
            search_bounds=stage_config.search_bounds,
        )
        random_seed = _normalized_stage_search_seed(
            campaign_seed=campaign_seed,
            stage_key=stage_key,
            random_seed=search_policy.random_seed or 0,
        )
    return replace(
        candidate,
        provenance=SearchCandidateProvenance(
            origin=provenance_origin,
            stage_key=stage_key,
            random_seed=random_seed,
            sampled_values=sampled_values,
            bounds_ref=(
                search_policy.bounds_ref
                if search_policy is not None
                else stage_config.search_bounds_ref
            ),
            config_ref=(
                search_policy.config_ref
                if search_policy is not None
                else stage_config.search_config_ref
            ),
        ),
    )


def _apply_search_policy_to_runner_kwargs(
    campaign_request: CampaignRequest,
    *,
    stage_key: StageKey,
    stage_config: PipelineStageConfig,
) -> dict[str, object]:
    search_policy = resolve_stage_search_policy(campaign_request, stage_key=stage_key)
    if search_policy is not None and stage_key not in SEARCH_POLICY_STAGE_KEYS:
        raise ContractValidationError(
            "Random-search orchestration is disabled for non-search canonical stages."
        )
    effective_runner_kwargs = dict(stage_config.runner_kwargs)
    if stage_key in (StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN):
        candidate = effective_runner_kwargs.get("promoted_candidate")
        if isinstance(candidate, FrontendPromotionCandidate):
            effective_runner_kwargs["promoted_candidate"] = _apply_frontend_candidate_provenance(
                candidate,
                stage_key=stage_key,
                search_policy=search_policy,
                campaign_seed=campaign_request.campaign_seed,
                stage_config=stage_config,
            )
    return effective_runner_kwargs


def _validate_stage_lineage_records(
    stages: Sequence[CampaignStageLineage],
    *,
    field_name: str,
    require_stage_request_ref: bool,
    require_stage_result_ref: bool,
    allow_stage_result_ref: bool,
    allow_emitted_artifact_refs: bool,
    allow_promoted_winner_refs: bool,
) -> None:
    available_stage_artifacts: set[ArtifactRef] = set()
    available_promoted_winners: set[ArtifactRef] = set()

    for index, stage in enumerate(stages):
        stage_field = f"{field_name}[{index}]"
        if require_stage_request_ref and stage.stage_request_ref is None:
            raise ContractValidationError(
                f"`{stage_field}.stage_request_ref` is required for canonical campaign lineage."
            )
        if require_stage_result_ref and stage.stage_result_ref is None:
            raise ContractValidationError(
                f"`{stage_field}.stage_result_ref` is required for executed campaign stages."
            )
        if not allow_stage_result_ref and stage.stage_result_ref is not None:
            raise ContractValidationError(
                f"`{stage_field}.stage_result_ref` is not valid in the campaign manifest."
            )
        if not allow_emitted_artifact_refs and stage.emitted_artifact_refs:
            raise ContractValidationError(
                f"`{stage_field}.emitted_artifact_refs` is not valid in the campaign manifest."
            )
        if not allow_promoted_winner_refs and stage.promoted_winner_refs:
            raise ContractValidationError(
                f"`{stage_field}.promoted_winner_refs` is not valid in the campaign manifest."
            )
        missing_stage_inputs = [
            ref
            for ref in stage.input_stage_artifact_refs
            if ref not in available_stage_artifacts
        ]
        if missing_stage_inputs:
            raise ContractValidationError(
                f"`{stage_field}.input_stage_artifact_refs` must reference earlier stage artifacts."
            )
        missing_winner_inputs = [
            ref
            for ref in stage.input_promoted_winner_refs
            if ref not in available_promoted_winners
        ]
        if missing_winner_inputs:
            raise ContractValidationError(
                f"`{stage_field}.input_promoted_winner_refs` must reference earlier promoted winners."
            )
        if any(ref not in stage.emitted_artifact_refs for ref in stage.promoted_winner_refs):
            raise ContractValidationError(
                f"`{stage_field}.promoted_winner_refs` must also appear in `emitted_artifact_refs`."
            )

        if stage.stage_request_ref is not None:
            available_stage_artifacts.add(stage.stage_request_ref)
        if stage.stage_result_ref is not None:
            available_stage_artifacts.add(stage.stage_result_ref)
        available_stage_artifacts.update(stage.emitted_artifact_refs)
        available_promoted_winners.update(stage.promoted_winner_refs)


def canonical_stage_sequence() -> tuple[StageKey, ...]:
    """Return the canonical V3 campaign stage sequence."""

    return CANONICAL_STAGE_SEQUENCE


def canonical_stage_name(stage_key: StageKey | str) -> str:
    """Return the canonical human-readable name for a V3 campaign stage."""

    resolved_stage_key = _coerce_stage_key(stage_key, field_name="stage_key")
    try:
        return CANONICAL_STAGE_NAMES[resolved_stage_key]
    except KeyError as exc:
        raise ContractValidationError(
            f"`stage_key` must be one of the canonical campaign stages, got `{resolved_stage_key.value}`."
        ) from exc


def validate_canonical_stage_order(
    stage_keys: Sequence[StageKey | str],
    *,
    field_name: str = "stage_keys",
    allow_empty: bool = False,
) -> tuple[StageKey, ...]:
    """Validate that stage keys form a contiguous prefix of the canonical V3 sequence."""

    resolved_stage_keys = tuple(
        _coerce_stage_key(stage_key, field_name=f"{field_name}[{index}]")
        for index, stage_key in enumerate(stage_keys)
    )
    if not resolved_stage_keys:
        if allow_empty:
            return ()
        raise ContractValidationError(f"`{field_name}` must not be empty.")
    if len(set(resolved_stage_keys)) != len(resolved_stage_keys):
        raise ContractValidationError(f"`{field_name}` must not contain duplicate stages.")
    expected_prefix = CANONICAL_STAGE_SEQUENCE[: len(resolved_stage_keys)]
    if resolved_stage_keys != expected_prefix:
        canonical_order = ", ".join(stage_key.value for stage_key in CANONICAL_STAGE_SEQUENCE)
        actual_order = ", ".join(stage_key.value for stage_key in resolved_stage_keys)
        raise ContractValidationError(
            f"`{field_name}` must be a contiguous prefix of the canonical V3 stage order "
            f"`{canonical_order}`; got `{actual_order}`."
        )
    return resolved_stage_keys


def _default_promotion_stage(stage_key: StageKey) -> PromotionStage | None:
    if stage_key in (StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN):
        return PromotionStage.MAIN_SCREEN
    if stage_key in (StageKey.CAPACITY_REFINEMENT, StageKey.WINNER_DEEPEN_FREEZE_EXPORT):
        return PromotionStage.CAPACITY_REFINEMENT
    return None


def _stage_output_dir(
    campaign_request: CampaignRequest,
    *,
    stage_index: int,
    stage_key: StageKey,
) -> Path:
    return Path(campaign_request.output_dir) / f"{stage_index:02d}_{stage_key.value}"


def _load_stage_request_from_manifest(
    manifest: CampaignManifest,
    *,
    stage_index: int,
    stage_key: StageKey,
) -> tuple[StageRequest, ArtifactRef]:
    try:
        stage = manifest.stages[stage_index]
    except IndexError as exc:
        raise ContractValidationError(
            "Campaign manifest is missing canonical stage lineage for the requested execution index."
        ) from exc
    if stage.stage_key != stage_key:
        raise ContractValidationError(
            "Campaign manifest stage ordering must match the canonical requested execution sequence."
        )
    stage_request_ref = stage.stage_request_ref
    if stage_request_ref is None:
        raise ContractValidationError(
            "Campaign manifest execution requires each stage to carry an emitted `stage_request_ref`."
        )
    request = load_json_artifact_ref(stage_request_ref)
    if not isinstance(request, StageRequest):
        raise ContractValidationError(
            "Campaign manifest `stage_request_ref` must resolve to a canonical V3 `StageRequest` artifact."
        )
    if request.stage_key != stage_key:
        raise ContractValidationError(
            "Loaded manifest stage-request artifact does not match the canonical stage key."
        )
    return request, stage_request_ref


def _load_previous_winner_artifact(
    previous_stage_execution: CampaignStageExecution | None,
) -> PromotedFrontendWinner | PromotedDeepResult | None:
    if previous_stage_execution is None or not previous_stage_execution.promoted_winner_refs:
        return None
    artifact = load_json_artifact_ref(previous_stage_execution.promoted_winner_refs[-1])
    if isinstance(artifact, (PromotedFrontendWinner, PromotedDeepResult)):
        return artifact
    raise ContractValidationError(
        "Canonical stage boundary winner lineage must resolve to a promoted frontend/deep artifact."
    )


def _derive_deep_inputs(
    *,
    stage_key: StageKey,
    stage_config: PipelineStageConfig,
    previous_stage_execution: CampaignStageExecution | None,
) -> tuple[DeepInputRef, ...]:
    if stage_config.deep_inputs:
        return tuple(stage_config.deep_inputs)
    if stage_key not in (
        StageKey.DEEP_SMOKE,
        StageKey.DEEP_MAIN_SCREEN,
        StageKey.CAPACITY_REFINEMENT,
        StageKey.WINNER_DEEPEN_FREEZE_EXPORT,
    ):
        return ()
    previous_winner = _load_previous_winner_artifact(previous_stage_execution)
    if previous_winner is None:
        return ()
    if isinstance(previous_winner, PromotedFrontendWinner):
        return (
            (previous_winner.downstream_deep_input,)
            if previous_winner.downstream_deep_input is not None
            else ()
        )
    return tuple(previous_winner.frontend_inputs)


def _build_stage_request(
    campaign_request: CampaignRequest,
    *,
    stage_index: int,
    stage_key: StageKey,
    stage_config: PipelineStageConfig,
    previous_stage_execution: CampaignStageExecution | None,
) -> tuple[StageRequest, ArtifactRef]:
    input_stage_refs = (
        (previous_stage_execution.stage_result_ref,)
        if previous_stage_execution is not None
        else ()
    )
    input_promoted_winner_refs = (
        previous_stage_execution.promoted_winner_refs
        if previous_stage_execution is not None
        else ()
    )
    request = StageRequest(
        stage_key=stage_key,
        stage_name=canonical_stage_name(stage_key),
        campaign_id=campaign_request.campaign_id,
        campaign_seed=campaign_request.campaign_seed,
        output_dir=str(
            _stage_output_dir(
                campaign_request,
                stage_index=stage_index,
                stage_key=stage_key,
            ).resolve()
        ),
        input_artifacts=_dedupe_artifact_refs(
            input_stage_refs,
            input_promoted_winner_refs,
            stage_config.input_artifacts,
        ),
        frontend_inputs=tuple(stage_config.frontend_inputs),
        deep_inputs=_derive_deep_inputs(
            stage_key=stage_key,
            stage_config=stage_config,
            previous_stage_execution=previous_stage_execution,
        ),
        parent_anchor_ref=stage_config.parent_anchor_ref,
        promotion_stage=(
            stage_config.promotion_stage
            if stage_config.promotion_stage is not None
            else _default_promotion_stage(stage_key)
        ),
        execution_trace=stage_config.execution_trace,
        notes=tuple(stage_config.notes),
    )
    request_ref = write_json_artifact(
        Path(request.output_dir) / STAGE_REQUEST_ARTIFACT_NAME,
        request,
    )
    return request, request_ref


def _run_stage(
    request: StageRequest,
    *,
    campaign_request: CampaignRequest,
    stage_key: StageKey,
    stage_config: PipelineStageConfig,
    previous_stage_execution: CampaignStageExecution | None,
) -> object:
    effective_runner_kwargs = _apply_search_policy_to_runner_kwargs(
        campaign_request,
        stage_key=stage_key,
        stage_config=stage_config,
    )

    if stage_key in (StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN):
        from bittrace.v3.frontend_stage import run_frontend_stage

        return run_frontend_stage(request, **effective_runner_kwargs)
    if stage_key in (StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN):
        from bittrace.v3.deep_stage import run_deep_stage

        return run_deep_stage(request, **effective_runner_kwargs)
    if stage_key == StageKey.CAPACITY_REFINEMENT:
        from bittrace.v3.deep_stage import run_capacity_refinement_stage

        return run_capacity_refinement_stage(request, **effective_runner_kwargs)
    if stage_key == StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
        if previous_stage_execution is None or not previous_stage_execution.promoted_winner_refs:
            raise ContractValidationError(
                "Canonical S6 freeze/export execution requires the promoted S5 Deep winner lineage."
            )
        resolved_source_ref = previous_stage_execution.promoted_winner_refs[-1]
        configured_source_ref = effective_runner_kwargs.get("source_promoted_deep_result_ref")
        if configured_source_ref is None:
            effective_runner_kwargs["source_promoted_deep_result_ref"] = resolved_source_ref
        elif configured_source_ref != resolved_source_ref:
            raise ContractValidationError(
                "Canonical S6 execution must use the immediately preceding promoted S5 Deep winner ref."
            )

        from bittrace.v3.freeze_export import run_winner_deepen_freeze_export

        return run_winner_deepen_freeze_export(request, **effective_runner_kwargs)
    raise ContractValidationError(
        f"Unsupported canonical stage runner for `{stage_key.value}`."
    )


def _stage_emitted_artifact_refs(stage_key: StageKey, runner_output: object) -> tuple[ArtifactRef, ...]:
    if stage_key in (StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN):
        return (runner_output.frontend_promotion_ref,)
    if stage_key in (StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN, StageKey.CAPACITY_REFINEMENT):
        return (runner_output.deep_promotion_ref,)
    if stage_key == StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
        return (
            runner_output.freeze_export_manifest_ref,
            runner_output.deep_anchor_artifact_ref,
            runner_output.frontend_export_reference_ref,
        )
    raise ContractValidationError(
        f"Unsupported canonical stage output extraction for `{stage_key.value}`."
    )


def _stage_promoted_winner_refs(stage_key: StageKey, runner_output: object) -> tuple[ArtifactRef, ...]:
    if stage_key in (StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN):
        return (runner_output.frontend_promotion_ref,)
    if stage_key in (StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN, StageKey.CAPACITY_REFINEMENT):
        return (runner_output.deep_promotion_ref,)
    if stage_key == StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
        return ()
    raise ContractValidationError(
        f"Unsupported canonical stage winner extraction for `{stage_key.value}`."
    )


def _build_stage_execution(
    *,
    stage_key: StageKey,
    stage_request: StageRequest,
    stage_request_ref: ArtifactRef,
    runner_output: object,
    previous_stage_execution: CampaignStageExecution | None,
) -> CampaignStageExecution:
    stage_result = runner_output.stage_result
    stage_result_ref = runner_output.stage_result_ref
    if not isinstance(stage_result, StageResult):
        raise ContractValidationError(
            "Canonical stage execution must resolve to a V3 `StageResult` artifact payload."
        )
    if stage_result.stage_key != stage_key:
        raise ContractValidationError(
            "Canonical stage execution returned a `StageResult` with the wrong `stage_key`."
        )
    emitted_artifact_refs = _stage_emitted_artifact_refs(stage_key, runner_output)
    promoted_winner_refs = _stage_promoted_winner_refs(stage_key, runner_output)
    lineage = CampaignStageLineage(
        stage_key=stage_key,
        stage_request_ref=stage_request_ref,
        stage_result_ref=stage_result_ref,
        input_stage_artifact_refs=(
            (previous_stage_execution.stage_result_ref,)
            if previous_stage_execution is not None
            else ()
        ),
        input_promoted_winner_refs=(
            previous_stage_execution.promoted_winner_refs
            if previous_stage_execution is not None
            else ()
        ),
        emitted_artifact_refs=emitted_artifact_refs,
        promoted_winner_refs=promoted_winner_refs,
    )
    return CampaignStageExecution(
        stage_key=stage_key,
        stage_request=stage_request,
        stage_request_ref=stage_request_ref,
        stage_result=stage_result,
        stage_result_ref=stage_result_ref,
        emitted_artifact_refs=emitted_artifact_refs,
        promoted_winner_refs=promoted_winner_refs,
        lineage=lineage,
        runner_output=runner_output,
    )


def _final_promoted_winner_refs(
    completed_stage_executions: Sequence[CampaignStageExecution],
) -> tuple[ArtifactRef, ...]:
    latest_by_kind: dict[str, ArtifactRef] = {}
    ordered_kinds: list[str] = []
    for stage_execution in completed_stage_executions:
        for ref in stage_execution.promoted_winner_refs:
            if ref.kind not in latest_by_kind:
                ordered_kinds.append(ref.kind)
            latest_by_kind[ref.kind] = ref
    return tuple(latest_by_kind[kind] for kind in ordered_kinds)


def _final_freeze_export_refs(
    completed_stage_executions: Sequence[CampaignStageExecution],
) -> tuple[ArtifactRef, ...]:
    for stage_execution in reversed(tuple(completed_stage_executions)):
        if stage_execution.stage_key == StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            return stage_execution.emitted_artifact_refs
    return ()


def validate_campaign_manifest_stages(
    campaign_request: CampaignRequest,
    stages: Sequence[CampaignStageLineage],
    *,
    field_name: str = "stages",
) -> None:
    """Validate canonical stage planning for a campaign manifest."""

    copied_stages = tuple(stages)
    stage_keys = validate_canonical_stage_order(
        [stage.stage_key for stage in copied_stages],
        field_name=field_name,
    )
    if stage_keys != campaign_request.stage_sequence:
        raise ContractValidationError(
            f"`{field_name}` must match `CampaignRequest.stage_sequence` exactly."
        )
    _validate_stage_lineage_records(
        copied_stages,
        field_name=field_name,
        require_stage_request_ref=True,
        require_stage_result_ref=False,
        allow_stage_result_ref=False,
        allow_emitted_artifact_refs=False,
        allow_promoted_winner_refs=False,
    )


def validate_campaign_result(
    *,
    campaign_request: CampaignRequest,
    completed_stages: Sequence[CampaignStageLineage],
    failed_stage: CampaignStageLineage | None,
    final_promoted_winner_refs: Sequence[ArtifactRef],
    freeze_export_refs: Sequence[ArtifactRef],
    verification_refs: Sequence[ArtifactRef],
    field_name: str = "CampaignResult",
) -> None:
    """Validate canonical stage completion, failure, and final lineage summary."""

    copied_completed_stages = tuple(completed_stages)
    completed_stage_keys = validate_canonical_stage_order(
        [stage.stage_key for stage in copied_completed_stages],
        field_name=f"{field_name}.completed_stages",
        allow_empty=True,
    )
    expected_completed_stage_keys = campaign_request.stage_sequence[: len(copied_completed_stages)]
    if completed_stage_keys != expected_completed_stage_keys:
        raise ContractValidationError(
            f"`{field_name}.completed_stages` must match the leading executed prefix of "
            "`CampaignRequest.stage_sequence`."
        )

    executed_stages = list(copied_completed_stages)
    if failed_stage is not None:
        if len(copied_completed_stages) >= len(campaign_request.stage_sequence):
            raise ContractValidationError(
                f"`{field_name}.failed_stage` is invalid once all requested stages are completed."
            )
        expected_failed_stage_key = campaign_request.stage_sequence[len(copied_completed_stages)]
        if failed_stage.stage_key != expected_failed_stage_key:
            raise ContractValidationError(
                f"`{field_name}.failed_stage.stage_key` must be the next canonical stage after "
                "`completed_stages`."
            )
        executed_stages.append(failed_stage)

    _validate_stage_lineage_records(
        executed_stages,
        field_name=f"{field_name}.executed_stages",
        require_stage_request_ref=True,
        require_stage_result_ref=True,
        allow_stage_result_ref=True,
        allow_emitted_artifact_refs=True,
        allow_promoted_winner_refs=True,
    )

    completed_promoted_winner_refs = _artifact_ref_set(
        ref
        for stage in copied_completed_stages
        for ref in stage.promoted_winner_refs
    )
    completed_emitted_artifact_refs = _artifact_ref_set(
        ref
        for stage in copied_completed_stages
        for ref in stage.emitted_artifact_refs
    )
    if any(ref not in completed_promoted_winner_refs for ref in final_promoted_winner_refs):
        raise ContractValidationError(
            f"`{field_name}.final_promoted_winner_refs` must resolve to promoted winners from completed stages."
        )
    if any(ref not in completed_emitted_artifact_refs for ref in freeze_export_refs):
        raise ContractValidationError(
            f"`{field_name}.freeze_export_refs` must resolve to emitted artifacts from completed stages."
        )
    if len(set(verification_refs)) != len(tuple(verification_refs)):
        raise ContractValidationError(
            f"`{field_name}.verification_refs` must not contain duplicate artifact refs."
        )


def build_campaign_manifest(
    campaign_request: CampaignRequest,
    *,
    stages: Sequence[CampaignStageLineage],
    notes: Sequence[str] = (),
) -> CampaignManifest:
    """Build a validated canonical campaign manifest artifact."""

    return CampaignManifest(
        campaign_request=campaign_request,
        stages=tuple(stages),
        notes=tuple(notes),
    )


def emit_campaign_manifest(
    campaign_request: CampaignRequest,
    *,
    stages: Sequence[CampaignStageLineage],
    notes: Sequence[str] = (),
    output_path: str | Path | None = None,
) -> tuple[CampaignManifest, ArtifactRef]:
    """Write the canonical campaign manifest artifact."""

    manifest = build_campaign_manifest(
        campaign_request,
        stages=stages,
        notes=notes,
    )
    manifest_ref = write_json_artifact(
        output_path or Path(campaign_request.output_dir) / CAMPAIGN_MANIFEST_ARTIFACT_NAME,
        manifest,
    )
    return manifest, manifest_ref


def build_campaign_result(
    campaign_request: CampaignRequest,
    *,
    campaign_manifest_ref: ArtifactRef | None = None,
    completed_stages: Sequence[CampaignStageLineage] = (),
    failed_stage: CampaignStageLineage | None = None,
    final_promoted_winner_refs: Sequence[ArtifactRef] = (),
    freeze_export_refs: Sequence[ArtifactRef] = (),
    verification_refs: Sequence[ArtifactRef] = (),
    notes: Sequence[str] = (),
) -> CampaignResult:
    """Build a validated canonical campaign result artifact."""

    return CampaignResult(
        campaign_request=campaign_request,
        campaign_manifest_ref=campaign_manifest_ref,
        completed_stages=tuple(completed_stages),
        failed_stage=failed_stage,
        final_promoted_winner_refs=tuple(final_promoted_winner_refs),
        freeze_export_refs=tuple(freeze_export_refs),
        verification_refs=tuple(verification_refs),
        notes=tuple(notes),
    )


def emit_campaign_result(
    campaign_request: CampaignRequest,
    *,
    campaign_manifest_ref: ArtifactRef | None = None,
    completed_stages: Sequence[CampaignStageLineage] = (),
    failed_stage: CampaignStageLineage | None = None,
    final_promoted_winner_refs: Sequence[ArtifactRef] = (),
    freeze_export_refs: Sequence[ArtifactRef] = (),
    verification_refs: Sequence[ArtifactRef] = (),
    notes: Sequence[str] = (),
    output_path: str | Path | None = None,
) -> tuple[CampaignResult, ArtifactRef]:
    """Write the canonical campaign result artifact."""

    result = build_campaign_result(
        campaign_request,
        campaign_manifest_ref=campaign_manifest_ref,
        completed_stages=completed_stages,
        failed_stage=failed_stage,
        final_promoted_winner_refs=final_promoted_winner_refs,
        freeze_export_refs=freeze_export_refs,
        verification_refs=verification_refs,
        notes=notes,
    )
    result_ref = write_json_artifact(
        output_path or Path(campaign_request.output_dir) / CAMPAIGN_RESULT_ARTIFACT_NAME,
        result,
    )
    return result, result_ref


def run_canonical_campaign(
    campaign: CampaignRequest | CampaignManifest,
    *,
    stage_configs: Mapping[StageKey | str, PipelineStageConfig],
    campaign_manifest_ref: ArtifactRef | None = None,
    output_path: str | Path | None = None,
) -> CampaignPipelineRunResult:
    """Execute the requested canonical V3 stage sequence and emit a campaign result."""

    manifest = campaign if isinstance(campaign, CampaignManifest) else None
    campaign_request = campaign.campaign_request if isinstance(campaign, CampaignManifest) else campaign
    resolved_stage_configs = _coerce_stage_configs(stage_configs)
    _validate_stage_configs(
        campaign_request.stage_sequence,
        stage_configs=resolved_stage_configs,
    )

    completed_stage_executions: list[CampaignStageExecution] = []
    failed_stage_execution: CampaignStageExecution | None = None
    previous_stage_execution: CampaignStageExecution | None = None

    for stage_index, stage_key in enumerate(campaign_request.stage_sequence, start=1):
        if manifest is not None:
            stage_request, stage_request_ref = _load_stage_request_from_manifest(
                manifest,
                stage_index=stage_index - 1,
                stage_key=stage_key,
            )
        else:
            stage_request, stage_request_ref = _build_stage_request(
                campaign_request,
                stage_index=stage_index,
                stage_key=stage_key,
                stage_config=resolved_stage_configs[stage_key],
                previous_stage_execution=previous_stage_execution,
            )

        runner_output = _run_stage(
            stage_request,
            campaign_request=campaign_request,
            stage_key=stage_key,
            stage_config=resolved_stage_configs[stage_key],
            previous_stage_execution=previous_stage_execution,
        )
        stage_execution = _build_stage_execution(
            stage_key=stage_key,
            stage_request=stage_request,
            stage_request_ref=stage_request_ref,
            runner_output=runner_output,
            previous_stage_execution=previous_stage_execution,
        )
        if stage_execution.stage_result.pass_fail == PassFail.FAIL:
            failed_stage_execution = stage_execution
            break
        completed_stage_executions.append(stage_execution)
        previous_stage_execution = stage_execution

    campaign_result, campaign_result_ref = emit_campaign_result(
        campaign_request,
        campaign_manifest_ref=campaign_manifest_ref,
        completed_stages=tuple(
            stage_execution.lineage for stage_execution in completed_stage_executions
        ),
        failed_stage=(
            failed_stage_execution.lineage if failed_stage_execution is not None else None
        ),
        final_promoted_winner_refs=_final_promoted_winner_refs(completed_stage_executions),
        freeze_export_refs=_final_freeze_export_refs(completed_stage_executions),
        verification_refs=(),
        output_path=output_path,
    )
    return CampaignPipelineRunResult(
        campaign_request=campaign_request,
        campaign_manifest=manifest,
        campaign_manifest_ref=campaign_manifest_ref,
        completed_stage_executions=tuple(completed_stage_executions),
        failed_stage_execution=failed_stage_execution,
        campaign_result=campaign_result,
        campaign_result_ref=campaign_result_ref,
    )


__all__ = [
    "CAMPAIGN_MANIFEST_ARTIFACT_NAME",
    "CAMPAIGN_RESULT_ARTIFACT_NAME",
    "CANONICAL_STAGE_NAMES",
    "CANONICAL_STAGE_SEQUENCE",
    "CampaignPipelineRunResult",
    "CampaignStageExecution",
    "PipelineStageConfig",
    "STAGE_REQUEST_ARTIFACT_NAME",
    "build_campaign_manifest",
    "build_campaign_result",
    "canonical_stage_name",
    "canonical_stage_sequence",
    "emit_campaign_manifest",
    "emit_campaign_result",
    "run_canonical_campaign",
    "resolve_stage_search_policy",
    "sample_bounded_search_values",
    "validate_campaign_manifest_stages",
    "validate_campaign_result",
    "validate_canonical_stage_order",
]
