"""Canonical V3 S6 freeze/export artifact emission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from bittrace.v3.artifacts import compute_json_sha256, load_json_artifact_ref, write_json_artifact
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepAnchorArtifact,
    DeepInputRef,
    DeployedDeepWinner,
    ExecutionTrace,
    FreezeExportDeployRuntime,
    FreezeExportManifest,
    FreezeExportProvenance,
    FrontendExportReference,
    PassFail,
    PromotedDeepResult,
    PromotionStage,
    StageKey,
    StageRequest,
    StageResult,
)


FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME = "bt3.freeze_export_manifest.json"
DEEP_ANCHOR_ARTIFACT_NAME = "bt3.deep_anchor_artifact.json"
FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME = "bt3.frontend_export_reference.json"
FREEZE_EXPORT_STAGE_RESULT_ARTIFACT_NAME = "bt3.stage_result.json"
_CANONICAL_SUMMARY = (
    "Winner-deepen freeze/export emitted a device-agnostic Deep anchor artifact, "
    "a frontend export reference, and a canonical S6 manifest locked to the promoted S5 winning k."
)
_CANONICAL_DEVICE_AGNOSTIC_EXPORT = {
    "portable": True,
    "execution_device": None,
    "hardware_binding": None,
    "note": (
        "Execution acceleration is recorded in the stage trace only; the exported Deep anchor "
        "remains device-agnostic and deployment-boundary safe."
    ),
}


@dataclass(frozen=True, slots=True)
class FreezeExportRunResult:
    """Written S6 freeze/export artifacts and their resolved refs."""

    freeze_export_manifest: FreezeExportManifest
    freeze_export_manifest_ref: ArtifactRef
    deep_anchor_artifact: DeepAnchorArtifact
    deep_anchor_artifact_ref: ArtifactRef
    frontend_export_reference: FrontendExportReference
    frontend_export_reference_ref: ArtifactRef
    stage_result: StageResult
    stage_result_ref: ArtifactRef
    source_promoted_deep_result: PromotedDeepResult
    source_promoted_deep_result_ref: ArtifactRef


def run_winner_deepen_freeze_export(
    request: StageRequest,
    *,
    source_promoted_deep_result_ref: ArtifactRef,
    execution_trace: ExecutionTrace | None = None,
    device_agnostic_export: Mapping[str, object] | None = None,
    summary: str = _CANONICAL_SUMMARY,
) -> FreezeExportRunResult:
    """Emit canonical V3 S6 freeze/export artifacts from the promoted S5 Deep winner."""

    return freeze_export(
        request,
        source_promoted_deep_result_ref=source_promoted_deep_result_ref,
        execution_trace=execution_trace,
        device_agnostic_export=device_agnostic_export,
        summary=summary,
    )


def freeze_export(
    request: StageRequest,
    *,
    source_promoted_deep_result_ref: ArtifactRef,
    execution_trace: ExecutionTrace | None = None,
    device_agnostic_export: Mapping[str, object] | None = None,
    summary: str = _CANONICAL_SUMMARY,
) -> FreezeExportRunResult:
    """Freeze the promoted S5 winner into canonical V3 S6 export artifacts."""

    _validate_freeze_export_request(request, source_promoted_deep_result_ref=source_promoted_deep_result_ref)
    source_promoted_deep_result = _load_source_promoted_deep_result(source_promoted_deep_result_ref)
    winner = source_promoted_deep_result.promoted_candidate
    if winner is None:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires a passing promoted S5 Deep winner."
        )
    selected_k = winner.selected_k_per_class
    if selected_k is None:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires the promoted S5 Deep winner to carry exactly one selected `k`."
        )

    effective_execution_trace = (
        execution_trace
        if execution_trace is not None
        else request.execution_trace or source_promoted_deep_result.execution_trace
    )
    effective_device_agnostic_export = _normalize_device_agnostic_export(
        source_mapping=device_agnostic_export or source_promoted_deep_result.device_agnostic_export,
    )
    frontend_lineage = _resolve_frontend_lineage(
        request,
        source_promoted_deep_result=source_promoted_deep_result,
    )
    source_frontend_promotion_ref = frontend_lineage.source_promotion_artifact_ref
    if source_frontend_promotion_ref is None:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires explicit lineage back to the promoted frontend winner."
        )

    provenance = FreezeExportProvenance(
        source_stage_request=request,
        source_promoted_deep_result_ref=source_promoted_deep_result_ref,
        source_promoted_deep_result=source_promoted_deep_result,
        source_frontend_input=frontend_lineage,
        source_frontend_promotion_ref=source_frontend_promotion_ref,
    )
    direct_restore_payload = _build_direct_restore_payload(
        winner_best_deep_artifact_ref=winner.best_deep_artifact_ref,
        selected_k_per_class=selected_k,
        device_agnostic_export=effective_device_agnostic_export,
    )
    winning_deep_candidate_fingerprint = _compute_winning_deep_candidate_fingerprint(
        promoted_winner=winner,
        direct_restore_payload=direct_restore_payload,
    )
    deployed_winner = DeployedDeepWinner(
        candidate_id=winner.candidate_id,
        candidate_order=winner.candidate_order,
        branch_mode=winner.branch_mode,
        selected_k_per_class=selected_k,
        frontend_input_id=winner.frontend_input_id,
        frontend_fingerprint=winner.frontend_fingerprint,
        parent_anchor_fingerprint=winner.parent_anchor_fingerprint,
        best_deep_artifact_ref=winner.best_deep_artifact_ref,
        metrics_summary_ref=winner.metrics_summary_ref,
        candidate_report_ref=winner.candidate_report_ref,
        checkpoint_ref=winner.checkpoint_ref,
        effective_engine_deep_config={"k_medoids_per_class": selected_k},
        direct_restore_payload=direct_restore_payload,
    )
    deep_anchor_artifact = DeepAnchorArtifact(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=PassFail.PASS,
        summary=summary,
        provenance=provenance,
        frontend_lineage=frontend_lineage,
        deployed_winner=deployed_winner,
        winning_deep_candidate_fingerprint=winning_deep_candidate_fingerprint,
        export_portability=effective_device_agnostic_export,
        execution_trace=effective_execution_trace,
    )
    deep_anchor_artifact_ref = write_json_artifact(
        Path(request.output_dir) / DEEP_ANCHOR_ARTIFACT_NAME,
        deep_anchor_artifact,
    )

    frontend_export_reference = FrontendExportReference(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=PassFail.PASS,
        summary=summary,
        source_stage_request=request,
        source_promoted_deep_result_ref=source_promoted_deep_result_ref,
        source_frontend_promotion_ref=source_frontend_promotion_ref,
        frontend_lineage=frontend_lineage,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        selected_k_per_class=selected_k,
        winning_deep_candidate_fingerprint=winning_deep_candidate_fingerprint,
        execution_trace=effective_execution_trace,
    )
    frontend_export_reference_ref = write_json_artifact(
        Path(request.output_dir) / FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME,
        frontend_export_reference,
    )

    deploy_runtime = FreezeExportDeployRuntime(
        selected_k_per_class=selected_k,
        anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference_ref=frontend_export_reference_ref,
        winner_artifact_refs=(deep_anchor_artifact_ref, frontend_export_reference_ref),
        winner_best_deep_artifact_ref=winner.best_deep_artifact_ref,
        winner_metrics_summary_ref=winner.metrics_summary_ref,
        winner_candidate_report_ref=winner.candidate_report_ref,
        winner_checkpoint_ref=winner.checkpoint_ref,
        device_agnostic_export=effective_device_agnostic_export,
        execution_trace=effective_execution_trace,
    )
    freeze_export_manifest = FreezeExportManifest(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=PassFail.PASS,
        summary=summary,
        placeholder=False,
        promotion_stage=PromotionStage.CAPACITY_REFINEMENT,
        anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference_ref=frontend_export_reference_ref,
        winner_artifact_refs=(deep_anchor_artifact_ref, frontend_export_reference_ref),
        source_promoted_deep_result_ref=source_promoted_deep_result_ref,
        provenance=provenance,
        deploy_runtime=deploy_runtime,
        device_agnostic_export=effective_device_agnostic_export,
        execution_trace=effective_execution_trace,
    )
    freeze_export_manifest_ref = write_json_artifact(
        Path(request.output_dir) / FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME,
        freeze_export_manifest,
    )

    stage_result = StageResult(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=PassFail.PASS,
        primary_artifact_ref=deep_anchor_artifact_ref,
        artifact_refs=(
            deep_anchor_artifact_ref,
            frontend_export_reference_ref,
            freeze_export_manifest_ref,
        ),
        execution_trace=effective_execution_trace,
        compliance_checks=_stage_result_compliance_checks(
            execution_trace=effective_execution_trace,
            device_agnostic_export=effective_device_agnostic_export,
        ),
    )
    stage_result_ref = write_json_artifact(
        Path(request.output_dir) / FREEZE_EXPORT_STAGE_RESULT_ARTIFACT_NAME,
        stage_result,
    )

    return FreezeExportRunResult(
        freeze_export_manifest=freeze_export_manifest,
        freeze_export_manifest_ref=freeze_export_manifest_ref,
        deep_anchor_artifact=deep_anchor_artifact,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference=frontend_export_reference,
        frontend_export_reference_ref=frontend_export_reference_ref,
        stage_result=stage_result,
        stage_result_ref=stage_result_ref,
        source_promoted_deep_result=source_promoted_deep_result,
        source_promoted_deep_result_ref=source_promoted_deep_result_ref,
    )


def _validate_freeze_export_request(
    request: StageRequest,
    *,
    source_promoted_deep_result_ref: ArtifactRef,
) -> None:
    if request.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
        raise ContractValidationError(
            "Canonical V3 S6 freeze/export only supports `winner_deepen_freeze_export`."
        )
    if source_promoted_deep_result_ref not in request.input_artifacts:
        raise ContractValidationError(
            "Canonical S6 stage requests must explicitly include the promoted S5 Deep result ref in `input_artifacts`."
        )
    if request.promotion_stage is not None and request.promotion_stage != PromotionStage.CAPACITY_REFINEMENT:
        raise ContractValidationError(
            "Canonical S6 freeze/export only accepts lineage from `promotion_stage=capacity_refinement`."
        )


def _load_source_promoted_deep_result(source_promoted_deep_result_ref: ArtifactRef) -> PromotedDeepResult:
    artifact = load_json_artifact_ref(source_promoted_deep_result_ref)
    if not isinstance(artifact, PromotedDeepResult):
        raise ContractValidationError(
            "Canonical S6 freeze/export requires a `PromotedDeepResult` artifact ref as input lineage."
        )
    if artifact.stage_key != StageKey.CAPACITY_REFINEMENT:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires lineage from canonical S5 `capacity_refinement`."
        )
    if artifact.promotion_stage != PromotionStage.CAPACITY_REFINEMENT:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires `promotion_stage=capacity_refinement` from the source Deep result."
        )
    if artifact.pass_fail != PassFail.PASS:
        raise ContractValidationError(
            "Canonical S6 freeze/export only supports a passing promoted S5 Deep result."
        )
    return artifact


def _resolve_frontend_lineage(
    request: StageRequest,
    *,
    source_promoted_deep_result: PromotedDeepResult,
) -> DeepInputRef:
    winner = source_promoted_deep_result.promoted_candidate
    if winner is None:
        raise ContractValidationError("Promoted S5 Deep result is missing its winner.")
    matching_inputs = tuple(
        frontend_input
        for frontend_input in source_promoted_deep_result.frontend_inputs
        if (
            (winner.frontend_input_id is None or frontend_input.frontend_input_id == winner.frontend_input_id)
            and (
                winner.frontend_fingerprint is None
                or frontend_input.frontend_fingerprint == winner.frontend_fingerprint
            )
        )
    )
    if not matching_inputs and len(source_promoted_deep_result.frontend_inputs) == 1:
        matching_inputs = tuple(source_promoted_deep_result.frontend_inputs)
    if len(matching_inputs) != 1:
        raise ContractValidationError(
            "Canonical S6 freeze/export requires exactly one resolved frontend lineage from the promoted S5 winner."
        )
    frontend_lineage = matching_inputs[0]
    if request.deep_inputs:
        request_matches = tuple(
            deep_input
            for deep_input in request.deep_inputs
            if deep_input.frontend_input_id == frontend_lineage.frontend_input_id
            and deep_input.frontend_fingerprint == frontend_lineage.frontend_fingerprint
        )
        if not request_matches:
            raise ContractValidationError(
                "Canonical S6 stage request deep-input lineage must match the promoted S5 winner frontend lineage."
            )
    return frontend_lineage


def _normalize_device_agnostic_export(*, source_mapping: Mapping[str, object] | None) -> dict[str, object]:
    merged = dict(_CANONICAL_DEVICE_AGNOSTIC_EXPORT)
    if source_mapping is not None:
        merged.update(dict(source_mapping))
    merged["portable"] = True
    merged["execution_device"] = None
    merged["hardware_binding"] = None
    return merged


def _build_direct_restore_payload(
    *,
    winner_best_deep_artifact_ref: ArtifactRef | None,
    selected_k_per_class: int,
    device_agnostic_export: Mapping[str, object],
) -> dict[str, object]:
    raw_payload: Mapping[str, object] = {}
    if winner_best_deep_artifact_ref is not None:
        artifact_path = Path(winner_best_deep_artifact_ref.path)
        if artifact_path.is_file():
            loaded_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_payload, dict):
                raise ContractValidationError("Winner best-Deep artifact payload must deserialize to a JSON object.")
            raw_payload = loaded_payload

    sanitized: dict[str, object] = {}
    for field_name in (
        "kind",
        "schema_version",
        "mode",
        "row_format",
        "bit_length",
        "class_labels",
        "best_candidate",
        "deep_engine_semantics",
        "smoke_engine_semantics",
        "execution_kind",
    ):
        if field_name in raw_payload:
            sanitized[field_name] = raw_payload[field_name]
    sanitized["deep_config"] = {"k_medoids_per_class": selected_k_per_class}
    sanitized["export_contract"] = dict(device_agnostic_export)
    sanitized["deploy_path"] = "pure_symbolic"
    sanitized["deep_owns_downstream_classification_export"] = True
    return sanitized


def _compute_winning_deep_candidate_fingerprint(
    *,
    promoted_winner,
    direct_restore_payload: Mapping[str, object],
) -> str:
    best_candidate = direct_restore_payload.get("best_candidate")
    if isinstance(best_candidate, Mapping):
        candidate_payload = best_candidate.get("candidate")
        if isinstance(candidate_payload, Mapping):
            return compute_json_sha256(candidate_payload)
    return compute_json_sha256(promoted_winner.to_dict())


def _stage_result_compliance_checks(
    *,
    execution_trace: ExecutionTrace | None,
    device_agnostic_export: Mapping[str, object],
) -> dict[str, bool]:
    return {
        "winner_present_when_promoted": True,
        "exactly_one_promoted_s6_winner": True,
        "freeze_export_manifest_emitted": True,
        "frontend_export_reference_emitted": True,
        "selected_k_per_class_locked_to_promoted_s5_winner": True,
        "device_agnostic_export_enforced": (
            device_agnostic_export.get("portable") is True
            and device_agnostic_export.get("execution_device") is None
            and device_agnostic_export.get("hardware_binding") is None
        ),
        "execution_trace_present": execution_trace is not None,
        "requested_backend_recorded": execution_trace is not None,
        "actual_backend_recorded": execution_trace is not None and execution_trace.backend_actual is not None,
    }


__all__ = [
    "DEEP_ANCHOR_ARTIFACT_NAME",
    "FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME",
    "FREEZE_EXPORT_STAGE_RESULT_ARTIFACT_NAME",
    "FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME",
    "FreezeExportRunResult",
    "freeze_export",
    "run_winner_deepen_freeze_export",
]
