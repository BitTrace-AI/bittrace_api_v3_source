"""Thin consumer bridge for parity verification on completed full-binary runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from bittrace.v3 import (
    ArtifactContract,
    ArtifactRef,
    CAMPAIGN_RESULT_ARTIFACT_NAME,
    CampaignResult,
    ContractValidationError,
    DEEP_ANCHOR_ARTIFACT_NAME,
    ExecutionAcceleration,
    ExecutionTrace,
    FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME,
    FreezeExportManifest,
    FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME,
    PromotionStage,
    STAGE_REQUEST_ARTIFACT_NAME,
    StageKey,
    StageRequest,
    compute_file_sha256,
    load_frozen_s6_golden_reference,
    load_frozen_s6_runtime,
    load_json_artifact,
    write_json_artifact,
)
from bittrace.v3.contracts import GoldenVectorEntry, VerificationLevel
from bittrace.v3.verify import (
    ParityObservation,
    emit_canonical_verification_artifacts,
    payload_snapshot,
)


DEFAULT_VERIFICATION_STAGE_DIRNAME = "07_parity_verification"
_PARITY_STAGE_NAME = "Parity Verification"


@dataclass(frozen=True, slots=True)
class ResolvedS6Artifacts:
    run_root: Path
    freeze_export_manifest_ref: ArtifactRef
    deep_anchor_artifact_ref: ArtifactRef
    frontend_export_reference_ref: ArtifactRef

    @property
    def freeze_export_dir(self) -> Path:
        return Path(self.freeze_export_manifest_ref.path).resolve().parent


@dataclass(frozen=True, slots=True)
class FullBinaryVerificationRunResult:
    run_root: Path
    output_dir: Path
    stage_request_ref: ArtifactRef
    verification_kit_manifest_ref: ArtifactRef
    golden_vector_manifest_ref: ArtifactRef
    parity_report_ref: ArtifactRef
    vector_count: int
    observation_count: int


def resolve_s6_artifacts(run_root: str | Path) -> ResolvedS6Artifacts:
    resolved_run_root = Path(run_root).resolve()
    if not resolved_run_root.exists():
        raise ContractValidationError(f"`run_root` does not exist: {resolved_run_root}")
    if not resolved_run_root.is_dir():
        raise ContractValidationError(f"`run_root` must be a directory: {resolved_run_root}")

    artifact_refs = _artifact_refs_from_campaign_result(resolved_run_root)
    if artifact_refs is None:
        artifact_refs = {
            FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME: _artifact_ref_from_unique_name(
                resolved_run_root,
                FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME,
            ),
            DEEP_ANCHOR_ARTIFACT_NAME: _artifact_ref_from_unique_name(
                resolved_run_root,
                DEEP_ANCHOR_ARTIFACT_NAME,
            ),
            FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME: _artifact_ref_from_unique_name(
                resolved_run_root,
                FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME,
            ),
        }

    return ResolvedS6Artifacts(
        run_root=resolved_run_root,
        freeze_export_manifest_ref=artifact_refs[FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME],
        deep_anchor_artifact_ref=artifact_refs[DEEP_ANCHOR_ARTIFACT_NAME],
        frontend_export_reference_ref=artifact_refs[FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME],
    )


def run_full_binary_verification(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> FullBinaryVerificationRunResult:
    resolved = resolve_s6_artifacts(run_root)
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else resolved.run_root / DEFAULT_VERIFICATION_STAGE_DIRNAME
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    stage_request = _build_stage_request(
        resolved,
        output_dir=resolved_output_dir,
    )
    stage_request_ref = write_json_artifact(
        resolved_output_dir / STAGE_REQUEST_ARTIFACT_NAME,
        stage_request,
    )

    golden_vector_entries, observations = _build_verification_inputs(resolved)
    verification_result = emit_canonical_verification_artifacts(
        stage_request,
        source_freeze_export_manifest_ref=resolved.freeze_export_manifest_ref,
        deep_anchor_artifact_ref=resolved.deep_anchor_artifact_ref,
        frontend_export_reference_ref=resolved.frontend_export_reference_ref,
        golden_vector_entries=golden_vector_entries,
        observations=observations,
    )

    return FullBinaryVerificationRunResult(
        run_root=resolved.run_root,
        output_dir=resolved_output_dir,
        stage_request_ref=stage_request_ref,
        verification_kit_manifest_ref=verification_result.verification_kit_manifest_ref,
        golden_vector_manifest_ref=verification_result.golden_vector_manifest_ref,
        parity_report_ref=verification_result.parity_report_ref,
        vector_count=len(golden_vector_entries),
        observation_count=len(observations),
    )


def _artifact_refs_from_campaign_result(
    run_root: Path,
) -> dict[str, ArtifactRef] | None:
    campaign_result_path = run_root / CAMPAIGN_RESULT_ARTIFACT_NAME
    if not campaign_result_path.exists():
        return None
    campaign_result = load_json_artifact(campaign_result_path)
    if not isinstance(campaign_result, CampaignResult):
        raise ContractValidationError(
            f"`{campaign_result_path}` must resolve to a `CampaignResult` artifact."
        )
    refs_by_name: dict[str, ArtifactRef] = {}
    for ref in campaign_result.freeze_export_refs:
        artifact_name = Path(ref.path).name
        if artifact_name in {
            FREEZE_EXPORT_MANIFEST_ARTIFACT_NAME,
            DEEP_ANCHOR_ARTIFACT_NAME,
            FRONTEND_EXPORT_REFERENCE_ARTIFACT_NAME,
        }:
            refs_by_name[artifact_name] = ref
    if len(refs_by_name) != 3:
        return None
    return refs_by_name


def _artifact_ref_from_unique_name(run_root: Path, artifact_name: str) -> ArtifactRef:
    matches = sorted(path.resolve() for path in run_root.rglob(artifact_name))
    if not matches:
        raise ContractValidationError(
            f"Could not locate `{artifact_name}` under completed run root `{run_root}`."
        )
    if len(matches) != 1:
        raise ContractValidationError(
            f"Expected exactly one `{artifact_name}` under `{run_root}`, found {len(matches)}."
        )
    return _artifact_ref_from_path(matches[0])


def _artifact_ref_from_path(path: str | Path) -> ArtifactRef:
    resolved_path = Path(path).resolve()
    artifact = load_json_artifact(resolved_path)
    if not isinstance(artifact, ArtifactContract):
        raise ContractValidationError(f"`{resolved_path}` must resolve to a V3 artifact.")
    return ArtifactRef(
        kind=artifact.kind,
        schema_version=artifact.schema_version,
        path=str(resolved_path),
        sha256=compute_file_sha256(resolved_path),
    )


def _build_stage_request(
    resolved: ResolvedS6Artifacts,
    *,
    output_dir: Path,
) -> StageRequest:
    freeze_export_manifest = load_json_artifact(resolved.freeze_export_manifest_ref.path)
    if not isinstance(freeze_export_manifest, FreezeExportManifest):
        raise ContractValidationError(
            "Resolved freeze/export manifest path must load as a `FreezeExportManifest`."
        )
    return StageRequest(
        stage_key=StageKey.PARITY_VERIFICATION,
        stage_name=_PARITY_STAGE_NAME,
        campaign_id=freeze_export_manifest.campaign_id,
        campaign_seed=freeze_export_manifest.campaign_seed,
        output_dir=str(output_dir.resolve()),
        input_artifacts=(
            resolved.freeze_export_manifest_ref,
            resolved.deep_anchor_artifact_ref,
            resolved.frontend_export_reference_ref,
        ),
        promotion_stage=PromotionStage.CAPACITY_REFINEMENT,
        execution_trace=ExecutionTrace(
            requested_execution_acceleration=ExecutionAcceleration.CPU,
            resolved_execution_acceleration=ExecutionAcceleration.CPU,
            backend_actual="consumer_verification_bridge",
            allow_backend_fallback=False,
        ),
        notes=(
            f"consumer_run_root={resolved.run_root}",
            f"source_freeze_export_dir={resolved.freeze_export_dir}",
        ),
    )


def _build_verification_inputs(
    resolved: ResolvedS6Artifacts,
) -> tuple[tuple[GoldenVectorEntry, ...], tuple[ParityObservation, ...]]:
    golden_reference = load_frozen_s6_golden_reference(
        freeze_export_manifest_ref=resolved.freeze_export_manifest_ref,
        deep_anchor_artifact_ref=resolved.deep_anchor_artifact_ref,
        frontend_export_reference_ref=resolved.frontend_export_reference_ref,
    )
    runtime = load_frozen_s6_runtime(
        freeze_export_manifest_ref=resolved.freeze_export_manifest_ref,
        deep_anchor_artifact_ref=resolved.deep_anchor_artifact_ref,
        frontend_export_reference_ref=resolved.frontend_export_reference_ref,
    )
    records = _load_record_payloads(
        golden_reference.frontend_export_reference.frontend_lineage.handoff_manifest_path
    )

    golden_vector_entries: list[GoldenVectorEntry] = []
    observations: list[ParityObservation] = []
    for record in records:
        vector_id = _require_record_id(record)
        canonical_input = payload_snapshot(record)
        record_label = _require_record_state_label(record)

        expected_frontend = golden_reference.expected_frontend_output(record)
        actual_frontend = runtime.frontend_infer(record)
        expected_deep = golden_reference.expected_deep_output(canonical_input=record)
        actual_deep = runtime.deep_infer(canonical_input=record)
        expected_end_to_end = golden_reference.expected_end_to_end_output(record)
        actual_end_to_end = runtime.end_to_end_infer(record)

        golden_vector_entries.extend(
            (
                GoldenVectorEntry(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.ADAPTER_PARITY,
                    canonical_input=canonical_input,
                    expected_class=record_label,
                    expected_reject=False,
                ),
                GoldenVectorEntry(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.FRONTEND_PARITY,
                    canonical_input=canonical_input,
                    packed_frontend_input=payload_snapshot(expected_frontend.payload),
                    expected_class=record_label,
                    expected_reject=False,
                ),
                GoldenVectorEntry(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.DEEP_PARITY,
                    canonical_input=canonical_input,
                    expected_deploy_runtime_output=payload_snapshot(expected_deep.payload),
                    expected_class=expected_deep.predicted_class,
                    expected_reject=expected_deep.reject,
                ),
                GoldenVectorEntry(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.END_TO_END_PARITY,
                    canonical_input=canonical_input,
                    expected_deploy_runtime_output=payload_snapshot(
                        expected_end_to_end.payload
                    ),
                    expected_class=expected_end_to_end.predicted_class,
                    expected_reject=expected_end_to_end.reject,
                ),
            )
        )
        observations.extend(
            (
                ParityObservation(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.ADAPTER_PARITY,
                    actual_payload=record,
                ),
                ParityObservation(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.FRONTEND_PARITY,
                    actual_payload=actual_frontend.payload,
                ),
                ParityObservation(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.DEEP_PARITY,
                    actual_payload=actual_deep.payload,
                    actual_class=actual_deep.predicted_class,
                    actual_reject=actual_deep.reject,
                ),
                ParityObservation(
                    vector_id=vector_id,
                    verification_level=VerificationLevel.END_TO_END_PARITY,
                    actual_payload=actual_end_to_end.payload,
                    actual_class=actual_end_to_end.predicted_class,
                    actual_reject=actual_end_to_end.reject,
                ),
            )
        )
    return tuple(golden_vector_entries), tuple(observations)


def _load_record_payloads(handoff_manifest_path: str | Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(Path(handoff_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            f"`{handoff_manifest_path}` must deserialize to a JSON object."
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(
        raw_records, (str, bytes, bytearray)
    ):
        raise ContractValidationError(
            f"`{handoff_manifest_path}` must include a JSON-array `records` field."
        )
    records: list[dict[str, object]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ContractValidationError(
                f"`{handoff_manifest_path}.records[{index}]` must be a JSON object."
            )
        records.append(dict(record))
    if not records:
        raise ContractValidationError(
            f"`{handoff_manifest_path}` must include at least one verification record."
        )
    return tuple(records)


def _require_record_id(record: Mapping[str, object]) -> str:
    value = record.get("source_record_id")
    if not isinstance(value, str) or value == "":
        raise ContractValidationError(
            "Verification record payloads must include non-empty `source_record_id`."
        )
    return value


def _require_record_state_label(record: Mapping[str, object]) -> str:
    value = record.get("state_label")
    if not isinstance(value, str) or value == "":
        raise ContractValidationError(
            "Verification record payloads must include non-empty `state_label`."
        )
    return value


__all__ = [
    "DEFAULT_VERIFICATION_STAGE_DIRNAME",
    "FullBinaryVerificationRunResult",
    "ResolvedS6Artifacts",
    "resolve_s6_artifacts",
    "run_full_binary_verification",
]
