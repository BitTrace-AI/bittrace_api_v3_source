"""Canonical V3 parity and golden-vector verification helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bittrace.v3.artifacts import compute_json_sha256, load_json_artifact_ref, write_json_artifact
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepAnchorArtifact,
    FreezeExportManifest,
    FrontendExportReference,
    GoldenVectorEntry,
    GoldenVectorManifest,
    JsonPrimitive,
    JsonValue,
    ParityComparisonStatus,
    ParityMismatchDetail,
    ParityReport,
    ParityReportEntry,
    PassFail,
    PayloadSnapshot,
    PromotedDeepResult,
    PromotionStage,
    StageKey,
    StageRequest,
    VerificationKitManifest,
    VerificationLevel,
)


VERIFICATION_KIT_MANIFEST_ARTIFACT_NAME = "bt3.verification_kit_manifest.json"
GOLDEN_VECTOR_MANIFEST_ARTIFACT_NAME = "bt3.golden_vector_manifest.json"
PARITY_REPORT_ARTIFACT_NAME = "bt3.parity_report.json"
CANONICAL_VERIFICATION_LEVELS: tuple[VerificationLevel, ...] = (
    VerificationLevel.ADAPTER_PARITY,
    VerificationLevel.FRONTEND_PARITY,
    VerificationLevel.DEEP_PARITY,
    VerificationLevel.END_TO_END_PARITY,
)
_DEFAULT_KIT_SUMMARY = (
    "Canonical V3 verification kit locked to the promoted S5 winner and the frozen/exported S6 deploy path."
)
_DEFAULT_GOLDEN_VECTOR_SUMMARY = (
    "Canonical V3 golden vectors for adapter, frontend, deep, and end-to-end parity verification."
)
_DEFAULT_REPORT_SUMMARY_PREFIX = "Canonical V3 parity verification completed"


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty string.")
    return value


def _require_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name=field_name)


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a bool.")
    return value


def _copy_distinct_str_tuple(value: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    copied = tuple(_require_str(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(set(copied)) != len(copied):
        raise ContractValidationError(f"`{field_name}` must not contain duplicate values.")
    return copied


def _coerce_verification_level(value: VerificationLevel | str, *, field_name: str) -> VerificationLevel:
    try:
        return value if isinstance(value, VerificationLevel) else VerificationLevel(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in VerificationLevel)
        raise ContractValidationError(f"`{field_name}` must be one of: {allowed}.") from exc


def _copy_verification_levels(
    levels: Sequence[VerificationLevel | str],
    *,
    field_name: str,
) -> tuple[VerificationLevel, ...]:
    copied = tuple(
        _coerce_verification_level(level, field_name=f"{field_name}[{index}]")
        for index, level in enumerate(levels)
    )
    if not copied:
        raise ContractValidationError(f"`{field_name}` must not be empty.")
    if len(set(copied)) != len(copied):
        raise ContractValidationError(f"`{field_name}` must not contain duplicate levels.")
    return copied


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {
            _require_str(key, field_name="json.<key>"): _to_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item) for item in value]
    raise ContractValidationError(f"Unsupported JSON value type `{type(value).__name__}`.")


def _load_typed_artifact(ref: ArtifactRef, artifact_type: type[object], *, label: str) -> object:
    artifact = load_json_artifact_ref(ref)
    if not isinstance(artifact, artifact_type):
        raise ContractValidationError(
            f"`{label}` must resolve to `{artifact_type.__name__}`."
        )
    return artifact


def _validate_verification_request(
    request: StageRequest,
    *,
    source_freeze_export_manifest_ref: ArtifactRef,
    deep_anchor_artifact_ref: ArtifactRef,
    frontend_export_reference_ref: ArtifactRef,
) -> None:
    if request.stage_key != StageKey.PARITY_VERIFICATION:
        raise ContractValidationError(
            "Canonical V3 verification only supports `parity_verification` stage requests."
        )
    if (
        request.promotion_stage is not None
        and request.promotion_stage != PromotionStage.CAPACITY_REFINEMENT
    ):
        raise ContractValidationError(
            "Canonical V3 verification only accepts lineage from `promotion_stage=capacity_refinement`."
        )
    for required_ref, label in (
        (source_freeze_export_manifest_ref, "source_freeze_export_manifest_ref"),
        (deep_anchor_artifact_ref, "deep_anchor_artifact_ref"),
        (frontend_export_reference_ref, "frontend_export_reference_ref"),
    ):
        if required_ref not in request.input_artifacts:
            raise ContractValidationError(
                f"Parity-verification stage requests must include `{label}` in `input_artifacts`."
            )


def _build_payload_snapshot(payload: Mapping[str, object]) -> PayloadSnapshot:
    normalized = {
        _require_str(key, field_name="payload.<key>"): _to_json_value(value)
        for key, value in _require_mapping(payload, field_name="payload").items()
    }
    return PayloadSnapshot(payload=normalized, sha256=compute_json_sha256(normalized))


def payload_snapshot(payload: Mapping[str, object]) -> PayloadSnapshot:
    """Create a canonical payload snapshot with its exact SHA-256 digest."""

    return _build_payload_snapshot(payload)


@dataclass(frozen=True, slots=True)
class ParityObservation:
    """Observed parity surface for one golden vector at one verification level."""

    vector_id: str
    verification_level: VerificationLevel | str
    actual_payload: Mapping[str, object] | None = None
    actual_class: JsonPrimitive = None
    actual_reject: bool | None = None
    unsupported_reason: str | None = None
    non_deploy_telemetry_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_id",
            _require_str(self.vector_id, field_name="ParityObservation.vector_id"),
        )
        object.__setattr__(
            self,
            "verification_level",
            _coerce_verification_level(
                self.verification_level,
                field_name="ParityObservation.verification_level",
            ),
        )
        if self.actual_payload is not None:
            normalized_payload = {
                _require_str(key, field_name="ParityObservation.actual_payload.<key>"): _to_json_value(value)
                for key, value in _require_mapping(
                    self.actual_payload,
                    field_name="ParityObservation.actual_payload",
                ).items()
            }
            object.__setattr__(self, "actual_payload", normalized_payload)
        if self.actual_reject is not None:
            object.__setattr__(
                self,
                "actual_reject",
                _require_bool(
                    self.actual_reject,
                    field_name="ParityObservation.actual_reject",
                ),
            )
        object.__setattr__(
            self,
            "unsupported_reason",
            _require_optional_str(
                self.unsupported_reason,
                field_name="ParityObservation.unsupported_reason",
            ),
        )
        object.__setattr__(
            self,
            "non_deploy_telemetry_fields",
            _copy_distinct_str_tuple(
                self.non_deploy_telemetry_fields,
                field_name="ParityObservation.non_deploy_telemetry_fields",
            ),
        )
        if self.unsupported_reason is None and self.actual_payload is None:
            raise ContractValidationError(
                "Supported parity observations must include `actual_payload`."
            )
        if self.unsupported_reason is not None and (
            self.actual_payload is not None
            or self.actual_class is not None
            or self.actual_reject is not None
        ):
            raise ContractValidationError(
                "Unsupported parity observations must not include observed payload/class/reject values."
            )


@dataclass(frozen=True, slots=True)
class VerificationArtifactsRunResult:
    """Written canonical verification artifacts and their resolved refs."""

    verification_kit_manifest: VerificationKitManifest
    verification_kit_manifest_ref: ArtifactRef
    golden_vector_manifest: GoldenVectorManifest
    golden_vector_manifest_ref: ArtifactRef
    parity_report: ParityReport
    parity_report_ref: ArtifactRef


def build_verification_kit_manifest(
    request: StageRequest,
    *,
    source_freeze_export_manifest_ref: ArtifactRef,
    deep_anchor_artifact_ref: ArtifactRef,
    frontend_export_reference_ref: ArtifactRef,
    verification_levels: Sequence[VerificationLevel | str] = CANONICAL_VERIFICATION_LEVELS,
    deploy_runtime_non_deploy_telemetry_fields: Sequence[str] = (),
    summary: str = _DEFAULT_KIT_SUMMARY,
) -> VerificationKitManifest:
    """Build the canonical V3 verification-kit manifest from frozen/exported S6 lineage."""

    _validate_verification_request(
        request,
        source_freeze_export_manifest_ref=source_freeze_export_manifest_ref,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference_ref=frontend_export_reference_ref,
    )
    freeze_export_manifest = cast(
        FreezeExportManifest,
        _load_typed_artifact(
            source_freeze_export_manifest_ref,
            FreezeExportManifest,
            label="source_freeze_export_manifest_ref",
        ),
    )
    deep_anchor_artifact = cast(
        DeepAnchorArtifact,
        _load_typed_artifact(
            deep_anchor_artifact_ref,
            DeepAnchorArtifact,
            label="deep_anchor_artifact_ref",
        ),
    )
    frontend_export_reference = cast(
        FrontendExportReference,
        _load_typed_artifact(
            frontend_export_reference_ref,
            FrontendExportReference,
            label="frontend_export_reference_ref",
        ),
    )
    promoted_deep_result_ref = freeze_export_manifest.source_promoted_deep_result_ref
    if promoted_deep_result_ref is None:
        raise ContractValidationError(
            "Canonical verification requires the S6 freeze/export manifest to preserve the promoted S5 Deep result ref."
        )
    promoted_deep_result = cast(
        PromotedDeepResult,
        _load_typed_artifact(
            promoted_deep_result_ref,
            PromotedDeepResult,
            label="freeze_export_manifest.source_promoted_deep_result_ref",
        ),
    )
    return VerificationKitManifest(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        summary=summary,
        source_stage_request=request,
        verification_levels=_copy_verification_levels(
            verification_levels,
            field_name="verification_levels",
        ),
        source_promoted_deep_result_ref=promoted_deep_result_ref,
        source_promoted_deep_result=promoted_deep_result,
        source_freeze_export_manifest_ref=source_freeze_export_manifest_ref,
        source_freeze_export_manifest=freeze_export_manifest,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        deep_anchor_artifact=deep_anchor_artifact,
        frontend_export_reference_ref=frontend_export_reference_ref,
        frontend_export_reference=frontend_export_reference,
        deploy_runtime_non_deploy_telemetry_fields=_copy_distinct_str_tuple(
            tuple(deploy_runtime_non_deploy_telemetry_fields),
            field_name="deploy_runtime_non_deploy_telemetry_fields",
        ),
        execution_trace=request.execution_trace,
    )


def emit_verification_kit_manifest(
    request: StageRequest,
    *,
    source_freeze_export_manifest_ref: ArtifactRef,
    deep_anchor_artifact_ref: ArtifactRef,
    frontend_export_reference_ref: ArtifactRef,
    verification_levels: Sequence[VerificationLevel | str] = CANONICAL_VERIFICATION_LEVELS,
    deploy_runtime_non_deploy_telemetry_fields: Sequence[str] = (),
    summary: str = _DEFAULT_KIT_SUMMARY,
) -> tuple[VerificationKitManifest, ArtifactRef]:
    """Build and write the canonical verification-kit manifest artifact."""

    manifest = build_verification_kit_manifest(
        request,
        source_freeze_export_manifest_ref=source_freeze_export_manifest_ref,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference_ref=frontend_export_reference_ref,
        verification_levels=verification_levels,
        deploy_runtime_non_deploy_telemetry_fields=deploy_runtime_non_deploy_telemetry_fields,
        summary=summary,
    )
    ref = write_json_artifact(
        Path(request.output_dir) / VERIFICATION_KIT_MANIFEST_ARTIFACT_NAME,
        manifest,
    )
    return manifest, ref


def build_golden_vector_manifest(
    request: StageRequest,
    *,
    source_verification_kit_ref: ArtifactRef,
    entries: Sequence[GoldenVectorEntry],
    source_verification_kit: VerificationKitManifest | None = None,
    summary: str = _DEFAULT_GOLDEN_VECTOR_SUMMARY,
) -> GoldenVectorManifest:
    """Build the canonical V3 golden-vector manifest."""

    verification_kit = source_verification_kit
    if verification_kit is None:
        verification_kit = cast(
            VerificationKitManifest,
            _load_typed_artifact(
                source_verification_kit_ref,
                VerificationKitManifest,
                label="source_verification_kit_ref",
            ),
        )
    return GoldenVectorManifest(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        summary=summary,
        source_verification_kit_ref=source_verification_kit_ref,
        source_verification_kit=verification_kit,
        entries=tuple(entries),
        execution_trace=request.execution_trace,
    )


def emit_golden_vector_manifest(
    request: StageRequest,
    *,
    source_verification_kit_ref: ArtifactRef,
    entries: Sequence[GoldenVectorEntry],
    source_verification_kit: VerificationKitManifest | None = None,
    summary: str = _DEFAULT_GOLDEN_VECTOR_SUMMARY,
) -> tuple[GoldenVectorManifest, ArtifactRef]:
    """Build and write the canonical golden-vector manifest artifact."""

    manifest = build_golden_vector_manifest(
        request,
        source_verification_kit_ref=source_verification_kit_ref,
        entries=entries,
        source_verification_kit=source_verification_kit,
        summary=summary,
    )
    ref = write_json_artifact(
        Path(request.output_dir) / GOLDEN_VECTOR_MANIFEST_ARTIFACT_NAME,
        manifest,
    )
    return manifest, ref


def _expected_payload_snapshot(entry: GoldenVectorEntry) -> PayloadSnapshot:
    if entry.verification_level == VerificationLevel.ADAPTER_PARITY:
        return entry.canonical_input
    if entry.verification_level == VerificationLevel.FRONTEND_PARITY:
        if entry.packed_frontend_input is None:
            raise ContractValidationError(
                f"Golden vector `{entry.vector_id}` is missing `packed_frontend_input`."
            )
        return entry.packed_frontend_input
    if entry.expected_deploy_runtime_output is None:
        raise ContractValidationError(
            f"Golden vector `{entry.vector_id}` is missing `expected_deploy_runtime_output`."
        )
    return entry.expected_deploy_runtime_output


def _is_ignored_path(path: str, ignored_paths: set[str]) -> bool:
    if path == "":
        return False
    return any(path == ignored or path.startswith(f"{ignored}.") or path.startswith(f"{ignored}[") for ignored in ignored_paths)


def _append_mismatch(
    mismatches: list[ParityMismatchDetail],
    *,
    field_path: str,
    expected: object,
    actual: object,
) -> None:
    mismatches.append(
        ParityMismatchDetail(
            field_path=field_path or "<root>",
            expected=_to_json_value(expected),
            actual=_to_json_value(actual),
        )
    )


def _compare_json_values(
    expected: JsonValue,
    actual: JsonValue,
    *,
    path: str,
    ignored_paths: set[str],
    mismatches: list[ParityMismatchDetail],
) -> None:
    if _is_ignored_path(path, ignored_paths):
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys | actual_keys):
            child_path = key if path == "" else f"{path}.{key}"
            if _is_ignored_path(child_path, ignored_paths):
                continue
            if key not in expected:
                _append_mismatch(
                    mismatches,
                    field_path=child_path,
                    expected="<missing>",
                    actual=actual[key],
                )
                continue
            if key not in actual:
                _append_mismatch(
                    mismatches,
                    field_path=child_path,
                    expected=expected[key],
                    actual="<missing>",
                )
                continue
            _compare_json_values(
                expected[key],
                actual[key],
                path=child_path,
                ignored_paths=ignored_paths,
                mismatches=mismatches,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        limit = max(len(expected), len(actual))
        for index in range(limit):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            if _is_ignored_path(child_path, ignored_paths):
                continue
            if index >= len(expected):
                _append_mismatch(
                    mismatches,
                    field_path=child_path,
                    expected="<missing>",
                    actual=actual[index],
                )
                continue
            if index >= len(actual):
                _append_mismatch(
                    mismatches,
                    field_path=child_path,
                    expected=expected[index],
                    actual="<missing>",
                )
                continue
            _compare_json_values(
                expected[index],
                actual[index],
                path=child_path,
                ignored_paths=ignored_paths,
                mismatches=mismatches,
            )
        return
    if expected != actual:
        _append_mismatch(
            mismatches,
            field_path=path,
            expected=expected,
            actual=actual,
        )


def build_parity_report(
    request: StageRequest,
    *,
    source_verification_kit_ref: ArtifactRef,
    source_golden_vector_manifest_ref: ArtifactRef,
    observations: Sequence[ParityObservation],
    source_verification_kit: VerificationKitManifest | None = None,
    source_golden_vector_manifest: GoldenVectorManifest | None = None,
    summary: str | None = None,
) -> ParityReport:
    """Build the canonical V3 parity report from observed parity surfaces."""

    verification_kit = source_verification_kit
    if verification_kit is None:
        verification_kit = cast(
            VerificationKitManifest,
            _load_typed_artifact(
                source_verification_kit_ref,
                VerificationKitManifest,
                label="source_verification_kit_ref",
            ),
        )
    golden_vector_manifest = source_golden_vector_manifest
    if golden_vector_manifest is None:
        golden_vector_manifest = cast(
            GoldenVectorManifest,
            _load_typed_artifact(
                source_golden_vector_manifest_ref,
                GoldenVectorManifest,
                label="source_golden_vector_manifest_ref",
            ),
        )

    entry_by_key = {
        (entry.vector_id, entry.verification_level): entry
        for entry in golden_vector_manifest.entries
    }
    results: list[ParityReportEntry] = []
    for observation in observations:
        level = cast(VerificationLevel, observation.verification_level)
        key = (observation.vector_id, level)
        try:
            entry = entry_by_key[key]
        except KeyError as exc:
            raise ContractValidationError(
                f"Parity observation `{observation.vector_id}` / `{level.value}` does not map to a golden-vector entry."
            ) from exc

        expected_payload = _expected_payload_snapshot(entry)
        effective_non_deploy_fields = tuple(
            dict.fromkeys(
                verification_kit.deploy_runtime_non_deploy_telemetry_fields
                + entry.non_deploy_telemetry_fields
                + observation.non_deploy_telemetry_fields
            )
        )
        if observation.unsupported_reason is not None:
            results.append(
                ParityReportEntry(
                    vector_id=entry.vector_id,
                    verification_level=entry.verification_level,
                    comparison_status=ParityComparisonStatus.UNSUPPORTED,
                    expected_payload_sha256=expected_payload.sha256,
                    expected_class=entry.expected_class,
                    expected_reject=entry.expected_reject,
                    non_deploy_telemetry_fields=effective_non_deploy_fields,
                    unsupported_reason=observation.unsupported_reason,
                )
            )
            continue

        actual_payload = cast(Mapping[str, object], observation.actual_payload)
        actual_payload_sha256 = compute_json_sha256(actual_payload)
        mismatches: list[ParityMismatchDetail] = []
        _compare_json_values(
            expected_payload.payload,
            cast(JsonValue, _to_json_value(actual_payload)),
            path="",
            ignored_paths=set(effective_non_deploy_fields),
            mismatches=mismatches,
        )

        compare_class = level in {
            VerificationLevel.DEEP_PARITY,
            VerificationLevel.END_TO_END_PARITY,
        } or observation.actual_class is not None
        compare_reject = level in {
            VerificationLevel.DEEP_PARITY,
            VerificationLevel.END_TO_END_PARITY,
        } or observation.actual_reject is not None
        if compare_class and entry.expected_class != observation.actual_class:
            _append_mismatch(
                mismatches,
                field_path="class",
                expected=entry.expected_class,
                actual=observation.actual_class,
            )
        if compare_reject:
            if observation.actual_reject is None:
                raise ContractValidationError(
                    f"Parity observation `{entry.vector_id}` / `{level.value}` is missing `actual_reject`."
                )
            if entry.expected_reject != observation.actual_reject:
                _append_mismatch(
                    mismatches,
                    field_path="reject",
                    expected=entry.expected_reject,
                    actual=observation.actual_reject,
                )

        results.append(
            ParityReportEntry(
                vector_id=entry.vector_id,
                verification_level=entry.verification_level,
                comparison_status=(
                    ParityComparisonStatus.MISMATCH
                    if mismatches
                    else ParityComparisonStatus.EXACT_MATCH
                ),
                expected_payload_sha256=expected_payload.sha256,
                actual_payload_sha256=actual_payload_sha256,
                expected_class=entry.expected_class if compare_class else None,
                actual_class=observation.actual_class if compare_class else None,
                expected_reject=entry.expected_reject if compare_reject else None,
                actual_reject=observation.actual_reject if compare_reject else None,
                non_deploy_telemetry_fields=effective_non_deploy_fields,
                mismatch_details=tuple(mismatches),
            )
        )

    exact_match_count = sum(
        1 for result in results if result.comparison_status == ParityComparisonStatus.EXACT_MATCH
    )
    mismatch_count = sum(
        1 for result in results if result.comparison_status == ParityComparisonStatus.MISMATCH
    )
    unsupported_count = sum(
        1 for result in results if result.comparison_status == ParityComparisonStatus.UNSUPPORTED
    )
    resolved_summary = summary or (
        f"{_DEFAULT_REPORT_SUMMARY_PREFIX}: "
        f"{exact_match_count} exact matches, {mismatch_count} mismatches, "
        f"{unsupported_count} unsupported comparisons."
    )
    return ParityReport(
        stage_key=request.stage_key,
        stage_name=request.stage_name,
        campaign_id=request.campaign_id,
        campaign_seed=request.campaign_seed,
        pass_fail=PassFail.FAIL if mismatch_count else PassFail.PASS,
        summary=resolved_summary,
        source_verification_kit_ref=source_verification_kit_ref,
        source_verification_kit=verification_kit,
        source_golden_vector_manifest_ref=source_golden_vector_manifest_ref,
        source_golden_vector_manifest=golden_vector_manifest,
        results=tuple(results),
        exact_match_count=exact_match_count,
        mismatch_count=mismatch_count,
        unsupported_count=unsupported_count,
        execution_trace=request.execution_trace,
    )


def emit_parity_report(
    request: StageRequest,
    *,
    source_verification_kit_ref: ArtifactRef,
    source_golden_vector_manifest_ref: ArtifactRef,
    observations: Sequence[ParityObservation],
    source_verification_kit: VerificationKitManifest | None = None,
    source_golden_vector_manifest: GoldenVectorManifest | None = None,
    summary: str | None = None,
) -> tuple[ParityReport, ArtifactRef]:
    """Build and write the canonical parity report artifact."""

    report = build_parity_report(
        request,
        source_verification_kit_ref=source_verification_kit_ref,
        source_golden_vector_manifest_ref=source_golden_vector_manifest_ref,
        observations=observations,
        source_verification_kit=source_verification_kit,
        source_golden_vector_manifest=source_golden_vector_manifest,
        summary=summary,
    )
    ref = write_json_artifact(Path(request.output_dir) / PARITY_REPORT_ARTIFACT_NAME, report)
    return report, ref


def emit_canonical_verification_artifacts(
    request: StageRequest,
    *,
    source_freeze_export_manifest_ref: ArtifactRef,
    deep_anchor_artifact_ref: ArtifactRef,
    frontend_export_reference_ref: ArtifactRef,
    golden_vector_entries: Sequence[GoldenVectorEntry],
    observations: Sequence[ParityObservation],
    verification_levels: Sequence[VerificationLevel | str] = CANONICAL_VERIFICATION_LEVELS,
    deploy_runtime_non_deploy_telemetry_fields: Sequence[str] = (),
    verification_kit_summary: str = _DEFAULT_KIT_SUMMARY,
    golden_vector_summary: str = _DEFAULT_GOLDEN_VECTOR_SUMMARY,
    parity_report_summary: str | None = None,
) -> VerificationArtifactsRunResult:
    """Emit the canonical V3 verification kit, golden vectors, and parity report."""

    verification_kit_manifest, verification_kit_manifest_ref = emit_verification_kit_manifest(
        request,
        source_freeze_export_manifest_ref=source_freeze_export_manifest_ref,
        deep_anchor_artifact_ref=deep_anchor_artifact_ref,
        frontend_export_reference_ref=frontend_export_reference_ref,
        verification_levels=verification_levels,
        deploy_runtime_non_deploy_telemetry_fields=deploy_runtime_non_deploy_telemetry_fields,
        summary=verification_kit_summary,
    )
    golden_vector_manifest, golden_vector_manifest_ref = emit_golden_vector_manifest(
        request,
        source_verification_kit_ref=verification_kit_manifest_ref,
        source_verification_kit=verification_kit_manifest,
        entries=golden_vector_entries,
        summary=golden_vector_summary,
    )
    parity_report, parity_report_ref = emit_parity_report(
        request,
        source_verification_kit_ref=verification_kit_manifest_ref,
        source_golden_vector_manifest_ref=golden_vector_manifest_ref,
        source_verification_kit=verification_kit_manifest,
        source_golden_vector_manifest=golden_vector_manifest,
        observations=observations,
        summary=parity_report_summary,
    )
    return VerificationArtifactsRunResult(
        verification_kit_manifest=verification_kit_manifest,
        verification_kit_manifest_ref=verification_kit_manifest_ref,
        golden_vector_manifest=golden_vector_manifest,
        golden_vector_manifest_ref=golden_vector_manifest_ref,
        parity_report=parity_report,
        parity_report_ref=parity_report_ref,
    )


__all__ = [
    "CANONICAL_VERIFICATION_LEVELS",
    "GOLDEN_VECTOR_MANIFEST_ARTIFACT_NAME",
    "PARITY_REPORT_ARTIFACT_NAME",
    "VERIFICATION_KIT_MANIFEST_ARTIFACT_NAME",
    "ParityObservation",
    "VerificationArtifactsRunResult",
    "build_golden_vector_manifest",
    "build_parity_report",
    "build_verification_kit_manifest",
    "emit_canonical_verification_artifacts",
    "emit_golden_vector_manifest",
    "emit_parity_report",
    "emit_verification_kit_manifest",
    "payload_snapshot",
]
