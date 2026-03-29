"""Generic waveform-backed frontend materialization helpers for canonical V3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from bittrace.core.config import LeanTrainingConfig
from bittrace.v3.artifacts import compute_file_sha256, compute_json_sha256
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepInputRef,
    FrontendInput,
    FrontendPromotionCandidate,
    PromotedFrontendWinner,
    StageRequest,
    WaveformDatasetRecord,
)
from bittrace.v3.dataset_inputs import (
    WAVEFORM_SOURCE_BUNDLE_KIND,
    WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION,
    WAVEFORM_SOURCE_HANDOFF_KIND,
    WAVEFORM_SOURCE_HANDOFF_SCHEMA_VERSION,
    WaveformDatasetBundle,
)
from bittrace.v3.frontend_encoding import (
    build_legacy_hash_frontend_summary,
    build_temporal_frontend_encoding,
)


BEST_FRONTEND_ARTIFACT_NAME = "bt3.best_frontend.json"
FRONTEND_CANDIDATE_REPORT_ARTIFACT_NAME = "bt3.frontend_candidate_report.json"
_BEST_FRONTEND_KIND = "bittrace_v3_materialized_frontend_artifact"
_BEST_FRONTEND_SCHEMA_VERSION = "bittrace-v3-materialized-frontend-artifact-1"
_FRONTEND_CANDIDATE_REPORT_KIND = "bittrace_v3_frontend_candidate_report"
_FRONTEND_CANDIDATE_REPORT_SCHEMA_VERSION = "bittrace-v3-frontend-candidate-report-1"
_CANONICAL_SPLIT_TARGET_COUNT = 3.0


@dataclass(frozen=True, slots=True)
class MaterializedWaveformFrontend:
    """Resolved generic frontend winner materialized from waveform-backed inputs."""

    promoted_candidate: FrontendPromotionCandidate
    downstream_deep_input: DeepInputRef


def materialize_waveform_frontend(
    request: StageRequest,
    *,
    frontend_input: FrontendInput,
    promotion_artifact_path: str | Path,
    canonical_frontend_path: str,
    lean_training_config: LeanTrainingConfig | None = None,
) -> MaterializedWaveformFrontend:
    """Materialize a single generic frontend candidate from a waveform-backed bundle."""

    bundle = _load_waveform_dataset_bundle(frontend_input)
    summary = _summarize_canonical_records(bundle.canonical_records)
    frontend_input_id = f"frontend-{request.stage_key.value}-{frontend_input.bundle_fingerprint[:12]}"
    candidate_id = frontend_input_id
    legacy_frontend_config_fingerprint = compute_json_sha256(
        {
            "canonical_frontend_path": canonical_frontend_path,
            "frontend_input_bundle_fingerprint": frontend_input.bundle_fingerprint,
            "include_test_metrics": frontend_input.include_test_metrics,
            "resolved_genome_identity": (
                frontend_input.resolved_genome_identity.to_dict()
                if frontend_input.resolved_genome_identity is not None
                else None
            ),
        }
    )
    legacy_frontend_fingerprint = compute_json_sha256(
        {
            "frontend_config_fingerprint": legacy_frontend_config_fingerprint,
            "frontend_input_id": frontend_input_id,
            "source_bundle_fingerprint": frontend_input.bundle_fingerprint,
            "stage_key": request.stage_key.value,
            "summary": summary,
        }
    )
    legacy_hash_summary = build_legacy_hash_frontend_summary(
        bundle.canonical_records,
        dataset_id=bundle.dataset_id,
        adapter_profile_id=bundle.adapter_profile_id,
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=legacy_frontend_fingerprint,
    )
    ranking_metrics_basis = {
        "healthy_unhealthy_margin": "one_minus_validation_false_positive_rate_from_train_medoids",
        "inter_class_separation": "train_class_medoid_hamming_distance_fraction",
        "intra_class_compactness": "one_minus_train_within_class_distance_fraction",
        "bit_balance": "mean_train_bit_balance_score",
        "bit_stability": "one_minus_cross_split_bit_rate_drift",
    }
    frontend_encoding_payload: dict[str, object] | None = None
    temporal_frontend_summary: dict[str, object] | None = None
    comparison_to_legacy_hash: dict[str, object] | None = None
    materialization_mode = "canonical_waveform_bundle"
    frontend_config_fingerprint = legacy_frontend_config_fingerprint
    promoted_frontend_fingerprint = legacy_frontend_fingerprint
    ranking_metrics = dict(legacy_hash_summary["ranking_metrics"])
    if _has_temporal_feature_payload(bundle.canonical_records):
        temporal_frontend = build_temporal_frontend_encoding(bundle.canonical_records)
        frontend_encoding_payload = dict(temporal_frontend["encoder"])
        temporal_frontend_summary = dict(temporal_frontend["summary"])
        materialization_mode = "temporal_threshold_waveform_bundle"
        frontend_config_fingerprint = compute_json_sha256(
            {
                "canonical_frontend_path": canonical_frontend_path,
                "frontend_encoder_fingerprint": frontend_encoding_payload["encoder_fingerprint"],
                "frontend_input_bundle_fingerprint": frontend_input.bundle_fingerprint,
                "include_test_metrics": frontend_input.include_test_metrics,
                "resolved_genome_identity": (
                    frontend_input.resolved_genome_identity.to_dict()
                    if frontend_input.resolved_genome_identity is not None
                    else None
                ),
            }
        )
        promoted_frontend_fingerprint = compute_json_sha256(
            {
                "frontend_config_fingerprint": frontend_config_fingerprint,
                "frontend_input_id": frontend_input_id,
                "frontend_kind": frontend_encoding_payload["frontend_kind"],
                "source_bundle_fingerprint": frontend_input.bundle_fingerprint,
                "stage_key": request.stage_key.value,
                "temporal_summary": temporal_frontend_summary,
                "waveform_summary": summary,
            }
        )
        ranking_metrics = dict(temporal_frontend["ranking_metrics"])
        comparison_to_legacy_hash = {
            "legacy_hash_baseline": legacy_hash_summary,
            "metric_deltas": {
                metric_name: round(
                    float(ranking_metrics.get(metric_name, 0.0))
                    - float(legacy_hash_summary["ranking_metrics"].get(metric_name, 0.0)),
                    6,
                )
                for metric_name in sorted(
                    set(ranking_metrics).union(legacy_hash_summary["ranking_metrics"])
                )
            },
        }

    best_frontend_ref = _write_support_artifact(
        Path(request.output_dir) / BEST_FRONTEND_ARTIFACT_NAME,
        kind=_BEST_FRONTEND_KIND,
        schema_version=_BEST_FRONTEND_SCHEMA_VERSION,
        payload={
            "materialization_mode": materialization_mode,
            "stage_key": request.stage_key.value,
            "stage_name": request.stage_name,
            "candidate_id": candidate_id,
            "frontend_input_id": frontend_input_id,
            "promoted_frontend_fingerprint": promoted_frontend_fingerprint,
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "canonical_frontend_path": canonical_frontend_path,
            "include_test_metrics": frontend_input.include_test_metrics,
            "source_bundle": {
                "bundle_dir": str(Path(frontend_input.bundle_dir).resolve()),
                "bundle_contract_path": str(Path(frontend_input.bundle_contract_path).resolve()),
                "bundle_fingerprint": frontend_input.bundle_fingerprint,
                "source_handoff_manifest_path": str(Path(frontend_input.source_handoff_manifest_path).resolve()),
            },
            "effective_engine_lean_config": _lean_training_config_payload(lean_training_config),
            "waveform_summary": summary,
            "frontend_encoding": frontend_encoding_payload,
            "temporal_frontend_summary": temporal_frontend_summary,
            "comparison_to_legacy_hash_frontend": comparison_to_legacy_hash,
            "resolved_genome_identity": (
                frontend_input.resolved_genome_identity.to_dict()
                if frontend_input.resolved_genome_identity is not None
                else None
            ),
        },
    )

    downstream_deep_input = bundle.materialize_deep_input_ref(
        output_dir=Path(request.output_dir) / "effective_deep_input",
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=promoted_frontend_fingerprint,
        include_test_metrics=frontend_input.include_test_metrics,
        resolved_genome_identity=frontend_input.resolved_genome_identity,
        source_promotion_artifact_ref=ArtifactRef(
            kind=PromotedFrontendWinner.KIND,
            schema_version=PromotedFrontendWinner.SCHEMA_VERSION,
            path=str(Path(promotion_artifact_path).resolve()),
        ),
        extra_handoff_fields=(
            {"frontend_encoding": frontend_encoding_payload}
            if frontend_encoding_payload is not None
            else None
        ),
        extra_contract_fields=(
            {"frontend_encoding": frontend_encoding_payload}
            if frontend_encoding_payload is not None
            else None
        ),
    )

    candidate_report_ref = _write_support_artifact(
        Path(request.output_dir) / FRONTEND_CANDIDATE_REPORT_ARTIFACT_NAME,
        kind=_FRONTEND_CANDIDATE_REPORT_KIND,
        schema_version=_FRONTEND_CANDIDATE_REPORT_SCHEMA_VERSION,
        payload={
            "materialization_mode": materialization_mode,
            "stage_key": request.stage_key.value,
            "stage_name": request.stage_name,
            "candidate_id": candidate_id,
            "frontend_input_id": frontend_input_id,
            "promoted_frontend_fingerprint": promoted_frontend_fingerprint,
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "ranking_metrics": ranking_metrics,
            "ranking_metrics_basis": ranking_metrics_basis,
            "effective_engine_lean_config": _lean_training_config_payload(lean_training_config),
            "waveform_summary": summary,
            "frontend_encoding": frontend_encoding_payload,
            "temporal_frontend_summary": temporal_frontend_summary,
            "comparison_to_legacy_hash_frontend": comparison_to_legacy_hash,
            "best_frontend_artifact_ref": best_frontend_ref.to_dict(),
            "downstream_deep_input": downstream_deep_input.to_dict(),
        },
    )

    return MaterializedWaveformFrontend(
        promoted_candidate=FrontendPromotionCandidate(
            candidate_id=candidate_id,
            candidate_order=1,
            promoted_frontend_fingerprint=promoted_frontend_fingerprint,
            ranking_eligible=True,
            ranking_metrics=ranking_metrics,
            candidate_report_ref=candidate_report_ref,
            best_frontend_artifact_ref=best_frontend_ref,
            frontend_config_fingerprint=frontend_config_fingerprint,
        ),
        downstream_deep_input=downstream_deep_input,
    )


def _lean_training_config_payload(
    lean_training_config: LeanTrainingConfig | None,
) -> dict[str, object]:
    if lean_training_config is None:
        return {}
    return {
        "backend": lean_training_config.backend,
        "allow_backend_fallback": lean_training_config.allow_backend_fallback,
    }


def _load_waveform_dataset_bundle(frontend_input: FrontendInput) -> WaveformDatasetBundle:
    bundle_dir = _require_existing_dir(
        frontend_input.bundle_dir,
        field_name="FrontendInput.bundle_dir",
    )
    bundle_contract_path = _require_existing_file(
        frontend_input.bundle_contract_path,
        field_name="FrontendInput.bundle_contract_path",
    )
    handoff_manifest_path = _require_existing_file(
        frontend_input.source_handoff_manifest_path,
        field_name="FrontendInput.source_handoff_manifest_path",
    )
    contract_payload = _load_json_mapping(
        bundle_contract_path,
        field_name="FrontendInput.bundle_contract_path",
    )
    handoff_payload = _load_json_mapping(
        handoff_manifest_path,
        field_name="FrontendInput.source_handoff_manifest_path",
    )

    if contract_payload.get("kind") != WAVEFORM_SOURCE_BUNDLE_KIND:
        raise ContractValidationError(
            "`FrontendInput.bundle_contract_path` must reference a generic V3 waveform source bundle."
        )
    if contract_payload.get("schema_version") != WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION:
        raise ContractValidationError(
            "`FrontendInput.bundle_contract_path` has an unsupported waveform source bundle schema version."
        )
    if handoff_payload.get("kind") != WAVEFORM_SOURCE_HANDOFF_KIND:
        raise ContractValidationError(
            "`FrontendInput.source_handoff_manifest_path` must reference a generic V3 waveform handoff manifest."
        )
    if handoff_payload.get("schema_version") != WAVEFORM_SOURCE_HANDOFF_SCHEMA_VERSION:
        raise ContractValidationError(
            "`FrontendInput.source_handoff_manifest_path` has an unsupported waveform handoff schema version."
        )

    resolved_handoff_path = str(handoff_manifest_path.resolve())
    if contract_payload.get("handoff_manifest_path") != resolved_handoff_path:
        raise ContractValidationError(
            "`FrontendInput.source_handoff_manifest_path` must match the waveform bundle contract lineage."
        )
    if compute_json_sha256(contract_payload) != frontend_input.bundle_fingerprint:
        raise ContractValidationError(
            "`FrontendInput.bundle_fingerprint` must match the referenced waveform bundle contract."
        )
    if contract_payload.get("handoff_manifest_sha256") != compute_json_sha256(handoff_payload):
        raise ContractValidationError(
            "Waveform bundle handoff manifest digest does not match the contract lineage."
        )

    raw_records = handoff_payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ContractValidationError(
            "Waveform bundle handoff manifests must include a `records` sequence."
        )
    canonical_records = tuple(_validate_record_payload(record) for record in raw_records)
    if not canonical_records:
        raise ContractValidationError(
            "Waveform-backed frontend materialization requires at least one canonical record."
        )

    source_record_ids = tuple(record["source_record_id"] for record in canonical_records)
    if tuple(contract_payload.get("source_record_ids", ())) != source_record_ids:
        raise ContractValidationError(
            "Waveform bundle contract `source_record_ids` must match the handoff manifest records."
        )
    if int(contract_payload.get("record_count", 0)) != len(canonical_records):
        raise ContractValidationError(
            "Waveform bundle contract `record_count` must match the handoff manifest records."
        )

    state_labels = tuple(contract_payload.get("state_labels", ()))
    channel_names = tuple(contract_payload.get("channel_names", ()))
    return WaveformDatasetBundle(
        bundle_dir=bundle_dir,
        bundle_contract_path=bundle_contract_path,
        handoff_manifest_path=handoff_manifest_path,
        bundle_fingerprint=frontend_input.bundle_fingerprint,
        record_count=len(canonical_records),
        source_record_ids=source_record_ids,
        state_labels=state_labels,
        channel_names=channel_names,
        dataset_id=_coerce_optional_string(contract_payload.get("dataset_id")),
        adapter_profile_id=_coerce_optional_string(contract_payload.get("adapter_profile_id")),
        canonical_records=canonical_records,
    )


def _validate_record_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Waveform bundle records must be JSON objects.")
    context = payload.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise ContractValidationError("Waveform bundle record `context` values must be JSON objects.")
    record = WaveformDatasetRecord(
        source_record_id=payload.get("source_record_id"),
        split=payload.get("split"),
        state_label=payload.get("state_label"),
        waveforms=payload.get("waveforms", {}),
        label_metadata=payload.get("label_metadata", {}),
        sampling_hz=context.get("sampling_hz"),
        rpm=context.get("rpm"),
        operating_condition=context.get("operating_condition"),
        context_metadata=context.get("metadata", {}),
        lineage_metadata=payload.get("lineage_metadata", {}),
    )
    for channel_name, waveform_ref in record.waveforms.items():
        if waveform_ref.waveform_path is None:
            continue
        waveform_path = Path(waveform_ref.waveform_path)
        if not waveform_path.is_file():
            raise ContractValidationError(
                "Waveform-backed frontend materialization requires existing `waveform_path` "
                f"values; missing file for record `{record.source_record_id}` channel `{channel_name}`."
            )
    return record.to_dict()


def _summarize_canonical_records(
    canonical_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    state_counts = Counter(str(record["state_label"]) for record in canonical_records)
    split_counts = Counter(str(record["split"]) for record in canonical_records)
    channel_counts = tuple(len(dict(record["waveforms"])) for record in canonical_records)
    channel_names = sorted(
        {
            channel_name
            for record in canonical_records
            for channel_name in dict(record["waveforms"]).keys()
        }
    )
    total_waveform_refs = sum(channel_counts)
    accessible_waveform_refs = 0
    payload_backed_waveforms = 0
    file_backed_waveforms = 0
    for record in canonical_records:
        for waveform_payload in dict(record["waveforms"]).values():
            waveform_payload_mapping = dict(waveform_payload)
            if waveform_payload_mapping.get("waveform_path") is not None:
                file_backed_waveforms += 1
                accessible_waveform_refs += 1
            elif waveform_payload_mapping.get("waveform_payload_ref") is not None:
                payload_backed_waveforms += 1
                accessible_waveform_refs += 1

    return {
        "record_count": len(canonical_records),
        "state_labels": sorted(state_counts),
        "state_counts": dict(sorted(state_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "channel_names": channel_names,
        "waveform_reference_counts": {
            "total": total_waveform_refs,
            "accessible": accessible_waveform_refs,
            "file_backed": file_backed_waveforms,
            "payload_backed": payload_backed_waveforms,
        },
        "channel_consistency": {
            "min_channels_per_record": min(channel_counts),
            "max_channels_per_record": max(channel_counts),
        },
    }


def _build_ranking_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    record_count = float(int(summary["record_count"]))
    state_counts = dict(summary["state_counts"])
    split_counts = dict(summary["split_counts"])
    waveform_reference_counts = dict(summary["waveform_reference_counts"])
    channel_consistency = dict(summary["channel_consistency"])

    min_state_support = min(int(value) for value in state_counts.values())
    max_state_support = max(int(value) for value in state_counts.values())
    min_channels_per_record = int(channel_consistency["min_channels_per_record"])
    max_channels_per_record = int(channel_consistency["max_channels_per_record"])
    total_waveform_refs = max(1, int(waveform_reference_counts["total"]))
    accessible_waveform_refs = int(waveform_reference_counts["accessible"])

    return {
        "healthy_unhealthy_margin": round(min_state_support / max_state_support, 6),
        "inter_class_separation": round(len(state_counts) / record_count, 6),
        "intra_class_compactness": round(min_channels_per_record / max_channels_per_record, 6),
        "bit_balance": round(accessible_waveform_refs / total_waveform_refs, 6),
        "bit_stability": round(min(1.0, len(split_counts) / _CANONICAL_SPLIT_TARGET_COUNT), 6),
    }


def _write_support_artifact(
    path: Path,
    *,
    kind: str,
    schema_version: str,
    payload: Mapping[str, object],
) -> ArtifactRef:
    artifact_path = path.resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    materialized_payload = {
        "kind": kind,
        "schema_version": schema_version,
        **dict(payload),
    }
    artifact_path.write_text(
        json.dumps(materialized_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ArtifactRef(
        kind=kind,
        schema_version=schema_version,
        path=str(artifact_path),
        sha256=compute_file_sha256(artifact_path),
    )


def _require_existing_file(value: str, *, field_name: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise ContractValidationError(f"`{field_name}` must point to an existing file.")
    return path


def _require_existing_dir(value: str, *, field_name: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise ContractValidationError(f"`{field_name}` must point to an existing directory.")
    return path


def _load_json_mapping(path: Path, *, field_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"`{field_name}` must contain valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{field_name}` must deserialize to a JSON object.")
    return payload


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractValidationError("Waveform bundle metadata values must be strings when present.")
    return value


def _has_temporal_feature_payload(canonical_records: Sequence[Mapping[str, object]]) -> bool:
    if not canonical_records:
        return False
    for record in canonical_records:
        context = record.get("context")
        if isinstance(context, Mapping):
            metadata = context.get("metadata", {})
            if isinstance(metadata, Mapping) and isinstance(metadata.get("temporal_features"), Mapping):
                continue
        context_metadata = record.get("context_metadata", {})
        if isinstance(context_metadata, Mapping) and isinstance(context_metadata.get("temporal_features"), Mapping):
            continue
        return False
    return True


__all__ = [
    "BEST_FRONTEND_ARTIFACT_NAME",
    "FRONTEND_CANDIDATE_REPORT_ARTIFACT_NAME",
    "MaterializedWaveformFrontend",
    "materialize_waveform_frontend",
]
