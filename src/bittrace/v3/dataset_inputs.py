"""Generic V3 dataset-ingestion helpers for waveform-backed canonical inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from bittrace.v3.artifacts import compute_json_sha256
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepInputRef,
    FrontendInput,
    ResolvedGenomeIdentity,
    WaveformDatasetRecord,
)


WAVEFORM_SOURCE_HANDOFF_KIND = "bittrace_v3_waveform_source_handoff_manifest"
WAVEFORM_SOURCE_HANDOFF_SCHEMA_VERSION = "bittrace-v3-waveform-source-handoff-1"
WAVEFORM_SOURCE_BUNDLE_KIND = "bittrace_v3_waveform_source_bundle"
WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION = "bittrace-v3-waveform-source-bundle-1"
WAVEFORM_DEEP_INPUT_HANDOFF_KIND = "bittrace_v3_waveform_deep_input_handoff"
WAVEFORM_DEEP_INPUT_HANDOFF_SCHEMA_VERSION = "bittrace-v3-waveform-deep-input-handoff-1"
WAVEFORM_DEEP_INPUT_BUNDLE_KIND = "bittrace_v3_waveform_deep_input_bundle"
WAVEFORM_DEEP_INPUT_BUNDLE_SCHEMA_VERSION = "bittrace-v3-waveform-deep-input-bundle-1"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_waveform_dataset_record(value: WaveformDatasetRecord | Mapping[str, object]) -> WaveformDatasetRecord:
    if isinstance(value, WaveformDatasetRecord):
        return value
    if isinstance(value, Mapping):
        return WaveformDatasetRecord.from_dict(value)
    raise ContractValidationError(
        "Waveform dataset ingestion records must be `WaveformDatasetRecord` instances or mappings."
    )


def _canonical_record_payload(record: WaveformDatasetRecord) -> dict[str, object]:
    context: dict[str, object] = {}
    if record.sampling_hz is not None:
        context["sampling_hz"] = record.sampling_hz
    if record.rpm is not None:
        context["rpm"] = record.rpm
    if record.operating_condition is not None:
        context["operating_condition"] = record.operating_condition
    if record.context_metadata:
        context["metadata"] = dict(record.context_metadata)

    payload: dict[str, object] = {
        "source_record_id": record.source_record_id,
        "split": record.split,
        "state_label": record.state_label,
        "waveforms": {
            channel_name: waveform_ref.to_dict()
            for channel_name, waveform_ref in sorted(record.waveforms.items())
        },
    }
    if record.label_metadata:
        payload["label_metadata"] = dict(record.label_metadata)
    if context:
        payload["context"] = context
    if record.lineage_metadata:
        payload["lineage_metadata"] = dict(record.lineage_metadata)
    return payload


@dataclass(frozen=True, slots=True)
class WaveformDatasetBundle:
    """Materialized shared V3 source bundle for waveform-backed dataset records."""

    bundle_dir: Path
    bundle_contract_path: Path
    handoff_manifest_path: Path
    bundle_fingerprint: str
    record_count: int
    source_record_ids: tuple[str, ...]
    state_labels: tuple[str, ...]
    channel_names: tuple[str, ...]
    dataset_id: str | None
    adapter_profile_id: str | None
    canonical_records: tuple[dict[str, object], ...]

    def to_frontend_input(
        self,
        *,
        include_test_metrics: bool,
        resolved_genome_identity: ResolvedGenomeIdentity | None = None,
    ) -> FrontendInput:
        return FrontendInput(
            bundle_dir=str(self.bundle_dir.resolve()),
            bundle_contract_path=str(self.bundle_contract_path.resolve()),
            bundle_fingerprint=self.bundle_fingerprint,
            source_handoff_manifest_path=str(self.handoff_manifest_path.resolve()),
            include_test_metrics=include_test_metrics,
            resolved_genome_identity=resolved_genome_identity,
            adapter_profile_id=self.adapter_profile_id,
        )

    def materialize_deep_input_ref(
        self,
        *,
        output_dir: str | Path,
        frontend_input_id: str,
        frontend_fingerprint: str,
        include_test_metrics: bool,
        resolved_genome_identity: ResolvedGenomeIdentity | None = None,
        source_promotion_artifact_ref: ArtifactRef | None = None,
        extra_handoff_fields: Mapping[str, object] | None = None,
        extra_contract_fields: Mapping[str, object] | None = None,
    ) -> DeepInputRef:
        deep_output_dir = Path(output_dir).resolve()
        handoff_manifest_path = deep_output_dir / "handoff.json"
        handoff_payload = {
            "kind": WAVEFORM_DEEP_INPUT_HANDOFF_KIND,
            "schema_version": WAVEFORM_DEEP_INPUT_HANDOFF_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "adapter_profile_id": self.adapter_profile_id,
            "frontend_input_id": frontend_input_id,
            "frontend_fingerprint": frontend_fingerprint,
            "source_bundle_fingerprint": self.bundle_fingerprint,
            "record_count": self.record_count,
            "records": list(self.canonical_records),
        }
        if extra_handoff_fields:
            handoff_payload.update(dict(extra_handoff_fields))
        _write_json(handoff_manifest_path, handoff_payload)

        bundle_contract_path = deep_output_dir / "contract.json"
        bundle_contract_payload = {
            "kind": WAVEFORM_DEEP_INPUT_BUNDLE_KIND,
            "schema_version": WAVEFORM_DEEP_INPUT_BUNDLE_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "adapter_profile_id": self.adapter_profile_id,
            "frontend_input_id": frontend_input_id,
            "frontend_fingerprint": frontend_fingerprint,
            "record_count": self.record_count,
            "source_record_ids": list(self.source_record_ids),
            "state_labels": list(self.state_labels),
            "channel_names": list(self.channel_names),
            "handoff_manifest_path": str(handoff_manifest_path.resolve()),
            "handoff_manifest_sha256": compute_json_sha256(handoff_payload),
        }
        if extra_contract_fields:
            bundle_contract_payload.update(dict(extra_contract_fields))
        _write_json(bundle_contract_path, bundle_contract_payload)
        bundle_fingerprint = compute_json_sha256(bundle_contract_payload)

        return DeepInputRef(
            bundle_dir=str(deep_output_dir),
            bundle_contract_path=str(bundle_contract_path.resolve()),
            bundle_fingerprint=bundle_fingerprint,
            handoff_manifest_path=str(handoff_manifest_path.resolve()),
            source_bundle_dir=str(self.bundle_dir.resolve()),
            source_bundle_contract_path=str(self.bundle_contract_path.resolve()),
            source_bundle_fingerprint=self.bundle_fingerprint,
            source_handoff_manifest_path=str(self.handoff_manifest_path.resolve()),
            include_test_metrics=include_test_metrics,
            frontend_input_id=frontend_input_id,
            frontend_fingerprint=frontend_fingerprint,
            resolved_genome_identity=resolved_genome_identity,
            source_promotion_artifact_ref=source_promotion_artifact_ref,
        )


def build_waveform_dataset_bundle(
    records: Iterable[WaveformDatasetRecord | Mapping[str, object]],
    *,
    output_dir: str | Path,
    dataset_id: str | None = None,
    adapter_profile_id: str | None = None,
) -> WaveformDatasetBundle:
    """Materialize a reusable V3 bundle from waveform-backed dataset records."""

    canonical_records = tuple(
        _canonical_record_payload(_coerce_waveform_dataset_record(record))
        for record in records
    )
    if not canonical_records:
        raise ContractValidationError("Cannot build a waveform dataset bundle from zero records.")

    source_record_ids = tuple(record["source_record_id"] for record in canonical_records)
    if len(set(source_record_ids)) != len(source_record_ids):
        raise ContractValidationError(
            "Waveform dataset bundles require unique `source_record_id` values."
        )

    output_path = Path(output_dir).resolve()
    handoff_manifest_path = output_path / "handoff.json"
    handoff_payload = {
        "kind": WAVEFORM_SOURCE_HANDOFF_KIND,
        "schema_version": WAVEFORM_SOURCE_HANDOFF_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "adapter_profile_id": adapter_profile_id,
        "record_count": len(canonical_records),
        "records": list(canonical_records),
    }
    _write_json(handoff_manifest_path, handoff_payload)

    state_labels = tuple(sorted({str(record["state_label"]) for record in canonical_records}))
    channel_names = tuple(
        sorted(
            {
                channel_name
                for record in canonical_records
                for channel_name in dict(record["waveforms"]).keys()
            }
        )
    )
    bundle_contract_path = output_path / "contract.json"
    bundle_contract_payload = {
        "kind": WAVEFORM_SOURCE_BUNDLE_KIND,
        "schema_version": WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "adapter_profile_id": adapter_profile_id,
        "record_count": len(canonical_records),
        "source_record_ids": list(source_record_ids),
        "state_labels": list(state_labels),
        "channel_names": list(channel_names),
        "handoff_manifest_path": str(handoff_manifest_path.resolve()),
        "handoff_manifest_sha256": compute_json_sha256(handoff_payload),
    }
    _write_json(bundle_contract_path, bundle_contract_payload)

    return WaveformDatasetBundle(
        bundle_dir=output_path,
        bundle_contract_path=bundle_contract_path,
        handoff_manifest_path=handoff_manifest_path,
        bundle_fingerprint=compute_json_sha256(bundle_contract_payload),
        record_count=len(canonical_records),
        source_record_ids=source_record_ids,
        state_labels=state_labels,
        channel_names=channel_names,
        dataset_id=dataset_id,
        adapter_profile_id=adapter_profile_id,
        canonical_records=canonical_records,
    )


__all__ = [
    "WAVEFORM_DEEP_INPUT_BUNDLE_KIND",
    "WAVEFORM_DEEP_INPUT_BUNDLE_SCHEMA_VERSION",
    "WAVEFORM_DEEP_INPUT_HANDOFF_KIND",
    "WAVEFORM_DEEP_INPUT_HANDOFF_SCHEMA_VERSION",
    "WAVEFORM_SOURCE_BUNDLE_KIND",
    "WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION",
    "WAVEFORM_SOURCE_HANDOFF_KIND",
    "WAVEFORM_SOURCE_HANDOFF_SCHEMA_VERSION",
    "WaveformDatasetBundle",
    "build_waveform_dataset_bundle",
]
