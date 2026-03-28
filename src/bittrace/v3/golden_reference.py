"""Independent frozen S6 golden/reference generation for canonical V3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
from types import MappingProxyType

from bittrace.v3.artifacts import compute_json_sha256, load_json_artifact_ref
from bittrace.v3.contracts import (
    ArtifactRef,
    ContractValidationError,
    DeepAnchorArtifact,
    FreezeExportManifest,
    FrontendExportReference,
    PassFail,
    WaveformDatasetRecord,
)
from bittrace.v3.frontend_encoding import encode_frontend_record


_REFERENCE_ROW_FORMAT = "packed_int_lsb0"
_REFERENCE_BIT_LENGTH = 64
_MAX_EXACT_MEDOID_COMBINATIONS = 4096


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json_mapping(path: str | Path, *, field_name: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"`{field_name}` must deserialize to a JSON object.")
    return dict(payload)


def _coerce_reference_record(
    record: WaveformDatasetRecord | Mapping[str, object],
) -> WaveformDatasetRecord:
    if isinstance(record, WaveformDatasetRecord):
        return record
    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Frozen golden/reference generation expects a `WaveformDatasetRecord` or JSON-object payload."
        )
    if "context" in record:
        context = record.get("context", {})
        if context is None:
            context = {}
        if not isinstance(context, Mapping):
            raise ContractValidationError("Frozen golden/reference record `context` must be a JSON object.")
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
    context_metadata = record.get("context_metadata", {})
    if context_metadata is None:
        context_metadata = {}
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
        sampling_hz=record.get("sampling_hz"),
        rpm=record.get("rpm"),
        operating_condition=record.get("operating_condition"),
        context_metadata=context_metadata,
        lineage_metadata=lineage_metadata,
    )


def _reference_encoding_payload(
    record: WaveformDatasetRecord,
    *,
    dataset_id: str | None,
    adapter_profile_id: str | None,
    frontend_input_id: str,
    frontend_fingerprint: str,
) -> dict[str, object]:
    return {
        "adapter_profile_id": adapter_profile_id,
        "context_metadata": dict(record.context_metadata),
        "dataset_id": dataset_id,
        "frontend_fingerprint": frontend_fingerprint,
        "frontend_input_id": frontend_input_id,
        "lineage_metadata": dict(record.lineage_metadata),
        "operating_condition": record.operating_condition,
        "rpm": record.rpm,
        "sampling_hz": record.sampling_hz,
        "waveforms": {
            channel_name: waveform_ref.to_dict()
            for channel_name, waveform_ref in sorted(record.waveforms.items())
        },
    }


def _pack_reference_row(payload: Mapping[str, object]) -> tuple[int, str]:
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False), digest.hex()


def _bit_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _select_reference_medoids(rows: Sequence[int], k_per_class: int) -> tuple[int, ...]:
    unique_rows = tuple(sorted(set(rows)))
    if not unique_rows:
        return ()
    selected_k = min(max(1, int(k_per_class)), len(unique_rows))
    if selected_k == 1:
        return (_select_reference_single_medoid(unique_rows),)
    if comb(len(unique_rows), selected_k) <= _MAX_EXACT_MEDOID_COMBINATIONS:
        return _select_reference_exact_medoids(unique_rows, selected_k)
    return _select_reference_greedy_medoids(unique_rows, selected_k)


def _select_reference_single_medoid(rows: Sequence[int]) -> int:
    return min(
        rows,
        key=lambda row: (
            sum(_bit_distance(row, other) for other in rows),
            row,
        ),
    )


def _select_reference_exact_medoids(rows: Sequence[int], selected_k: int) -> tuple[int, ...]:
    best_rows: tuple[int, ...] | None = None
    best_cost: int | None = None
    for candidate_rows in itertools.combinations(rows, selected_k):
        total_cost = sum(
            min(_bit_distance(row, medoid) for medoid in candidate_rows)
            for row in rows
        )
        if best_cost is None or total_cost < best_cost or (
            total_cost == best_cost and candidate_rows < best_rows
        ):
            best_rows = candidate_rows
            best_cost = total_cost
    if best_rows is None:
        raise ContractValidationError("Frozen golden/reference generation failed to select exact medoids.")
    return best_rows


def _select_reference_greedy_medoids(rows: Sequence[int], selected_k: int) -> tuple[int, ...]:
    selected = [_select_reference_single_medoid(rows)]
    while len(selected) < selected_k:
        remaining = [row for row in rows if row not in selected]
        next_row = max(
            remaining,
            key=lambda row: (
                min(_bit_distance(row, chosen) for chosen in selected),
                -row,
            ),
        )
        selected.append(next_row)
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class GoldenFrontendOutput:
    """Deterministic expected frontend deploy output for one canonical record."""

    payload: Mapping[str, object]
    packed_row_int: int
    packed_row_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class GoldenDeployOutput:
    """Deterministic expected deep deploy output for one canonical record."""

    payload: Mapping[str, object]
    predicted_class: str | None
    reject: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class FrozenS6GoldenReference:
    """Independent frozen S6 reference surface backed by canonical exported artifacts."""

    deep_anchor_artifact: DeepAnchorArtifact
    frontend_export_reference: FrontendExportReference
    freeze_export_manifest: FreezeExportManifest | None = None
    _deep_input_contract: Mapping[str, object] = field(init=False, repr=False)
    _deep_input_records: tuple[WaveformDatasetRecord, ...] = field(init=False, repr=False)
    _state_labels: tuple[str, ...] = field(init=False, repr=False)
    _prototype_rows_by_class: Mapping[str, tuple[int, ...]] = field(init=False, repr=False)
    _prototype_record_ids_by_class: Mapping[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.deep_anchor_artifact.pass_fail != PassFail.PASS:
            raise ContractValidationError("Frozen golden/reference requires a passing Deep anchor artifact.")
        if self.frontend_export_reference.pass_fail != PassFail.PASS:
            raise ContractValidationError(
                "Frozen golden/reference requires a passing frontend export reference."
            )
        if self.deep_anchor_artifact.deployed_winner is None:
            raise ContractValidationError(
                "Frozen golden/reference requires an emitted deployed winner."
            )
        if self.freeze_export_manifest is not None:
            if self.freeze_export_manifest.pass_fail != PassFail.PASS:
                raise ContractValidationError(
                    "Frozen golden/reference requires a passing freeze/export manifest."
                )
            if (
                self.freeze_export_manifest.anchor_artifact_ref
                != self.frontend_export_reference.deep_anchor_artifact_ref
            ):
                raise ContractValidationError(
                    "Freeze/export manifest and frontend export reference must point to the same Deep anchor artifact."
                )

        deep_input_contract, deep_input_records = _load_reference_deep_input_bundle(
            self.frontend_export_reference.frontend_lineage.bundle_contract_path,
            self.frontend_export_reference.frontend_lineage.handoff_manifest_path,
            expected_bundle_fingerprint=self.frontend_export_reference.frontend_lineage.bundle_fingerprint,
        )
        object.__setattr__(self, "_deep_input_contract", MappingProxyType(dict(deep_input_contract)))
        object.__setattr__(self, "_deep_input_records", tuple(deep_input_records))

        state_labels = tuple(str(label) for label in deep_input_contract.get("state_labels", ()))
        if not state_labels:
            state_labels = tuple(sorted({record.state_label for record in deep_input_records}))
        object.__setattr__(self, "_state_labels", state_labels)

        prototype_rows_by_class: dict[str, tuple[int, ...]] = {}
        prototype_record_ids_by_class: dict[str, tuple[str, ...]] = {}
        training_records = tuple(
            record for record in deep_input_records if record.split.lower() == "train"
        ) or deep_input_records
        encoded_training_rows = {
            record.source_record_id: self.expected_frontend_output(record)
            for record in training_records
        }
        selected_k = self.deep_anchor_artifact.deployed_winner.selected_k_per_class
        for label in self._state_labels:
            label_records = tuple(record for record in training_records if record.state_label == label)
            rows = tuple(
                encoded_training_rows[record.source_record_id].packed_row_int
                for record in label_records
            )
            medoids = _select_reference_medoids(rows, selected_k)
            prototype_rows_by_class[label] = medoids
            prototype_record_ids_by_class[label] = tuple(
                record.source_record_id
                for medoid in medoids
                for record in label_records
                if encoded_training_rows[record.source_record_id].packed_row_int == medoid
            )
        object.__setattr__(self, "_prototype_rows_by_class", MappingProxyType(prototype_rows_by_class))
        object.__setattr__(
            self,
            "_prototype_record_ids_by_class",
            MappingProxyType(prototype_record_ids_by_class),
        )

    @property
    def state_labels(self) -> tuple[str, ...]:
        return self._state_labels

    @property
    def selected_k_per_class(self) -> int:
        deployed_winner = self.deep_anchor_artifact.deployed_winner
        if deployed_winner is None:
            raise ContractValidationError("Frozen golden/reference requires an emitted deployed winner.")
        return deployed_winner.selected_k_per_class

    def expected_frontend_output(
        self,
        canonical_input: WaveformDatasetRecord | Mapping[str, object],
    ) -> GoldenFrontendOutput:
        record = _coerce_reference_record(canonical_input)
        frontend_input_id = self.frontend_export_reference.frontend_lineage.frontend_input_id
        frontend_fingerprint = self.frontend_export_reference.frontend_lineage.frontend_fingerprint
        if frontend_input_id is None or frontend_fingerprint is None:
            raise ContractValidationError("Frozen golden/reference requires frontend lineage identifiers.")
        encoded = encode_frontend_record(
            record,
            dataset_id=_optional_string(self._deep_input_contract.get("dataset_id")),
            adapter_profile_id=_optional_string(self._deep_input_contract.get("adapter_profile_id")),
            frontend_input_id=frontend_input_id,
            frontend_fingerprint=frontend_fingerprint,
            contract_payload=self._deep_input_contract,
        )
        return GoldenFrontendOutput(
            payload=encoded.payload,
            packed_row_int=encoded.packed_row_int,
            packed_row_sha256=encoded.packed_row_sha256,
        )

    def expected_deep_output(
        self,
        *,
        canonical_input: WaveformDatasetRecord | Mapping[str, object] | None = None,
        frontend_output: GoldenFrontendOutput | Mapping[str, object] | None = None,
    ) -> GoldenDeployOutput:
        resolved_frontend = _coerce_expected_frontend_output(
            self.expected_frontend_output(canonical_input) if frontend_output is None else frontend_output
        )
        class_distances = {
            label: min(
                (_bit_distance(resolved_frontend.packed_row_int, prototype) for prototype in prototypes),
                default=None,
            )
            for label, prototypes in self._prototype_rows_by_class.items()
        }
        available_distances = {
            label: distance for label, distance in class_distances.items() if distance is not None
        }
        if not available_distances:
            payload = {
                "bit_length": _REFERENCE_BIT_LENGTH,
                "decision": "reject",
                "deploy_path": _reference_deploy_path(self.freeze_export_manifest),
                "frontend_fingerprint": self.frontend_export_reference.frontend_lineage.frontend_fingerprint,
                "frontend_input_id": self.frontend_export_reference.frontend_lineage.frontend_input_id,
                "reason": "no_prototypes_available",
                "reject": True,
                "row_format": _REFERENCE_ROW_FORMAT,
                "selected_k_per_class": self.selected_k_per_class,
                "source_record_id": resolved_frontend.payload["source_record_id"],
            }
            return GoldenDeployOutput(payload=payload, predicted_class=None, reject=True)

        label_order = {label: index for index, label in enumerate(self._state_labels)}
        ranked_labels = sorted(
            available_distances.items(),
            key=lambda item: (item[1], label_order.get(item[0], len(label_order)), item[0]),
        )
        winning_class, winning_distance = ranked_labels[0]
        runner_up_distance = ranked_labels[1][1] if len(ranked_labels) > 1 else None
        payload = {
            "bit_length": _REFERENCE_BIT_LENGTH,
            "class_distances": {
                label: available_distances[label]
                for label in self._state_labels
                if label in available_distances
            },
            "decision": "classify",
            "deploy_path": _reference_deploy_path(self.freeze_export_manifest),
            "distance_margin": (
                None if runner_up_distance is None else runner_up_distance - winning_distance
            ),
            "frontend_fingerprint": self.frontend_export_reference.frontend_lineage.frontend_fingerprint,
            "frontend_input_id": self.frontend_export_reference.frontend_lineage.frontend_input_id,
            "packed_row_hex": resolved_frontend.payload["packed_row_hex"],
            "packed_row_int": resolved_frontend.payload["packed_row_int"],
            "prototype_record_ids": {
                label: list(self._prototype_record_ids_by_class.get(label, ()))
                for label in self._state_labels
                if label in available_distances
            },
            "reject": False,
            "row_format": _REFERENCE_ROW_FORMAT,
            "runner_up_distance": runner_up_distance,
            "selected_k_per_class": self.selected_k_per_class,
            "source_record_id": resolved_frontend.payload["source_record_id"],
            "winning_class": winning_class,
            "winning_distance": winning_distance,
        }
        return GoldenDeployOutput(payload=payload, predicted_class=winning_class, reject=False)

    def expected_end_to_end_output(
        self,
        canonical_input: WaveformDatasetRecord | Mapping[str, object],
    ) -> GoldenDeployOutput:
        return self.expected_deep_output(canonical_input=canonical_input)


def load_frozen_s6_golden_reference(
    *,
    freeze_export_manifest_ref: ArtifactRef | None = None,
    deep_anchor_artifact_ref: ArtifactRef | None = None,
    frontend_export_reference_ref: ArtifactRef | None = None,
) -> FrozenS6GoldenReference:
    """Load the independent golden/reference surface from emitted frozen S6 artifacts."""

    manifest: FreezeExportManifest | None = None
    if freeze_export_manifest_ref is not None:
        loaded_manifest = load_json_artifact_ref(freeze_export_manifest_ref)
        if not isinstance(loaded_manifest, FreezeExportManifest):
            raise ContractValidationError(
                "`freeze_export_manifest_ref` must resolve to a `FreezeExportManifest`."
            )
        manifest = loaded_manifest
        if deep_anchor_artifact_ref is None:
            deep_anchor_artifact_ref = manifest.anchor_artifact_ref
        if frontend_export_reference_ref is None:
            frontend_export_reference_ref = manifest.frontend_export_reference_ref

    if deep_anchor_artifact_ref is None or frontend_export_reference_ref is None:
        raise ContractValidationError(
            "Frozen golden/reference loading requires the emitted Deep anchor and frontend export references."
        )

    deep_anchor = load_json_artifact_ref(deep_anchor_artifact_ref)
    if not isinstance(deep_anchor, DeepAnchorArtifact):
        raise ContractValidationError(
            "`deep_anchor_artifact_ref` must resolve to a `DeepAnchorArtifact`."
        )
    frontend_export_reference = load_json_artifact_ref(frontend_export_reference_ref)
    if not isinstance(frontend_export_reference, FrontendExportReference):
        raise ContractValidationError(
            "`frontend_export_reference_ref` must resolve to a `FrontendExportReference`."
        )
    if frontend_export_reference.deep_anchor_artifact_ref != deep_anchor_artifact_ref:
        raise ContractValidationError(
            "Frozen golden/reference artifact lineage mismatch: frontend export reference points to a different Deep anchor."
        )

    return FrozenS6GoldenReference(
        deep_anchor_artifact=deep_anchor,
        frontend_export_reference=frontend_export_reference,
        freeze_export_manifest=manifest,
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractValidationError(
            "Frozen golden/reference contract fields must preserve string values."
        )
    return value


def _load_reference_deep_input_bundle(
    bundle_contract_path: str | Path,
    handoff_manifest_path: str | Path,
    *,
    expected_bundle_fingerprint: str,
) -> tuple[dict[str, object], tuple[WaveformDatasetRecord, ...]]:
    contract_payload = _load_json_mapping(
        bundle_contract_path,
        field_name="FrozenGoldenReference.deep_input_contract",
    )
    handoff_payload = _load_json_mapping(
        handoff_manifest_path,
        field_name="FrozenGoldenReference.deep_input_handoff",
    )
    expected_handoff_path = str(Path(handoff_manifest_path).resolve())
    if contract_payload.get("handoff_manifest_path") != expected_handoff_path:
        raise ContractValidationError(
            "Frozen golden/reference deep-input contract lineage does not match the emitted handoff manifest."
        )
    if compute_json_sha256(contract_payload) != expected_bundle_fingerprint:
        raise ContractValidationError(
            "Frozen golden/reference deep-input contract fingerprint does not match the emitted frontend lineage."
        )
    if contract_payload.get("handoff_manifest_sha256") != compute_json_sha256(handoff_payload):
        raise ContractValidationError(
            "Frozen golden/reference deep-input handoff digest does not match the emitted contract lineage."
        )
    raw_records = handoff_payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ContractValidationError(
            "Frozen golden/reference deep-input handoff must include a `records` sequence."
        )
    records = tuple(_coerce_reference_record(record) for record in raw_records)
    if not records:
        raise ContractValidationError(
            "Frozen golden/reference deep-input handoff must contain at least one record."
        )
    return contract_payload, records


def _reference_deploy_path(manifest: FreezeExportManifest | None) -> str:
    if manifest is None or manifest.deploy_runtime is None:
        return "pure_symbolic"
    return manifest.deploy_runtime.deploy_path


def _coerce_expected_frontend_output(
    output: GoldenFrontendOutput | Mapping[str, object],
) -> GoldenFrontendOutput:
    if isinstance(output, GoldenFrontendOutput):
        return output
    if not isinstance(output, Mapping):
        raise ContractValidationError(
            "Frozen golden/reference deep output expects a `GoldenFrontendOutput` or JSON-object payload."
        )
    packed_row_int = output.get("packed_row_int")
    packed_row_sha256 = output.get("encoding_sha256")
    if not isinstance(packed_row_int, int) or isinstance(packed_row_int, bool):
        raise ContractValidationError(
            "Frozen golden/reference frontend output must include integer `packed_row_int`."
        )
    if not isinstance(packed_row_sha256, str) or packed_row_sha256 == "":
        raise ContractValidationError(
            "Frozen golden/reference frontend output must include string `encoding_sha256`."
        )
    return GoldenFrontendOutput(
        payload=dict(output),
        packed_row_int=packed_row_int,
        packed_row_sha256=packed_row_sha256,
    )


__all__ = [
    "FrozenS6GoldenReference",
    "GoldenDeployOutput",
    "GoldenFrontendOutput",
    "load_frozen_s6_golden_reference",
]
