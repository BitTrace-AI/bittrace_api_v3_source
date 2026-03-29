"""Shared frontend encoding helpers for canonical waveform-backed V3 records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from types import MappingProxyType

from bittrace.v3.artifacts import compute_json_sha256
from bittrace.v3.contracts import ContractValidationError, WaveformDatasetRecord


PACKED_ROW_FORMAT = "packed_int_lsb0"
PACKED_BIT_LENGTH = 64
LEGACY_HASH_FRONTEND_KIND = "canonical_waveform_hash_encoder"
TEMPORAL_THRESHOLD_FRONTEND_KIND = "temporal_threshold_encoder"
TEMPORAL_THRESHOLD_ENCODING_KIND = "bittrace_v3_temporal_threshold_encoding"
TEMPORAL_THRESHOLD_ENCODING_SCHEMA_VERSION = "bittrace-v3-temporal-threshold-encoding-1"
_TEMPORAL_FEATURE_KEY = "temporal_features"
_TEMPORAL_FEATURE_PRIORITY = (
    "variance",
    "rms",
    "mean_abs_deviation",
    "peak_to_peak",
    "max_delta",
    "count_above_threshold",
    "max_spike_above_mean",
    "slope",
    "diff_sign_change_rate",
    "delta_variance",
    "delta_rms",
    "mean",
    "median",
    "min",
    "max",
    "first_diff_abs_mean",
    "first_diff_mean",
    "cumulative_delta",
)
_LABEL_ORDER = {"healthy": 0, "unhealthy": 1}


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _legacy_hash_payload(
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


def _pack_hash_payload(payload: Mapping[str, object]) -> tuple[int, str]:
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False), digest.hex()


def _bit_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ContractValidationError("Temporal threshold encoding encountered zero denominator.")
    if numerator == 0:
        return 0
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    return sign * ((absolute + (denominator // 2)) // denominator)


@dataclass(frozen=True, slots=True)
class FrontendEncodingResult:
    payload: Mapping[str, object]
    packed_row_int: int
    packed_row_sha256: str
    bit_length: int
    row_format: str
    bit_feature_names: tuple[str, ...]
    frontend_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class _TemporalRecord:
    source_record_id: str
    split: str
    state_label: str
    feature_names: tuple[str, ...]
    feature_values: tuple[int, ...]
    config_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class _EncodedRow:
    source_record_id: str
    split: str
    state_label: str
    packed_row_int: int


def build_temporal_frontend_encoding(
    canonical_records: Sequence[Mapping[str, object]],
    *,
    bit_length: int = PACKED_BIT_LENGTH,
) -> dict[str, object]:
    temporal_records = tuple(_temporal_record_from_payload(record) for record in canonical_records)
    if not temporal_records:
        raise ContractValidationError("Temporal threshold encoding requires at least one canonical record.")
    feature_names = temporal_records[0].feature_names
    if any(record.feature_names != feature_names for record in temporal_records[1:]):
        raise ContractValidationError("Temporal threshold encoding requires an identical feature schema for every record.")
    if len(feature_names) * 2 > bit_length:
        raise ContractValidationError(
            f"Temporal threshold encoding requires at most {bit_length // 2} features for a {bit_length}-bit bundle."
        )

    threshold_counts = _allocate_threshold_counts(feature_names, bit_length=bit_length)
    train_records = tuple(record for record in temporal_records if record.split == "train") or temporal_records
    threshold_specs = [
        {
            "feature_name": feature_name,
            "threshold_count": threshold_counts[feature_name],
            "thresholds": _quantile_thresholds(
                [record.feature_values[index] for record in train_records],
                threshold_counts[feature_name],
            ),
        }
        for index, feature_name in enumerate(feature_names)
    ]
    bit_feature_names = tuple(
        f"{spec['feature_name']}__t{threshold_index + 1}"
        for spec in threshold_specs
        for threshold_index in range(int(spec["threshold_count"]))
    )
    encoder_payload = {
        "kind": TEMPORAL_THRESHOLD_ENCODING_KIND,
        "schema_version": TEMPORAL_THRESHOLD_ENCODING_SCHEMA_VERSION,
        "frontend_kind": TEMPORAL_THRESHOLD_FRONTEND_KIND,
        "bit_length": bit_length,
        "row_format": PACKED_ROW_FORMAT,
        "feature_names": list(feature_names),
        "bit_feature_names": list(bit_feature_names),
        "feature_thresholds": threshold_specs,
        "config_fingerprints": sorted(
            {
                record.config_fingerprint
                for record in temporal_records
                if isinstance(record.config_fingerprint, str) and record.config_fingerprint != ""
            }
        ),
    }
    encoder_payload["encoder_fingerprint"] = compute_json_sha256(encoder_payload)

    encoded_rows = tuple(
        _EncodedRow(
            source_record_id=record.source_record_id,
            split=record.split,
            state_label=record.state_label,
            packed_row_int=_pack_temporal_feature_values(record.feature_values, encoder_payload=encoder_payload),
        )
        for record in temporal_records
    )
    summary = _summarize_encoded_rows(encoded_rows, bit_length=bit_length)

    feature_ranges = [
        {
            "feature_name": feature_name,
            "train_min": min(record.feature_values[index] for record in train_records),
            "train_max": max(record.feature_values[index] for record in train_records),
        }
        for index, feature_name in enumerate(feature_names)
    ]
    return {
        "encoder": encoder_payload,
        "summary": {
            **summary,
            "feature_ranges": feature_ranges,
        },
        "ranking_metrics": summary["ranking_metrics"],
    }


def build_legacy_hash_frontend_summary(
    canonical_records: Sequence[Mapping[str, object]],
    *,
    dataset_id: str | None,
    adapter_profile_id: str | None,
    frontend_input_id: str,
    frontend_fingerprint: str,
    bit_length: int = PACKED_BIT_LENGTH,
) -> dict[str, object]:
    encoded_rows = tuple(
        _EncodedRow(
            source_record_id=_require_record_id(record),
            split=_require_record_split(record),
            state_label=_require_state_label(record),
            packed_row_int=_pack_hash_payload(
                _legacy_hash_payload(
                    _coerce_waveform_record(record),
                    dataset_id=dataset_id,
                    adapter_profile_id=adapter_profile_id,
                    frontend_input_id=frontend_input_id,
                    frontend_fingerprint=frontend_fingerprint,
                )
            )[0],
        )
        for record in canonical_records
    )
    return _summarize_encoded_rows(encoded_rows, bit_length=bit_length)


def encode_frontend_record(
    record: WaveformDatasetRecord,
    *,
    dataset_id: str | None,
    adapter_profile_id: str | None,
    frontend_input_id: str,
    frontend_fingerprint: str,
    contract_payload: Mapping[str, object] | None = None,
) -> FrontendEncodingResult:
    encoder_payload = resolve_temporal_threshold_encoder(contract_payload)
    if encoder_payload is not None:
        temporal_payload = _temporal_feature_payload_from_record(record)
        packed_row_int = _pack_temporal_feature_values(
            _feature_values_from_temporal_payload(temporal_payload),
            encoder_payload=encoder_payload,
        )
        hex_width = max(1, int(encoder_payload["bit_length"]) // 4)
        payload = {
            "bit_length": int(encoder_payload["bit_length"]),
            "deploy_frontend_kind": TEMPORAL_THRESHOLD_FRONTEND_KIND,
            "encoder_fingerprint": str(encoder_payload["encoder_fingerprint"]),
            "frontend_fingerprint": frontend_fingerprint,
            "frontend_input_id": frontend_input_id,
            "packed_row_hex": f"{packed_row_int:0{hex_width}x}",
            "packed_row_int": packed_row_int,
            "row_format": PACKED_ROW_FORMAT,
            "source_record_id": record.source_record_id,
            "temporal_feature_config_fingerprint": temporal_payload.get("config_fingerprint"),
        }
        payload["encoding_sha256"] = compute_json_sha256(payload)
        return FrontendEncodingResult(
            payload=payload,
            packed_row_int=packed_row_int,
            packed_row_sha256=str(payload["encoding_sha256"]),
            bit_length=int(encoder_payload["bit_length"]),
            row_format=PACKED_ROW_FORMAT,
            bit_feature_names=tuple(str(name) for name in encoder_payload["bit_feature_names"]),
            frontend_kind=TEMPORAL_THRESHOLD_FRONTEND_KIND,
        )

    legacy_payload = _legacy_hash_payload(
        record,
        dataset_id=dataset_id,
        adapter_profile_id=adapter_profile_id,
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=frontend_fingerprint,
    )
    packed_row_int, packed_row_sha256 = _pack_hash_payload(legacy_payload)
    payload = {
        "bit_length": PACKED_BIT_LENGTH,
        "deploy_frontend_kind": LEGACY_HASH_FRONTEND_KIND,
        "encoding_sha256": packed_row_sha256,
        "frontend_fingerprint": frontend_fingerprint,
        "frontend_input_id": frontend_input_id,
        "packed_row_hex": f"{packed_row_int:016x}",
        "packed_row_int": packed_row_int,
        "row_format": PACKED_ROW_FORMAT,
        "source_record_id": record.source_record_id,
    }
    return FrontendEncodingResult(
        payload=payload,
        packed_row_int=packed_row_int,
        packed_row_sha256=packed_row_sha256,
        bit_length=PACKED_BIT_LENGTH,
        row_format=PACKED_ROW_FORMAT,
        bit_feature_names=tuple(f"bit_{index}" for index in range(PACKED_BIT_LENGTH)),
        frontend_kind=LEGACY_HASH_FRONTEND_KIND,
    )


def resolve_temporal_threshold_encoder(
    contract_payload: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if contract_payload is None:
        return None
    raw = contract_payload.get("frontend_encoding")
    if not isinstance(raw, Mapping):
        return None
    if raw.get("kind") != TEMPORAL_THRESHOLD_ENCODING_KIND:
        return None
    if raw.get("schema_version") != TEMPORAL_THRESHOLD_ENCODING_SCHEMA_VERSION:
        raise ContractValidationError(
            "Unsupported `frontend_encoding.schema_version` for the temporal threshold encoder."
        )
    return dict(raw)


def _coerce_waveform_record(record: WaveformDatasetRecord | Mapping[str, object]) -> WaveformDatasetRecord:
    if isinstance(record, WaveformDatasetRecord):
        return record
    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Frontend encoding requires a `WaveformDatasetRecord` or JSON-object waveform record."
        )
    return WaveformDatasetRecord.from_dict(record)


def _record_context_metadata(record: WaveformDatasetRecord | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(record, WaveformDatasetRecord):
        return record.context_metadata
    if not isinstance(record, Mapping):
        raise ContractValidationError("Waveform record context extraction requires a mapping.")
    if "context" in record:
        context = record.get("context", {})
        if context is None:
            return {}
        if not isinstance(context, Mapping):
            raise ContractValidationError("Waveform record `context` must be a mapping.")
        metadata = context.get("metadata", {})
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise ContractValidationError("Waveform record `context.metadata` must be a mapping.")
        return metadata
    metadata = record.get("context_metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ContractValidationError("Waveform record `context_metadata` must be a mapping.")
    return metadata


def _temporal_feature_payload_from_record(
    record: WaveformDatasetRecord | Mapping[str, object],
) -> Mapping[str, object]:
    metadata = _record_context_metadata(record)
    payload = metadata.get(_TEMPORAL_FEATURE_KEY)
    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            "Temporal threshold encoding requires `context_metadata.temporal_features` on every record."
        )
    return dict(payload)


def _feature_values_from_temporal_payload(payload: Mapping[str, object]) -> tuple[int, ...]:
    feature_names = payload.get("feature_names")
    feature_values = payload.get("feature_values")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes, bytearray)):
        raise ContractValidationError("Temporal feature payload must include a `feature_names` sequence.")
    if not isinstance(feature_values, Sequence) or isinstance(feature_values, (str, bytes, bytearray)):
        raise ContractValidationError("Temporal feature payload must include a `feature_values` sequence.")
    if len(feature_names) != len(feature_values):
        raise ContractValidationError("Temporal feature payload `feature_names` and `feature_values` must align.")
    normalized_values: list[int] = []
    for index, value in enumerate(feature_values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(
                f"Temporal feature payload `feature_values[{index}]` must be an integer."
            )
        normalized_values.append(value)
    return tuple(normalized_values)


def _temporal_record_from_payload(record: Mapping[str, object]) -> _TemporalRecord:
    payload = _temporal_feature_payload_from_record(record)
    feature_names_raw = payload.get("feature_names")
    if not isinstance(feature_names_raw, Sequence) or isinstance(feature_names_raw, (str, bytes, bytearray)):
        raise ContractValidationError("Temporal feature payload must include a `feature_names` sequence.")
    feature_names = tuple(str(name) for name in feature_names_raw)
    if not feature_names:
        raise ContractValidationError("Temporal feature payload must include at least one feature name.")
    return _TemporalRecord(
        source_record_id=_require_record_id(record),
        split=_require_record_split(record),
        state_label=_require_state_label(record),
        feature_names=feature_names,
        feature_values=_feature_values_from_temporal_payload(payload),
        config_fingerprint=(
            str(payload.get("config_fingerprint"))
            if isinstance(payload.get("config_fingerprint"), str)
            else None
        ),
    )


def _require_record_id(record: Mapping[str, object]) -> str:
    value = record.get("source_record_id")
    if not isinstance(value, str) or value == "":
        raise ContractValidationError("Waveform records must include a non-empty `source_record_id`.")
    return value


def _require_record_split(record: Mapping[str, object]) -> str:
    value = record.get("split")
    if not isinstance(value, str) or value == "":
        raise ContractValidationError("Waveform records must include a non-empty `split`.")
    return value.lower()


def _require_state_label(record: Mapping[str, object]) -> str:
    value = record.get("state_label")
    if not isinstance(value, str) or value == "":
        raise ContractValidationError("Waveform records must include a non-empty `state_label`.")
    return value


def _allocate_threshold_counts(
    feature_names: Sequence[str],
    *,
    bit_length: int,
) -> dict[str, int]:
    counts = {feature_name: 2 for feature_name in feature_names}
    remaining = bit_length - (2 * len(feature_names))
    if remaining < 0:
        raise ContractValidationError(
            f"Temporal threshold encoding requires at most {bit_length // 2} features for a {bit_length}-bit export."
        )
    priority = [name for name in _TEMPORAL_FEATURE_PRIORITY if name in counts] + [
        name for name in feature_names if name not in _TEMPORAL_FEATURE_PRIORITY
    ]
    while remaining > 0:
        for feature_name in priority:
            if remaining <= 0:
                break
            counts[feature_name] += 1
            remaining -= 1
    return counts


def _quantile_thresholds(values: Sequence[int], threshold_count: int) -> list[int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ContractValidationError("Temporal threshold encoding requires non-empty threshold values.")
    if threshold_count < 1:
        raise ContractValidationError("Temporal threshold encoding requires at least one threshold per feature.")
    if len(ordered) == 1:
        return [ordered[0] for _ in range(threshold_count)]
    thresholds: list[int] = []
    divisor = threshold_count + 1
    max_index = len(ordered) - 1
    for step in range(1, threshold_count + 1):
        index = _round_ratio(max_index * step, divisor)
        thresholds.append(int(ordered[index]))
    return thresholds


def _pack_temporal_feature_values(
    feature_values: Sequence[int],
    *,
    encoder_payload: Mapping[str, object],
) -> int:
    feature_names = tuple(str(name) for name in encoder_payload["feature_names"])
    threshold_specs = tuple(encoder_payload["feature_thresholds"])
    if len(feature_values) != len(feature_names):
        raise ContractValidationError(
            "Temporal threshold encoding received a feature vector that does not match the encoder schema."
        )
    packed_row = 0
    bit_index = 0
    for feature_index, spec in enumerate(threshold_specs):
        thresholds = spec.get("thresholds")
        if not isinstance(thresholds, Sequence) or isinstance(thresholds, (str, bytes, bytearray)):
            raise ContractValidationError("Temporal threshold encoding requires sequence `thresholds` values.")
        feature_value = int(feature_values[feature_index])
        for threshold in thresholds:
            if isinstance(threshold, bool) or not isinstance(threshold, int):
                raise ContractValidationError("Temporal threshold values must be integers.")
            if feature_value >= threshold:
                packed_row |= 1 << bit_index
            bit_index += 1
    if bit_index != int(encoder_payload["bit_length"]):
        raise ContractValidationError("Temporal threshold encoding bit packing did not fill the expected bit length.")
    return packed_row


def _select_single_medoid(rows: Sequence[int]) -> int:
    return min(
        rows,
        key=lambda row: (
            sum(_bit_distance(row, other) for other in rows),
            row,
        ),
    )


def _summarize_encoded_rows(
    rows: Sequence[_EncodedRow],
    *,
    bit_length: int,
) -> dict[str, object]:
    split_counts = Counter(row.split for row in rows)
    state_counts = Counter(row.state_label for row in rows)
    train_rows = tuple(row for row in rows if row.split == "train") or tuple(rows)
    rows_by_label: dict[str, tuple[_EncodedRow, ...]] = {
        label: tuple(row for row in train_rows if row.state_label == label)
        for label in sorted({row.state_label for row in rows}, key=lambda label: (_LABEL_ORDER.get(label, 999), label))
    }
    medoids = {
        label: _select_single_medoid(tuple(row.packed_row_int for row in label_rows))
        for label, label_rows in rows_by_label.items()
        if label_rows
    }

    within_class_distances = [
        _bit_distance(row.packed_row_int, medoids[row.state_label]) / float(bit_length)
        for row in train_rows
        if row.state_label in medoids
    ]
    intra_class_compactness = max(
        0.0,
        1.0 - (
            sum(within_class_distances) / float(len(within_class_distances))
            if within_class_distances
            else 0.0
        ),
    )

    inter_class_distances = [
        _bit_distance(medoids[left], medoids[right]) / float(bit_length)
        for left, right in itertools.combinations(sorted(medoids), 2)
    ]
    inter_class_separation = (
        sum(inter_class_distances) / float(len(inter_class_distances))
        if inter_class_distances
        else 0.0
    )

    reference_rows = train_rows or rows
    bit_balance = _bit_balance(reference_rows, bit_length=bit_length)
    bit_stability = _bit_stability(rows, bit_length=bit_length)
    validation_summary = _binary_validation_summary(rows, medoids=medoids, bit_length=bit_length)

    return {
        "split_counts": dict(sorted(split_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "medoids": {
            label: {
                "packed_row_hex": f"{medoid:0{max(1, bit_length // 4)}x}",
                "packed_row_int": medoid,
            }
            for label, medoid in medoids.items()
        },
        "bit_statistics": {
            "bit_balance": round(bit_balance, 6),
            "bit_stability": round(bit_stability, 6),
        },
        "binary_validation": validation_summary,
        "ranking_metrics": {
            "healthy_unhealthy_margin": round(float(validation_summary["margin_proxy"]), 6),
            "inter_class_separation": round(inter_class_separation, 6),
            "intra_class_compactness": round(intra_class_compactness, 6),
            "bit_balance": round(bit_balance, 6),
            "bit_stability": round(bit_stability, 6),
        },
    }


def _bit_balance(rows: Sequence[_EncodedRow], *, bit_length: int) -> float:
    if not rows:
        return 0.0
    total = float(len(rows))
    scores: list[float] = []
    for bit_index in range(bit_length):
        ones = sum(1 for row in rows if row.packed_row_int & (1 << bit_index))
        rate = float(ones) / total
        scores.append(1.0 - (abs(rate - 0.5) / 0.5))
    return max(0.0, min(1.0, sum(scores) / float(len(scores))))


def _bit_stability(rows: Sequence[_EncodedRow], *, bit_length: int) -> float:
    split_groups: dict[str, tuple[_EncodedRow, ...]] = {
        split: tuple(row for row in rows if row.split == split)
        for split in sorted({row.split for row in rows})
    }
    active_groups = [group for group in split_groups.values() if group]
    if len(active_groups) < 2:
        return 1.0
    stabilities: list[float] = []
    for bit_index in range(bit_length):
        rates = []
        for group in active_groups:
            ones = sum(1 for row in group if row.packed_row_int & (1 << bit_index))
            rates.append(float(ones) / float(len(group)))
        stabilities.append(1.0 - (max(rates) - min(rates)))
    return max(0.0, min(1.0, sum(stabilities) / float(len(stabilities))))


def _binary_validation_summary(
    rows: Sequence[_EncodedRow],
    *,
    medoids: Mapping[str, int],
    bit_length: int,
) -> dict[str, object]:
    validation_rows = tuple(row for row in rows if row.split == "val") or tuple(rows)
    if not validation_rows or not medoids:
        return {
            "accuracy": 0.0,
            "false_positive_rate": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "margin_proxy": 0.0,
            "confusion_matrix": {"matrix": [[0, 0], [0, 0]], "labels": ["healthy", "unhealthy"]},
        }

    labels = sorted(medoids, key=lambda label: (_LABEL_ORDER.get(label, 999), label))
    predictions = [
        (
            row,
            min(
                labels,
                key=lambda label: (
                    _bit_distance(row.packed_row_int, medoids[label]),
                    _LABEL_ORDER.get(label, 999),
                    label,
                ),
            ),
        )
        for row in validation_rows
    ]
    total = float(len(predictions))
    accuracy = sum(1 for row, prediction in predictions if row.state_label == prediction) / total

    if "healthy" in labels and "unhealthy" in labels:
        tn = sum(1 for row, prediction in predictions if row.state_label == "healthy" and prediction == "healthy")
        fp = sum(1 for row, prediction in predictions if row.state_label == "healthy" and prediction == "unhealthy")
        fn = sum(1 for row, prediction in predictions if row.state_label == "unhealthy" and prediction == "healthy")
        tp = sum(1 for row, prediction in predictions if row.state_label == "unhealthy" and prediction == "unhealthy")
        negative_total = tn + fp
        positive_total = tp + fn
        false_positive_rate = float(fp) / float(negative_total) if negative_total else 1.0
        precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        recall = float(tp) / float(positive_total) if positive_total else 0.0
        margin_proxy = 1.0 - false_positive_rate
        confusion_matrix = {
            "matrix": [[tn, fp], [fn, tp]],
            "labels": ["healthy", "unhealthy"],
        }
    else:
        false_positive_rate = 1.0 - accuracy
        precision = accuracy
        recall = accuracy
        margin_proxy = accuracy
        confusion_matrix = {
            "matrix": [],
            "labels": labels,
        }

    return {
        "accuracy": round(accuracy, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "margin_proxy": round(margin_proxy, 6),
        "confusion_matrix": confusion_matrix,
        "normalized_bit_length": bit_length,
    }


__all__ = [
    "FrontendEncodingResult",
    "LEGACY_HASH_FRONTEND_KIND",
    "PACKED_BIT_LENGTH",
    "PACKED_ROW_FORMAT",
    "TEMPORAL_THRESHOLD_FRONTEND_KIND",
    "build_legacy_hash_frontend_summary",
    "build_temporal_frontend_encoding",
    "encode_frontend_record",
    "resolve_temporal_threshold_encoder",
]
