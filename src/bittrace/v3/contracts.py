"""Stable V3 contract models and invariant validators for phase 2 slice 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
import types
import typing
from typing import Any, ClassVar, TypeAlias, TypeVar, get_args, get_origin, get_type_hints


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class ContractValidationError(ValueError):
    """Raised when a V3 contract violates the canonical phase-2 surface."""


class PassFail(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class StageKey(str, Enum):
    LEAN_SMOKE = "lean_smoke"
    DEEP_SMOKE = "deep_smoke"
    LEAN_MAIN_SCREEN = "lean_main_screen"
    DEEP_MAIN_SCREEN = "deep_main_screen"
    CAPACITY_REFINEMENT = "capacity_refinement"
    WINNER_DEEPEN_FREEZE_EXPORT = "winner_deepen_freeze_export"
    PARITY_VERIFICATION = "parity_verification"


class ExecutionAcceleration(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class FrontendRankingMode(str, Enum):
    ENCODER_PROXY_QUALITY = "encoder_proxy_quality"


class DeepRankingMode(str, Enum):
    SCOUT_UNHEALTHY_ALERT = "scout_unhealthy_alert"


class PromotionStage(str, Enum):
    MAIN_SCREEN = "main_screen"
    CAPACITY_REFINEMENT = "capacity_refinement"


class ScoutAlertabilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    DEAD_DETECTOR = "dead_detector"
    BLOCKED = "blocked"


class RetainedAlternateKind(str, Enum):
    FRONTEND_CANDIDATE = "frontend_candidate"
    DEEP_CANDIDATE = "deep_candidate"


class VerificationLevel(str, Enum):
    ADAPTER_PARITY = "adapter_parity"
    FRONTEND_PARITY = "frontend_parity"
    DEEP_PARITY = "deep_parity"
    END_TO_END_PARITY = "end_to_end_parity"


class ParityComparisonStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    UNSUPPORTED = "unsupported"


class SearchExplorationMode(str, Enum):
    BOUNDED_RANDOM = "bounded_random"


class SearchRefinementMode(str, Enum):
    LOCAL_TIGHTENING = "local_tightening"


class CandidateProvenanceOrigin(str, Enum):
    LOCAL_REFINE = "local_refine"
    BOUNDED_RANDOM = "bounded_random"
    WINNER_REPLAY = "winner_replay"
    WINNER_MUTATION = "winner_mutation"


E = TypeVar("E", bound=Enum)
T = TypeVar("T", bound="SerializableModel")
_UNION_ORIGINS = {types.UnionType, typing.Union}


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _require_sequence(value: object, *, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"`{field_name}` must be a sequence.")
    return value


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


def _require_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"`{field_name}` must be numeric.")
    resolved = float(value)
    if minimum is not None and resolved < minimum:
        raise ContractValidationError(f"`{field_name}` must be >= {minimum}.")
    if maximum is not None and resolved > maximum:
        raise ContractValidationError(f"`{field_name}` must be <= {maximum}.")
    return resolved


def _require_str(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"`{field_name}` must be a string.")
    if not allow_empty and value == "":
        raise ContractValidationError(f"`{field_name}` cannot be empty.")
    return value


def _require_optional_str(value: object, *, field_name: str, allow_empty: bool = False) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name=field_name, allow_empty=allow_empty)


def _require_optional_sha256(value: object, *, field_name: str) -> str | None:
    text = _require_optional_str(value, field_name=field_name)
    if text is None:
        return None
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractValidationError(f"`{field_name}` must be a lowercase 64-char SHA-256 digest.")
    return text


def _require_sha256(value: object, *, field_name: str) -> str:
    text = _require_optional_sha256(value, field_name=field_name)
    if text is None:
        raise ContractValidationError(f"`{field_name}` is required.")
    return text


def _coerce_enum(value: object, enum_type: type[E], *, field_name: str) -> E:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ContractValidationError(f"`{field_name}` must be a `{enum_type.__name__}` value.")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractValidationError(
            f"`{field_name}` must be one of: {allowed}."
        ) from exc


def _copy_json_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, JsonValue]:
    return {
        _require_str(key, field_name=f"{field_name}.<key>"): _to_plain_data(item)
        for key, item in value.items()
    }


def _copy_bool_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, bool]:
    return {
        _require_str(key, field_name=f"{field_name}.<key>"): _require_bool(
            item,
            field_name=f"{field_name}.{key}",
        )
        for key, item in value.items()
    }


def _copy_float_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        field_path = f"{field_name}.{key}"
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractValidationError(f"`{field_path}` must be numeric.")
        result[_require_str(key, field_name=f"{field_name}.<key>")] = float(item)
    return result


def _copy_int_tuple(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    return tuple(
        _require_int(item, field_name=f"{field_name}[{index}]", minimum=minimum)
        for index, item in enumerate(_require_sequence(value, field_name=field_name))
    )


def _copy_str_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        _require_str(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(_require_sequence(value, field_name=field_name))
    )


def _copy_distinct_str_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    items = _copy_str_tuple(value, field_name=field_name)
    if len(set(items)) != len(items):
        raise ContractValidationError(f"`{field_name}` must not contain duplicate values.")
    return items


def _copy_verification_level_tuple(
    value: object,
    *,
    field_name: str,
    require_non_empty: bool,
) -> tuple[VerificationLevel, ...]:
    items = tuple(
        _coerce_enum(
            item,
            VerificationLevel,
            field_name=f"{field_name}[{index}]",
        )
        for index, item in enumerate(_require_sequence(value, field_name=field_name))
    )
    if require_non_empty and not items:
        raise ContractValidationError(f"`{field_name}` must not be empty.")
    if len(set(items)) != len(items):
        raise ContractValidationError(f"`{field_name}` must not contain duplicate levels.")
    return items


def _walk_json_values(value: object) -> typing.Iterator[tuple[str | None, object]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key if isinstance(key, str) else None, item
            yield from _walk_json_values(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_json_values(item)


def _require_absent_json_key(
    value: object,
    *,
    key_name: str,
    field_name: str,
) -> None:
    for nested_key, _ in _walk_json_values(value):
        if nested_key == key_name:
            raise ContractValidationError(
                f"`{field_name}` must not contain `{key_name}` in deploy/runtime fields."
            )


def _copy_device_agnostic_export_mapping(
    value: object,
    *,
    field_name: str,
    require_portable: bool,
) -> dict[str, JsonValue]:
    copied = _copy_json_mapping(
        _require_mapping(value, field_name=field_name),
        field_name=field_name,
    )
    portable_value = copied.get("portable")
    if require_portable and portable_value is not True:
        raise ContractValidationError(f"`{field_name}.portable` must be `true` for canonical S6 export.")
    for null_field in ("execution_device", "hardware_binding"):
        if null_field in copied and copied[null_field] is not None:
            raise ContractValidationError(
                f"`{field_name}.{null_field}` must be null for a device-agnostic export surface."
            )
    return copied


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def _to_plain_data(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SerializableModel):
        return value.to_dict()
    if is_dataclass(value):
        return {
            field_info.name: _to_plain_data(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            _require_str(key, field_name="mapping.<key>"): _to_plain_data(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain_data(item) for item in value]
    raise ContractValidationError(f"Unsupported JSON value type `{type(value).__name__}`.")


@lru_cache(maxsize=None)
def _type_hints(model_type: type[object]) -> dict[str, object]:
    return get_type_hints(model_type)


def _coerce_typed_value(value: object, expected_type: object, *, field_name: str) -> object:
    origin = get_origin(expected_type)
    if expected_type in {Any, object}:
        return _to_plain_data(value)
    if origin in _UNION_ORIGINS:
        options = get_args(expected_type)
        for option in options:
            try:
                return _coerce_typed_value(value, option, field_name=field_name)
            except ContractValidationError:
                continue
        raise ContractValidationError(f"`{field_name}` does not match any allowed type.")
    if origin in {tuple, list, Sequence}:
        items = _require_sequence(value, field_name=field_name)
        args = get_args(expected_type)
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(
                    _coerce_typed_value(item, args[0], field_name=f"{field_name}[{index}]")
                    for index, item in enumerate(items)
                )
            if args and len(args) != len(items):
                raise ContractValidationError(
                    f"`{field_name}` must contain exactly {len(args)} items."
                )
            if not args:
                return tuple(items)
            return tuple(
                _coerce_typed_value(item, args[index], field_name=f"{field_name}[{index}]")
                for index, item in enumerate(items)
            )
        item_type = args[0] if args else object
        return [
            _coerce_typed_value(item, item_type, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(items)
        ]
    if origin in {dict, Mapping}:
        mapping = _require_mapping(value, field_name=field_name)
        args = get_args(expected_type)
        if len(args) != 2:
            return dict(mapping)
        key_type, value_type = args
        if key_type is not str:
            raise ContractValidationError(f"`{field_name}` only supports string-key mappings.")
        return {
            _require_str(key, field_name=f"{field_name}.<key>"): _coerce_typed_value(
                item,
                value_type,
                field_name=f"{field_name}.{key}",
            )
            for key, item in mapping.items()
        }
    if isinstance(expected_type, type):
        if issubclass(expected_type, Enum):
            return _coerce_enum(value, expected_type, field_name=field_name)
        if is_dataclass(expected_type) and issubclass(expected_type, SerializableModel):
            mapping = _require_mapping(value, field_name=field_name)
            return expected_type.from_dict(mapping)
        if expected_type is bool:
            return _require_bool(value, field_name=field_name)
        if expected_type is int:
            return _require_int(value, field_name=field_name)
        if expected_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractValidationError(f"`{field_name}` must be a float.")
            return float(value)
        if expected_type is str:
            return _require_str(value, field_name=field_name, allow_empty=True)
        if expected_type is type(None):
            if value is not None:
                raise ContractValidationError(f"`{field_name}` must be null.")
            return None
    return value


def _build_model(model_type: type[T], payload: Mapping[str, object], *, model_name: str) -> T:
    allowed_fields = tuple(field_info.name for field_info in fields(model_type) if field_info.init)
    unexpected = sorted(set(payload).difference(allowed_fields))
    if unexpected:
        raise ContractValidationError(
            f"`{model_name}` contains unexpected fields: {', '.join(unexpected)}."
        )

    hints = _type_hints(model_type)
    kwargs: dict[str, object] = {}
    for field_info in fields(model_type):
        if not field_info.init:
            continue
        field_name = field_info.name
        if field_name in payload:
            kwargs[field_name] = _coerce_typed_value(
                payload[field_name],
                hints.get(field_name, field_info.type),
                field_name=f"{model_name}.{field_name}",
            )
            continue
        if field_info.default is not MISSING or field_info.default_factory is not MISSING:
            continue
        raise ContractValidationError(f"`{model_name}.{field_name}` is required.")
    return model_type(**kwargs)


class SerializableModel:
    """JSON-serializable frozen dataclass mixin."""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            field_info.name: _to_plain_data(getattr(self, field_info.name))
            for field_info in fields(self)
            if field_info.init
        }

    @classmethod
    def from_dict(cls: type[T], payload: Mapping[str, object]) -> T:
        return _build_model(cls, payload, model_name=cls.__name__)


class ArtifactContract(SerializableModel):
    """Top-level artifact contract with stable kind and schema metadata."""

    KIND: ClassVar[str]
    SCHEMA_VERSION: ClassVar[str]

    @property
    def kind(self) -> str:
        return type(self).KIND

    @property
    def schema_version(self) -> str:
        return type(self).SCHEMA_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        payload = super().to_dict()
        payload["kind"] = self.kind
        payload["schema_version"] = self.schema_version
        return payload

    @classmethod
    def from_dict(cls: type[T], payload: Mapping[str, object]) -> T:
        kind = _require_str(payload.get("kind"), field_name=f"{cls.__name__}.kind")
        schema_version = _require_str(
            payload.get("schema_version"),
            field_name=f"{cls.__name__}.schema_version",
        )
        if kind != cls.KIND:
            raise ContractValidationError(
                f"`{cls.__name__}.kind` must be `{cls.KIND}`, got `{kind}`."
            )
        if schema_version != cls.SCHEMA_VERSION:
            raise ContractValidationError(
                f"`{cls.__name__}.schema_version` must be `{cls.SCHEMA_VERSION}`, got "
                f"`{schema_version}`."
            )
        body = {key: value for key, value in payload.items() if key not in {"kind", "schema_version"}}
        return _build_model(cls, body, model_name=cls.__name__)


@dataclass(frozen=True, slots=True)
class ArtifactRef(SerializableModel):
    kind: str
    path: str
    schema_version: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_str(self.kind, field_name="ArtifactRef.kind"))
        object.__setattr__(self, "path", _require_str(self.path, field_name="ArtifactRef.path"))
        object.__setattr__(
            self,
            "schema_version",
            _require_optional_str(
                self.schema_version,
                field_name="ArtifactRef.schema_version",
            ),
        )
        object.__setattr__(
            self,
            "sha256",
            _require_optional_sha256(self.sha256, field_name="ArtifactRef.sha256"),
        )


def _artifact_ref_matches_lineage(left: ArtifactRef | None, right: ArtifactRef | None) -> bool:
    if left is None or right is None:
        return left is right
    if left.kind != right.kind or left.path != right.path:
        return False
    if left.schema_version is not None and right.schema_version is not None:
        return left.schema_version == right.schema_version
    return True


@dataclass(frozen=True, slots=True)
class ResolvedGenomeIdentity(SerializableModel):
    candidate_name: str
    genome_fingerprint: str
    descriptor_family: str | None = None
    encoding_family: str | None = None
    genome_file: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_name",
            _require_str(
                self.candidate_name,
                field_name="ResolvedGenomeIdentity.candidate_name",
            ),
        )
        object.__setattr__(
            self,
            "genome_fingerprint",
            _require_str(
                self.genome_fingerprint,
                field_name="ResolvedGenomeIdentity.genome_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "descriptor_family",
            _require_optional_str(
                self.descriptor_family,
                field_name="ResolvedGenomeIdentity.descriptor_family",
            ),
        )
        object.__setattr__(
            self,
            "encoding_family",
            _require_optional_str(
                self.encoding_family,
                field_name="ResolvedGenomeIdentity.encoding_family",
            ),
        )
        object.__setattr__(
            self,
            "genome_file",
            _require_optional_str(
                self.genome_file,
                field_name="ResolvedGenomeIdentity.genome_file",
            ),
        )


@dataclass(frozen=True, slots=True)
class FrontendInput(SerializableModel):
    bundle_dir: str
    bundle_contract_path: str
    bundle_fingerprint: str
    source_handoff_manifest_path: str
    include_test_metrics: bool
    resolved_genome_identity: ResolvedGenomeIdentity | None = None
    adapter_profile_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_dir", _require_str(self.bundle_dir, field_name="FrontendInput.bundle_dir"))
        object.__setattr__(
            self,
            "bundle_contract_path",
            _require_str(
                self.bundle_contract_path,
                field_name="FrontendInput.bundle_contract_path",
            ),
        )
        object.__setattr__(
            self,
            "bundle_fingerprint",
            _require_str(
                self.bundle_fingerprint,
                field_name="FrontendInput.bundle_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "source_handoff_manifest_path",
            _require_str(
                self.source_handoff_manifest_path,
                field_name="FrontendInput.source_handoff_manifest_path",
            ),
        )
        object.__setattr__(
            self,
            "include_test_metrics",
            _require_bool(
                self.include_test_metrics,
                field_name="FrontendInput.include_test_metrics",
            ),
        )
        object.__setattr__(
            self,
            "adapter_profile_id",
            _require_optional_str(
                self.adapter_profile_id,
                field_name="FrontendInput.adapter_profile_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class DeepInputRef(SerializableModel):
    bundle_dir: str
    bundle_contract_path: str
    bundle_fingerprint: str
    handoff_manifest_path: str
    source_bundle_dir: str
    source_bundle_contract_path: str
    source_bundle_fingerprint: str
    source_handoff_manifest_path: str
    include_test_metrics: bool
    frontend_input_id: str | None = None
    frontend_fingerprint: str | None = None
    resolved_genome_identity: ResolvedGenomeIdentity | None = None
    source_promotion_artifact_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        field_names = (
            "bundle_dir",
            "bundle_contract_path",
            "bundle_fingerprint",
            "handoff_manifest_path",
            "source_bundle_dir",
            "source_bundle_contract_path",
            "source_bundle_fingerprint",
            "source_handoff_manifest_path",
        )
        for field_name in field_names:
            object.__setattr__(
                self,
                field_name,
                _require_str(
                    getattr(self, field_name),
                    field_name=f"DeepInputRef.{field_name}",
                ),
            )
        object.__setattr__(
            self,
            "include_test_metrics",
            _require_bool(
                self.include_test_metrics,
                field_name="DeepInputRef.include_test_metrics",
            ),
        )
        object.__setattr__(
            self,
            "frontend_input_id",
            _require_optional_str(
                self.frontend_input_id,
                field_name="DeepInputRef.frontend_input_id",
            ),
        )
        object.__setattr__(
            self,
            "frontend_fingerprint",
            _require_optional_str(
                self.frontend_fingerprint,
                field_name="DeepInputRef.frontend_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class WaveformPayloadRef(SerializableModel):
    waveform_path: str | None = None
    waveform_payload_ref: ArtifactRef | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.waveform_payload_ref is not None:
            if isinstance(self.waveform_payload_ref, ArtifactRef):
                normalized_payload_ref = self.waveform_payload_ref
            elif isinstance(self.waveform_payload_ref, Mapping):
                normalized_payload_ref = ArtifactRef.from_dict(self.waveform_payload_ref)
            else:
                raise ContractValidationError(
                    "`WaveformPayloadRef.waveform_payload_ref` must be an `ArtifactRef` or mapping."
                )
            object.__setattr__(self, "waveform_payload_ref", normalized_payload_ref)
        object.__setattr__(
            self,
            "waveform_path",
            _require_optional_str(
                self.waveform_path,
                field_name="WaveformPayloadRef.waveform_path",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.metadata,
                    field_name="WaveformPayloadRef.metadata",
                ),
                field_name="WaveformPayloadRef.metadata",
            ),
        )
        if self.waveform_path is None and self.waveform_payload_ref is None:
            raise ContractValidationError(
                "Waveform payload refs must include either `waveform_path` or `waveform_payload_ref`."
            )
        if self.waveform_path is not None and self.waveform_payload_ref is not None:
            raise ContractValidationError(
                "Waveform payload refs must not include both `waveform_path` and `waveform_payload_ref`."
            )


@dataclass(frozen=True, slots=True)
class WaveformDatasetRecord(SerializableModel):
    source_record_id: str
    split: str
    state_label: str
    waveforms: Mapping[str, WaveformPayloadRef]
    label_metadata: Mapping[str, object] = field(default_factory=dict)
    sampling_hz: float | None = None
    rpm: float | None = None
    operating_condition: str | None = None
    context_metadata: Mapping[str, object] = field(default_factory=dict)
    lineage_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_record_id",
            _require_str(
                self.source_record_id,
                field_name="WaveformDatasetRecord.source_record_id",
            ),
        )
        object.__setattr__(
            self,
            "split",
            _require_str(
                self.split,
                field_name="WaveformDatasetRecord.split",
            ),
        )
        object.__setattr__(
            self,
            "state_label",
            _require_str(
                self.state_label,
                field_name="WaveformDatasetRecord.state_label",
            ),
        )
        waveforms_raw = _require_mapping(
            self.waveforms,
            field_name="WaveformDatasetRecord.waveforms",
        )
        if not waveforms_raw:
            raise ContractValidationError("`WaveformDatasetRecord.waveforms` must not be empty.")
        normalized_waveforms: dict[str, WaveformPayloadRef] = {}
        for channel_name, payload_ref in waveforms_raw.items():
            channel_key = _require_str(
                channel_name,
                field_name="WaveformDatasetRecord.waveforms.<key>",
            )
            if isinstance(payload_ref, WaveformPayloadRef):
                normalized_waveforms[channel_key] = payload_ref
                continue
            if isinstance(payload_ref, Mapping):
                normalized_waveforms[channel_key] = WaveformPayloadRef.from_dict(payload_ref)
                continue
            raise ContractValidationError(
                "`WaveformDatasetRecord.waveforms` values must be `WaveformPayloadRef` mappings."
            )
        object.__setattr__(self, "waveforms", normalized_waveforms)
        object.__setattr__(
            self,
            "label_metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.label_metadata,
                    field_name="WaveformDatasetRecord.label_metadata",
                ),
                field_name="WaveformDatasetRecord.label_metadata",
            ),
        )
        if self.sampling_hz is not None:
            sampling_hz = _require_float(
                self.sampling_hz,
                field_name="WaveformDatasetRecord.sampling_hz",
            )
            if sampling_hz <= 0.0:
                raise ContractValidationError(
                    "`WaveformDatasetRecord.sampling_hz` must be > 0 when provided."
                )
            object.__setattr__(self, "sampling_hz", sampling_hz)
        if self.rpm is not None:
            rpm = _require_float(
                self.rpm,
                field_name="WaveformDatasetRecord.rpm",
            )
            if rpm <= 0.0:
                raise ContractValidationError("`WaveformDatasetRecord.rpm` must be > 0 when provided.")
            object.__setattr__(self, "rpm", rpm)
        object.__setattr__(
            self,
            "operating_condition",
            _require_optional_str(
                self.operating_condition,
                field_name="WaveformDatasetRecord.operating_condition",
            ),
        )
        object.__setattr__(
            self,
            "context_metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.context_metadata,
                    field_name="WaveformDatasetRecord.context_metadata",
                ),
                field_name="WaveformDatasetRecord.context_metadata",
            ),
        )
        object.__setattr__(
            self,
            "lineage_metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.lineage_metadata,
                    field_name="WaveformDatasetRecord.lineage_metadata",
                ),
                field_name="WaveformDatasetRecord.lineage_metadata",
            ),
        )


@dataclass(frozen=True, slots=True)
class ParentAnchorRef(SerializableModel):
    anchor_artifact_ref: ArtifactRef
    contract_kind: str
    contract_schema_version: str
    direct_restore_supported: bool
    parent_anchor_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_kind",
            _require_str(
                self.contract_kind,
                field_name="ParentAnchorRef.contract_kind",
            ),
        )
        object.__setattr__(
            self,
            "contract_schema_version",
            _require_str(
                self.contract_schema_version,
                field_name="ParentAnchorRef.contract_schema_version",
            ),
        )
        object.__setattr__(
            self,
            "direct_restore_supported",
            _require_bool(
                self.direct_restore_supported,
                field_name="ParentAnchorRef.direct_restore_supported",
            ),
        )
        object.__setattr__(
            self,
            "parent_anchor_fingerprint",
            _require_str(
                self.parent_anchor_fingerprint,
                field_name="ParentAnchorRef.parent_anchor_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionTrace(SerializableModel):
    requested_execution_acceleration: ExecutionAcceleration = ExecutionAcceleration.AUTO
    resolved_execution_acceleration: ExecutionAcceleration | None = None
    backend_actual: str | None = None
    execution_device: str | None = None
    fallback_reason: str | None = None
    blocked_reason: str | None = None
    allow_backend_fallback: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_execution_acceleration",
            _coerce_enum(
                self.requested_execution_acceleration,
                ExecutionAcceleration,
                field_name="ExecutionTrace.requested_execution_acceleration",
            ),
        )
        if self.resolved_execution_acceleration is not None:
            object.__setattr__(
                self,
                "resolved_execution_acceleration",
                _coerce_enum(
                    self.resolved_execution_acceleration,
                    ExecutionAcceleration,
                    field_name="ExecutionTrace.resolved_execution_acceleration",
                ),
            )
        object.__setattr__(
            self,
            "backend_actual",
            _require_optional_str(
                self.backend_actual,
                field_name="ExecutionTrace.backend_actual",
            ),
        )
        object.__setattr__(
            self,
            "execution_device",
            _require_optional_str(
                self.execution_device,
                field_name="ExecutionTrace.execution_device",
            ),
        )
        object.__setattr__(
            self,
            "fallback_reason",
            _require_optional_str(
                self.fallback_reason,
                field_name="ExecutionTrace.fallback_reason",
            ),
        )
        object.__setattr__(
            self,
            "blocked_reason",
            _require_optional_str(
                self.blocked_reason,
                field_name="ExecutionTrace.blocked_reason",
            ),
        )
        if self.allow_backend_fallback is not None:
            object.__setattr__(
                self,
                "allow_backend_fallback",
                _require_bool(
                    self.allow_backend_fallback,
                    field_name="ExecutionTrace.allow_backend_fallback",
                ),
            )


@dataclass(frozen=True, slots=True)
class RetainedAlternateRef(SerializableModel):
    candidate_kind: RetainedAlternateKind
    candidate_id: str
    artifact_ref: ArtifactRef
    reason: str
    rank: int | None = None
    metadata_only: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_kind",
            _coerce_enum(
                self.candidate_kind,
                RetainedAlternateKind,
                field_name="RetainedAlternateRef.candidate_kind",
            ),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(
                self.candidate_id,
                field_name="RetainedAlternateRef.candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "reason",
            _require_str(self.reason, field_name="RetainedAlternateRef.reason"),
        )
        if self.rank is not None:
            object.__setattr__(
                self,
                "rank",
                _require_int(
                    self.rank,
                    field_name="RetainedAlternateRef.rank",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "metadata_only",
            _require_bool(
                self.metadata_only,
                field_name="RetainedAlternateRef.metadata_only",
            ),
        )
        if not self.metadata_only:
            raise ContractValidationError(
                "Retained alternates must remain metadata-only in V3 slice 1."
            )
        object.__setattr__(
            self,
            "metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.metadata,
                    field_name="RetainedAlternateRef.metadata",
                ),
                field_name="RetainedAlternateRef.metadata",
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchCandidateProvenance(SerializableModel):
    origin: CandidateProvenanceOrigin
    stage_key: StageKey | None = None
    random_seed: int | None = None
    sampled_values: Mapping[str, object] = field(default_factory=dict)
    bounds_ref: str | None = None
    config_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin",
            _coerce_enum(
                self.origin,
                CandidateProvenanceOrigin,
                field_name="SearchCandidateProvenance.origin",
            ),
        )
        if self.stage_key is not None:
            object.__setattr__(
                self,
                "stage_key",
                _coerce_enum(
                    self.stage_key,
                    StageKey,
                    field_name="SearchCandidateProvenance.stage_key",
                ),
            )
        if self.random_seed is not None:
            object.__setattr__(
                self,
                "random_seed",
                _require_int(
                    self.random_seed,
                    field_name="SearchCandidateProvenance.random_seed",
                    minimum=0,
                ),
            )
        object.__setattr__(
            self,
            "sampled_values",
            _copy_json_mapping(
                _require_mapping(
                    self.sampled_values,
                    field_name="SearchCandidateProvenance.sampled_values",
                ),
                field_name="SearchCandidateProvenance.sampled_values",
            ),
        )
        object.__setattr__(
            self,
            "bounds_ref",
            _require_optional_str(
                self.bounds_ref,
                field_name="SearchCandidateProvenance.bounds_ref",
            ),
        )
        object.__setattr__(
            self,
            "config_ref",
            _require_optional_str(
                self.config_ref,
                field_name="SearchCandidateProvenance.config_ref",
            ),
        )


@dataclass(frozen=True, slots=True)
class StageSearchPolicy(SerializableModel):
    exploration_fraction: float = 0.0
    exploration_mode: SearchExplorationMode | None = None
    refinement_mode: SearchRefinementMode | None = None
    random_seed: int | None = None
    bounds_ref: str | None = None
    config_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exploration_fraction",
            _require_float(
                self.exploration_fraction,
                field_name="StageSearchPolicy.exploration_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.exploration_mode is not None:
            object.__setattr__(
                self,
                "exploration_mode",
                _coerce_enum(
                    self.exploration_mode,
                    SearchExplorationMode,
                    field_name="StageSearchPolicy.exploration_mode",
                ),
            )
        if self.refinement_mode is not None:
            object.__setattr__(
                self,
                "refinement_mode",
                _coerce_enum(
                    self.refinement_mode,
                    SearchRefinementMode,
                    field_name="StageSearchPolicy.refinement_mode",
                ),
            )
        if self.random_seed is not None:
            object.__setattr__(
                self,
                "random_seed",
                _require_int(
                    self.random_seed,
                    field_name="StageSearchPolicy.random_seed",
                    minimum=0,
                ),
            )
        object.__setattr__(
            self,
            "bounds_ref",
            _require_optional_str(
                self.bounds_ref,
                field_name="StageSearchPolicy.bounds_ref",
            ),
        )
        object.__setattr__(
            self,
            "config_ref",
            _require_optional_str(
                self.config_ref,
                field_name="StageSearchPolicy.config_ref",
            ),
        )
        if self.exploration_fraction > 0.0:
            if self.exploration_mode != SearchExplorationMode.BOUNDED_RANDOM:
                raise ContractValidationError(
                    "Positive `exploration_fraction` requires `exploration_mode=bounded_random`."
                )
            if self.random_seed is None:
                raise ContractValidationError(
                    "Positive `exploration_fraction` requires an explicit `random_seed`."
                )
        if self.exploration_mode == SearchExplorationMode.BOUNDED_RANDOM and self.random_seed is None:
            raise ContractValidationError(
                "`exploration_mode=bounded_random` requires an explicit `random_seed`."
            )

    @property
    def random_exploration_enabled(self) -> bool:
        return (
            self.exploration_mode == SearchExplorationMode.BOUNDED_RANDOM
            and self.exploration_fraction > 0.0
        )


@dataclass(frozen=True, slots=True)
class FrontendPromotionCandidate(SerializableModel):
    candidate_id: str
    candidate_order: int
    promoted_frontend_fingerprint: str
    ranking_eligible: bool
    ranking_metrics: Mapping[str, object]
    candidate_report_ref: ArtifactRef | None = None
    best_frontend_artifact_ref: ArtifactRef | None = None
    frontend_config_fingerprint: str | None = None
    provenance: SearchCandidateProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(
                self.candidate_id,
                field_name="FrontendPromotionCandidate.candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "candidate_order",
            _require_int(
                self.candidate_order,
                field_name="FrontendPromotionCandidate.candidate_order",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "promoted_frontend_fingerprint",
            _require_str(
                self.promoted_frontend_fingerprint,
                field_name="FrontendPromotionCandidate.promoted_frontend_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "ranking_eligible",
            _require_bool(
                self.ranking_eligible,
                field_name="FrontendPromotionCandidate.ranking_eligible",
            ),
        )
        object.__setattr__(
            self,
            "ranking_metrics",
            _copy_float_mapping(
                _require_mapping(
                    self.ranking_metrics,
                    field_name="FrontendPromotionCandidate.ranking_metrics",
                ),
                field_name="FrontendPromotionCandidate.ranking_metrics",
            ),
        )
        object.__setattr__(
            self,
            "frontend_config_fingerprint",
            _require_optional_str(
                self.frontend_config_fingerprint,
                field_name="FrontendPromotionCandidate.frontend_config_fingerprint",
            ),
        )
        if self.provenance is not None and self.provenance.stage_key is not None:
            if self.provenance.stage_key not in (StageKey.LEAN_SMOKE, StageKey.LEAN_MAIN_SCREEN):
                raise ContractValidationError(
                    "Frontend candidate provenance can only target frontend search stages."
                )


def _resolve_single_selected_k(
    *,
    explicit_value: object,
    legacy_value: object,
    effective_engine_deep_config: Mapping[str, object],
    field_name_prefix: str,
) -> int | None:
    selected_values: list[int] = []
    if explicit_value is not None:
        selected_values.append(
            _require_int(
                explicit_value,
                field_name=f"{field_name_prefix}.selected_k_per_class",
                minimum=1,
            )
        )
    if legacy_value is not None:
        selected_values.append(
            _require_int(
                legacy_value,
                field_name=f"{field_name_prefix}.k_medoids_per_class",
                minimum=1,
            )
        )
    config_k = effective_engine_deep_config.get("k_medoids_per_class")
    if config_k is not None:
        selected_values.append(
            _require_int(
                config_k,
                field_name=f"{field_name_prefix}.effective_engine_deep_config.k_medoids_per_class",
                minimum=1,
            )
        )
    if not selected_values:
        return None
    if any(k_value != selected_values[0] for k_value in selected_values[1:]):
        raise ContractValidationError(
            f"`{field_name_prefix}` must keep `selected_k_per_class` and `k_medoids_per_class` consistent."
        )
    return selected_values[0]


def _require_candidate_selected_k(
    candidate: "DeepPromotionCandidate",
    *,
    field_name: str,
    required: bool,
) -> int | None:
    selected_k = candidate.selected_k_per_class
    if selected_k is None:
        selected_k = _resolve_single_selected_k(
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
        duplicate_values = ", ".join(str(value) for value in duplicates)
        raise ContractValidationError(
            f"`{field_name}` must not repeat `k` values; duplicated: {duplicate_values}."
        )


@dataclass(frozen=True, slots=True)
class DeepPromotionCandidate(SerializableModel):
    candidate_id: str
    candidate_order: int
    branch_mode: str
    ranking_eligible: bool
    scout_alertability_status: ScoutAlertabilityStatus
    scout_alertability_guardrail_triggered: bool = False
    scout_alertability_reason: str | None = None
    effective_engine_deep_config: Mapping[str, object] = field(default_factory=dict)
    best_deep_artifact_ref: ArtifactRef | None = None
    metrics_summary_ref: ArtifactRef | None = None
    candidate_report_ref: ArtifactRef | None = None
    checkpoint_ref: ArtifactRef | None = None
    frontend_input_id: str | None = None
    frontend_fingerprint: str | None = None
    parent_anchor_fingerprint: str | None = None
    k_medoids_per_class: int | None = None
    selected_k_per_class: int | None = None
    k_medoids_search_values: tuple[int, ...] = ()
    provenance: SearchCandidateProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(
                self.candidate_id,
                field_name="DeepPromotionCandidate.candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "candidate_order",
            _require_int(
                self.candidate_order,
                field_name="DeepPromotionCandidate.candidate_order",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "branch_mode",
            _require_str(
                self.branch_mode,
                field_name="DeepPromotionCandidate.branch_mode",
            ),
        )
        object.__setattr__(
            self,
            "ranking_eligible",
            _require_bool(
                self.ranking_eligible,
                field_name="DeepPromotionCandidate.ranking_eligible",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_status",
            _coerce_enum(
                self.scout_alertability_status,
                ScoutAlertabilityStatus,
                field_name="DeepPromotionCandidate.scout_alertability_status",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_guardrail_triggered",
            _require_bool(
                self.scout_alertability_guardrail_triggered,
                field_name="DeepPromotionCandidate.scout_alertability_guardrail_triggered",
            ),
        )
        object.__setattr__(
            self,
            "scout_alertability_reason",
            _require_optional_str(
                self.scout_alertability_reason,
                field_name="DeepPromotionCandidate.scout_alertability_reason",
            ),
        )
        object.__setattr__(
            self,
            "effective_engine_deep_config",
            _copy_json_mapping(
                _require_mapping(
                    self.effective_engine_deep_config,
                    field_name="DeepPromotionCandidate.effective_engine_deep_config",
                ),
                field_name="DeepPromotionCandidate.effective_engine_deep_config",
            ),
        )
        object.__setattr__(
            self,
            "frontend_input_id",
            _require_optional_str(
                self.frontend_input_id,
                field_name="DeepPromotionCandidate.frontend_input_id",
            ),
        )
        object.__setattr__(
            self,
            "frontend_fingerprint",
            _require_optional_str(
                self.frontend_fingerprint,
                field_name="DeepPromotionCandidate.frontend_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "parent_anchor_fingerprint",
            _require_optional_str(
                self.parent_anchor_fingerprint,
                field_name="DeepPromotionCandidate.parent_anchor_fingerprint",
            ),
        )
        if self.k_medoids_per_class is not None:
            object.__setattr__(
                self,
                "k_medoids_per_class",
                _require_int(
                    self.k_medoids_per_class,
                    field_name="DeepPromotionCandidate.k_medoids_per_class",
                    minimum=1,
                ),
            )
        resolved_selected_k = _resolve_single_selected_k(
            explicit_value=self.selected_k_per_class,
            legacy_value=self.k_medoids_per_class,
            effective_engine_deep_config=self.effective_engine_deep_config,
            field_name_prefix="DeepPromotionCandidate",
        )
        object.__setattr__(self, "selected_k_per_class", resolved_selected_k)
        object.__setattr__(
            self,
            "k_medoids_search_values",
            _copy_int_tuple(
                self.k_medoids_search_values,
                field_name="DeepPromotionCandidate.k_medoids_search_values",
                minimum=1,
            ),
        )
        if self.provenance is not None and self.provenance.stage_key is not None:
            if self.provenance.stage_key not in (StageKey.DEEP_SMOKE, StageKey.DEEP_MAIN_SCREEN):
                raise ContractValidationError(
                    "Deep candidate provenance can only target deep search stages in slice 6C."
                )
        if (
            self.scout_alertability_status == ScoutAlertabilityStatus.DEAD_DETECTOR
            and self.ranking_eligible
        ):
            raise ContractValidationError(
                "Dead-detector deep candidates must have `ranking_eligible=False`."
            )


@dataclass(frozen=True, slots=True)
class StageRequest(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_stage_request"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-stage-request-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    output_dir: str
    input_artifacts: tuple[ArtifactRef, ...] = ()
    frontend_inputs: tuple[FrontendInput, ...] = ()
    deep_inputs: tuple[DeepInputRef, ...] = ()
    parent_anchor_ref: ParentAnchorRef | None = None
    promotion_stage: PromotionStage | None = None
    execution_trace: ExecutionTrace | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(self.stage_key, StageKey, field_name="StageRequest.stage_key"),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(self.stage_name, field_name="StageRequest.stage_name"),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(self.campaign_id, field_name="StageRequest.campaign_id"),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="StageRequest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "output_dir",
            _require_str(self.output_dir, field_name="StageRequest.output_dir"),
        )
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        object.__setattr__(self, "frontend_inputs", tuple(self.frontend_inputs))
        object.__setattr__(self, "deep_inputs", tuple(self.deep_inputs))
        object.__setattr__(self, "notes", _copy_str_tuple(self.notes, field_name="StageRequest.notes"))
        if self.promotion_stage is not None:
            object.__setattr__(
                self,
                "promotion_stage",
                _coerce_enum(
                    self.promotion_stage,
                    PromotionStage,
                    field_name="StageRequest.promotion_stage",
                ),
            )


@dataclass(frozen=True, slots=True)
class StageResult(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_stage_result"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-stage-result-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    exact_blocker: str | None = None
    primary_artifact_ref: ArtifactRef | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()
    retained_alternates: tuple[RetainedAlternateRef, ...] = ()
    execution_trace: ExecutionTrace | None = None
    compliance_checks: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(self.stage_key, StageKey, field_name="StageResult.stage_key"),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(self.stage_name, field_name="StageResult.stage_name"),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(self.campaign_id, field_name="StageResult.campaign_id"),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="StageResult.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(self.pass_fail, PassFail, field_name="StageResult.pass_fail"),
        )
        object.__setattr__(
            self,
            "exact_blocker",
            _require_optional_str(
                self.exact_blocker,
                field_name="StageResult.exact_blocker",
            ),
        )
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "retained_alternates", tuple(self.retained_alternates))
        object.__setattr__(
            self,
            "compliance_checks",
            _copy_bool_mapping(
                _require_mapping(
                    self.compliance_checks,
                    field_name="StageResult.compliance_checks",
                ),
                field_name="StageResult.compliance_checks",
            ),
        )


@dataclass(frozen=True, slots=True)
class CampaignRequest(SerializableModel):
    campaign_id: str
    campaign_seed: int
    output_dir: str
    stage_sequence: tuple[StageKey, ...] = ()
    search_policy: StageSearchPolicy | None = None
    stage_search_policies: Mapping[str, StageSearchPolicy] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="CampaignRequest.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="CampaignRequest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "output_dir",
            _require_str(self.output_dir, field_name="CampaignRequest.output_dir"),
        )
        object.__setattr__(
            self,
            "notes",
            _copy_str_tuple(self.notes, field_name="CampaignRequest.notes"),
        )
        object.__setattr__(
            self,
            "metadata",
            _copy_json_mapping(
                _require_mapping(
                    self.metadata,
                    field_name="CampaignRequest.metadata",
                ),
                field_name="CampaignRequest.metadata",
            ),
        )
        from bittrace.v3.pipeline import canonical_stage_sequence, validate_canonical_stage_order

        stage_sequence = tuple(self.stage_sequence)
        if not stage_sequence:
            stage_sequence = canonical_stage_sequence()
        resolved_stage_sequence = validate_canonical_stage_order(
            stage_sequence,
            field_name="CampaignRequest.stage_sequence",
        )
        object.__setattr__(
            self,
            "stage_sequence",
            resolved_stage_sequence,
        )
        resolved_stage_search_policies: dict[StageKey, StageSearchPolicy] = {}
        for raw_stage_key, policy in self.stage_search_policies.items():
            stage_key = _coerce_enum(
                raw_stage_key,
                StageKey,
                field_name="CampaignRequest.stage_search_policies.<key>",
            )
            if stage_key not in (
                StageKey.LEAN_SMOKE,
                StageKey.DEEP_SMOKE,
                StageKey.LEAN_MAIN_SCREEN,
                StageKey.DEEP_MAIN_SCREEN,
            ):
                raise ContractValidationError(
                    "Stage search policy overrides are only valid for canonical search stages."
                )
            if stage_key not in resolved_stage_sequence:
                raise ContractValidationError(
                    "Stage search policy overrides must only target requested canonical stages."
                )
            resolved_stage_search_policies[stage_key] = policy
        object.__setattr__(self, "stage_search_policies", resolved_stage_search_policies)


@dataclass(frozen=True, slots=True)
class CampaignStageLineage(SerializableModel):
    stage_key: StageKey
    stage_request_ref: ArtifactRef | None = None
    stage_result_ref: ArtifactRef | None = None
    input_stage_artifact_refs: tuple[ArtifactRef, ...] = ()
    input_promoted_winner_refs: tuple[ArtifactRef, ...] = ()
    emitted_artifact_refs: tuple[ArtifactRef, ...] = ()
    promoted_winner_refs: tuple[ArtifactRef, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="CampaignStageLineage.stage_key",
            ),
        )
        if self.stage_request_ref is not None and self.stage_request_ref.kind != StageRequest.KIND:
            raise ContractValidationError(
                "Campaign stage lineage `stage_request_ref` must reference a V3 stage request artifact."
            )
        if self.stage_result_ref is not None and self.stage_result_ref.kind != StageResult.KIND:
            raise ContractValidationError(
                "Campaign stage lineage `stage_result_ref` must reference a V3 stage result artifact."
            )
        for field_name in (
            "input_stage_artifact_refs",
            "input_promoted_winner_refs",
            "emitted_artifact_refs",
            "promoted_winner_refs",
        ):
            refs = tuple(getattr(self, field_name))
            if len(set(refs)) != len(refs):
                raise ContractValidationError(
                    f"`CampaignStageLineage.{field_name}` must not contain duplicate artifact refs."
                )
            object.__setattr__(self, field_name, refs)
        object.__setattr__(
            self,
            "notes",
            _copy_str_tuple(self.notes, field_name="CampaignStageLineage.notes"),
        )


@dataclass(frozen=True, slots=True)
class CampaignManifest(ArtifactContract):
    KIND: ClassVar[str] = "bt3.campaign_manifest"
    SCHEMA_VERSION: ClassVar[str] = "bt3.campaign_manifest.v1"

    campaign_request: CampaignRequest
    stages: tuple[CampaignStageLineage, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self,
            "notes",
            _copy_str_tuple(self.notes, field_name="CampaignManifest.notes"),
        )
        from bittrace.v3.pipeline import validate_campaign_manifest_stages

        validate_campaign_manifest_stages(
            self.campaign_request,
            self.stages,
            field_name="CampaignManifest.stages",
        )


@dataclass(frozen=True, slots=True)
class CampaignResult(ArtifactContract):
    KIND: ClassVar[str] = "bt3.campaign_result"
    SCHEMA_VERSION: ClassVar[str] = "bt3.campaign_result.v1"

    campaign_request: CampaignRequest
    campaign_manifest_ref: ArtifactRef | None = None
    completed_stages: tuple[CampaignStageLineage, ...] = ()
    failed_stage: CampaignStageLineage | None = None
    final_promoted_winner_refs: tuple[ArtifactRef, ...] = ()
    freeze_export_refs: tuple[ArtifactRef, ...] = ()
    verification_refs: tuple[ArtifactRef, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_stages", tuple(self.completed_stages))
        for field_name in (
            "final_promoted_winner_refs",
            "freeze_export_refs",
            "verification_refs",
        ):
            refs = tuple(getattr(self, field_name))
            if len(set(refs)) != len(refs):
                raise ContractValidationError(
                    f"`CampaignResult.{field_name}` must not contain duplicate artifact refs."
                )
            object.__setattr__(self, field_name, refs)
        object.__setattr__(
            self,
            "notes",
            _copy_str_tuple(self.notes, field_name="CampaignResult.notes"),
        )
        from bittrace.v3.pipeline import validate_campaign_result

        validate_campaign_result(
            campaign_request=self.campaign_request,
            completed_stages=self.completed_stages,
            failed_stage=self.failed_stage,
            final_promoted_winner_refs=self.final_promoted_winner_refs,
            freeze_export_refs=self.freeze_export_refs,
            verification_refs=self.verification_refs,
            field_name="CampaignResult",
        )


@dataclass(frozen=True, slots=True)
class PromotedFrontendWinner(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_promoted_frontend_winner"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-promoted-frontend-winner-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    frontend_input_ref: FrontendInput | None = None
    promoted_candidate: FrontendPromotionCandidate | None = None
    downstream_deep_input: DeepInputRef | None = None
    exact_blocker: str | None = None
    ranking_mode: FrontendRankingMode = FrontendRankingMode.ENCODER_PROXY_QUALITY
    ranking_policy: Mapping[str, object] = field(default_factory=dict)
    retained_alternates: tuple[RetainedAlternateRef, ...] = ()
    execution_trace: ExecutionTrace | None = None
    compliance_checks: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="PromotedFrontendWinner.stage_key",
            ),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="PromotedFrontendWinner.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="PromotedFrontendWinner.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="PromotedFrontendWinner.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(
                self.pass_fail,
                PassFail,
                field_name="PromotedFrontendWinner.pass_fail",
            ),
        )
        object.__setattr__(
            self,
            "exact_blocker",
            _require_optional_str(
                self.exact_blocker,
                field_name="PromotedFrontendWinner.exact_blocker",
            ),
        )
        object.__setattr__(
            self,
            "ranking_mode",
            _coerce_enum(
                self.ranking_mode,
                FrontendRankingMode,
                field_name="PromotedFrontendWinner.ranking_mode",
            ),
        )
        if self.ranking_mode != FrontendRankingMode.ENCODER_PROXY_QUALITY:
            raise ContractValidationError(
                "Frontend ranking mode must be `encoder_proxy_quality`."
            )
        object.__setattr__(
            self,
            "ranking_policy",
            _copy_json_mapping(
                _require_mapping(
                    self.ranking_policy,
                    field_name="PromotedFrontendWinner.ranking_policy",
                ),
                field_name="PromotedFrontendWinner.ranking_policy",
            ),
        )
        object.__setattr__(self, "retained_alternates", tuple(self.retained_alternates))
        object.__setattr__(
            self,
            "compliance_checks",
            _copy_bool_mapping(
                _require_mapping(
                    self.compliance_checks,
                    field_name="PromotedFrontendWinner.compliance_checks",
                ),
                field_name="PromotedFrontendWinner.compliance_checks",
            ),
        )
        if self.pass_fail == PassFail.PASS:
            if self.promoted_candidate is None:
                raise ContractValidationError(
                    "Passing frontend promotions must include `promoted_candidate`."
                )
            if self.frontend_input_ref is None:
                raise ContractValidationError(
                    "Passing frontend promotions must include `frontend_input_ref`."
                )
            if self.downstream_deep_input is None:
                raise ContractValidationError(
                    "Passing frontend promotions must include `downstream_deep_input`."
                )


@dataclass(frozen=True, slots=True)
class PromotedDeepResult(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_promoted_deep_result"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-promoted-deep-result-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    promotion_stage: PromotionStage
    ranking_mode: DeepRankingMode = DeepRankingMode.SCOUT_UNHEALTHY_ALERT
    promoted_candidate: DeepPromotionCandidate | None = None
    frontend_inputs: tuple[DeepInputRef, ...] = ()
    parent_anchor_ref: ParentAnchorRef | None = None
    tested_k_candidates: tuple[DeepPromotionCandidate, ...] = ()
    k_candidates_tested: tuple[int, ...] = ()
    retained_alternates: tuple[RetainedAlternateRef, ...] = ()
    canonical_downstream_reference: ArtifactRef | None = None
    exact_blocker: str | None = None
    device_agnostic_export: Mapping[str, object] = field(default_factory=dict)
    ranking_policy: Mapping[str, object] = field(default_factory=dict)
    execution_trace: ExecutionTrace | None = None
    compliance_checks: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="PromotedDeepResult.stage_key",
            ),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(self.stage_name, field_name="PromotedDeepResult.stage_name"),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(self.campaign_id, field_name="PromotedDeepResult.campaign_id"),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="PromotedDeepResult.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(
                self.pass_fail,
                PassFail,
                field_name="PromotedDeepResult.pass_fail",
            ),
        )
        object.__setattr__(
            self,
            "promotion_stage",
            _coerce_enum(
                self.promotion_stage,
                PromotionStage,
                field_name="PromotedDeepResult.promotion_stage",
            ),
        )
        object.__setattr__(
            self,
            "ranking_mode",
            _coerce_enum(
                self.ranking_mode,
                DeepRankingMode,
                field_name="PromotedDeepResult.ranking_mode",
            ),
        )
        if self.ranking_mode != DeepRankingMode.SCOUT_UNHEALTHY_ALERT:
            raise ContractValidationError(
                "Deep ranking mode must be `scout_unhealthy_alert`."
            )
        object.__setattr__(self, "frontend_inputs", tuple(self.frontend_inputs))
        object.__setattr__(self, "tested_k_candidates", tuple(self.tested_k_candidates))
        tested_k_values_from_candidates = tuple(
            _require_candidate_selected_k(
                candidate,
                field_name=f"PromotedDeepResult.tested_k_candidates[{index}]",
                required=self.promotion_stage == PromotionStage.CAPACITY_REFINEMENT,
            )
            for index, candidate in enumerate(self.tested_k_candidates)
        )
        if any(value is not None for value in tested_k_values_from_candidates):
            _validate_unique_k_values(
                tuple(value for value in tested_k_values_from_candidates if value is not None),
                field_name="PromotedDeepResult.tested_k_candidates",
            )
        resolved_k_candidates_tested = _copy_int_tuple(
            self.k_candidates_tested,
            field_name="PromotedDeepResult.k_candidates_tested",
            minimum=1,
        )
        if not resolved_k_candidates_tested and tested_k_values_from_candidates:
            resolved_k_candidates_tested = tuple(
                value for value in tested_k_values_from_candidates if value is not None
            )
        if resolved_k_candidates_tested:
            _validate_unique_k_values(
                resolved_k_candidates_tested,
                field_name="PromotedDeepResult.k_candidates_tested",
            )
        object.__setattr__(self, "k_candidates_tested", resolved_k_candidates_tested)
        object.__setattr__(self, "retained_alternates", tuple(self.retained_alternates))
        object.__setattr__(
            self,
            "exact_blocker",
            _require_optional_str(
                self.exact_blocker,
                field_name="PromotedDeepResult.exact_blocker",
            ),
        )
        object.__setattr__(
            self,
            "device_agnostic_export",
            _copy_json_mapping(
                _require_mapping(
                    self.device_agnostic_export,
                    field_name="PromotedDeepResult.device_agnostic_export",
                ),
                field_name="PromotedDeepResult.device_agnostic_export",
            ),
        )
        object.__setattr__(
            self,
            "ranking_policy",
            _copy_json_mapping(
                _require_mapping(
                    self.ranking_policy,
                    field_name="PromotedDeepResult.ranking_policy",
                ),
                field_name="PromotedDeepResult.ranking_policy",
            ),
        )
        object.__setattr__(
            self,
            "compliance_checks",
            _copy_bool_mapping(
                _require_mapping(
                    self.compliance_checks,
                    field_name="PromotedDeepResult.compliance_checks",
                ),
                field_name="PromotedDeepResult.compliance_checks",
            ),
        )
        if self.pass_fail == PassFail.PASS and self.promoted_candidate is None:
            raise ContractValidationError(
                "Passing deep promotions must include `promoted_candidate`."
            )
        if (
            self.promotion_stage == PromotionStage.CAPACITY_REFINEMENT
            and not self.tested_k_candidates
        ):
            raise ContractValidationError(
                "Capacity-refinement promotions must carry tested `k` candidates."
            )
        if (
            self.promotion_stage == PromotionStage.CAPACITY_REFINEMENT
            and not self.k_candidates_tested
        ):
            raise ContractValidationError(
                "Capacity-refinement promotions must record `k_candidates_tested`."
            )
        if (
            self.promotion_stage == PromotionStage.CAPACITY_REFINEMENT
            and tested_k_values_from_candidates
            and tuple(value for value in tested_k_values_from_candidates if value is not None)
            != self.k_candidates_tested
        ):
            raise ContractValidationError(
                "Capacity-refinement promotions must keep `k_candidates_tested` consistent "
                "with the per-`k` candidate summaries."
            )
        if (
            self.promotion_stage == PromotionStage.CAPACITY_REFINEMENT
            and self.promoted_candidate is not None
        ):
            winning_k = _require_candidate_selected_k(
                self.promoted_candidate,
                field_name="PromotedDeepResult.promoted_candidate",
                required=True,
            )
            if winning_k not in self.k_candidates_tested:
                raise ContractValidationError(
                    "Capacity-refinement promotions must include the winning `k` in "
                    "`tested_k_candidates`."
                )


@dataclass(frozen=True, slots=True)
class FreezeExportProvenance(SerializableModel):
    source_stage_request: StageRequest
    source_promoted_deep_result_ref: ArtifactRef
    source_promoted_deep_result: PromotedDeepResult
    source_frontend_input: DeepInputRef | None = None
    source_frontend_promotion_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.source_promoted_deep_result_ref.kind != PromotedDeepResult.KIND:
            raise ContractValidationError(
                "Freeze/export provenance must reference a promoted Deep result artifact."
            )
        if self.source_promoted_deep_result.stage_key != StageKey.CAPACITY_REFINEMENT:
            raise ContractValidationError(
                "Freeze/export provenance must come from canonical S5 `capacity_refinement`."
            )
        if self.source_promoted_deep_result.promotion_stage != PromotionStage.CAPACITY_REFINEMENT:
            raise ContractValidationError(
                "Freeze/export provenance must come from `promotion_stage=capacity_refinement`."
            )
        if (
            self.source_promoted_deep_result.pass_fail == PassFail.PASS
            and self.source_promoted_deep_result.promoted_candidate is None
        ):
            raise ContractValidationError(
                "Passing freeze/export provenance must include the promoted S5 Deep winner."
            )
        if self.source_frontend_input is not None:
            matching_inputs = tuple(
                frontend_input
                for frontend_input in self.source_promoted_deep_result.frontend_inputs
                if frontend_input.frontend_input_id == self.source_frontend_input.frontend_input_id
                and frontend_input.frontend_fingerprint == self.source_frontend_input.frontend_fingerprint
            )
            if not matching_inputs:
                raise ContractValidationError(
                    "Freeze/export provenance frontend lineage must be present in the source S5 promoted Deep result."
                )
        if self.source_frontend_promotion_ref is not None and self.source_frontend_input is not None:
            expected_ref = self.source_frontend_input.source_promotion_artifact_ref
            if expected_ref is not None and not _artifact_ref_matches_lineage(
                expected_ref,
                self.source_frontend_promotion_ref,
            ):
                raise ContractValidationError(
                    "Freeze/export provenance frontend-promotion ref must match the promoted frontend lineage."
                )


@dataclass(frozen=True, slots=True)
class DeployedDeepWinner(SerializableModel):
    candidate_id: str
    candidate_order: int
    branch_mode: str
    selected_k_per_class: int
    frontend_input_id: str | None = None
    frontend_fingerprint: str | None = None
    parent_anchor_fingerprint: str | None = None
    best_deep_artifact_ref: ArtifactRef | None = None
    metrics_summary_ref: ArtifactRef | None = None
    candidate_report_ref: ArtifactRef | None = None
    checkpoint_ref: ArtifactRef | None = None
    effective_engine_deep_config: Mapping[str, object] = field(default_factory=dict)
    direct_restore_payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_str(self.candidate_id, field_name="DeployedDeepWinner.candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_order",
            _require_int(
                self.candidate_order,
                field_name="DeployedDeepWinner.candidate_order",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "branch_mode",
            _require_str(self.branch_mode, field_name="DeployedDeepWinner.branch_mode"),
        )
        object.__setattr__(
            self,
            "selected_k_per_class",
            _require_int(
                self.selected_k_per_class,
                field_name="DeployedDeepWinner.selected_k_per_class",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "frontend_input_id",
            _require_optional_str(
                self.frontend_input_id,
                field_name="DeployedDeepWinner.frontend_input_id",
            ),
        )
        object.__setattr__(
            self,
            "frontend_fingerprint",
            _require_optional_str(
                self.frontend_fingerprint,
                field_name="DeployedDeepWinner.frontend_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "parent_anchor_fingerprint",
            _require_optional_str(
                self.parent_anchor_fingerprint,
                field_name="DeployedDeepWinner.parent_anchor_fingerprint",
            ),
        )
        effective_engine_deep_config = _copy_json_mapping(
            _require_mapping(
                self.effective_engine_deep_config,
                field_name="DeployedDeepWinner.effective_engine_deep_config",
            ),
            field_name="DeployedDeepWinner.effective_engine_deep_config",
        )
        _require_absent_json_key(
            effective_engine_deep_config,
            key_name="k_candidates_tested",
            field_name="DeployedDeepWinner.effective_engine_deep_config",
        )
        _require_absent_json_key(
            effective_engine_deep_config,
            key_name="k_medoids_search_values",
            field_name="DeployedDeepWinner.effective_engine_deep_config",
        )
        _require_absent_json_key(
            effective_engine_deep_config,
            key_name="promotion_stage",
            field_name="DeployedDeepWinner.effective_engine_deep_config",
        )
        for forbidden_key in (
            "exploration_fraction",
            "exploration_mode",
            "refinement_mode",
            "random_seed",
            "sampled_values",
        ):
            _require_absent_json_key(
                effective_engine_deep_config,
                key_name=forbidden_key,
                field_name="DeployedDeepWinner.effective_engine_deep_config",
            )
        config_k = effective_engine_deep_config.get("k_medoids_per_class")
        if config_k is not None and _require_int(
            config_k,
            field_name="DeployedDeepWinner.effective_engine_deep_config.k_medoids_per_class",
            minimum=1,
        ) != self.selected_k_per_class:
            raise ContractValidationError(
                "Deployed Deep winner config must lock `k_medoids_per_class` to the promoted S5 winning `k`."
            )
        object.__setattr__(self, "effective_engine_deep_config", effective_engine_deep_config)
        direct_restore_payload = _copy_json_mapping(
            _require_mapping(
                self.direct_restore_payload,
                field_name="DeployedDeepWinner.direct_restore_payload",
            ),
            field_name="DeployedDeepWinner.direct_restore_payload",
        )
        _require_absent_json_key(
            direct_restore_payload,
            key_name="k_candidates_tested",
            field_name="DeployedDeepWinner.direct_restore_payload",
        )
        _require_absent_json_key(
            direct_restore_payload,
            key_name="k_medoids_search_values",
            field_name="DeployedDeepWinner.direct_restore_payload",
        )
        _require_absent_json_key(
            direct_restore_payload,
            key_name="promotion_stage",
            field_name="DeployedDeepWinner.direct_restore_payload",
        )
        for forbidden_key in (
            "exploration_fraction",
            "exploration_mode",
            "refinement_mode",
            "random_seed",
            "sampled_values",
        ):
            _require_absent_json_key(
                direct_restore_payload,
                key_name=forbidden_key,
                field_name="DeployedDeepWinner.direct_restore_payload",
            )
        object.__setattr__(self, "direct_restore_payload", direct_restore_payload)


@dataclass(frozen=True, slots=True)
class FreezeExportDeployRuntime(SerializableModel):
    selected_k_per_class: int
    deploy_path: str = "pure_symbolic"
    deep_owns_downstream_classification_export: bool = True
    direct_restore_supported: bool = True
    anchor_artifact_ref: ArtifactRef | None = None
    frontend_export_reference_ref: ArtifactRef | None = None
    winner_artifact_refs: tuple[ArtifactRef, ...] = ()
    winner_best_deep_artifact_ref: ArtifactRef | None = None
    winner_metrics_summary_ref: ArtifactRef | None = None
    winner_candidate_report_ref: ArtifactRef | None = None
    winner_checkpoint_ref: ArtifactRef | None = None
    device_agnostic_export: Mapping[str, object] = field(default_factory=dict)
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_k_per_class",
            _require_int(
                self.selected_k_per_class,
                field_name="FreezeExportDeployRuntime.selected_k_per_class",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "deploy_path",
            _require_str(self.deploy_path, field_name="FreezeExportDeployRuntime.deploy_path"),
        )
        if self.deploy_path != "pure_symbolic":
            raise ContractValidationError("Canonical S6 deploy path must remain `pure_symbolic`.")
        object.__setattr__(
            self,
            "deep_owns_downstream_classification_export",
            _require_bool(
                self.deep_owns_downstream_classification_export,
                field_name="FreezeExportDeployRuntime.deep_owns_downstream_classification_export",
            ),
        )
        if not self.deep_owns_downstream_classification_export:
            raise ContractValidationError(
                "Canonical S6 requires Deep to own downstream classification/export."
            )
        object.__setattr__(
            self,
            "direct_restore_supported",
            _require_bool(
                self.direct_restore_supported,
                field_name="FreezeExportDeployRuntime.direct_restore_supported",
            ),
        )
        if not self.direct_restore_supported:
            raise ContractValidationError(
                "Canonical S6 requires a direct-restore-compatible Deep winner payload."
            )
        object.__setattr__(self, "winner_artifact_refs", tuple(self.winner_artifact_refs))
        object.__setattr__(
            self,
            "device_agnostic_export",
            _copy_device_agnostic_export_mapping(
                self.device_agnostic_export,
                field_name="FreezeExportDeployRuntime.device_agnostic_export",
                require_portable=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class DeepAnchorArtifact(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_deep_anchor_artifact"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-deep-anchor-artifact-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    summary: str
    provenance: FreezeExportProvenance
    frontend_lineage: DeepInputRef
    deployed_winner: DeployedDeepWinner | None = None
    winning_deep_candidate_fingerprint: str | None = None
    export_portability: Mapping[str, object] = field(default_factory=dict)
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(self.stage_key, StageKey, field_name="DeepAnchorArtifact.stage_key"),
        )
        if self.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Deep anchor artifacts are only valid for `winner_deepen_freeze_export`."
            )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(self.stage_name, field_name="DeepAnchorArtifact.stage_name"),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(self.campaign_id, field_name="DeepAnchorArtifact.campaign_id"),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="DeepAnchorArtifact.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(self.pass_fail, PassFail, field_name="DeepAnchorArtifact.pass_fail"),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="DeepAnchorArtifact.summary"),
        )
        object.__setattr__(
            self,
            "winning_deep_candidate_fingerprint",
            _require_optional_str(
                self.winning_deep_candidate_fingerprint,
                field_name="DeepAnchorArtifact.winning_deep_candidate_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "export_portability",
            _copy_device_agnostic_export_mapping(
                self.export_portability,
                field_name="DeepAnchorArtifact.export_portability",
                require_portable=self.pass_fail == PassFail.PASS,
            ),
        )
        if self.pass_fail == PassFail.PASS:
            if self.deployed_winner is None:
                raise ContractValidationError(
                    "Passing deep anchor artifacts must include `deployed_winner`."
                )
            if self.winning_deep_candidate_fingerprint is None:
                raise ContractValidationError(
                    "Passing deep anchor artifacts must include `winning_deep_candidate_fingerprint`."
                )
            source_winner = self.provenance.source_promoted_deep_result.promoted_candidate
            if source_winner is None:
                raise ContractValidationError(
                    "Passing deep anchor artifacts require a passing promoted S5 Deep result in provenance."
                )
            if self.deployed_winner.selected_k_per_class != source_winner.selected_k_per_class:
                raise ContractValidationError(
                    "Deep anchor artifacts must lock `selected_k_per_class` to the actual promoted S5 winner."
                )
            if not _artifact_ref_matches_lineage(
                self.frontend_lineage.source_promotion_artifact_ref,
                self.provenance.source_frontend_promotion_ref,
            ):
                raise ContractValidationError(
                    "Deep anchor frontend lineage must preserve the promoted frontend artifact ref explicitly."
                )


@dataclass(frozen=True, slots=True)
class FrontendExportReference(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_frontend_export_reference"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-frontend-export-reference-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    summary: str
    source_stage_request: StageRequest
    source_promoted_deep_result_ref: ArtifactRef
    source_frontend_promotion_ref: ArtifactRef
    frontend_lineage: DeepInputRef
    deep_anchor_artifact_ref: ArtifactRef | None = None
    selected_k_per_class: int | None = None
    winning_deep_candidate_fingerprint: str | None = None
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="FrontendExportReference.stage_key",
            ),
        )
        if self.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Frontend export references are only valid for `winner_deepen_freeze_export`."
            )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="FrontendExportReference.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="FrontendExportReference.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="FrontendExportReference.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(
                self.pass_fail,
                PassFail,
                field_name="FrontendExportReference.pass_fail",
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="FrontendExportReference.summary"),
        )
        if self.source_promoted_deep_result_ref.kind != PromotedDeepResult.KIND:
            raise ContractValidationError(
                "Frontend export references must point back to the promoted S5 Deep result."
            )
        if (
            self.frontend_lineage.source_promotion_artifact_ref is not None
            and not _artifact_ref_matches_lineage(
                self.frontend_lineage.source_promotion_artifact_ref,
                self.source_frontend_promotion_ref,
            )
        ):
            raise ContractValidationError(
                "Frontend export reference lineage must match the promoted frontend artifact ref."
            )
        if self.selected_k_per_class is not None:
            object.__setattr__(
                self,
                "selected_k_per_class",
                _require_int(
                    self.selected_k_per_class,
                    field_name="FrontendExportReference.selected_k_per_class",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "winning_deep_candidate_fingerprint",
            _require_optional_str(
                self.winning_deep_candidate_fingerprint,
                field_name="FrontendExportReference.winning_deep_candidate_fingerprint",
            ),
        )
        if self.pass_fail == PassFail.PASS:
            if self.deep_anchor_artifact_ref is None:
                raise ContractValidationError(
                    "Passing frontend export references must point to the emitted Deep anchor artifact."
                )
            if self.selected_k_per_class is None:
                raise ContractValidationError(
                    "Passing frontend export references must record the deployed selected `k`."
                )


@dataclass(frozen=True, slots=True)
class RetainedAlternatesManifest(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_retained_alternates_manifest"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-retained-alternates-manifest-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    alternates: tuple[RetainedAlternateRef, ...]
    summary: str = ""
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="RetainedAlternatesManifest.stage_key",
            ),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="RetainedAlternatesManifest.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="RetainedAlternatesManifest.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="RetainedAlternatesManifest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(self, "alternates", tuple(self.alternates))
        object.__setattr__(
            self,
            "summary",
            _require_str(
                self.summary,
                field_name="RetainedAlternatesManifest.summary",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class FreezeExportManifest(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_freeze_export_manifest"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-freeze-export-manifest-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    summary: str
    placeholder: bool = True
    promotion_stage: PromotionStage | None = None
    anchor_artifact_ref: ArtifactRef | None = None
    frontend_export_reference_ref: ArtifactRef | None = None
    winner_artifact_refs: tuple[ArtifactRef, ...] = ()
    source_promoted_deep_result_ref: ArtifactRef | None = None
    provenance: FreezeExportProvenance | None = None
    deploy_runtime: FreezeExportDeployRuntime | None = None
    exact_blocker: str | None = None
    device_agnostic_export: Mapping[str, object] = field(default_factory=dict)
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="FreezeExportManifest.stage_key",
            ),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="FreezeExportManifest.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="FreezeExportManifest.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="FreezeExportManifest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(
                self.pass_fail,
                PassFail,
                field_name="FreezeExportManifest.pass_fail",
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="FreezeExportManifest.summary"),
        )
        object.__setattr__(
            self,
            "placeholder",
            _require_bool(
                self.placeholder,
                field_name="FreezeExportManifest.placeholder",
            ),
        )
        if self.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Freeze/export manifests are only valid for `winner_deepen_freeze_export`."
            )
        object.__setattr__(self, "winner_artifact_refs", tuple(self.winner_artifact_refs))
        object.__setattr__(
            self,
            "exact_blocker",
            _require_optional_str(
                self.exact_blocker,
                field_name="FreezeExportManifest.exact_blocker",
            ),
        )
        object.__setattr__(
            self,
            "device_agnostic_export",
            _copy_device_agnostic_export_mapping(
                self.device_agnostic_export,
                field_name="FreezeExportManifest.device_agnostic_export",
                require_portable=self.pass_fail == PassFail.PASS and not self.placeholder,
            ),
        )
        if self.promotion_stage is not None:
            object.__setattr__(
                self,
                "promotion_stage",
                _coerce_enum(
                    self.promotion_stage,
                    PromotionStage,
                    field_name="FreezeExportManifest.promotion_stage",
                ),
            )
        if self.pass_fail == PassFail.PASS and not self.placeholder:
            if self.anchor_artifact_ref is None:
                raise ContractValidationError(
                    "Passing freeze/export manifests must include `anchor_artifact_ref`."
                )
            if self.frontend_export_reference_ref is None:
                raise ContractValidationError(
                    "Passing freeze/export manifests must include `frontend_export_reference_ref`."
                )
            if self.source_promoted_deep_result_ref is None:
                raise ContractValidationError(
                    "Passing freeze/export manifests must include `source_promoted_deep_result_ref`."
                )
            if self.provenance is None:
                raise ContractValidationError(
                    "Passing freeze/export manifests must distinguish `provenance` explicitly."
                )
            if self.deploy_runtime is None:
                raise ContractValidationError(
                    "Passing freeze/export manifests must distinguish deploy/runtime fields explicitly."
                )
            if self.deploy_runtime.anchor_artifact_ref != self.anchor_artifact_ref:
                raise ContractValidationError(
                    "Freeze/export manifest deploy/runtime anchor ref must match `anchor_artifact_ref`."
                )
            if self.deploy_runtime.frontend_export_reference_ref != self.frontend_export_reference_ref:
                raise ContractValidationError(
                    "Freeze/export manifest deploy/runtime frontend export ref must match the top-level ref."
                )
            if self.provenance.source_promoted_deep_result_ref != self.source_promoted_deep_result_ref:
                raise ContractValidationError(
                    "Freeze/export manifest provenance must preserve the promoted S5 Deep result ref explicitly."
                )
            if self.anchor_artifact_ref not in self.winner_artifact_refs:
                raise ContractValidationError(
                    "Freeze/export manifest winner refs must include the emitted Deep anchor artifact."
                )
            if self.frontend_export_reference_ref not in self.winner_artifact_refs:
                raise ContractValidationError(
                    "Freeze/export manifest winner refs must include the emitted frontend export reference."
                )


@dataclass(frozen=True, slots=True)
class ParityVerificationBundle(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_parity_verification_bundle"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-parity-verification-bundle-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    source_fixture_ids: tuple[str, ...] = ()
    expected_row_digest_refs: tuple[ArtifactRef, ...] = ()
    placeholder: bool = True
    adapter_profile_version: str | None = None
    canonical_input_contract_version: str | None = None
    bundle_fingerprint: str | None = None
    frontend_fingerprint: str | None = None
    deep_anchor_fingerprint: str | None = None
    exact_mismatch_location: str | None = None
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="ParityVerificationBundle.stage_key",
            ),
        )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="ParityVerificationBundle.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="ParityVerificationBundle.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="ParityVerificationBundle.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(
                self.pass_fail,
                PassFail,
                field_name="ParityVerificationBundle.pass_fail",
            ),
        )
        object.__setattr__(
            self,
            "source_fixture_ids",
            _copy_str_tuple(
                self.source_fixture_ids,
                field_name="ParityVerificationBundle.source_fixture_ids",
            ),
        )
        object.__setattr__(
            self,
            "expected_row_digest_refs",
            tuple(self.expected_row_digest_refs),
        )
        object.__setattr__(
            self,
            "placeholder",
            _require_bool(
                self.placeholder,
                field_name="ParityVerificationBundle.placeholder",
            ),
        )
        optional_fields = (
            "adapter_profile_version",
            "canonical_input_contract_version",
            "bundle_fingerprint",
            "frontend_fingerprint",
            "deep_anchor_fingerprint",
            "exact_mismatch_location",
        )
        for field_name in optional_fields:
            object.__setattr__(
                self,
                field_name,
                _require_optional_str(
                    getattr(self, field_name),
                    field_name=f"ParityVerificationBundle.{field_name}",
                ),
            )


@dataclass(frozen=True, slots=True)
class PayloadSnapshot(SerializableModel):
    payload: Mapping[str, object]
    sha256: str

    def __post_init__(self) -> None:
        copied_payload = _copy_json_mapping(
            _require_mapping(self.payload, field_name="PayloadSnapshot.payload"),
            field_name="PayloadSnapshot.payload",
        )
        object.__setattr__(self, "payload", copied_payload)
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field_name="PayloadSnapshot.sha256"),
        )
        if self.sha256 != _canonical_json_sha256(copied_payload):
            raise ContractValidationError(
                "Payload snapshots must store the canonical SHA-256 of `payload` exactly."
            )


@dataclass(frozen=True, slots=True)
class GoldenVectorEntry(SerializableModel):
    vector_id: str
    verification_level: VerificationLevel
    canonical_input: PayloadSnapshot
    packed_frontend_input: PayloadSnapshot | None = None
    expected_class: JsonPrimitive = None
    expected_reject: bool = False
    expected_deploy_runtime_output: PayloadSnapshot | None = None
    non_deploy_telemetry_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_id",
            _require_str(self.vector_id, field_name="GoldenVectorEntry.vector_id"),
        )
        object.__setattr__(
            self,
            "verification_level",
            _coerce_enum(
                self.verification_level,
                VerificationLevel,
                field_name="GoldenVectorEntry.verification_level",
            ),
        )
        object.__setattr__(
            self,
            "expected_reject",
            _require_bool(
                self.expected_reject,
                field_name="GoldenVectorEntry.expected_reject",
            ),
        )
        object.__setattr__(
            self,
            "non_deploy_telemetry_fields",
            _copy_distinct_str_tuple(
                self.non_deploy_telemetry_fields,
                field_name="GoldenVectorEntry.non_deploy_telemetry_fields",
            ),
        )
        if not self.expected_reject and self.expected_class is None:
            raise ContractValidationError(
                "Golden-vector entries must preserve `expected_class` when `expected_reject` is false."
            )
        if (
            self.verification_level == VerificationLevel.FRONTEND_PARITY
            and self.packed_frontend_input is None
        ):
            raise ContractValidationError(
                "Frontend-parity golden vectors must include `packed_frontend_input`."
            )
        if (
            self.verification_level in {VerificationLevel.DEEP_PARITY, VerificationLevel.END_TO_END_PARITY}
            and self.expected_deploy_runtime_output is None
        ):
            raise ContractValidationError(
                "Deep and end-to-end parity golden vectors must include `expected_deploy_runtime_output`."
            )
        if (
            self.expected_deploy_runtime_output is None
            and self.non_deploy_telemetry_fields
        ):
            raise ContractValidationError(
                "Non-deploy telemetry fields may only be declared when deploy/runtime output parity is present."
            )


@dataclass(frozen=True, slots=True)
class VerificationKitManifest(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_verification_kit_manifest"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-verification-kit-manifest-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    summary: str
    source_stage_request: StageRequest
    verification_levels: tuple[VerificationLevel, ...]
    source_promoted_deep_result_ref: ArtifactRef
    source_promoted_deep_result: PromotedDeepResult
    source_freeze_export_manifest_ref: ArtifactRef
    source_freeze_export_manifest: FreezeExportManifest
    deep_anchor_artifact_ref: ArtifactRef
    deep_anchor_artifact: DeepAnchorArtifact
    frontend_export_reference_ref: ArtifactRef
    frontend_export_reference: FrontendExportReference
    deploy_runtime_non_deploy_telemetry_fields: tuple[str, ...] = ()
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="VerificationKitManifest.stage_key",
            ),
        )
        if self.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Verification-kit manifests are only valid for `parity_verification`."
            )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="VerificationKitManifest.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="VerificationKitManifest.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="VerificationKitManifest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="VerificationKitManifest.summary"),
        )
        object.__setattr__(
            self,
            "verification_levels",
            _copy_verification_level_tuple(
                self.verification_levels,
                field_name="VerificationKitManifest.verification_levels",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "deploy_runtime_non_deploy_telemetry_fields",
            _copy_distinct_str_tuple(
                self.deploy_runtime_non_deploy_telemetry_fields,
                field_name="VerificationKitManifest.deploy_runtime_non_deploy_telemetry_fields",
            ),
        )
        if self.source_stage_request.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Verification-kit manifests must preserve a parity-verification source stage request."
            )
        if self.stage_name != self.source_stage_request.stage_name:
            raise ContractValidationError(
                "Verification-kit `stage_name` must match the source stage request."
            )
        if self.campaign_id != self.source_stage_request.campaign_id:
            raise ContractValidationError(
                "Verification-kit `campaign_id` must match the source stage request."
            )
        if self.campaign_seed != self.source_stage_request.campaign_seed:
            raise ContractValidationError(
                "Verification-kit `campaign_seed` must match the source stage request."
            )
        if (
            self.source_stage_request.promotion_stage is not None
            and self.source_stage_request.promotion_stage != PromotionStage.CAPACITY_REFINEMENT
        ):
            raise ContractValidationError(
                "Verification-kit manifests only accept lineage from `promotion_stage=capacity_refinement`."
            )
        if self.source_freeze_export_manifest_ref.kind != FreezeExportManifest.KIND:
            raise ContractValidationError(
                "Verification-kit manifests must preserve the canonical S6 freeze/export manifest ref."
            )
        if self.source_freeze_export_manifest_ref.kind != self.source_freeze_export_manifest.kind:
            raise ContractValidationError(
                "Verification-kit freeze/export manifest ref kind must align with the embedded manifest."
            )
        if self.deep_anchor_artifact_ref.kind != DeepAnchorArtifact.KIND:
            raise ContractValidationError(
                "Verification-kit manifests must preserve the emitted Deep anchor artifact ref."
            )
        if self.deep_anchor_artifact_ref.kind != self.deep_anchor_artifact.kind:
            raise ContractValidationError(
                "Verification-kit Deep anchor ref kind must align with the embedded artifact."
            )
        if self.frontend_export_reference_ref.kind != FrontendExportReference.KIND:
            raise ContractValidationError(
                "Verification-kit manifests must preserve the emitted frontend export reference ref."
            )
        if self.frontend_export_reference_ref.kind != self.frontend_export_reference.kind:
            raise ContractValidationError(
                "Verification-kit frontend export ref kind must align with the embedded reference."
            )
        if self.source_promoted_deep_result_ref.kind != PromotedDeepResult.KIND:
            raise ContractValidationError(
                "Verification-kit manifests must preserve the promoted S5 Deep result ref."
            )
        if self.source_promoted_deep_result_ref.kind != self.source_promoted_deep_result.kind:
            raise ContractValidationError(
                "Verification-kit promoted-Deep ref kind must align with the embedded promoted Deep result."
            )
        for required_ref, label in (
            (self.source_freeze_export_manifest_ref, "source_freeze_export_manifest_ref"),
            (self.deep_anchor_artifact_ref, "deep_anchor_artifact_ref"),
            (self.frontend_export_reference_ref, "frontend_export_reference_ref"),
        ):
            if required_ref not in self.source_stage_request.input_artifacts:
                raise ContractValidationError(
                    f"Verification-kit source stage request must include `{label}` in `input_artifacts`."
                )
        if self.source_promoted_deep_result.stage_key != StageKey.CAPACITY_REFINEMENT:
            raise ContractValidationError(
                "Verification-kit manifests must preserve lineage to canonical S5 `capacity_refinement`."
            )
        if self.source_promoted_deep_result.pass_fail != PassFail.PASS:
            raise ContractValidationError(
                "Verification-kit manifests require a passing promoted S5 Deep result."
            )
        if self.source_freeze_export_manifest.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Verification-kit manifests must preserve lineage to canonical S6 freeze/export."
            )
        if self.source_freeze_export_manifest.pass_fail != PassFail.PASS:
            raise ContractValidationError(
                "Verification-kit manifests require a passing S6 freeze/export manifest."
            )
        if self.source_freeze_export_manifest.placeholder:
            raise ContractValidationError(
                "Verification-kit manifests require the frozen/exported S6 manifest, not a placeholder."
            )
        if self.source_freeze_export_manifest.anchor_artifact_ref != self.deep_anchor_artifact_ref:
            raise ContractValidationError(
                "Verification-kit manifests must preserve the exact Deep anchor ref from S6 freeze/export."
            )
        if (
            self.source_freeze_export_manifest.frontend_export_reference_ref
            != self.frontend_export_reference_ref
        ):
            raise ContractValidationError(
                "Verification-kit manifests must preserve the exact frontend export ref from S6 freeze/export."
            )
        if (
            self.source_freeze_export_manifest.source_promoted_deep_result_ref
            != self.source_promoted_deep_result_ref
        ):
            raise ContractValidationError(
                "Verification-kit manifests must preserve the promoted S5 Deep result ref explicitly."
            )
        if self.source_freeze_export_manifest.provenance is None:
            raise ContractValidationError(
                "Verification-kit manifests require the canonical S6 provenance payload."
            )
        if (
            self.source_freeze_export_manifest.provenance.source_promoted_deep_result_ref
            != self.source_promoted_deep_result_ref
        ):
            raise ContractValidationError(
                "Verification-kit manifests must keep S6 provenance aligned to the promoted S5 winner."
            )
        if self.deep_anchor_artifact.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Verification-kit manifests must embed the canonical S6 Deep anchor artifact."
            )
        if (
            self.deep_anchor_artifact.provenance.source_promoted_deep_result_ref
            != self.source_promoted_deep_result_ref
        ):
            raise ContractValidationError(
                "Verification-kit manifests must preserve the Deep anchor lineage to the promoted S5 winner."
            )
        if self.frontend_export_reference.stage_key != StageKey.WINNER_DEEPEN_FREEZE_EXPORT:
            raise ContractValidationError(
                "Verification-kit manifests must embed the canonical S6 frontend export reference."
            )
        if (
            self.frontend_export_reference.source_promoted_deep_result_ref
            != self.source_promoted_deep_result_ref
        ):
            raise ContractValidationError(
                "Verification-kit manifests must preserve the frontend export lineage to the promoted S5 winner."
            )
        if self.frontend_export_reference.deep_anchor_artifact_ref != self.deep_anchor_artifact_ref:
            raise ContractValidationError(
                "Verification-kit manifests must keep the frontend export reference aligned to the Deep anchor artifact."
            )


@dataclass(frozen=True, slots=True)
class GoldenVectorManifest(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_golden_vector_manifest"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-golden-vector-manifest-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    summary: str
    source_verification_kit_ref: ArtifactRef
    source_verification_kit: VerificationKitManifest
    entries: tuple[GoldenVectorEntry, ...] = ()
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="GoldenVectorManifest.stage_key",
            ),
        )
        if self.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Golden-vector manifests are only valid for `parity_verification`."
            )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(
                self.stage_name,
                field_name="GoldenVectorManifest.stage_name",
            ),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(
                self.campaign_id,
                field_name="GoldenVectorManifest.campaign_id",
            ),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="GoldenVectorManifest.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="GoldenVectorManifest.summary"),
        )
        if self.source_verification_kit_ref.kind != self.source_verification_kit.kind:
            raise ContractValidationError(
                "Golden-vector verification-kit ref kind must align with the embedded verification kit."
            )
        if self.source_verification_kit_ref.kind != VerificationKitManifest.KIND:
            raise ContractValidationError(
                "Golden-vector manifests must preserve lineage to a verification-kit manifest."
            )
        if self.source_verification_kit.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Golden-vector manifests must embed a parity-verification kit manifest."
            )
        if self.stage_name != self.source_verification_kit.stage_name:
            raise ContractValidationError(
                "Golden-vector `stage_name` must match the source verification kit."
            )
        if self.campaign_id != self.source_verification_kit.campaign_id:
            raise ContractValidationError(
                "Golden-vector `campaign_id` must match the source verification kit."
            )
        if self.campaign_seed != self.source_verification_kit.campaign_seed:
            raise ContractValidationError(
                "Golden-vector `campaign_seed` must match the source verification kit."
            )
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ContractValidationError(
                "Golden-vector manifests must contain at least one typed golden-vector entry."
            )
        seen: set[tuple[str, VerificationLevel]] = set()
        for index, entry in enumerate(self.entries):
            key = (entry.vector_id, entry.verification_level)
            if key in seen:
                raise ContractValidationError(
                    "Golden-vector manifests must not contain duplicate `(vector_id, verification_level)` entries."
                )
            seen.add(key)
            if entry.verification_level not in self.source_verification_kit.verification_levels:
                raise ContractValidationError(
                    f"`GoldenVectorManifest.entries[{index}]` uses a verification level that is not enabled in the source kit."
                )


@dataclass(frozen=True, slots=True)
class ParityMismatchDetail(SerializableModel):
    field_path: str
    expected: JsonValue
    actual: JsonValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_path",
            _require_str(
                self.field_path,
                field_name="ParityMismatchDetail.field_path",
            ),
        )


@dataclass(frozen=True, slots=True)
class ParityReportEntry(SerializableModel):
    vector_id: str
    verification_level: VerificationLevel
    comparison_status: ParityComparisonStatus
    expected_payload_sha256: str | None = None
    actual_payload_sha256: str | None = None
    expected_class: JsonPrimitive = None
    actual_class: JsonPrimitive = None
    expected_reject: bool | None = None
    actual_reject: bool | None = None
    non_deploy_telemetry_fields: tuple[str, ...] = ()
    mismatch_details: tuple[ParityMismatchDetail, ...] = ()
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_id",
            _require_str(self.vector_id, field_name="ParityReportEntry.vector_id"),
        )
        object.__setattr__(
            self,
            "verification_level",
            _coerce_enum(
                self.verification_level,
                VerificationLevel,
                field_name="ParityReportEntry.verification_level",
            ),
        )
        object.__setattr__(
            self,
            "comparison_status",
            _coerce_enum(
                self.comparison_status,
                ParityComparisonStatus,
                field_name="ParityReportEntry.comparison_status",
            ),
        )
        object.__setattr__(
            self,
            "expected_payload_sha256",
            _require_optional_sha256(
                self.expected_payload_sha256,
                field_name="ParityReportEntry.expected_payload_sha256",
            ),
        )
        object.__setattr__(
            self,
            "actual_payload_sha256",
            _require_optional_sha256(
                self.actual_payload_sha256,
                field_name="ParityReportEntry.actual_payload_sha256",
            ),
        )
        if self.expected_reject is not None:
            object.__setattr__(
                self,
                "expected_reject",
                _require_bool(
                    self.expected_reject,
                    field_name="ParityReportEntry.expected_reject",
                ),
            )
        if self.actual_reject is not None:
            object.__setattr__(
                self,
                "actual_reject",
                _require_bool(
                    self.actual_reject,
                    field_name="ParityReportEntry.actual_reject",
                ),
            )
        object.__setattr__(
            self,
            "non_deploy_telemetry_fields",
            _copy_distinct_str_tuple(
                self.non_deploy_telemetry_fields,
                field_name="ParityReportEntry.non_deploy_telemetry_fields",
            ),
        )
        object.__setattr__(self, "mismatch_details", tuple(self.mismatch_details))
        object.__setattr__(
            self,
            "unsupported_reason",
            _require_optional_str(
                self.unsupported_reason,
                field_name="ParityReportEntry.unsupported_reason",
            ),
        )
        if self.comparison_status == ParityComparisonStatus.EXACT_MATCH:
            if self.mismatch_details:
                raise ContractValidationError(
                    "Exact-match parity report entries must not contain mismatch details."
                )
            if self.unsupported_reason is not None:
                raise ContractValidationError(
                    "Exact-match parity report entries must not contain `unsupported_reason`."
                )
            if (
                self.expected_payload_sha256 is not None
                and self.actual_payload_sha256 is not None
                and self.expected_payload_sha256 != self.actual_payload_sha256
                and not self.non_deploy_telemetry_fields
            ):
                raise ContractValidationError(
                    "Exact-match parity report entries must keep payload hashes identical."
                )
            if self.expected_class != self.actual_class:
                raise ContractValidationError(
                    "Exact-match parity report entries must keep `expected_class` and `actual_class` identical."
                )
            if self.expected_reject != self.actual_reject:
                raise ContractValidationError(
                    "Exact-match parity report entries must keep `expected_reject` and `actual_reject` identical."
                )
        elif self.comparison_status == ParityComparisonStatus.MISMATCH:
            if self.unsupported_reason is not None:
                raise ContractValidationError(
                    "Mismatch parity report entries must not contain `unsupported_reason`."
                )
            if not self.mismatch_details:
                raise ContractValidationError(
                    "Mismatch parity report entries must record at least one mismatch detail."
                )
        else:
            if self.unsupported_reason is None:
                raise ContractValidationError(
                    "Unsupported parity report entries must record `unsupported_reason`."
                )
            if self.mismatch_details:
                raise ContractValidationError(
                    "Unsupported parity report entries must not contain mismatch details."
                )


@dataclass(frozen=True, slots=True)
class ParityReport(ArtifactContract):
    KIND: ClassVar[str] = "bittrace_v3_parity_report"
    SCHEMA_VERSION: ClassVar[str] = "bittrace-v3-parity-report-1"

    stage_key: StageKey
    stage_name: str
    campaign_id: str
    campaign_seed: int
    pass_fail: PassFail
    summary: str
    source_verification_kit_ref: ArtifactRef
    source_verification_kit: VerificationKitManifest
    source_golden_vector_manifest_ref: ArtifactRef
    source_golden_vector_manifest: GoldenVectorManifest
    results: tuple[ParityReportEntry, ...] = ()
    exact_match_count: int = 0
    mismatch_count: int = 0
    unsupported_count: int = 0
    execution_trace: ExecutionTrace | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_key",
            _coerce_enum(
                self.stage_key,
                StageKey,
                field_name="ParityReport.stage_key",
            ),
        )
        if self.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Parity reports are only valid for `parity_verification`."
            )
        object.__setattr__(
            self,
            "stage_name",
            _require_str(self.stage_name, field_name="ParityReport.stage_name"),
        )
        object.__setattr__(
            self,
            "campaign_id",
            _require_str(self.campaign_id, field_name="ParityReport.campaign_id"),
        )
        object.__setattr__(
            self,
            "campaign_seed",
            _require_int(
                self.campaign_seed,
                field_name="ParityReport.campaign_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "pass_fail",
            _coerce_enum(self.pass_fail, PassFail, field_name="ParityReport.pass_fail"),
        )
        object.__setattr__(
            self,
            "summary",
            _require_str(self.summary, field_name="ParityReport.summary"),
        )
        if self.source_verification_kit_ref.kind != self.source_verification_kit.kind:
            raise ContractValidationError(
                "Parity-report verification-kit ref kind must align with the embedded verification kit."
            )
        if self.source_verification_kit_ref.kind != VerificationKitManifest.KIND:
            raise ContractValidationError(
                "Parity reports must preserve lineage to a verification-kit manifest."
            )
        if self.source_golden_vector_manifest_ref.kind != self.source_golden_vector_manifest.kind:
            raise ContractValidationError(
                "Parity-report golden-vector ref kind must align with the embedded golden-vector manifest."
            )
        if self.source_golden_vector_manifest_ref.kind != GoldenVectorManifest.KIND:
            raise ContractValidationError(
                "Parity reports must preserve lineage to a golden-vector manifest."
            )
        if self.stage_name != self.source_verification_kit.stage_name:
            raise ContractValidationError(
                "Parity-report `stage_name` must match the source verification kit."
            )
        if self.campaign_id != self.source_verification_kit.campaign_id:
            raise ContractValidationError(
                "Parity-report `campaign_id` must match the source verification kit."
            )
        if self.campaign_seed != self.source_verification_kit.campaign_seed:
            raise ContractValidationError(
                "Parity-report `campaign_seed` must match the source verification kit."
            )
        if self.stage_name != self.source_golden_vector_manifest.stage_name:
            raise ContractValidationError(
                "Parity-report `stage_name` must match the source golden-vector manifest."
            )
        if self.campaign_id != self.source_golden_vector_manifest.campaign_id:
            raise ContractValidationError(
                "Parity-report `campaign_id` must match the source golden-vector manifest."
            )
        if self.campaign_seed != self.source_golden_vector_manifest.campaign_seed:
            raise ContractValidationError(
                "Parity-report `campaign_seed` must match the source golden-vector manifest."
            )
        if self.source_golden_vector_manifest.stage_key != StageKey.PARITY_VERIFICATION:
            raise ContractValidationError(
                "Parity reports must embed a parity-verification golden-vector manifest."
            )
        if (
            self.source_golden_vector_manifest.source_verification_kit_ref
            != self.source_verification_kit_ref
        ):
            raise ContractValidationError(
                "Parity reports must keep the golden-vector lineage aligned to the verification kit."
            )
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "exact_match_count",
            _require_int(
                self.exact_match_count,
                field_name="ParityReport.exact_match_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "mismatch_count",
            _require_int(
                self.mismatch_count,
                field_name="ParityReport.mismatch_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "unsupported_count",
            _require_int(
                self.unsupported_count,
                field_name="ParityReport.unsupported_count",
                minimum=0,
            ),
        )
        status_counts = {
            ParityComparisonStatus.EXACT_MATCH: 0,
            ParityComparisonStatus.MISMATCH: 0,
            ParityComparisonStatus.UNSUPPORTED: 0,
        }
        seen: set[tuple[str, VerificationLevel]] = set()
        vector_keys = {
            (entry.vector_id, entry.verification_level)
            for entry in self.source_golden_vector_manifest.entries
        }
        for index, result in enumerate(self.results):
            status_counts[result.comparison_status] += 1
            key = (result.vector_id, result.verification_level)
            if key in seen:
                raise ContractValidationError(
                    "Parity reports must not contain duplicate `(vector_id, verification_level)` results."
                )
            seen.add(key)
            if key not in vector_keys:
                raise ContractValidationError(
                    f"`ParityReport.results[{index}]` does not map to a golden-vector entry."
                )
        if self.exact_match_count != status_counts[ParityComparisonStatus.EXACT_MATCH]:
            raise ContractValidationError(
                "`ParityReport.exact_match_count` must match the number of exact-match results."
            )
        if self.mismatch_count != status_counts[ParityComparisonStatus.MISMATCH]:
            raise ContractValidationError(
                "`ParityReport.mismatch_count` must match the number of mismatch results."
            )
        if self.unsupported_count != status_counts[ParityComparisonStatus.UNSUPPORTED]:
            raise ContractValidationError(
                "`ParityReport.unsupported_count` must match the number of unsupported results."
            )
        expected_pass_fail = PassFail.FAIL if self.mismatch_count else PassFail.PASS
        if self.pass_fail != expected_pass_fail:
            raise ContractValidationError(
                "Parity-report `pass_fail` must be `FAIL` when mismatches exist and `PASS` otherwise."
            )


TOP_LEVEL_ARTIFACT_TYPES: tuple[type[ArtifactContract], ...] = (
    CampaignManifest,
    CampaignResult,
    StageRequest,
    StageResult,
    PromotedFrontendWinner,
    PromotedDeepResult,
    RetainedAlternatesManifest,
    DeepAnchorArtifact,
    FrontendExportReference,
    FreezeExportManifest,
    ParityVerificationBundle,
    VerificationKitManifest,
    GoldenVectorManifest,
    ParityReport,
)


__all__ = [
    "ArtifactContract",
    "ArtifactRef",
    "CandidateProvenanceOrigin",
    "CampaignManifest",
    "CampaignRequest",
    "CampaignResult",
    "CampaignStageLineage",
    "ContractValidationError",
    "DeepInputRef",
    "DeepAnchorArtifact",
    "DeployedDeepWinner",
    "DeepPromotionCandidate",
    "DeepRankingMode",
    "ExecutionAcceleration",
    "ExecutionTrace",
    "FreezeExportDeployRuntime",
    "FreezeExportManifest",
    "FreezeExportProvenance",
    "GoldenVectorEntry",
    "GoldenVectorManifest",
    "FrontendInput",
    "FrontendExportReference",
    "FrontendPromotionCandidate",
    "FrontendRankingMode",
    "JsonValue",
    "ParentAnchorRef",
    "ParityComparisonStatus",
    "ParityMismatchDetail",
    "ParityReport",
    "ParityReportEntry",
    "ParityVerificationBundle",
    "PayloadSnapshot",
    "PassFail",
    "PromotedDeepResult",
    "PromotedFrontendWinner",
    "PromotionStage",
    "ResolvedGenomeIdentity",
    "RetainedAlternateKind",
    "RetainedAlternateRef",
    "RetainedAlternatesManifest",
    "SearchCandidateProvenance",
    "SearchExplorationMode",
    "SearchRefinementMode",
    "ScoutAlertabilityStatus",
    "StageKey",
    "StageSearchPolicy",
    "StageRequest",
    "StageResult",
    "TOP_LEVEL_ARTIFACT_TYPES",
    "VerificationKitManifest",
    "VerificationLevel",
    "WaveformDatasetRecord",
    "WaveformPayloadRef",
]
