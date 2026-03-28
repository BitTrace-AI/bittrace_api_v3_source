"""Deterministic temporal feature extraction for waveform-backed consumer records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scipy.io import loadmat

from bittrace.v3 import ContractValidationError


_TEMPORAL_FEATURE_VERSION = "bittrace-bearings-v3-source-temporal-features-1"
_DEFAULT_CHANNEL_NAME = "vibration_1"
_DEFAULT_WINDOW_ANCHOR = "last_complete"
_SUPPORTED_WINDOW_ANCHORS = frozenset({"last_complete", "center"})
_BASE_FEATURE_NAMES = (
    "mean",
    "median",
    "min",
    "max",
    "peak_to_peak",
    "variance",
    "rms",
    "mean_abs_deviation",
    "first_diff_mean",
    "first_diff_abs_mean",
    "max_delta",
    "diff_sign_change_rate",
    "slope",
    "cumulative_delta",
    "count_above_threshold",
    "max_spike_above_mean",
)
_DEFAULT_PERSISTENCE_DELTAS = ("rms", "variance")
_DEFAULT_FEATURE_SCALE_SHIFTS = {
    "variance": 8,
    "delta_variance": 8,
}


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _config_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_positive_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"`{field_name}` must be an integer.")
    return value


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return int(value + 0.5)
    return int(value - 0.5)


def _div_round_nearest(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ContractValidationError("Temporal feature division encountered zero denominator.")
    if numerator == 0:
        return 0
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    return sign * ((absolute + (denominator // 2)) // denominator)


def _clamp(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class TemporalFeatureConfig:
    enabled: bool = False
    channel_name: str = _DEFAULT_CHANNEL_NAME
    window_size: int = 256
    window_anchor: str = _DEFAULT_WINDOW_ANCHOR
    sample_scale: int = 4096
    sample_clip: int = 32767
    approx_median_stride: int = 4
    spike_multiplier_numerator: int = 2
    spike_multiplier_denominator: int = 1
    rate_scale: int = 1024
    slope_scale: int = 4096
    clamp_min: int = -32768
    clamp_max: int = 32767
    selected_persistence_deltas: tuple[str, ...] = _DEFAULT_PERSISTENCE_DELTAS
    feature_scale_shifts: Mapping[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_FEATURE_SCALE_SHIFTS)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_name", _require_non_empty_str(self.channel_name, field_name="temporal_features.channel_name"))
        object.__setattr__(
            self,
            "window_size",
            _require_positive_int(self.window_size, field_name="temporal_features.window_size", minimum=32),
        )
        if self.window_anchor not in _SUPPORTED_WINDOW_ANCHORS:
            raise ContractValidationError(
                "`temporal_features.window_anchor` must be one of "
                + ", ".join(sorted(_SUPPORTED_WINDOW_ANCHORS))
                + "."
            )
        object.__setattr__(
            self,
            "sample_scale",
            _require_positive_int(self.sample_scale, field_name="temporal_features.sample_scale"),
        )
        object.__setattr__(
            self,
            "sample_clip",
            _require_positive_int(self.sample_clip, field_name="temporal_features.sample_clip"),
        )
        object.__setattr__(
            self,
            "approx_median_stride",
            _require_positive_int(
                self.approx_median_stride,
                field_name="temporal_features.approx_median_stride",
            ),
        )
        object.__setattr__(
            self,
            "spike_multiplier_numerator",
            _require_positive_int(
                self.spike_multiplier_numerator,
                field_name="temporal_features.spike_multiplier_numerator",
            ),
        )
        object.__setattr__(
            self,
            "spike_multiplier_denominator",
            _require_positive_int(
                self.spike_multiplier_denominator,
                field_name="temporal_features.spike_multiplier_denominator",
            ),
        )
        object.__setattr__(
            self,
            "rate_scale",
            _require_positive_int(self.rate_scale, field_name="temporal_features.rate_scale"),
        )
        object.__setattr__(
            self,
            "slope_scale",
            _require_positive_int(self.slope_scale, field_name="temporal_features.slope_scale"),
        )
        object.__setattr__(self, "clamp_min", _require_int(self.clamp_min, field_name="temporal_features.clamp_min"))
        object.__setattr__(self, "clamp_max", _require_int(self.clamp_max, field_name="temporal_features.clamp_max"))
        if self.clamp_min >= self.clamp_max:
            raise ContractValidationError(
                "`temporal_features.clamp_min` must be strictly less than `temporal_features.clamp_max`."
            )
        persistence_deltas = tuple(str(value).strip() for value in self.selected_persistence_deltas)
        if not persistence_deltas:
            raise ContractValidationError(
                "`temporal_features.selected_persistence_deltas` must include at least one feature name."
            )
        invalid = [name for name in persistence_deltas if name not in _BASE_FEATURE_NAMES]
        if invalid:
            raise ContractValidationError(
                "`temporal_features.selected_persistence_deltas` contains unsupported values: "
                + ", ".join(invalid)
                + "."
            )
        object.__setattr__(self, "selected_persistence_deltas", persistence_deltas)
        scale_shifts = dict(self.feature_scale_shifts)
        for feature_name, shift_value in scale_shifts.items():
            if isinstance(shift_value, bool) or not isinstance(shift_value, int) or shift_value < 0:
                raise ContractValidationError(
                    f"`temporal_features.feature_scale_shifts.{feature_name}` must be an integer >= 0."
                )
        object.__setattr__(self, "feature_scale_shifts", scale_shifts)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _BASE_FEATURE_NAMES + tuple(
            f"delta_{feature_name}" for feature_name in self.selected_persistence_deltas
        )

    @property
    def config_payload(self) -> dict[str, object]:
        return {
            "version": _TEMPORAL_FEATURE_VERSION,
            "channel_name": self.channel_name,
            "window_size": self.window_size,
            "window_anchor": self.window_anchor,
            "sample_scale": self.sample_scale,
            "sample_clip": self.sample_clip,
            "approx_median_stride": self.approx_median_stride,
            "spike_multiplier_numerator": self.spike_multiplier_numerator,
            "spike_multiplier_denominator": self.spike_multiplier_denominator,
            "rate_scale": self.rate_scale,
            "slope_scale": self.slope_scale,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "selected_persistence_deltas": list(self.selected_persistence_deltas),
            "feature_scale_shifts": dict(sorted(self.feature_scale_shifts.items())),
        }

    @property
    def fingerprint(self) -> str:
        return _config_fingerprint(self.config_payload)


@dataclass(frozen=True, slots=True)
class _WindowSelection:
    current_start: int
    current_end: int
    previous_start: int | None
    previous_end: int | None


def load_temporal_feature_config(profile: Mapping[str, Any]) -> TemporalFeatureConfig:
    raw = profile.get("temporal_features", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`temporal_features` must be a mapping when present.")

    enabled = raw.get("enabled", profile.get("enable_temporal_features", False))
    if not isinstance(enabled, bool):
        raise ContractValidationError("`enable_temporal_features` must be a boolean when present.")

    spike_rule = raw.get("spike_rule", {})
    if spike_rule is None:
        spike_rule = {}
    if not isinstance(spike_rule, Mapping):
        raise ContractValidationError("`temporal_features.spike_rule` must be a mapping when present.")

    scale_shifts = raw.get("feature_scale_shifts", _DEFAULT_FEATURE_SCALE_SHIFTS)
    if not isinstance(scale_shifts, Mapping):
        raise ContractValidationError("`temporal_features.feature_scale_shifts` must be a mapping.")

    return TemporalFeatureConfig(
        enabled=enabled,
        channel_name=str(raw.get("channel_name", _DEFAULT_CHANNEL_NAME)),
        window_size=int(raw.get("window_size", 256)),
        window_anchor=str(raw.get("window_anchor", _DEFAULT_WINDOW_ANCHOR)),
        sample_scale=int(raw.get("sample_scale", 4096)),
        sample_clip=int(raw.get("sample_clip", 32767)),
        approx_median_stride=int(raw.get("approx_median_stride", 4)),
        spike_multiplier_numerator=int(spike_rule.get("multiplier_numerator", raw.get("spike_multiplier_numerator", 2))),
        spike_multiplier_denominator=int(spike_rule.get("multiplier_denominator", raw.get("spike_multiplier_denominator", 1))),
        rate_scale=int(raw.get("rate_scale", 1024)),
        slope_scale=int(raw.get("slope_scale", 4096)),
        clamp_min=int(raw.get("clamp_min", -32768)),
        clamp_max=int(raw.get("clamp_max", 32767)),
        selected_persistence_deltas=tuple(raw.get("selected_persistence_deltas", _DEFAULT_PERSISTENCE_DELTAS)),
        feature_scale_shifts={str(key): int(value) for key, value in scale_shifts.items()},
    )


def build_temporal_feature_payload(
    waveform_path: str | Path,
    *,
    config: TemporalFeatureConfig,
) -> dict[str, object]:
    if not config.enabled:
        raise ContractValidationError("Temporal feature payload requested while the feature stage is disabled.")
    return _build_temporal_feature_payload_cached(
        str(Path(waveform_path).resolve()),
        json.dumps(config.config_payload, sort_keys=True),
    )


@lru_cache(maxsize=4096)
def _build_temporal_feature_payload_cached(
    waveform_path: str,
    config_payload_json: str,
) -> dict[str, object]:
    config_payload = json.loads(config_payload_json)
    config = TemporalFeatureConfig(
        enabled=True,
        channel_name=config_payload["channel_name"],
        window_size=config_payload["window_size"],
        window_anchor=config_payload["window_anchor"],
        sample_scale=config_payload["sample_scale"],
        sample_clip=config_payload["sample_clip"],
        approx_median_stride=config_payload["approx_median_stride"],
        spike_multiplier_numerator=config_payload["spike_multiplier_numerator"],
        spike_multiplier_denominator=config_payload["spike_multiplier_denominator"],
        rate_scale=config_payload["rate_scale"],
        slope_scale=config_payload["slope_scale"],
        clamp_min=config_payload["clamp_min"],
        clamp_max=config_payload["clamp_max"],
        selected_persistence_deltas=tuple(config_payload["selected_persistence_deltas"]),
        feature_scale_shifts=config_payload["feature_scale_shifts"],
    )
    samples = _load_channel_samples(Path(waveform_path), channel_name=config.channel_name)
    selection = _select_windows(total_sample_count=len(samples), config=config)
    current_window = samples[selection.current_start : selection.current_end]
    previous_window = (
        samples[selection.previous_start : selection.previous_end]
        if selection.previous_start is not None and selection.previous_end is not None
        else None
    )
    current_quantized = _quantize_samples(current_window, config=config)
    previous_quantized = _quantize_samples(previous_window, config=config) if previous_window is not None else None

    current_raw = _raw_window_features(current_quantized, config=config)
    previous_raw = _raw_window_features(previous_quantized, config=config) if previous_quantized is not None else None

    scaled_features = {
        feature_name: _scale_feature_value(current_raw[feature_name], feature_name=feature_name, config=config)
        for feature_name in _BASE_FEATURE_NAMES
    }
    for feature_name in config.selected_persistence_deltas:
        delta_name = f"delta_{feature_name}"
        previous_value = previous_raw[feature_name] if previous_raw is not None else 0
        scaled_features[delta_name] = _scale_feature_value(
            current_raw[feature_name] - previous_value,
            feature_name=delta_name,
            config=config,
        )

    feature_names = config.feature_names
    feature_values = [int(scaled_features[feature_name]) for feature_name in feature_names]
    return {
        "version": _TEMPORAL_FEATURE_VERSION,
        "config_fingerprint": config.fingerprint,
        "config_payload": config.config_payload,
        "channel_name": config.channel_name,
        "feature_names": list(feature_names),
        "feature_values": feature_values,
        "window_size": config.window_size,
        "window_anchor": config.window_anchor,
        "total_sample_count": len(samples),
        "current_window_start": selection.current_start,
        "current_window_end": selection.current_end,
        "previous_window_start": selection.previous_start,
        "previous_window_end": selection.previous_end,
        "sample_scale": config.sample_scale,
        "sample_clip": config.sample_clip,
    }


def _load_channel_samples(path: Path, *, channel_name: str) -> tuple[float, ...]:
    if not path.is_file():
        raise ContractValidationError(f"Waveform file does not exist: {path}")
    try:
        mat = loadmat(path)
    except Exception as exc:
        raise ContractValidationError(f"Could not decode waveform file `{path}`: {exc}") from exc
    record_keys = [key for key in mat.keys() if not key.startswith("__")]
    if len(record_keys) != 1:
        raise ContractValidationError(
            f"`{path}` must expose exactly one top-level measurement key, found {len(record_keys)}."
        )
    entry = mat[record_keys[0]]
    if getattr(entry, "shape", None) != (1, 1):
        raise ContractValidationError(
            f"`{path}` must expose a single Paderborn measurement struct."
        )
    channels = entry[0, 0]["Y"]
    if channels is None:
        raise ContractValidationError(f"`{path}` is missing the expected `Y` channel array.")
    for index in range(channels.shape[1]):
        channel = channels[0, index]
        channel_label = _mat_text(channel["Name"])
        if channel_label != channel_name:
            continue
        data = channel["Data"]
        if data is None:
            break
        flattened = tuple(float(value) for value in data.reshape(-1))
        if not flattened:
            break
        return flattened
    raise ContractValidationError(
        f"`{path}` does not expose channel `{channel_name}` in the Paderborn `Y` payload."
    )


def _mat_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            return ""
        if len(value) == 1:
            return _mat_text(value[0])
        return "".join(_mat_text(item) for item in value)
    return str(value)


def _select_windows(*, total_sample_count: int, config: TemporalFeatureConfig) -> _WindowSelection:
    window_size = min(config.window_size, total_sample_count)
    if window_size < 8:
        raise ContractValidationError("Temporal feature extraction requires at least 8 samples.")

    complete_window_count = max(1, total_sample_count // window_size)
    if config.window_anchor == "center":
        current_index = 0 if complete_window_count == 1 else max(1, complete_window_count // 2)
    else:
        current_index = complete_window_count - 1
    current_start = current_index * window_size
    current_end = min(total_sample_count, current_start + window_size)
    previous_start = None
    previous_end = None
    if current_start >= window_size:
        previous_start = current_start - window_size
        previous_end = current_start
    return _WindowSelection(
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )


def _quantize_samples(
    samples: Sequence[float] | None,
    *,
    config: TemporalFeatureConfig,
) -> tuple[int, ...]:
    if samples is None:
        return ()
    quantized: list[int] = []
    for value in samples:
        scaled = _round_half_away_from_zero(float(value) * float(config.sample_scale))
        quantized.append(_clamp(scaled, minimum=-config.sample_clip, maximum=config.sample_clip))
    return tuple(quantized)


def _approximate_median(samples: Sequence[int], *, stride: int) -> int:
    sampled = list(samples[:: max(1, stride)])
    if sampled[-1] != samples[-1]:
        sampled.append(samples[-1])
    sampled.sort()
    midpoint = len(sampled) // 2
    if len(sampled) % 2 == 1:
        return sampled[midpoint]
    return _div_round_nearest(sampled[midpoint - 1] + sampled[midpoint], 2)


def _diff_sign_change_rate(diffs: Sequence[int], *, scale: int) -> int:
    signs: list[int] = []
    for value in diffs:
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
    if len(signs) < 2:
        return 0
    sign_changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return _div_round_nearest(sign_changes * scale, len(signs) - 1)


def _window_slope(samples: Sequence[int], *, slope_scale: int) -> int:
    if len(samples) < 2:
        return 0
    numerator = 0
    denominator = 0
    width = len(samples)
    for index, value in enumerate(samples):
        centered_index = (2 * index) - (width - 1)
        numerator += centered_index * value
        denominator += centered_index * centered_index
    return _div_round_nearest(numerator * slope_scale, denominator)


def _raw_window_features(
    samples: Sequence[int] | None,
    *,
    config: TemporalFeatureConfig,
) -> dict[str, int]:
    if not samples:
        raise ContractValidationError("Temporal feature extraction requires a non-empty window.")
    if len(samples) < 2:
        raise ContractValidationError("Temporal feature extraction requires at least two samples per window.")

    sample_min = min(samples)
    sample_max = max(samples)
    sample_sum = sum(samples)
    mean = _div_round_nearest(sample_sum, len(samples))
    median = _approximate_median(samples, stride=config.approx_median_stride)
    peak_to_peak = sample_max - sample_min

    sum_sq = sum(value * value for value in samples)
    rms = math.isqrt(max(0, _div_round_nearest(sum_sq, len(samples))))
    mean_abs_deviation = _div_round_nearest(
        sum(abs(value - mean) for value in samples),
        len(samples),
    )
    variance = _div_round_nearest(
        sum((value - mean) * (value - mean) for value in samples),
        len(samples),
    )

    diffs = [right - left for left, right in zip(samples, samples[1:])]
    first_diff_mean = _div_round_nearest(sum(diffs), len(diffs))
    first_diff_abs_mean = _div_round_nearest(sum(abs(value) for value in diffs), len(diffs))
    max_delta = max(abs(value) for value in diffs)
    diff_sign_change_rate = _diff_sign_change_rate(diffs, scale=config.rate_scale)
    slope = _window_slope(samples, slope_scale=config.slope_scale)
    cumulative_delta = samples[-1] - samples[0]

    spike_threshold = mean + _div_round_nearest(
        mean_abs_deviation * config.spike_multiplier_numerator,
        config.spike_multiplier_denominator,
    )
    count_above_threshold = sum(1 for value in samples if value > spike_threshold)
    max_spike_above_mean = max(0, sample_max - mean)

    return {
        "mean": mean,
        "median": median,
        "min": sample_min,
        "max": sample_max,
        "peak_to_peak": peak_to_peak,
        "variance": variance,
        "rms": rms,
        "mean_abs_deviation": mean_abs_deviation,
        "first_diff_mean": first_diff_mean,
        "first_diff_abs_mean": first_diff_abs_mean,
        "max_delta": max_delta,
        "diff_sign_change_rate": diff_sign_change_rate,
        "slope": slope,
        "cumulative_delta": cumulative_delta,
        "count_above_threshold": count_above_threshold,
        "max_spike_above_mean": max_spike_above_mean,
    }


def _scale_feature_value(
    raw_value: int,
    *,
    feature_name: str,
    config: TemporalFeatureConfig,
) -> int:
    shift = int(config.feature_scale_shifts.get(feature_name, 0))
    if shift > 0:
        scaled = _div_round_nearest(raw_value, 1 << shift)
    else:
        scaled = raw_value
    return _clamp(scaled, minimum=config.clamp_min, maximum=config.clamp_max)


__all__ = [
    "TemporalFeatureConfig",
    "build_temporal_feature_payload",
    "load_temporal_feature_config",
]
