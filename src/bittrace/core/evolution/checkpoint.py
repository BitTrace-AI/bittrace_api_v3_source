"""Shared checkpoint helpers for the engine-agnostic evolution loop."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path


_CHECKPOINT_SCHEMA_VERSION = 1


def checkpoint_schema_version() -> int:
    """Return the current shared checkpoint schema version."""

    return _CHECKPOINT_SCHEMA_VERSION


def save_checkpoint(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write one checkpoint payload as formatted JSON."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def load_checkpoint(path: str | Path) -> Mapping[str, object]:
    """Load one checkpoint payload from JSON."""

    checkpoint_path = Path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"`{checkpoint_path}` must contain a JSON object checkpoint.")
    return payload


def serialize_rng_state(state: object) -> object:
    """Convert `random.Random.getstate()` output into JSON-safe values."""

    if isinstance(state, tuple):
        return [serialize_rng_state(item) for item in state]
    if isinstance(state, list):
        return [serialize_rng_state(item) for item in state]
    if state is None or isinstance(state, int | float | str | bool):
        return state
    raise TypeError(
        "The shared evolution loop encountered a non-serializable RNG state value."
    )


def deserialize_rng_state(value: object) -> object:
    """Restore a JSON-safe RNG state payload into tuple form for `setstate(...)`."""

    if isinstance(value, list):
        return tuple(deserialize_rng_state(item) for item in value)
    return value


__all__ = [
    "checkpoint_schema_version",
    "deserialize_rng_state",
    "load_checkpoint",
    "save_checkpoint",
    "serialize_rng_state",
]
