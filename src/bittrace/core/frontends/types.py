"""Shared in-memory feature table and frontend result contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Self

from bittrace.core.config import FrontendMode


class FrontendError(ValueError):
    """Raised when a frontend cannot validate or transform a feature table."""


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """Validated in-memory feature table passed between frontend and engines."""

    rows: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    labels: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        _validate_rows(self.rows)
        expected = len(self.rows[0])
        _validate_feature_names(self.feature_names, expected=expected)
        if self.labels is not None and len(self.labels) != len(self.rows):
            raise FrontendError(
                f"`labels` row mismatch: expected {len(self.rows)}, got {len(self.labels)}."
            )

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[object]],
        *,
        feature_names: Sequence[str] | None = None,
        labels: Sequence[object] | None = None,
    ) -> Self:
        normalized_rows = _normalize_rows(rows)
        expected = len(normalized_rows[0])
        normalized_names = (
            _normalize_feature_names(feature_names, expected=expected)
            if feature_names is not None
            else tuple(f"feature_{index}" for index in range(expected))
        )
        normalized_labels = tuple(labels) if labels is not None else None
        return cls(
            rows=normalized_rows,
            feature_names=normalized_names,
            labels=normalized_labels,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return `(row_count, feature_count)` for the table."""

        return (len(self.rows), len(self.feature_names))


@dataclass(frozen=True, slots=True)
class FrontendResult:
    """Transformed feature table plus frontend artifacts/report payloads."""

    mode: FrontendMode
    table: FeatureTable
    artifacts: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


def _normalize_rows(rows: Sequence[Sequence[object]]) -> tuple[tuple[float, ...], ...]:
    if not rows:
        raise FrontendError("Feature tables must contain at least one row.")

    normalized_rows: list[tuple[float, ...]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(rows):
        if not row:
            raise FrontendError(
                f"`rows[{row_index}]` must contain at least one numeric feature."
            )
        normalized_row = tuple(
            _require_finite_float(value, path=f"rows[{row_index}][{column_index}]")
            for column_index, value in enumerate(row)
        )
        if expected_width is None:
            expected_width = len(normalized_row)
        elif len(normalized_row) != expected_width:
            raise FrontendError(
                f"`rows[{row_index}]` width mismatch: expected {expected_width}, "
                f"got {len(normalized_row)}."
            )
        normalized_rows.append(normalized_row)

    return tuple(normalized_rows)


def _validate_rows(rows: tuple[tuple[float, ...], ...]) -> None:
    if not rows:
        raise FrontendError("Feature tables must contain at least one row.")
    expected_width: int | None = None
    for row_index, row in enumerate(rows):
        if not row:
            raise FrontendError(
                f"`rows[{row_index}]` must contain at least one numeric feature."
            )
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise FrontendError(
                f"`rows[{row_index}]` width mismatch: expected {expected_width}, "
                f"got {len(row)}."
            )
        for column_index, value in enumerate(row):
            _require_finite_float(value, path=f"rows[{row_index}][{column_index}]")


def _normalize_feature_names(
    feature_names: Sequence[str],
    *,
    expected: int,
) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in feature_names)
    _validate_feature_names(names, expected=expected)
    return names


def _validate_feature_names(feature_names: tuple[str, ...], *, expected: int) -> None:
    if len(feature_names) != expected:
        raise FrontendError(
            f"`feature_names` length mismatch: expected {expected}, got {len(feature_names)}."
        )
    if any(not name for name in feature_names):
        raise FrontendError("`feature_names` entries must be non-empty strings.")
    if len(set(feature_names)) != len(feature_names):
        raise FrontendError("`feature_names` entries must be unique.")


def _require_finite_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FrontendError(f"`{path}` must be a finite numeric value.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise FrontendError(f"`{path}` must be finite.")
    return normalized
