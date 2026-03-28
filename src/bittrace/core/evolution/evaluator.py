"""Shared evaluator contract for engine-agnostic evolution code.

The shared evolution loop should only depend on `evaluate(...) ->
CandidateEvaluation`. Lean and deep evaluator internals stay behind this
protocol. Serialization hooks are intentionally omitted until a concrete caller
actually needs them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, runtime_checkable


MetricMap = Mapping[str, float]
TCandidate = TypeVar("TCandidate")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Minimal shared evaluation result returned by candidate evaluators."""

    fitness: float
    metrics: MetricMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.fitness, bool) or not isinstance(self.fitness, int | float):
            raise TypeError("`fitness` must be numeric.")

        normalized_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key:
                raise TypeError("Metric names must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"Metric `{key}` must be numeric.")
            normalized_metrics[key] = float(value)

        object.__setattr__(self, "fitness", float(self.fitness))
        object.__setattr__(self, "metrics", MappingProxyType(normalized_metrics))


@runtime_checkable
class CandidateEvaluator(Protocol, Generic[TCandidate]):
    """Engine-agnostic evaluator interface consumed by shared evolution code."""

    def evaluate(self, candidate: TCandidate) -> CandidateEvaluation:
        """Score one candidate and return its shared evaluation result."""


__all__ = ["CandidateEvaluation", "CandidateEvaluator", "MetricMap"]
