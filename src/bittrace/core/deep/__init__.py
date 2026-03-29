"""Deep engine package wired through the shared evolution loop."""

from bittrace.core.deep.engine import (
    DeepBackendSummary,
    DeepBundle,
    DeepCandidate,
    DeepEvaluator,
    DeepEvolutionResult,
    DeepLayer,
    DeepSplitMetrics,
    DeepState,
    build_deep_initializer,
    build_deep_mutator,
    count_deep_layers,
    deserialize_deep_candidate,
    load_deep_bundle,
    mutate_deep_candidate,
    run_deep_evolution,
    serialize_deep_candidate,
)

__all__ = [
    "DeepBackendSummary",
    "DeepBundle",
    "DeepCandidate",
    "DeepEvaluator",
    "DeepEvolutionResult",
    "DeepLayer",
    "DeepSplitMetrics",
    "DeepState",
    "build_deep_initializer",
    "build_deep_mutator",
    "count_deep_layers",
    "deserialize_deep_candidate",
    "load_deep_bundle",
    "mutate_deep_candidate",
    "run_deep_evolution",
    "serialize_deep_candidate",
]
