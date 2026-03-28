"""Evolution package for shared search contracts and the shared outer loop."""

from bittrace.core.evolution.checkpoint import (
    checkpoint_schema_version,
    deserialize_rng_state,
    load_checkpoint,
    save_checkpoint,
    serialize_rng_state,
)
from bittrace.core.evolution.evaluator import CandidateEvaluation, CandidateEvaluator
from bittrace.core.evolution.loop import (
    CandidateInitializer,
    CandidateDeserializer,
    CandidateMutator,
    CandidateSerializer,
    EvaluatedCandidate,
    EvolutionRunResult,
    GenerationSummary,
    LayerCounter,
    SelectionSpec,
    SurvivorSelection,
    rank_population,
    run_evolution_loop,
    schedule_mutation_rate,
    select_elites,
    select_survivors,
    tournament_select,
)

__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluator",
    "CandidateInitializer",
    "CandidateDeserializer",
    "CandidateMutator",
    "CandidateSerializer",
    "EvaluatedCandidate",
    "EvolutionRunResult",
    "GenerationSummary",
    "LayerCounter",
    "SelectionSpec",
    "SurvivorSelection",
    "rank_population",
    "run_evolution_loop",
    "schedule_mutation_rate",
    "checkpoint_schema_version",
    "deserialize_rng_state",
    "load_checkpoint",
    "save_checkpoint",
    "select_elites",
    "select_survivors",
    "serialize_rng_state",
    "tournament_select",
]
