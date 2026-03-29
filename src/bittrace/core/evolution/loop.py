"""Shared outer evolution loop for BitTrace API 3.0.

This module intentionally stops at the engine-agnostic outer loop. Lean and
deep candidates plug in through initializer, mutator, and evaluator callables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from statistics import fmean
from typing import Generic, TypeVar

from bittrace.core.config import EvolutionCheckpointConfig, EvolutionConfig
from bittrace.core.evolution.checkpoint import (
    checkpoint_schema_version,
    deserialize_rng_state,
    load_checkpoint,
    save_checkpoint,
    serialize_rng_state,
)
from bittrace.core.evolution.evaluator import CandidateEvaluation, CandidateEvaluator


TCandidate = TypeVar("TCandidate")
CandidateInitializer = Callable[[random.Random, int, EvolutionConfig], TCandidate]
CandidateMutator = Callable[[TCandidate, random.Random, float, int, EvolutionConfig], TCandidate]
LayerCounter = Callable[[TCandidate], int]
CandidateSerializer = Callable[[TCandidate], Mapping[str, object]]
CandidateDeserializer = Callable[[Mapping[str, object]], TCandidate]

_SUPPORTED_MUTATION_SCHEDULES = frozenset({"constant", "linear_decay"})
_SUPPORTED_SELECTION_MODES = frozenset({"tournament"})
_HISTORY_JSON_NAME = "history.json"
_HISTORY_CSV_NAME = "history.csv"
_DEFAULT_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """Ranking spec used for elites, survivors, and final best-candidate picks."""

    primary_metric: str = "fitness"
    tiebreak_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        primary_metric = _normalize_metric_name(
            self.primary_metric,
            field_name="primary_metric",
        )
        tiebreak_metrics = tuple(
            _normalize_metric_name(metric, field_name="tiebreak_metrics")
            for metric in self.tiebreak_metrics
        )
        if primary_metric in tiebreak_metrics:
            raise ValueError("`primary_metric` cannot also appear in `tiebreak_metrics`.")
        if len(set(tiebreak_metrics)) != len(tiebreak_metrics):
            raise ValueError("`tiebreak_metrics` cannot contain duplicates.")

        object.__setattr__(self, "primary_metric", primary_metric)
        object.__setattr__(self, "tiebreak_metrics", tiebreak_metrics)


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate(Generic[TCandidate]):
    """One candidate paired with its evaluation and simple lineage metadata."""

    candidate_id: int
    candidate: TCandidate
    evaluation: CandidateEvaluation
    birth_generation: int
    birth_origin: str
    parent_id: int | None
    total_layers: int | None = None

    def __post_init__(self) -> None:
        if self.birth_origin not in {"initial", "offspring"}:
            raise ValueError(
                "`birth_origin` must be either `initial` or `offspring`."
            )


@dataclass(frozen=True, slots=True)
class SurvivorSelection(Generic[TCandidate]):
    """Elites plus the full survivor pool chosen from one evaluated population."""

    elites: tuple[EvaluatedCandidate[TCandidate], ...]
    survivors: tuple[EvaluatedCandidate[TCandidate], ...]


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Compact generation-level history written to JSON and emitted to logs."""

    generation: int
    mutation_rate: float
    best_candidate_id: int
    best_fitness: float
    best_primary_metric: str
    best_primary_value: float
    best_total_layers: int | None
    mean_fitness: float
    elite_ids: tuple[int, ...]
    survivor_ids: tuple[int, ...]
    offspring_ids: tuple[int, ...]
    population_ids: tuple[int, ...]
    stagnation_generations: int
    stopped_early: bool = False


@dataclass(frozen=True, slots=True)
class EvolutionRunResult(Generic[TCandidate]):
    """Final shared-loop result returned to engine-specific callers."""

    best_candidate: EvaluatedCandidate[TCandidate]
    final_population: tuple[EvaluatedCandidate[TCandidate], ...]
    generation_summaries: tuple[GenerationSummary, ...]
    history_json_path: Path
    history_csv_path: Path
    completed_generations: int
    stopped_early: bool


@dataclass(frozen=True, slots=True)
class _PopulationHistoryRow:
    generation: int
    candidate_id: int
    birth_generation: int
    birth_origin: str
    parent_id: int | None
    fitness: float
    total_layers: int | None
    is_elite: bool
    is_survivor: bool
    is_offspring: bool
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class _ResumeState(Generic[TCandidate]):
    population: tuple[EvaluatedCandidate[TCandidate], ...]
    best_candidate: EvaluatedCandidate[TCandidate]
    generation_summaries: tuple[GenerationSummary, ...]
    history_rows: tuple[_PopulationHistoryRow, ...]
    rng_state: object
    next_candidate_id: int
    completed_generations: int
    stagnation_generations: int
    stopped_early: bool


def run_evolution_loop(
    evolution_config: EvolutionConfig,
    *,
    initialize_candidate: CandidateInitializer[TCandidate],
    mutate_candidate: CandidateMutator[TCandidate],
    evaluator: CandidateEvaluator[TCandidate],
    output_dir: str | Path,
    selection_spec: SelectionSpec | None = None,
    layer_counter: LayerCounter[TCandidate] | None = None,
    candidate_serializer: CandidateSerializer[TCandidate] | None = None,
    candidate_deserializer: CandidateDeserializer[TCandidate] | None = None,
    artifact_identity: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> EvolutionRunResult[TCandidate]:
    """Run the shared outer evolution loop and write `history.json` / `history.csv`."""

    _validate_runtime_config(evolution_config)
    selection = selection_spec if selection_spec is not None else SelectionSpec()
    prefer_fewer_layers = layer_counter is not None
    history_dir = Path(output_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    active_logger = logger if logger is not None else _DEFAULT_LOGGER
    checkpoint_config = evolution_config.checkpoint
    checkpoint_save_path = _resolve_checkpoint_path(checkpoint_config.save_path)
    checkpoint_resume_path = _resolve_checkpoint_path(checkpoint_config.resume_from)
    checkpoint_artifact_identity = _normalize_checkpoint_json_mapping(
        artifact_identity,
        field_name="artifact_identity",
    )
    _validate_checkpoint_configuration(
        checkpoint_config,
        checkpoint_save_path=checkpoint_save_path,
        checkpoint_resume_path=checkpoint_resume_path,
        candidate_serializer=candidate_serializer,
        candidate_deserializer=candidate_deserializer,
        artifact_identity=checkpoint_artifact_identity,
    )

    if checkpoint_resume_path is not None:
        resume_state = _load_resume_state(
            checkpoint_resume_path,
            evolution_config=evolution_config,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
            layer_counter=layer_counter,
            candidate_deserializer=candidate_deserializer,
            artifact_identity=checkpoint_artifact_identity,
        )
        rng = random.Random()
        rng.setstate(resume_state.rng_state)
        population = list(resume_state.population)
        best_candidate = resume_state.best_candidate
        generation_summaries = list(resume_state.generation_summaries)
        history_rows = list(resume_state.history_rows)
        next_candidate_id = resume_state.next_candidate_id
        completed_generations = resume_state.completed_generations
        stagnation_generations = resume_state.stagnation_generations
        stopped_early = resume_state.stopped_early
        active_logger.info(
            "Resuming shared evolution from generation %d using checkpoint `%s`.",
            completed_generations,
            checkpoint_resume_path.resolve(),
        )
    else:
        rng = random.Random(evolution_config.seed)
        next_candidate_id = 1
        population = []
        for candidate_index in range(evolution_config.population_size):
            candidate = initialize_candidate(rng, candidate_index, evolution_config)
            record, next_candidate_id = _evaluate_candidate(
                candidate,
                evaluator=evaluator,
                evolution_config=evolution_config,
                layer_counter=layer_counter,
                candidate_id=next_candidate_id,
                birth_generation=0,
                birth_origin="initial",
                parent_id=None,
            )
            population.append(record)

        population = rank_population(
            population,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
        )

        history_rows = _build_history_rows(
            generation=0,
            population=population,
            elite_ids=frozenset(),
            survivor_ids=frozenset(),
        )
        generation_summaries = [
            _build_generation_summary(
                generation=0,
                population=population,
                mutation_rate=0.0,
                selection_spec=selection,
                elite_ids=(),
                survivor_ids=(),
                offspring_ids=(),
                stagnation_generations=0,
            )
        ]
        _log_generation_summary(active_logger, generation_summaries[-1])

        best_candidate = population[0]
        stagnation_generations = 0
        stopped_early = False
        completed_generations = 0
        _write_checkpoint_if_configured(
            checkpoint_save_path,
            evolution_config=evolution_config,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
            artifact_identity=checkpoint_artifact_identity,
            population=population,
            best_candidate=best_candidate,
            generation_summaries=generation_summaries,
            history_rows=history_rows,
            rng=rng,
            next_candidate_id=next_candidate_id,
            completed_generations=completed_generations,
            stagnation_generations=stagnation_generations,
            stopped_early=stopped_early,
            candidate_serializer=candidate_serializer,
        )

    for generation in range(completed_generations + 1, evolution_config.generations + 1):
        mutation_rate = schedule_mutation_rate(
            evolution_config.mutation_rate,
            evolution_config.mutation_rate_schedule,
            generation=generation,
            total_generations=evolution_config.generations,
        )
        survivor_selection = select_survivors(
            population,
            survivor_count=evolution_config.mu,
            elite_count=evolution_config.elite_count,
            selection_mode=evolution_config.selection_mode,
            tournament_k=evolution_config.tournament_k,
            rng=rng,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
        )
        parents = tournament_select(
            survivor_selection.survivors,
            count=evolution_config.lam,
            tournament_k=evolution_config.tournament_k,
            rng=rng,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
            replace=True,
        )

        offspring: list[EvaluatedCandidate[TCandidate]] = []
        for parent in parents:
            child_candidate = mutate_candidate(
                parent.candidate,
                rng,
                mutation_rate,
                generation,
                evolution_config,
            )
            child_record, next_candidate_id = _evaluate_candidate(
                child_candidate,
                evaluator=evaluator,
                evolution_config=evolution_config,
                layer_counter=layer_counter,
                candidate_id=next_candidate_id,
                birth_generation=generation,
                birth_origin="offspring",
                parent_id=parent.candidate_id,
            )
            offspring.append(child_record)

        next_population = _select_next_population(
            survivor_selection=survivor_selection,
            offspring=offspring,
            population_size=evolution_config.population_size,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
        )

        current_best = next_population[0]
        if _is_better_candidate(
            current_best,
            than_candidate=best_candidate,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
        ):
            best_candidate = current_best
            stagnation_generations = 0
        else:
            stagnation_generations += 1

        stopped_early = (
            evolution_config.early_stopping_patience > 0
            and stagnation_generations >= evolution_config.early_stopping_patience
        )
        generation_summary = _build_generation_summary(
            generation=generation,
            population=next_population,
            mutation_rate=mutation_rate,
            selection_spec=selection,
            elite_ids=tuple(
                record.candidate_id for record in survivor_selection.elites
            ),
            survivor_ids=tuple(
                record.candidate_id for record in survivor_selection.survivors
            ),
            offspring_ids=tuple(record.candidate_id for record in offspring),
            stagnation_generations=stagnation_generations,
            stopped_early=stopped_early,
        )
        generation_summaries.append(generation_summary)
        history_rows.extend(
            _build_history_rows(
                generation=generation,
                population=next_population,
                elite_ids=frozenset(generation_summary.elite_ids),
                survivor_ids=frozenset(generation_summary.survivor_ids),
            )
        )
        _log_generation_summary(active_logger, generation_summary)

        population = next_population
        completed_generations = generation
        _write_checkpoint_if_configured(
            checkpoint_save_path,
            evolution_config=evolution_config,
            selection_spec=selection,
            prefer_fewer_layers=prefer_fewer_layers,
            artifact_identity=checkpoint_artifact_identity,
            population=population,
            best_candidate=best_candidate,
            generation_summaries=generation_summaries,
            history_rows=history_rows,
            rng=rng,
            next_candidate_id=next_candidate_id,
            completed_generations=completed_generations,
            stagnation_generations=stagnation_generations,
            stopped_early=stopped_early,
            candidate_serializer=candidate_serializer,
        )
        if stopped_early:
            break

    history_json_path = history_dir / _HISTORY_JSON_NAME
    history_csv_path = history_dir / _HISTORY_CSV_NAME
    _write_history_json(
        history_json_path,
        evolution_config=evolution_config,
        selection_spec=selection,
        generation_summaries=generation_summaries,
        best_candidate=best_candidate,
        completed_generations=completed_generations,
        stopped_early=stopped_early,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    _write_history_csv(history_csv_path, history_rows)

    return EvolutionRunResult(
        best_candidate=best_candidate,
        final_population=tuple(population),
        generation_summaries=tuple(generation_summaries),
        history_json_path=history_json_path,
        history_csv_path=history_csv_path,
        completed_generations=completed_generations,
        stopped_early=stopped_early,
    )


def _resolve_checkpoint_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    return Path(raw_path).expanduser()


def _validate_checkpoint_configuration(
    checkpoint_config: EvolutionCheckpointConfig,
    *,
    checkpoint_save_path: Path | None,
    checkpoint_resume_path: Path | None,
    candidate_serializer: CandidateSerializer[TCandidate] | None,
    candidate_deserializer: CandidateDeserializer[TCandidate] | None,
    artifact_identity: Mapping[str, object] | None,
) -> None:
    del checkpoint_config
    if checkpoint_save_path is None and checkpoint_resume_path is None:
        return
    if artifact_identity is None or not artifact_identity:
        raise ValueError(
            "Checkpointing requires a non-empty `artifact_identity` for honest resume validation."
        )
    if checkpoint_save_path is not None and candidate_serializer is None:
        raise ValueError(
            "Checkpoint saving requires a candidate serializer from the engine."
        )
    if checkpoint_resume_path is not None and candidate_deserializer is None:
        raise ValueError(
            "Checkpoint resume requires a candidate deserializer from the engine."
        )


def _write_checkpoint_if_configured(
    checkpoint_path: Path | None,
    *,
    evolution_config: EvolutionConfig,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool,
    artifact_identity: Mapping[str, object] | None,
    population: Sequence[EvaluatedCandidate[TCandidate]],
    best_candidate: EvaluatedCandidate[TCandidate],
    generation_summaries: Sequence[GenerationSummary],
    history_rows: Sequence[_PopulationHistoryRow],
    rng: random.Random,
    next_candidate_id: int,
    completed_generations: int,
    stagnation_generations: int,
    stopped_early: bool,
    candidate_serializer: CandidateSerializer[TCandidate] | None,
) -> None:
    if checkpoint_path is None:
        return
    if candidate_serializer is None or artifact_identity is None:
        raise ValueError("Checkpoint saving is missing required shared-loop hooks.")

    save_checkpoint(
        checkpoint_path,
        {
            "schema_version": checkpoint_schema_version(),
            "config": _serialize_evolution_config(evolution_config),
            "selection": _serialize_selection(
                selection_spec,
                prefer_fewer_layers=prefer_fewer_layers,
            ),
            "artifact_identity": dict(artifact_identity),
            "completed_generations": completed_generations,
            "next_candidate_id": next_candidate_id,
            "stagnation_generations": stagnation_generations,
            "stopped_early": stopped_early,
            "best_candidate": _serialize_evaluated_candidate(
                best_candidate,
                candidate_serializer=candidate_serializer,
            ),
            "active_population": [
                _serialize_evaluated_candidate(
                    record,
                    candidate_serializer=candidate_serializer,
                )
                for record in population
            ],
            "generation_summaries": [
                _serialize_generation_summary(summary)
                for summary in generation_summaries
            ],
            "history_rows": [
                _serialize_history_row(row)
                for row in history_rows
            ],
            "rng_state": serialize_rng_state(rng.getstate()),
        },
    )


def _load_resume_state(
    checkpoint_path: Path,
    *,
    evolution_config: EvolutionConfig,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool,
    layer_counter: LayerCounter[TCandidate] | None,
    candidate_deserializer: CandidateDeserializer[TCandidate] | None,
    artifact_identity: Mapping[str, object] | None,
) -> _ResumeState[TCandidate]:
    if candidate_deserializer is None or artifact_identity is None:
        raise ValueError("Checkpoint resume is missing required shared-loop hooks.")

    payload = load_checkpoint(checkpoint_path)
    schema_version = _checkpoint_int(
        payload.get("schema_version"),
        path="checkpoint.schema_version",
        minimum=1,
    )
    if schema_version != checkpoint_schema_version():
        raise ValueError(
            f"Unsupported checkpoint schema version `{schema_version}`. "
            f"Expected `{checkpoint_schema_version()}`."
        )

    checkpoint_config = _checkpoint_mapping(
        payload.get("config"),
        path="checkpoint.config",
    )
    current_config = _serialize_evolution_config(evolution_config)
    mismatched_fields = [
        field_name
        for field_name, field_value in current_config.items()
        if field_name != "generations"
        and checkpoint_config.get(field_name) != field_value
    ]
    if mismatched_fields:
        raise ValueError(
            "Checkpoint config does not match the current evolution config for "
            + ", ".join(f"`{field_name}`" for field_name in mismatched_fields)
            + "."
        )

    checkpoint_selection = _checkpoint_mapping(
        payload.get("selection"),
        path="checkpoint.selection",
    )
    current_selection = _serialize_selection(
        selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    if checkpoint_selection != current_selection:
        raise ValueError(
            "Checkpoint selection settings do not match the current shared-loop selection."
        )

    checkpoint_artifact_identity = _checkpoint_mapping(
        payload.get("artifact_identity"),
        path="checkpoint.artifact_identity",
    )
    if checkpoint_artifact_identity != artifact_identity:
        raise ValueError(
            "Checkpoint artifact identity does not match the current engine inputs."
        )

    completed_generations = _checkpoint_int(
        payload.get("completed_generations"),
        path="checkpoint.completed_generations",
        minimum=0,
    )
    if completed_generations >= evolution_config.generations:
        raise ValueError(
            "Checkpoint already reached or exceeded the requested total generations."
        )
    next_candidate_id = _checkpoint_int(
        payload.get("next_candidate_id"),
        path="checkpoint.next_candidate_id",
        minimum=1,
    )
    stagnation_generations = _checkpoint_int(
        payload.get("stagnation_generations"),
        path="checkpoint.stagnation_generations",
        minimum=0,
    )
    stopped_early = _checkpoint_bool(
        payload.get("stopped_early"),
        path="checkpoint.stopped_early",
    )
    if stopped_early:
        raise ValueError(
            "Checkpoint already represents an early-stopped run and cannot be resumed honestly."
        )

    best_candidate = _deserialize_evaluated_candidate(
        payload.get("best_candidate"),
        path="checkpoint.best_candidate",
        candidate_deserializer=candidate_deserializer,
        evolution_config=evolution_config,
        layer_counter=layer_counter,
    )
    population_payload = _checkpoint_list(
        payload.get("active_population"),
        path="checkpoint.active_population",
    )
    population = tuple(
        _deserialize_evaluated_candidate(
            record_payload,
            path=f"checkpoint.active_population[{index}]",
            candidate_deserializer=candidate_deserializer,
            evolution_config=evolution_config,
            layer_counter=layer_counter,
        )
        for index, record_payload in enumerate(population_payload)
    )
    if len(population) != evolution_config.population_size:
        raise ValueError(
            "Checkpoint active population size does not match "
            "`training.evolution.population_size`."
        )
    population_candidate_ids = [record.candidate_id for record in population]
    if len(set(population_candidate_ids)) != len(population_candidate_ids):
        raise ValueError("Checkpoint active population contains duplicate candidate IDs.")
    max_candidate_id = max(population_candidate_ids + [best_candidate.candidate_id])
    if next_candidate_id <= max_candidate_id:
        raise ValueError(
            "Checkpoint `next_candidate_id` must be greater than every restored candidate ID."
        )

    generation_summaries_payload = _checkpoint_list(
        payload.get("generation_summaries"),
        path="checkpoint.generation_summaries",
    )
    generation_summaries = tuple(
        _deserialize_generation_summary(
            summary_payload,
            path=f"checkpoint.generation_summaries[{index}]",
        )
        for index, summary_payload in enumerate(generation_summaries_payload)
    )
    if len(generation_summaries) != completed_generations + 1:
        raise ValueError(
            "Checkpoint generation summaries do not match the completed generation count."
        )
    if generation_summaries and generation_summaries[-1].generation != completed_generations:
        raise ValueError(
            "Checkpoint generation summaries do not end at the completed generation."
        )

    history_rows_payload = _checkpoint_list(
        payload.get("history_rows"),
        path="checkpoint.history_rows",
    )
    history_rows = tuple(
        _deserialize_history_row(
            row_payload,
            path=f"checkpoint.history_rows[{index}]",
        )
        for index, row_payload in enumerate(history_rows_payload)
    )

    rng_state = deserialize_rng_state(payload.get("rng_state"))
    rng_probe = random.Random()
    try:
        rng_probe.setstate(rng_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint RNG state could not be restored.") from exc

    ranked_population = tuple(
        rank_population(
            population,
            selection_spec=selection_spec,
            prefer_fewer_layers=prefer_fewer_layers,
        )
    )
    if _is_better_candidate(
        ranked_population[0],
        than_candidate=best_candidate,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    ):
        raise ValueError(
            "Checkpoint best-so-far candidate is worse than the restored active population."
        )

    return _ResumeState(
        population=ranked_population,
        best_candidate=best_candidate,
        generation_summaries=generation_summaries,
        history_rows=history_rows,
        rng_state=rng_state,
        next_candidate_id=next_candidate_id,
        completed_generations=completed_generations,
        stagnation_generations=stagnation_generations,
        stopped_early=stopped_early,
    )


def _serialize_evolution_config(evolution_config: EvolutionConfig) -> dict[str, object]:
    return {
        "seed": evolution_config.seed,
        "generations": evolution_config.generations,
        "population_size": evolution_config.population_size,
        "mu": evolution_config.mu,
        "lam": evolution_config.lam,
        "elite_count": evolution_config.elite_count,
        "min_layers": evolution_config.min_layers,
        "max_layers": evolution_config.max_layers,
        "mutation_rate": evolution_config.mutation_rate,
        "mutation_rate_schedule": evolution_config.mutation_rate_schedule,
        "selection_mode": evolution_config.selection_mode,
        "tournament_k": evolution_config.tournament_k,
        "early_stopping_patience": evolution_config.early_stopping_patience,
    }


def _serialize_selection(
    selection_spec: SelectionSpec,
    *,
    prefer_fewer_layers: bool,
) -> dict[str, object]:
    return {
        "primary_metric": selection_spec.primary_metric,
        "tiebreak_metrics": list(selection_spec.tiebreak_metrics),
        "prefer_fewer_layers": prefer_fewer_layers,
    }


def _serialize_evaluated_candidate(
    candidate: EvaluatedCandidate[TCandidate],
    *,
    candidate_serializer: CandidateSerializer[TCandidate],
) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": _normalize_checkpoint_json_mapping(
            candidate_serializer(candidate.candidate),
            field_name="candidate",
        ),
        "evaluation": _serialize_candidate_evaluation(candidate.evaluation),
        "birth_generation": candidate.birth_generation,
        "birth_origin": candidate.birth_origin,
        "parent_id": candidate.parent_id,
        "total_layers": candidate.total_layers,
    }


def _deserialize_evaluated_candidate(
    payload: object,
    *,
    path: str,
    candidate_deserializer: CandidateDeserializer[TCandidate],
    evolution_config: EvolutionConfig,
    layer_counter: LayerCounter[TCandidate] | None,
) -> EvaluatedCandidate[TCandidate]:
    record = _checkpoint_mapping(payload, path=path)
    candidate_payload = _checkpoint_mapping(
        record.get("candidate"),
        path=f"{path}.candidate",
    )
    candidate = candidate_deserializer(candidate_payload)
    evaluation = _deserialize_candidate_evaluation(
        record.get("evaluation"),
        path=f"{path}.evaluation",
    )
    total_layers = _checkpoint_optional_int(
        record.get("total_layers"),
        path=f"{path}.total_layers",
        minimum=0,
    )
    if layer_counter is not None:
        actual_total_layers = layer_counter(candidate)
        if total_layers != actual_total_layers:
            raise ValueError(
                f"{path}.total_layers does not match the restored candidate layer count."
            )
        if actual_total_layers < evolution_config.min_layers:
            raise ValueError(
                f"{path} has {actual_total_layers} layers, below "
                f"`training.evolution.min_layers`={evolution_config.min_layers}."
            )
        if actual_total_layers > evolution_config.max_layers:
            raise ValueError(
                f"{path} has {actual_total_layers} layers, above "
                f"`training.evolution.max_layers`={evolution_config.max_layers}."
            )

    return EvaluatedCandidate(
        candidate_id=_checkpoint_int(
            record.get("candidate_id"),
            path=f"{path}.candidate_id",
            minimum=1,
        ),
        candidate=candidate,
        evaluation=evaluation,
        birth_generation=_checkpoint_int(
            record.get("birth_generation"),
            path=f"{path}.birth_generation",
            minimum=0,
        ),
        birth_origin=_checkpoint_str(
            record.get("birth_origin"),
            path=f"{path}.birth_origin",
        ),
        parent_id=_checkpoint_optional_int(
            record.get("parent_id"),
            path=f"{path}.parent_id",
            minimum=1,
        ),
        total_layers=total_layers,
    )


def _serialize_candidate_evaluation(evaluation: CandidateEvaluation) -> dict[str, object]:
    return {
        "fitness": evaluation.fitness,
        "metrics": dict(evaluation.metrics),
    }


def _deserialize_candidate_evaluation(payload: object, *, path: str) -> CandidateEvaluation:
    record = _checkpoint_mapping(payload, path=path)
    metrics = _checkpoint_mapping(record.get("metrics"), path=f"{path}.metrics")
    normalized_metrics: dict[str, float] = {}
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f"{path}.metrics keys must be non-empty strings.")
        if isinstance(metric_value, bool) or not isinstance(metric_value, int | float):
            raise ValueError(f"{path}.metrics[{metric_name!r}] must be numeric.")
        normalized_metrics[metric_name] = float(metric_value)
    return CandidateEvaluation(
        fitness=_checkpoint_float(record.get("fitness"), path=f"{path}.fitness"),
        metrics=normalized_metrics,
    )


def _serialize_generation_summary(summary: GenerationSummary) -> dict[str, object]:
    return {
        "generation": summary.generation,
        "mutation_rate": summary.mutation_rate,
        "best_candidate_id": summary.best_candidate_id,
        "best_fitness": summary.best_fitness,
        "best_primary_metric": summary.best_primary_metric,
        "best_primary_value": summary.best_primary_value,
        "best_total_layers": summary.best_total_layers,
        "mean_fitness": summary.mean_fitness,
        "elite_ids": list(summary.elite_ids),
        "survivor_ids": list(summary.survivor_ids),
        "offspring_ids": list(summary.offspring_ids),
        "population_ids": list(summary.population_ids),
        "stagnation_generations": summary.stagnation_generations,
        "stopped_early": summary.stopped_early,
    }


def _deserialize_generation_summary(payload: object, *, path: str) -> GenerationSummary:
    record = _checkpoint_mapping(payload, path=path)
    return GenerationSummary(
        generation=_checkpoint_int(record.get("generation"), path=f"{path}.generation", minimum=0),
        mutation_rate=_checkpoint_float(
            record.get("mutation_rate"),
            path=f"{path}.mutation_rate",
        ),
        best_candidate_id=_checkpoint_int(
            record.get("best_candidate_id"),
            path=f"{path}.best_candidate_id",
            minimum=1,
        ),
        best_fitness=_checkpoint_float(
            record.get("best_fitness"),
            path=f"{path}.best_fitness",
        ),
        best_primary_metric=_checkpoint_str(
            record.get("best_primary_metric"),
            path=f"{path}.best_primary_metric",
        ),
        best_primary_value=_checkpoint_float(
            record.get("best_primary_value"),
            path=f"{path}.best_primary_value",
        ),
        best_total_layers=_checkpoint_optional_int(
            record.get("best_total_layers"),
            path=f"{path}.best_total_layers",
            minimum=0,
        ),
        mean_fitness=_checkpoint_float(
            record.get("mean_fitness"),
            path=f"{path}.mean_fitness",
        ),
        elite_ids=_checkpoint_int_tuple(
            record.get("elite_ids"),
            path=f"{path}.elite_ids",
            minimum=1,
        ),
        survivor_ids=_checkpoint_int_tuple(
            record.get("survivor_ids"),
            path=f"{path}.survivor_ids",
            minimum=1,
        ),
        offspring_ids=_checkpoint_int_tuple(
            record.get("offspring_ids"),
            path=f"{path}.offspring_ids",
            minimum=1,
        ),
        population_ids=_checkpoint_int_tuple(
            record.get("population_ids"),
            path=f"{path}.population_ids",
            minimum=1,
        ),
        stagnation_generations=_checkpoint_int(
            record.get("stagnation_generations"),
            path=f"{path}.stagnation_generations",
            minimum=0,
        ),
        stopped_early=_checkpoint_bool(
            record.get("stopped_early"),
            path=f"{path}.stopped_early",
        ),
    )


def _serialize_history_row(row: _PopulationHistoryRow) -> dict[str, object]:
    return {
        "generation": row.generation,
        "candidate_id": row.candidate_id,
        "birth_generation": row.birth_generation,
        "birth_origin": row.birth_origin,
        "parent_id": row.parent_id,
        "fitness": row.fitness,
        "total_layers": row.total_layers,
        "is_elite": row.is_elite,
        "is_survivor": row.is_survivor,
        "is_offspring": row.is_offspring,
        "metrics": dict(row.metrics),
    }


def _deserialize_history_row(payload: object, *, path: str) -> _PopulationHistoryRow:
    record = _checkpoint_mapping(payload, path=path)
    metrics = _checkpoint_mapping(record.get("metrics"), path=f"{path}.metrics")
    normalized_metrics: dict[str, float] = {}
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f"{path}.metrics keys must be non-empty strings.")
        if isinstance(metric_value, bool) or not isinstance(metric_value, int | float):
            raise ValueError(f"{path}.metrics[{metric_name!r}] must be numeric.")
        normalized_metrics[metric_name] = float(metric_value)
    return _PopulationHistoryRow(
        generation=_checkpoint_int(record.get("generation"), path=f"{path}.generation", minimum=0),
        candidate_id=_checkpoint_int(
            record.get("candidate_id"),
            path=f"{path}.candidate_id",
            minimum=1,
        ),
        birth_generation=_checkpoint_int(
            record.get("birth_generation"),
            path=f"{path}.birth_generation",
            minimum=0,
        ),
        birth_origin=_checkpoint_str(record.get("birth_origin"), path=f"{path}.birth_origin"),
        parent_id=_checkpoint_optional_int(
            record.get("parent_id"),
            path=f"{path}.parent_id",
            minimum=1,
        ),
        fitness=_checkpoint_float(record.get("fitness"), path=f"{path}.fitness"),
        total_layers=_checkpoint_optional_int(
            record.get("total_layers"),
            path=f"{path}.total_layers",
            minimum=0,
        ),
        is_elite=_checkpoint_bool(record.get("is_elite"), path=f"{path}.is_elite"),
        is_survivor=_checkpoint_bool(
            record.get("is_survivor"),
            path=f"{path}.is_survivor",
        ),
        is_offspring=_checkpoint_bool(
            record.get("is_offspring"),
            path=f"{path}.is_offspring",
        ),
        metrics=normalized_metrics,
    )


def _normalize_checkpoint_json_mapping(
    value: Mapping[str, object] | None,
    *,
    field_name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True))
    except TypeError as exc:
        raise TypeError(f"`{field_name}` must be JSON-serializable.") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"`{field_name}` must serialize to a JSON object.")
    for key in normalized:
        if not isinstance(key, str):
            raise TypeError(f"`{field_name}` keys must be strings.")
    return normalized


def _checkpoint_mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} keys must be strings.")
        normalized[key] = item
    return normalized


def _checkpoint_list(value: object, *, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array.")
    return list(value)


def _checkpoint_int(
    value: object,
    *,
    path: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be greater than or equal to {minimum}.")
    return value


def _checkpoint_optional_int(
    value: object,
    *,
    path: str,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _checkpoint_int(value, path=path, minimum=minimum)


def _checkpoint_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric.")
    return float(value)


def _checkpoint_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean.")
    return value


def _checkpoint_str(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _checkpoint_int_tuple(
    value: object,
    *,
    path: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    values = _checkpoint_list(value, path=path)
    return tuple(
        _checkpoint_int(item, path=f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(values)
    )


def select_elites(
    population: Sequence[EvaluatedCandidate[TCandidate]],
    *,
    elite_count: int,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool = False,
) -> tuple[EvaluatedCandidate[TCandidate], ...]:
    """Return the top-ranked elite slice from one evaluated population."""

    if elite_count < 0:
        raise ValueError("`elite_count` must be greater than or equal to 0.")
    if elite_count > len(population):
        raise ValueError("`elite_count` cannot exceed the population size.")
    ranked = rank_population(
        population,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    return tuple(ranked[:elite_count])


def select_survivors(
    population: Sequence[EvaluatedCandidate[TCandidate]],
    *,
    survivor_count: int,
    elite_count: int,
    selection_mode: str,
    tournament_k: int,
    rng: random.Random,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool = False,
) -> SurvivorSelection[TCandidate]:
    """Select elites plus the full survivor pool for the next generation."""

    if survivor_count < 1:
        raise ValueError("`survivor_count` must be greater than or equal to 1.")
    if survivor_count > len(population):
        raise ValueError("`survivor_count` cannot exceed the population size.")
    if elite_count > survivor_count:
        raise ValueError("`elite_count` cannot exceed `survivor_count`.")

    elites = select_elites(
        population,
        elite_count=elite_count,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    survivor_ids = {record.candidate_id for record in elites}
    survivors: list[EvaluatedCandidate[TCandidate]] = list(elites)
    remaining_count = survivor_count - len(survivors)
    if remaining_count == 0:
        return SurvivorSelection(elites=elites, survivors=tuple(survivors))

    remaining_population = [
        record for record in population if record.candidate_id not in survivor_ids
    ]
    if selection_mode == "tournament":
        survivors.extend(
            tournament_select(
                remaining_population,
                count=remaining_count,
                tournament_k=tournament_k,
                rng=rng,
                selection_spec=selection_spec,
                prefer_fewer_layers=prefer_fewer_layers,
                replace=False,
            )
        )
    else:
        raise ValueError(
            f"Unsupported `training.evolution.selection_mode` "
            f"`{selection_mode}`. Supported values: "
            f"{', '.join(sorted(_SUPPORTED_SELECTION_MODES))}."
        )

    survivors = rank_population(
        survivors,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    return SurvivorSelection(elites=elites, survivors=tuple(survivors))


def tournament_select(
    population: Sequence[EvaluatedCandidate[TCandidate]],
    *,
    count: int,
    tournament_k: int,
    rng: random.Random,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool = False,
    replace: bool = True,
) -> list[EvaluatedCandidate[TCandidate]]:
    """Select candidates by repeated tournament draws."""

    if count < 0:
        raise ValueError("`count` must be greater than or equal to 0.")
    if tournament_k < 1:
        raise ValueError("`tournament_k` must be greater than or equal to 1.")
    if count == 0:
        return []
    if not population:
        raise ValueError("Cannot run tournament selection on an empty population.")
    if not replace and count > len(population):
        raise ValueError(
            "`count` cannot exceed the population size when `replace=False`."
        )

    selected: list[EvaluatedCandidate[TCandidate]] = []
    available = list(population)
    for _ in range(count):
        pool = list(population) if replace else available
        if not pool:
            break
        draw_size = min(tournament_k, len(pool))
        contestants = rng.sample(pool, k=draw_size)
        winner = rank_population(
            contestants,
            selection_spec=selection_spec,
            prefer_fewer_layers=prefer_fewer_layers,
        )[0]
        selected.append(winner)
        if not replace:
            available = [
                candidate
                for candidate in available
                if candidate.candidate_id != winner.candidate_id
            ]
    return selected


def rank_population(
    population: Sequence[EvaluatedCandidate[TCandidate]],
    *,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool = False,
) -> list[EvaluatedCandidate[TCandidate]]:
    """Return a deterministically ranked population."""

    unique_population = _dedupe_by_candidate_id(population)
    return sorted(
        unique_population,
        key=lambda record: _candidate_sort_key(
            record,
            selection_spec=selection_spec,
            prefer_fewer_layers=prefer_fewer_layers,
        ),
        reverse=True,
    )


def schedule_mutation_rate(
    base_mutation_rate: float,
    schedule: str,
    *,
    generation: int,
    total_generations: int,
) -> float:
    """Resolve the mutation rate for one real generation."""

    if generation < 1:
        raise ValueError("`generation` must be greater than or equal to 1.")
    if total_generations < 1:
        raise ValueError("`total_generations` must be greater than or equal to 1.")

    if schedule == "constant":
        return float(base_mutation_rate)
    if schedule == "linear_decay":
        if total_generations == 1:
            return float(base_mutation_rate)
        progress = (generation - 1) / (total_generations - 1)
        return max(0.0, float(base_mutation_rate) * (1.0 - progress))

    raise ValueError(
        f"Unsupported `training.evolution.mutation_rate_schedule` `{schedule}`. "
        f"Supported values: {', '.join(sorted(_SUPPORTED_MUTATION_SCHEDULES))}."
    )


def _evaluate_candidate(
    candidate: TCandidate,
    *,
    evaluator: CandidateEvaluator[TCandidate],
    evolution_config: EvolutionConfig,
    layer_counter: LayerCounter[TCandidate] | None,
    candidate_id: int,
    birth_generation: int,
    birth_origin: str,
    parent_id: int | None,
) -> tuple[EvaluatedCandidate[TCandidate], int]:
    evaluation = evaluator.evaluate(candidate)
    if not isinstance(evaluation, CandidateEvaluation):
        raise TypeError(
            "`CandidateEvaluator.evaluate(...)` must return `CandidateEvaluation`."
        )

    total_layers: int | None = None
    if layer_counter is not None:
        total_layers = layer_counter(candidate)
        if isinstance(total_layers, bool) or not isinstance(total_layers, int):
            raise TypeError("`layer_counter` must return an integer layer count.")
        if total_layers < evolution_config.min_layers:
            raise ValueError(
                f"Candidate has {total_layers} layers, below "
                f"`training.evolution.min_layers`={evolution_config.min_layers}."
            )
        if total_layers > evolution_config.max_layers:
            raise ValueError(
                f"Candidate has {total_layers} layers, above "
                f"`training.evolution.max_layers`={evolution_config.max_layers}."
            )

    return (
        EvaluatedCandidate(
            candidate_id=candidate_id,
            candidate=candidate,
            evaluation=evaluation,
            birth_generation=birth_generation,
            birth_origin=birth_origin,
            parent_id=parent_id,
            total_layers=total_layers,
        ),
        candidate_id + 1,
    )


def _select_next_population(
    *,
    survivor_selection: SurvivorSelection[TCandidate],
    offspring: Sequence[EvaluatedCandidate[TCandidate]],
    population_size: int,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool,
) -> list[EvaluatedCandidate[TCandidate]]:
    candidate_pool = list(survivor_selection.survivors) + list(offspring)
    ranked_pool = rank_population(
        candidate_pool,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )
    elite_ids = {record.candidate_id for record in survivor_selection.elites}
    next_population: list[EvaluatedCandidate[TCandidate]] = list(
        survivor_selection.elites
    )
    for candidate in ranked_pool:
        if candidate.candidate_id in elite_ids:
            continue
        next_population.append(candidate)
        if len(next_population) == population_size:
            break

    if len(next_population) < population_size:
        raise ValueError(
            "The shared evolution loop could not refill the requested population. "
            "Ensure `training.evolution.population_size <= mu + lam`."
        )

    return rank_population(
        next_population,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )


def _validate_runtime_config(evolution_config: EvolutionConfig) -> None:
    if evolution_config.selection_mode not in _SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"Unsupported `training.evolution.selection_mode` "
            f"`{evolution_config.selection_mode}`. Supported values: "
            f"{', '.join(sorted(_SUPPORTED_SELECTION_MODES))}."
        )
    if evolution_config.mutation_rate_schedule not in _SUPPORTED_MUTATION_SCHEDULES:
        raise ValueError(
            f"Unsupported `training.evolution.mutation_rate_schedule` "
            f"`{evolution_config.mutation_rate_schedule}`. Supported values: "
            f"{', '.join(sorted(_SUPPORTED_MUTATION_SCHEDULES))}."
        )
    if evolution_config.elite_count > evolution_config.mu:
        raise ValueError(
            "`training.evolution.elite_count` cannot exceed "
            "`training.evolution.mu` for survivor selection."
        )
    if evolution_config.population_size > evolution_config.mu + evolution_config.lam:
        raise ValueError(
            "`training.evolution.population_size` cannot exceed `mu + lam` "
            "for the shared survivor-plus-offspring outer loop."
        )


def _build_generation_summary(
    *,
    generation: int,
    population: Sequence[EvaluatedCandidate[TCandidate]],
    mutation_rate: float,
    selection_spec: SelectionSpec,
    elite_ids: Sequence[int],
    survivor_ids: Sequence[int],
    offspring_ids: Sequence[int],
    stagnation_generations: int,
    stopped_early: bool = False,
) -> GenerationSummary:
    best_candidate = rank_population(
        population,
        selection_spec=selection_spec,
        prefer_fewer_layers=population[0].total_layers is not None,
    )[0]
    return GenerationSummary(
        generation=generation,
        mutation_rate=mutation_rate,
        best_candidate_id=best_candidate.candidate_id,
        best_fitness=best_candidate.evaluation.fitness,
        best_primary_metric=selection_spec.primary_metric,
        best_primary_value=_metric_value(
            best_candidate.evaluation,
            selection_spec.primary_metric,
        ),
        best_total_layers=best_candidate.total_layers,
        mean_fitness=fmean(candidate.evaluation.fitness for candidate in population),
        elite_ids=tuple(elite_ids),
        survivor_ids=tuple(survivor_ids),
        offspring_ids=tuple(offspring_ids),
        population_ids=tuple(candidate.candidate_id for candidate in population),
        stagnation_generations=stagnation_generations,
        stopped_early=stopped_early,
    )


def _build_history_rows(
    *,
    generation: int,
    population: Sequence[EvaluatedCandidate[TCandidate]],
    elite_ids: frozenset[int],
    survivor_ids: frozenset[int],
) -> list[_PopulationHistoryRow]:
    return [
        _PopulationHistoryRow(
            generation=generation,
            candidate_id=candidate.candidate_id,
            birth_generation=candidate.birth_generation,
            birth_origin=candidate.birth_origin,
            parent_id=candidate.parent_id,
            fitness=candidate.evaluation.fitness,
            total_layers=candidate.total_layers,
            is_elite=candidate.candidate_id in elite_ids,
            is_survivor=candidate.candidate_id in survivor_ids,
            is_offspring=candidate.birth_generation == generation
            and candidate.birth_origin == "offspring",
            metrics=dict(candidate.evaluation.metrics),
        )
        for candidate in population
    ]


def _write_history_json(
    path: Path,
    *,
    evolution_config: EvolutionConfig,
    selection_spec: SelectionSpec,
    generation_summaries: Sequence[GenerationSummary],
    best_candidate: EvaluatedCandidate[TCandidate],
    completed_generations: int,
    stopped_early: bool,
    prefer_fewer_layers: bool,
) -> None:
    payload = {
        "config": {
            "seed": evolution_config.seed,
            "generations": evolution_config.generations,
            "population_size": evolution_config.population_size,
            "mu": evolution_config.mu,
            "lam": evolution_config.lam,
            "elite_count": evolution_config.elite_count,
            "min_layers": evolution_config.min_layers,
            "max_layers": evolution_config.max_layers,
            "mutation_rate": evolution_config.mutation_rate,
            "mutation_rate_schedule": evolution_config.mutation_rate_schedule,
            "selection_mode": evolution_config.selection_mode,
            "tournament_k": evolution_config.tournament_k,
            "early_stopping_patience": evolution_config.early_stopping_patience,
        },
        "selection": {
            "primary_metric": selection_spec.primary_metric,
            "tiebreak_metrics": list(selection_spec.tiebreak_metrics),
            "prefer_fewer_layers": prefer_fewer_layers,
        },
        "completed_generations": completed_generations,
        "stopped_early": stopped_early,
        "best_candidate": {
            "candidate_id": best_candidate.candidate_id,
            "birth_generation": best_candidate.birth_generation,
            "birth_origin": best_candidate.birth_origin,
            "parent_id": best_candidate.parent_id,
            "fitness": best_candidate.evaluation.fitness,
            "metrics": dict(best_candidate.evaluation.metrics),
            "total_layers": best_candidate.total_layers,
        },
        "generation_summaries": [
            {
                "generation": summary.generation,
                "mutation_rate": summary.mutation_rate,
                "best_candidate_id": summary.best_candidate_id,
                "best_fitness": summary.best_fitness,
                "best_primary_metric": summary.best_primary_metric,
                "best_primary_value": summary.best_primary_value,
                "best_total_layers": summary.best_total_layers,
                "mean_fitness": summary.mean_fitness,
                "elite_ids": list(summary.elite_ids),
                "survivor_ids": list(summary.survivor_ids),
                "offspring_ids": list(summary.offspring_ids),
                "population_ids": list(summary.population_ids),
                "stagnation_generations": summary.stagnation_generations,
                "stopped_early": summary.stopped_early,
            }
            for summary in generation_summaries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_history_csv(
    path: Path,
    rows: Sequence[_PopulationHistoryRow],
) -> None:
    metric_columns = sorted(
        {
            metric_name
            for row in rows
            for metric_name in row.metrics
        }
    )
    fieldnames = [
        "generation",
        "candidate_id",
        "birth_generation",
        "birth_origin",
        "parent_id",
        "fitness",
        "total_layers",
        "is_elite",
        "is_survivor",
        "is_offspring",
        *[f"metric_{metric_name}" for metric_name in metric_columns],
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                "generation": row.generation,
                "candidate_id": row.candidate_id,
                "birth_generation": row.birth_generation,
                "birth_origin": row.birth_origin,
                "parent_id": row.parent_id,
                "fitness": row.fitness,
                "total_layers": row.total_layers,
                "is_elite": row.is_elite,
                "is_survivor": row.is_survivor,
                "is_offspring": row.is_offspring,
            }
            for metric_name in metric_columns:
                serialized[f"metric_{metric_name}"] = row.metrics.get(metric_name)
            writer.writerow(serialized)


def _log_generation_summary(
    logger: logging.Logger,
    summary: GenerationSummary,
) -> None:
    logger.info(
        "Generation %d: best_candidate=%d primary[%s]=%.6f fitness=%.6f "
        "mean_fitness=%.6f mutation_rate=%.6f elites=%d survivors=%d offspring=%d",
        summary.generation,
        summary.best_candidate_id,
        summary.best_primary_metric,
        summary.best_primary_value,
        summary.best_fitness,
        summary.mean_fitness,
        summary.mutation_rate,
        len(summary.elite_ids),
        len(summary.survivor_ids),
        len(summary.offspring_ids),
    )


def _candidate_sort_key(
    candidate: EvaluatedCandidate[TCandidate],
    *,
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool,
) -> tuple[float, ...]:
    values = [
        _metric_value(candidate.evaluation, selection_spec.primary_metric),
        *(
            _metric_value(candidate.evaluation, metric_name)
            for metric_name in selection_spec.tiebreak_metrics
        ),
    ]
    if prefer_fewer_layers:
        if candidate.total_layers is None:
            raise ValueError(
                "Layer-count tie-breaking requires `total_layers` for every candidate."
            )
        values.append(-float(candidate.total_layers))
    values.append(-float(candidate.candidate_id))
    return tuple(values)


def _is_better_candidate(
    candidate: EvaluatedCandidate[TCandidate],
    *,
    than_candidate: EvaluatedCandidate[TCandidate],
    selection_spec: SelectionSpec,
    prefer_fewer_layers: bool,
) -> bool:
    return _candidate_sort_key(
        candidate,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    ) > _candidate_sort_key(
        than_candidate,
        selection_spec=selection_spec,
        prefer_fewer_layers=prefer_fewer_layers,
    )


def _metric_value(evaluation: CandidateEvaluation, metric_name: str) -> float:
    if metric_name == "fitness":
        return evaluation.fitness
    if metric_name not in evaluation.metrics:
        raise ValueError(
            f"Metric `{metric_name}` was requested for selection but the evaluator "
            "did not provide it."
        )
    return evaluation.metrics[metric_name]


def _normalize_metric_name(metric_name: str, *, field_name: str) -> str:
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise TypeError(f"`{field_name}` entries must be non-empty strings.")
    return metric_name.strip()


def _dedupe_by_candidate_id(
    population: Sequence[EvaluatedCandidate[TCandidate]],
) -> list[EvaluatedCandidate[TCandidate]]:
    deduped: list[EvaluatedCandidate[TCandidate]] = []
    seen_candidate_ids: set[int] = set()
    for candidate in population:
        if candidate.candidate_id in seen_candidate_ids:
            continue
        deduped.append(candidate)
        seen_candidate_ids.add(candidate.candidate_id)
    return deduped


__all__ = [
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
    "select_elites",
    "select_survivors",
    "tournament_select",
]
