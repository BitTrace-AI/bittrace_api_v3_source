"""Minimal deep engine wired through the shared evolution loop.

This proof uses the same local JSON bit bundle format as lean:
- `contract.json`
- `train_bits.json`
- `val_bits.json`
- `test_bits.json`

When `contract.row_format = packed_int_lsb0`, each split JSON stores:
{
  "X_packed": [123, 456, ...],
  "y": [0, 1, ...]
}

Deep scoring differs from lean at the evaluator boundary only. The candidate is
still just a symbolic layer stack, but the evaluator materializes an all-layer
readout embedding by concatenating the residue produced after each layer.
Prototype construction, prediction, metrics, and artifacts use that full
layer-residue embedding rather than the final layer alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import comb
from pathlib import Path
import random
from types import MappingProxyType

from bittrace.core.config import DeepTrainingConfig, EvolutionConfig
from bittrace.core.deep.gpu_backend import (
    DeepBackendSummary,
    GpuMaterialization,
    PackedGpuDeepBackend,
    resolve_deep_backend,
)
from bittrace.core.evolution import (
    CandidateEvaluation,
    EvaluatedCandidate,
    EvolutionRunResult,
    SelectionSpec,
    run_evolution_loop,
)


_CONTRACT_NAME = "contract.json"
_SPLIT_FILE_NAMES = {
    "train": "train_bits.json",
    "val": "val_bits.json",
    "test": "test_bits.json",
}
_ARTIFACT_NAME = "best_deep_artifact.json"
_METRICS_SUMMARY_NAME = "metrics_summary.json"
_SUPPORTED_OPS = frozenset({"xor", "nand", "or", "and", "not", "set0", "set1", "rule3"})
_MASKED_OPS = frozenset({"xor", "nand", "or", "and", "set0", "set1"})
_MAX_EXACT_MEDOID_COMBINATIONS = 4096
_LEGACY_ROW_FORMAT = "bit_list_lsb0"
_PACKED_ROW_FORMAT = "packed_int_lsb0"
_SUPPORTED_ROW_FORMATS = frozenset({_LEGACY_ROW_FORMAT, _PACKED_ROW_FORMAT})


@dataclass(frozen=True, slots=True)
class DeepBundle:
    """Minimal local bit bundle used by the deep proof."""

    bundle_dir: Path
    bit_length: int
    row_format: str
    feature_names: tuple[str, ...]
    train_bits: tuple[int, ...]
    train_labels: tuple[int, ...]
    val_bits: tuple[int, ...]
    val_labels: tuple[int, ...]
    test_bits: tuple[int, ...]
    test_labels: tuple[int, ...]
    class_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeepLayer:
    """Single symbolic deep layer."""

    op: str
    shift: int = 0
    mask: int | None = None
    rule: int | None = None

    def __post_init__(self) -> None:
        if self.op not in _SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported deep op `{self.op}`. Supported ops: "
                f"{', '.join(sorted(_SUPPORTED_OPS))}."
            )
        if self.shift < 0:
            raise ValueError("`shift` must be greater than or equal to 0.")
        if self.op in _MASKED_OPS:
            if self.mask is None:
                raise ValueError(f"`{self.op}` layers require a mask.")
            if self.mask < 0:
                raise ValueError("Layer masks must be non-negative integers.")
        elif self.mask is not None:
            raise ValueError(f"`{self.op}` layers do not accept a mask.")
        if self.op == "rule3":
            if self.rule is None:
                raise ValueError("`rule3` layers require a rule value.")
            if self.rule < 0 or self.rule > 255:
                raise ValueError("`rule3` must be between 0 and 255.")
        elif self.rule is not None:
            raise ValueError(f"`{self.op}` layers do not accept a rule value.")

    def to_dict(self, *, bit_length: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "op": self.op,
            "shift": self.shift,
        }
        if self.mask is not None:
            payload["mask_bits"] = _int_to_bitstring(self.mask, bit_length=bit_length)
        if self.rule is not None:
            payload["rule"] = self.rule
        return payload


@dataclass(frozen=True, slots=True)
class DeepCandidate:
    """Deep search candidate expressed only as a layer stack."""

    layers: tuple[DeepLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Deep candidates must contain at least one layer.")


@dataclass(frozen=True, slots=True)
class DeepSplitMetrics:
    """Compact split metrics kept numeric for history and summaries."""

    n_rows: int
    accuracy: float
    macro_f1: float
    mean_margin: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": self.n_rows,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "mean_margin": self.mean_margin,
        }


@dataclass(frozen=True, slots=True)
class DeepState:
    """Materialized deep state for one candidate."""

    candidate: DeepCandidate
    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
    k_per_class: Mapping[int, int]
    embedding_bit_length: int
    split_metrics: Mapping[str, DeepSplitMetrics]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_per_class", MappingProxyType(dict(self.k_per_class)))
        object.__setattr__(self, "split_metrics", MappingProxyType(dict(self.split_metrics)))


@dataclass(frozen=True, slots=True)
class DeepEvolutionResult:
    """Deep run result plus the artifact paths written after the loop."""

    evolution_result: EvolutionRunResult[DeepCandidate]
    artifact_path: Path
    metrics_summary_path: Path
    backend_summary: DeepBackendSummary


def load_deep_bundle(path: str | Path) -> DeepBundle:
    """Load a minimal local deep bundle."""

    bundle_dir = Path(path)
    contract_path = bundle_dir / _CONTRACT_NAME
    contract_raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract_raw, Mapping):
        raise ValueError(f"`{contract_path}` must contain a JSON object.")

    bit_length = _read_positive_int(contract_raw.get("bit_length"), path="contract.bit_length")
    row_format = _read_row_format(contract_raw.get("row_format"), path="contract.row_format")
    feature_names_raw = contract_raw.get("feature_names")
    if feature_names_raw is None:
        feature_names = tuple(f"bit_{index}" for index in range(bit_length))
    else:
        if not isinstance(feature_names_raw, list) or not all(
            isinstance(name, str) and name for name in feature_names_raw
        ):
            raise ValueError("`contract.feature_names` must be a list of non-empty strings.")
        if len(feature_names_raw) != bit_length:
            raise ValueError(
                "`contract.feature_names` length must match `contract.bit_length`."
            )
        feature_names = tuple(feature_names_raw)

    split_payloads = {
        split_name: _load_split_file(
            bundle_dir / file_name,
            split_name=split_name,
            bit_length=bit_length,
            row_format=row_format,
        )
        for split_name, file_name in _SPLIT_FILE_NAMES.items()
    }
    train_labels = split_payloads["train"][1]
    class_labels = tuple(sorted(set(train_labels)))
    if not class_labels:
        raise ValueError("Deep bundles require at least one training class.")

    for split_name in ("val", "test"):
        unexpected = sorted(set(split_payloads[split_name][1]).difference(class_labels))
        if unexpected:
            raise ValueError(
                f"`{split_name}` labels {unexpected!r} do not exist in the train split."
            )

    return DeepBundle(
        bundle_dir=bundle_dir,
        bit_length=bit_length,
        row_format=row_format if row_format is not None else _PACKED_ROW_FORMAT,
        feature_names=feature_names,
        train_bits=split_payloads["train"][0],
        train_labels=train_labels,
        val_bits=split_payloads["val"][0],
        val_labels=split_payloads["val"][1],
        test_bits=split_payloads["test"][0],
        test_labels=split_payloads["test"][1],
        class_labels=class_labels,
    )


class DeepEvaluator:
    """Deep evaluator that keeps all-layer readout behind the shared boundary."""

    def __init__(
        self,
        bundle: DeepBundle,
        *,
        deep_config: DeepTrainingConfig | None = None,
        backend: str | None = None,
        allow_backend_fallback: bool | None = None,
        include_test_metrics: bool = True,
    ) -> None:
        self._bundle = bundle
        self._deep_config = deep_config if deep_config is not None else DeepTrainingConfig()
        requested_backend = self._deep_config.backend if backend is None else backend
        allow_fallback = (
            self._deep_config.allow_backend_fallback
            if allow_backend_fallback is None
            else allow_backend_fallback
        )
        self._backend_summary = resolve_deep_backend(
            row_format=bundle.row_format,
            bit_length=bundle.bit_length,
            requested_backend=requested_backend,
            allow_backend_fallback=allow_fallback,
        )
        self._include_test_metrics = include_test_metrics
        self._gpu_backend = (
            None
            if self._backend_summary.backend_actual != "gpu"
            else PackedGpuDeepBackend(
                bit_length=bundle.bit_length,
                train_rows=bundle.train_bits,
                train_labels=bundle.train_labels,
                val_rows=bundle.val_bits,
                val_labels=bundle.val_labels,
                test_rows=bundle.test_bits,
                test_labels=bundle.test_labels,
                class_labels=bundle.class_labels,
            )
        )
        self._cache: dict[DeepCandidate, DeepState] = {}

    @property
    def backend_summary(self) -> DeepBackendSummary:
        return self._backend_summary

    def evaluate(self, candidate: DeepCandidate) -> CandidateEvaluation:
        state = self.materialize(candidate)
        val_metrics = state.split_metrics["val"]
        return CandidateEvaluation(
            fitness=val_metrics.macro_f1,
            metrics={
                "accuracy": val_metrics.accuracy,
                "mean_margin": val_metrics.mean_margin,
            },
        )

    def materialize(self, candidate: DeepCandidate) -> DeepState:
        cached = self._cache.get(candidate)
        if cached is not None:
            return cached

        if self._backend_summary.backend_actual == "gpu":
            if self._gpu_backend is None:
                raise RuntimeError("Deep GPU backend was selected but not initialized.")
            materialized = self._gpu_backend.materialize(
                candidate_layers=candidate.layers,
                deep_config=self._deep_config,
                include_test_metrics=self._include_test_metrics,
            )
            state = DeepState(
                candidate=candidate,
                prototypes=materialized.prototypes,
                prototype_labels=materialized.prototype_labels,
                k_per_class=materialized.k_per_class,
                embedding_bit_length=materialized.embedding_bit_length,
                split_metrics={
                    split_name: DeepSplitMetrics(
                        n_rows=int(metrics["n_rows"]),
                        accuracy=float(metrics["accuracy"]),
                        macro_f1=float(metrics["macro_f1"]),
                        mean_margin=float(metrics["mean_margin"]),
                    )
                    for split_name, metrics in materialized.split_metrics.items()
                },
            )
            self._cache[candidate] = state
            return state

        train_embeddings = _apply_layers_as_embedding(
            self._bundle.train_bits,
            candidate.layers,
            bit_length=self._bundle.bit_length,
        )
        val_embeddings = _apply_layers_as_embedding(
            self._bundle.val_bits,
            candidate.layers,
            bit_length=self._bundle.bit_length,
        )
        test_embeddings = _apply_layers_as_embedding(
            self._bundle.test_bits,
            candidate.layers,
            bit_length=self._bundle.bit_length,
        )
        prototypes, prototype_labels, k_per_class = _select_prototypes(
            train_embeddings=train_embeddings,
            train_labels=self._bundle.train_labels,
            val_embeddings=val_embeddings,
            val_labels=self._bundle.val_labels,
            class_labels=self._bundle.class_labels,
            deep_config=self._deep_config,
        )
        embedding_bit_length = self._bundle.bit_length * len(candidate.layers)
        split_metrics = {
            "train": _evaluate_split(
                train_embeddings,
                self._bundle.train_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
                class_labels=self._bundle.class_labels,
            ),
            "val": _evaluate_split(
                val_embeddings,
                self._bundle.val_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
                class_labels=self._bundle.class_labels,
            ),
        }
        if self._include_test_metrics:
            split_metrics["test"] = _evaluate_split(
                test_embeddings,
                self._bundle.test_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
                class_labels=self._bundle.class_labels,
            )
        state = DeepState(
            candidate=candidate,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            k_per_class=k_per_class,
            embedding_bit_length=embedding_bit_length,
            split_metrics=split_metrics,
        )
        self._cache[candidate] = state
        return state


def build_deep_initializer(bundle: DeepBundle):
    """Return the shared-loop initializer for deep candidates."""

    def initialize_candidate(
        rng: random.Random,
        index: int,
        evolution_config: EvolutionConfig,
    ) -> DeepCandidate:
        base = DeepCandidate(
            layers=tuple(_identity_layer() for _ in range(evolution_config.min_layers))
        )
        if index == 0:
            return base

        candidate = base
        warmup_steps = 1 + (index % max(1, evolution_config.max_layers))
        for _ in range(warmup_steps):
            candidate = mutate_deep_candidate(
                candidate,
                rng,
                max(0.6, evolution_config.mutation_rate),
                0,
                evolution_config,
                bit_length=bundle.bit_length,
            )
        return candidate

    return initialize_candidate


def build_deep_mutator(bundle: DeepBundle):
    """Return the shared-loop mutator for deep candidates."""

    def mutate_candidate(
        parent: DeepCandidate,
        rng: random.Random,
        mutation_rate: float,
        generation: int,
        evolution_config: EvolutionConfig,
    ) -> DeepCandidate:
        return mutate_deep_candidate(
            parent,
            rng,
            mutation_rate,
            generation,
            evolution_config,
            bit_length=bundle.bit_length,
        )

    return mutate_candidate


def mutate_deep_candidate(
    parent: DeepCandidate,
    rng: random.Random,
    mutation_rate: float,
    generation: int,
    evolution_config: EvolutionConfig,
    *,
    bit_length: int,
) -> DeepCandidate:
    """Mutate a deep layer stack while staying within shared bounds."""

    del generation
    layers = list(parent.layers)
    clamped_rate = min(1.0, max(0.0, mutation_rate))

    mutated_any = False
    if len(layers) < evolution_config.max_layers and rng.random() < clamped_rate:
        insert_at = rng.randrange(len(layers) + 1)
        layers.insert(insert_at, _random_layer(bit_length, rng))
        mutated_any = True

    if len(layers) > evolution_config.min_layers and rng.random() < (clamped_rate / 2.0):
        del layers[rng.randrange(len(layers))]
        mutated_any = True

    for index, layer in enumerate(list(layers)):
        if rng.random() < clamped_rate:
            layers[index] = _mutate_layer(layer, bit_length=bit_length, rng=rng)
            mutated_any = True

    if not mutated_any:
        target = rng.randrange(len(layers))
        layers[target] = _mutate_layer(layers[target], bit_length=bit_length, rng=rng)

    if len(layers) < evolution_config.min_layers:
        while len(layers) < evolution_config.min_layers:
            layers.append(_identity_layer())
    if len(layers) > evolution_config.max_layers:
        layers = layers[: evolution_config.max_layers]

    return DeepCandidate(layers=tuple(layers))


def count_deep_layers(candidate: DeepCandidate) -> int:
    """Return the honest deep layer count used for shared tie-breaking."""

    return len(candidate.layers)


def serialize_deep_candidate(candidate: DeepCandidate) -> dict[str, object]:
    """Serialize the honest deep candidate state required for resume."""

    return {
        "layers": [
            {
                "op": layer.op,
                "shift": layer.shift,
                "mask": layer.mask,
                "rule": layer.rule,
            }
            for layer in candidate.layers
        ]
    }


def deserialize_deep_candidate(payload: Mapping[str, object]) -> DeepCandidate:
    """Restore one deep candidate from checkpoint state."""

    if not isinstance(payload, Mapping):
        raise ValueError("Deep checkpoint candidate payload must be a mapping.")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Deep checkpoint candidates must contain a non-empty `layers` list.")
    layers = tuple(
        _deserialize_deep_layer(layer_payload, path=f"candidate.layers[{index}]")
        for index, layer_payload in enumerate(raw_layers)
    )
    return DeepCandidate(layers=layers)


def _deserialize_deep_layer(payload: object, *, path: str) -> DeepLayer:
    if not isinstance(payload, Mapping):
        raise ValueError(f"`{path}` must be a mapping.")
    op = payload.get("op")
    shift = payload.get("shift")
    mask = payload.get("mask")
    rule = payload.get("rule")
    if not isinstance(op, str) or not op:
        raise ValueError(f"`{path}.op` must be a non-empty string.")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ValueError(f"`{path}.shift` must be an integer.")
    if mask is not None and (isinstance(mask, bool) or not isinstance(mask, int)):
        raise ValueError(f"`{path}.mask` must be an integer or null.")
    if rule is not None and (isinstance(rule, bool) or not isinstance(rule, int)):
        raise ValueError(f"`{path}.rule` must be an integer or null.")
    return DeepLayer(op=op, shift=shift, mask=mask, rule=rule)


def _build_deep_artifact_identity(
    bundle: DeepBundle,
    deep_config: DeepTrainingConfig,
) -> dict[str, object]:
    return {
        "mode": "deep",
        "bundle_dir": str(bundle.bundle_dir.resolve()),
        "bundle_sha256": {
            file_name: _sha256_file(bundle.bundle_dir / file_name)
            for file_name in (_CONTRACT_NAME, *_SPLIT_FILE_NAMES.values())
        },
        "deep_config": {
            "k_medoids_per_class": deep_config.k_medoids_per_class,
            "adaptive_k": deep_config.adaptive_k,
            "adaptive_k_candidates": list(deep_config.adaptive_k_candidates),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run_deep_evolution(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    evolution_config: EvolutionConfig,
    deep_config: DeepTrainingConfig | None = None,
    backend: str | None = None,
    allow_backend_fallback: bool | None = None,
    selection_spec: SelectionSpec | None = None,
    include_test_metrics: bool = True,
) -> DeepEvolutionResult:
    """Run deep through the shared evolution loop and write real artifacts."""

    bundle = load_deep_bundle(bundle_dir)
    resolved_deep_config = deep_config if deep_config is not None else DeepTrainingConfig()
    evaluator = DeepEvaluator(
        bundle,
        deep_config=resolved_deep_config,
        backend=backend,
        allow_backend_fallback=allow_backend_fallback,
        include_test_metrics=include_test_metrics,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_selection = (
        selection_spec
        if selection_spec is not None
        else SelectionSpec(
            primary_metric="fitness",
            tiebreak_metrics=("accuracy", "mean_margin"),
        )
    )
    evolution_result = run_evolution_loop(
        evolution_config,
        initialize_candidate=build_deep_initializer(bundle),
        mutate_candidate=build_deep_mutator(bundle),
        evaluator=evaluator,
        output_dir=output_path,
        selection_spec=resolved_selection,
        layer_counter=count_deep_layers,
        candidate_serializer=serialize_deep_candidate,
        candidate_deserializer=deserialize_deep_candidate,
        artifact_identity=_build_deep_artifact_identity(
            bundle,
            resolved_deep_config,
        ),
    )
    best_record = evolution_result.best_candidate
    best_state = evaluator.materialize(best_record.candidate)

    artifact_path = output_path / _ARTIFACT_NAME
    metrics_summary_path = output_path / _METRICS_SUMMARY_NAME
    _write_json(
        artifact_path,
        _build_artifact_payload(
            bundle=bundle,
            state=best_state,
            deep_config=resolved_deep_config,
            backend_summary=evaluator.backend_summary,
            best_candidate=best_record,
            evolution_config=evolution_config,
            selection_spec=resolved_selection,
        ),
    )
    _write_json(
        metrics_summary_path,
        _build_metrics_summary_payload(
            state=best_state,
            best_candidate=best_record,
            evolution_result=evolution_result,
            artifact_path=artifact_path,
            backend_summary=evaluator.backend_summary,
        ),
    )

    return DeepEvolutionResult(
        evolution_result=evolution_result,
        artifact_path=artifact_path,
        metrics_summary_path=metrics_summary_path,
        backend_summary=evaluator.backend_summary,
    )


def _load_split_file(
    path: Path,
    *,
    split_name: str,
    bit_length: int,
    row_format: str | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"`{path}` must contain a JSON object.")
    raw_labels = payload.get("y")
    if not isinstance(raw_labels, list):
        raise ValueError(f"`{path}` must provide a `y` list.")

    resolved_row_format, raw_rows = _resolve_raw_rows(
        payload,
        path=path,
        row_format=row_format,
    )
    if len(raw_rows) != len(raw_labels):
        raise ValueError(
            f"`{split_name}` row count mismatch: "
            f"{'X_packed' if resolved_row_format == _PACKED_ROW_FORMAT else 'X_bits'}="
            f"{len(raw_rows)} y={len(raw_labels)}."
        )

    if resolved_row_format == _PACKED_ROW_FORMAT:
        rows = tuple(
            _read_packed_row(
                row,
                split_name=split_name,
                row_index=index,
                bit_length=bit_length,
            )
            for index, row in enumerate(raw_rows)
        )
    else:
        rows = tuple(
            _row_to_int(row, split_name=split_name, row_index=index, bit_length=bit_length)
            for index, row in enumerate(raw_rows)
        )
    labels = tuple(
        _read_int(label, path=f"{split_name}.y[{index}]")
        for index, label in enumerate(raw_labels)
    )
    return rows, labels


def _resolve_raw_rows(
    payload: Mapping[str, object],
    *,
    path: Path,
    row_format: str | None,
) -> tuple[str, list[object]]:
    packed_rows = payload.get("X_packed")
    unpacked_rows = payload.get("X_bits")
    resolved_row_format = row_format

    if resolved_row_format is None:
        if isinstance(packed_rows, list) and unpacked_rows is None:
            resolved_row_format = _PACKED_ROW_FORMAT
        elif isinstance(unpacked_rows, list) and packed_rows is None:
            resolved_row_format = _LEGACY_ROW_FORMAT
        elif packed_rows is None and unpacked_rows is None:
            raise ValueError(f"`{path}` must provide `X_packed` or `X_bits` plus `y`.")
        else:
            raise ValueError(
                f"`{path}` must not provide both `X_packed` and `X_bits` without "
                "an explicit `contract.row_format`."
            )

    if resolved_row_format == _PACKED_ROW_FORMAT:
        if not isinstance(packed_rows, list):
            raise ValueError(
                f"`{path}` declares packed rows but does not provide an `X_packed` list."
            )
        return resolved_row_format, packed_rows
    if resolved_row_format == _LEGACY_ROW_FORMAT:
        if not isinstance(unpacked_rows, list):
            raise ValueError(
                f"`{path}` declares unpacked rows but does not provide an `X_bits` list."
            )
        return resolved_row_format, unpacked_rows
    raise ValueError(f"Unsupported row format `{resolved_row_format}`.")


def _read_packed_row(
    row: object,
    *,
    split_name: str,
    row_index: int,
    bit_length: int,
) -> int:
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError(f"`{split_name}.X_packed[{row_index}]` must be an integer.")
    if row < 0:
        raise ValueError(f"`{split_name}.X_packed[{row_index}]` must be non-negative.")
    max_value = (1 << bit_length) - 1
    if row > max_value:
        raise ValueError(
            f"`{split_name}.X_packed[{row_index}]` exceeds {bit_length} bits."
        )
    return row


def _row_to_int(
    row: object,
    *,
    split_name: str,
    row_index: int,
    bit_length: int,
) -> int:
    if not isinstance(row, list) or len(row) != bit_length:
        raise ValueError(
            f"`{split_name}.X_bits[{row_index}]` must be a list of length {bit_length}."
        )
    value = 0
    for bit_index, bit in enumerate(row):
        if bit not in (0, 1):
            raise ValueError(
                f"`{split_name}.X_bits[{row_index}][{bit_index}]` must be 0 or 1."
            )
        if bit == 1:
            value |= 1 << bit_index
    return value


def _apply_layers_as_embedding(
    rows: Sequence[int],
    layers: Sequence[DeepLayer],
    *,
    bit_length: int,
) -> tuple[int, ...]:
    return tuple(
        _apply_row_layers_as_embedding(row, layers, bit_length=bit_length)
        for row in rows
    )


def _apply_row_layers_as_embedding(
    row: int,
    layers: Sequence[DeepLayer],
    *,
    bit_length: int,
) -> int:
    current = row
    embedding = 0
    offset = 0
    for layer in layers:
        current = _apply_layer(current, layer, bit_length=bit_length)
        embedding |= current << offset
        offset += bit_length
    return embedding


def _apply_layer(row: int, layer: DeepLayer, *, bit_length: int) -> int:
    mask_all = (1 << bit_length) - 1
    rotated = _rotate_left(row, layer.shift, bit_length=bit_length)
    if layer.op == "not":
        return (~rotated) & mask_all
    if layer.op == "rule3":
        if layer.rule is None:
            raise ValueError("`rule3` requires a rule value.")
        return _apply_rule3(rotated, layer.rule, bit_length=bit_length)
    if layer.mask is None:
        raise ValueError(f"Layer `{layer.op}` requires a mask.")
    mask = layer.mask & mask_all
    if layer.op == "xor":
        return rotated ^ mask
    if layer.op == "and":
        return rotated & mask
    if layer.op == "or":
        return rotated | mask
    if layer.op == "nand":
        return (~(rotated & mask)) & mask_all
    if layer.op == "set0":
        return rotated & (~mask & mask_all)
    if layer.op == "set1":
        return rotated | mask
    raise ValueError(f"Unsupported deep op `{layer.op}`.")


def _apply_rule3(row: int, rule: int, *, bit_length: int) -> int:
    result = 0
    for bit_index in range(bit_length):
        left = 1 if row & (1 << ((bit_index - 1) % bit_length)) else 0
        center = 1 if row & (1 << bit_index) else 0
        right = 1 if row & (1 << ((bit_index + 1) % bit_length)) else 0
        pattern = (left << 2) | (center << 1) | right
        if (rule >> pattern) & 1:
            result |= 1 << bit_index
    return result


def _rotate_left(value: int, shift: int, *, bit_length: int) -> int:
    if bit_length < 1:
        raise ValueError("`bit_length` must be greater than or equal to 1.")
    if shift == 0:
        return value & ((1 << bit_length) - 1)
    amount = shift % bit_length
    mask_all = (1 << bit_length) - 1
    return ((value << amount) | (value >> (bit_length - amount))) & mask_all


def _select_prototypes(
    *,
    train_embeddings: Sequence[int],
    train_labels: Sequence[int],
    val_embeddings: Sequence[int],
    val_labels: Sequence[int],
    class_labels: Sequence[int],
    deep_config: DeepTrainingConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], Mapping[int, int]]:
    prototypes: list[int] = []
    prototype_labels: list[int] = []
    k_per_class: dict[int, int] = {}

    for class_label in class_labels:
        class_train = [
            row
            for row, label in zip(train_embeddings, train_labels, strict=True)
            if label == class_label
        ]
        if not class_train:
            raise ValueError(
                f"Train split produced no rows for class `{class_label}`."
            )
        class_val = [
            row
            for row, label in zip(val_embeddings, val_labels, strict=True)
            if label == class_label
        ]
        medoids = _select_class_medoids(
            train_rows=class_train,
            val_rows=class_val,
            base_k=deep_config.k_medoids_per_class,
            adaptive_k=deep_config.adaptive_k,
            adaptive_k_candidates=deep_config.adaptive_k_candidates,
        )
        prototypes.extend(medoids)
        prototype_labels.extend([class_label] * len(medoids))
        k_per_class[class_label] = len(medoids)

    return tuple(prototypes), tuple(prototype_labels), MappingProxyType(k_per_class)


def _select_class_medoids(
    *,
    train_rows: Sequence[int],
    val_rows: Sequence[int],
    base_k: int,
    adaptive_k: bool,
    adaptive_k_candidates: Sequence[int],
) -> tuple[int, ...]:
    candidate_ks = [base_k]
    if adaptive_k:
        candidate_ks.extend(adaptive_k_candidates)
    ordered_ks = sorted(set(candidate_ks))

    eval_rows = tuple(val_rows) if val_rows else tuple(train_rows)
    best_medoids: tuple[int, ...] | None = None
    best_score: float | None = None
    best_k: int | None = None
    for k_candidate in ordered_ks:
        medoids = _compute_class_medoids(train_rows, k_candidate)
        score = _mean_min_distance(eval_rows, medoids)
        if (
            best_score is None
            or score < best_score
            or (score == best_score and (best_k is None or len(medoids) < best_k))
        ):
            best_medoids = medoids
            best_score = score
            best_k = len(medoids)

    if best_medoids is None:
        raise ValueError("Failed to build class medoids.")
    return best_medoids


def _compute_class_medoids(rows: Sequence[int], k_medoids: int) -> tuple[int, ...]:
    unique_rows = tuple(dict.fromkeys(rows))
    if not unique_rows:
        raise ValueError("Cannot compute medoids on an empty class.")
    k_eff = min(max(1, int(k_medoids)), len(unique_rows))
    if k_eff == len(unique_rows):
        return unique_rows
    if k_eff == 1:
        return (_select_medoid(unique_rows),)

    if comb(len(unique_rows), k_eff) <= _MAX_EXACT_MEDOID_COMBINATIONS:
        return _compute_exact_medoids(unique_rows, k_eff)
    return _compute_greedy_medoids(unique_rows, k_eff)


def _compute_exact_medoids(rows: Sequence[int], k_eff: int) -> tuple[int, ...]:
    from itertools import combinations

    distance_matrix = _pairwise_distance_matrix(rows)
    best_indices: tuple[int, ...] | None = None
    best_cost: int | None = None
    best_rows: tuple[int, ...] | None = None
    for indices in combinations(range(len(rows)), k_eff):
        total_cost = sum(
            min(distance_matrix[row_index][medoid_index] for medoid_index in indices)
            for row_index in range(len(rows))
        )
        medoid_rows = tuple(rows[index] for index in indices)
        if (
            best_cost is None
            or total_cost < best_cost
            or (total_cost == best_cost and (best_rows is None or medoid_rows < best_rows))
        ):
            best_indices = indices
            best_cost = total_cost
            best_rows = medoid_rows

    if best_indices is None:
        raise ValueError("Exact medoid search failed.")
    return tuple(rows[index] for index in best_indices)


def _compute_greedy_medoids(rows: Sequence[int], k_eff: int) -> tuple[int, ...]:
    distance_matrix = _pairwise_distance_matrix(rows)
    total_costs = [sum(distances) for distances in distance_matrix]
    first_index = min(
        range(len(rows)),
        key=lambda index: (total_costs[index], rows[index]),
    )
    medoid_indices = [first_index]

    while len(medoid_indices) < k_eff:
        remaining = [index for index in range(len(rows)) if index not in medoid_indices]
        next_index = max(
            remaining,
            key=lambda index: (
                min(distance_matrix[index][medoid] for medoid in medoid_indices),
                -total_costs[index],
                -rows[index],
            ),
        )
        medoid_indices.append(next_index)

    for _ in range(4):
        assignments: dict[int, list[int]] = {index: [] for index in medoid_indices}
        for row_index in range(len(rows)):
            nearest = min(
                medoid_indices,
                key=lambda medoid_index: (
                    distance_matrix[row_index][medoid_index],
                    rows[medoid_index],
                ),
            )
            assignments[nearest].append(row_index)

        new_indices: list[int] = []
        for medoid_index in medoid_indices:
            members = assignments.get(medoid_index, [])
            if not members:
                new_indices.append(medoid_index)
                continue
            best_local = min(
                members,
                key=lambda candidate_index: (
                    sum(distance_matrix[candidate_index][member] for member in members),
                    rows[candidate_index],
                ),
            )
            if best_local not in new_indices:
                new_indices.append(best_local)
        if set(new_indices) == set(medoid_indices):
            break
        medoid_indices = _fill_missing_medoids(
            new_indices,
            target_count=k_eff,
            total_costs=total_costs,
            rows=rows,
        )

    medoid_indices = sorted(
        medoid_indices,
        key=lambda index: (rows[index], index),
    )
    return tuple(rows[index] for index in medoid_indices)


def _fill_missing_medoids(
    medoid_indices: Sequence[int],
    *,
    target_count: int,
    total_costs: Sequence[int],
    rows: Sequence[int],
) -> list[int]:
    filled = list(dict.fromkeys(medoid_indices))
    if len(filled) >= target_count:
        return filled[:target_count]

    remaining = [
        index
        for index in range(len(rows))
        if index not in filled
    ]
    remaining.sort(key=lambda index: (total_costs[index], rows[index], index))
    for index in remaining:
        filled.append(index)
        if len(filled) == target_count:
            break
    return filled


def _pairwise_distance_matrix(rows: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    matrix: list[tuple[int, ...]] = []
    for left in rows:
        matrix.append(tuple(_hamming_distance(left, right) for right in rows))
    return tuple(matrix)


def _select_medoid(rows: Sequence[int]) -> int:
    best_row = rows[0]
    best_cost: int | None = None
    for candidate in rows:
        cost = sum(_hamming_distance(candidate, other) for other in rows)
        if best_cost is None or cost < best_cost or (cost == best_cost and candidate < best_row):
            best_row = candidate
            best_cost = cost
    return best_row


def _evaluate_split(
    embeddings: Sequence[int],
    labels: Sequence[int],
    *,
    prototypes: Sequence[int],
    prototype_labels: Sequence[int],
    class_labels: Sequence[int],
) -> DeepSplitMetrics:
    predictions: list[int] = []
    margins: list[int] = []
    for embedding in embeddings:
        predicted_label, margin = _predict_row(
            embedding,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            class_labels=class_labels,
        )
        predictions.append(predicted_label)
        margins.append(margin)

    accuracy = _compute_accuracy(labels, predictions)
    macro_f1 = _compute_macro_f1(labels, predictions)
    mean_margin = (
        sum(float(margin) for margin in margins) / float(len(margins))
        if margins
        else 0.0
    )
    return DeepSplitMetrics(
        n_rows=len(labels),
        accuracy=accuracy,
        macro_f1=macro_f1,
        mean_margin=mean_margin,
    )


def _predict_row(
    embedding: int,
    *,
    prototypes: Sequence[int],
    prototype_labels: Sequence[int],
    class_labels: Sequence[int],
) -> tuple[int, int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for prototype, label in zip(prototypes, prototype_labels, strict=True):
        grouped[label].append(prototype)

    ranked = sorted(
        (
            (
                min(_hamming_distance(embedding, prototype) for prototype in grouped[class_label]),
                class_label,
            )
            for class_label in class_labels
        ),
        key=lambda item: (item[0], item[1]),
    )
    best_distance, best_label = ranked[0]
    margin = 0 if len(ranked) < 2 else ranked[1][0] - best_distance
    return best_label, margin


def _mean_min_distance(rows: Sequence[int], medoids: Sequence[int]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += float(min(_hamming_distance(row, medoid) for medoid in medoids))
    return total / float(len(rows))


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _compute_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    if not y_true:
        return 0.0
    matches = sum(
        1
        for expected, predicted in zip(y_true, y_pred, strict=True)
        if expected == predicted
    )
    return float(matches) / float(len(y_true))


def _compute_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    labels = sorted(set(y_true).union(y_pred))
    if not labels:
        return 0.0

    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(
            1
            for expected, predicted in zip(y_true, y_pred, strict=True)
            if expected == label and predicted == label
        )
        false_positive = sum(
            1
            for expected, predicted in zip(y_true, y_pred, strict=True)
            if expected != label and predicted == label
        )
        false_negative = sum(
            1
            for expected, predicted in zip(y_true, y_pred, strict=True)
            if expected == label and predicted != label
        )
        precision = (
            float(true_positive) / float(true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            float(true_positive) / float(true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        if precision == 0.0 and recall == 0.0:
            f1_values.append(0.0)
            continue
        f1_values.append((2.0 * precision * recall) / (precision + recall))
    return sum(f1_values) / float(len(f1_values))


def _identity_layer() -> DeepLayer:
    return DeepLayer(op="xor", shift=0, mask=0)


def _random_layer(bit_length: int, rng: random.Random) -> DeepLayer:
    op = ("xor", "nand", "or", "and", "not", "set0", "set1", "rule3")[rng.randrange(8)]
    shift = rng.randrange(bit_length)
    if op == "rule3":
        return DeepLayer(op=op, shift=shift, rule=rng.randrange(256))
    if op in _MASKED_OPS:
        return DeepLayer(op=op, shift=shift, mask=_random_mask(bit_length, rng))
    return DeepLayer(op=op, shift=shift)


def _mutate_layer(layer: DeepLayer, *, bit_length: int, rng: random.Random) -> DeepLayer:
    choice = rng.randrange(4)
    if choice == 0:
        return _random_layer(bit_length, rng)
    if choice == 1:
        if layer.op == "rule3":
            return DeepLayer(op="rule3", shift=rng.randrange(bit_length), rule=layer.rule)
        if layer.op in _MASKED_OPS:
            return DeepLayer(op=layer.op, shift=rng.randrange(bit_length), mask=layer.mask)
        return DeepLayer(op=layer.op, shift=rng.randrange(bit_length))
    if choice == 2 and layer.op in _MASKED_OPS and layer.mask is not None:
        flip_count = max(1, bit_length // 4)
        return DeepLayer(
            op=layer.op,
            shift=layer.shift,
            mask=_flip_mask_bits(layer.mask, bit_length=bit_length, flip_count=flip_count, rng=rng),
        )
    if choice == 3 and layer.op == "rule3":
        return DeepLayer(op="rule3", shift=layer.shift, rule=rng.randrange(256))
    return _random_layer(bit_length, rng)


def _random_mask(bit_length: int, rng: random.Random) -> int:
    return rng.getrandbits(bit_length) & ((1 << bit_length) - 1)


def _flip_mask_bits(
    mask: int,
    *,
    bit_length: int,
    flip_count: int,
    rng: random.Random,
) -> int:
    updated = mask
    for bit_index in rng.sample(range(bit_length), k=min(flip_count, bit_length)):
        updated ^= 1 << bit_index
    return updated & ((1 << bit_length) - 1)


def _int_to_bitstring(value: int, *, bit_length: int) -> str:
    return "".join(
        "1" if value & (1 << bit_index) else "0"
        for bit_index in range(bit_length)
    )


def _build_artifact_payload(
    *,
    bundle: DeepBundle,
    state: DeepState,
    deep_config: DeepTrainingConfig,
    backend_summary: DeepBackendSummary,
    best_candidate: EvaluatedCandidate[DeepCandidate],
    evolution_config: EvolutionConfig,
    selection_spec: SelectionSpec,
) -> dict[str, object]:
    return {
        "schema_version": "3.0-deep-proof-1",
        "mode": "deep",
        "scoring_mode": "all_layer_residue_readout",
        "embedding_mode": "concatenated_layer_residues",
        "inputs": {
            "bundle_dir": str(bundle.bundle_dir.resolve()),
        },
        "bit_length": bundle.bit_length,
        "embedding_bit_length": state.embedding_bit_length,
        "feature_names": list(bundle.feature_names),
        "class_labels": list(bundle.class_labels),
        "selection": {
            "primary_metric": selection_spec.primary_metric,
            "tiebreak_metrics": list(selection_spec.tiebreak_metrics),
        },
        "execution": backend_summary.to_dict(),
        "training": {
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
            "deep": {
                "k_medoids_per_class": deep_config.k_medoids_per_class,
                "adaptive_k": deep_config.adaptive_k,
                "adaptive_k_candidates": list(deep_config.adaptive_k_candidates),
            },
        },
        "best_candidate": {
            "candidate_id": best_candidate.candidate_id,
            "birth_generation": best_candidate.birth_generation,
            "birth_origin": best_candidate.birth_origin,
            "parent_id": best_candidate.parent_id,
            "fitness": best_candidate.evaluation.fitness,
            "metrics": dict(best_candidate.evaluation.metrics),
            "total_layers": best_candidate.total_layers,
        },
        "model": {
            "layers": [
                layer.to_dict(bit_length=bundle.bit_length)
                for layer in state.candidate.layers
            ],
            "embedding_bit_length": state.embedding_bit_length,
            "prototypes": [
                _int_to_bitstring(prototype, bit_length=state.embedding_bit_length)
                for prototype in state.prototypes
            ],
            "prototype_labels": list(state.prototype_labels),
            "k_per_class": {
                str(class_label): k_value
                for class_label, k_value in state.k_per_class.items()
            },
        },
    }


def _build_metrics_summary_payload(
    *,
    state: DeepState,
    best_candidate: EvaluatedCandidate[DeepCandidate],
    evolution_result: EvolutionRunResult[DeepCandidate],
    artifact_path: Path,
    backend_summary: DeepBackendSummary,
) -> dict[str, object]:
    return {
        "mode": "deep",
        "scoring_mode": "all_layer_residue_readout",
        "execution": backend_summary.to_dict(),
        "artifact_path": str(artifact_path.resolve()),
        "history_json_path": str(evolution_result.history_json_path.resolve()),
        "history_csv_path": str(evolution_result.history_csv_path.resolve()),
        "completed_generations": evolution_result.completed_generations,
        "stopped_early": evolution_result.stopped_early,
        "best_candidate": {
            "candidate_id": best_candidate.candidate_id,
            "fitness": best_candidate.evaluation.fitness,
            "metrics": dict(best_candidate.evaluation.metrics),
            "total_layers": best_candidate.total_layers,
            "prototype_count": len(state.prototype_labels),
        },
        "embedding_bit_length": state.embedding_bit_length,
        "k_per_class": {
            str(class_label): k_value
            for class_label, k_value in state.k_per_class.items()
        },
        "splits": {
            split_name: metrics.to_dict()
            for split_name, metrics in state.split_metrics.items()
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_positive_int(value: object, *, path: str) -> int:
    out = _read_int(value, path=path)
    if out < 1:
        raise ValueError(f"`{path}` must be greater than or equal to 1.")
    return out


def _read_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{path}` must be an integer.")
    return value


def _read_row_format(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"`{path}` must be a non-empty string when provided.")
    if value not in _SUPPORTED_ROW_FORMATS:
        raise ValueError(
            f"`{path}` must be one of {', '.join(sorted(_SUPPORTED_ROW_FORMATS))}."
        )
    return value


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
