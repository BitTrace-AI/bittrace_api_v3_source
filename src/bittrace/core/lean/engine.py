"""Minimal lean engine wired through the shared evolution loop.

This proof keeps the data path local and simple:
- `contract.json`
- `train_bits.json`
- `val_bits.json`
- `test_bits.json`

Each split JSON stores:
{
  "X_packed": [123, 456, ...],
  "y": [0, 1, ...]
}

Lean scoring uses only the final-layer output. Prototype construction,
prediction, metrics, and exported artifacts all operate on that final layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from types import MappingProxyType

from bittrace.core.config import EvolutionConfig, LeanTrainingConfig
from bittrace.core.evolution import (
    CandidateEvaluation,
    EvaluatedCandidate,
    EvolutionRunResult,
    SelectionSpec,
    run_evolution_loop,
)
from bittrace.core.lean.gpu_backend import (
    LeanBackendSummary,
    PackedGpuLeanBackend,
    resolve_lean_backend,
)


_CONTRACT_NAME = "contract.json"
_SPLIT_FILE_NAMES = {
    "train": "train_bits.json",
    "val": "val_bits.json",
    "test": "test_bits.json",
}
_ARTIFACT_NAME = "best_lean_artifact.json"
_METRICS_SUMMARY_NAME = "metrics_summary.json"
_SUPPORTED_OPS = frozenset({"xor", "and", "or", "not"})
_LEGACY_ROW_FORMAT = "bit_list_lsb0"
_PACKED_ROW_FORMAT = "packed_int_lsb0"
_SUPPORTED_ROW_FORMATS = frozenset({_LEGACY_ROW_FORMAT, _PACKED_ROW_FORMAT})


@dataclass(frozen=True, slots=True)
class LeanBundle:
    """Minimal local bit bundle used by the lean proof."""

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
class LeanLayer:
    """Single lean layer op applied to the bit row before scoring."""

    op: str
    shift: int = 0
    mask: int | None = None

    def __post_init__(self) -> None:
        if self.op not in _SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported lean op `{self.op}`. Supported ops: "
                f"{', '.join(sorted(_SUPPORTED_OPS))}."
            )
        if self.shift < 0:
            raise ValueError("`shift` must be greater than or equal to 0.")
        if self.op == "not" and self.mask is not None:
            raise ValueError("`not` layers do not accept a mask.")
        if self.op != "not":
            if self.mask is None:
                raise ValueError(f"`{self.op}` layers require a mask.")
            if self.mask < 0:
                raise ValueError("Layer masks must be non-negative integers.")

    def to_dict(self, *, bit_length: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "op": self.op,
            "shift": self.shift,
        }
        if self.mask is not None:
            payload["mask_bits"] = _int_to_bitstring(self.mask, bit_length=bit_length)
        return payload


@dataclass(frozen=True, slots=True)
class LeanCandidate:
    """Lean search candidate expressed only as a layer stack."""

    layers: tuple[LeanLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Lean candidates must contain at least one layer.")


@dataclass(frozen=True, slots=True)
class LeanSplitMetrics:
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
class LeanState:
    """Materialized lean state for one candidate."""

    candidate: LeanCandidate
    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
    split_metrics: Mapping[str, LeanSplitMetrics]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_metrics",
            MappingProxyType(dict(self.split_metrics)),
        )


@dataclass(frozen=True, slots=True)
class LeanEvolutionResult:
    """Lean run result plus the real artifact paths written after the loop."""

    evolution_result: EvolutionRunResult[LeanCandidate]
    artifact_path: Path
    metrics_summary_path: Path
    backend_summary: LeanBackendSummary


def load_lean_bundle(path: str | Path) -> LeanBundle:
    """Load a minimal local lean bundle."""

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
        raise ValueError("Lean bundles require at least one training class.")

    for split_name in ("val", "test"):
        unexpected = sorted(set(split_payloads[split_name][1]).difference(class_labels))
        if unexpected:
            raise ValueError(
                f"`{split_name}` labels {unexpected!r} do not exist in the train split."
            )

    return LeanBundle(
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


class LeanEvaluator:
    """Lean evaluator that keeps final-layer-only scoring behind the boundary."""

    def __init__(
        self,
        bundle: LeanBundle,
        *,
        lean_config: LeanTrainingConfig | None = None,
        backend: str | None = None,
        allow_backend_fallback: bool | None = None,
        include_test_metrics: bool = True,
    ) -> None:
        self._bundle = bundle
        self._lean_config = lean_config if lean_config is not None else LeanTrainingConfig()
        requested_backend = self._lean_config.backend if backend is None else backend
        allow_fallback = (
            self._lean_config.allow_backend_fallback
            if allow_backend_fallback is None
            else allow_backend_fallback
        )
        self._backend_summary = resolve_lean_backend(
            row_format=bundle.row_format,
            bit_length=bundle.bit_length,
            requested_backend=requested_backend,
            allow_backend_fallback=allow_fallback,
        )
        self._include_test_metrics = include_test_metrics
        self._gpu_backend = (
            None
            if self._backend_summary.backend_actual != "gpu"
            else PackedGpuLeanBackend(
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
        self._cache: dict[LeanCandidate, LeanState] = {}

    @property
    def backend_summary(self) -> LeanBackendSummary:
        return self._backend_summary

    def evaluate(self, candidate: LeanCandidate) -> CandidateEvaluation:
        state = self.materialize(candidate)
        val_metrics = state.split_metrics["val"]
        return CandidateEvaluation(
            fitness=val_metrics.macro_f1,
            metrics={
                "accuracy": val_metrics.accuracy,
                "mean_margin": val_metrics.mean_margin,
            },
        )

    def materialize(self, candidate: LeanCandidate) -> LeanState:
        cached = self._cache.get(candidate)
        if cached is not None:
            return cached

        if self._backend_summary.backend_actual == "gpu":
            if self._gpu_backend is None:
                raise RuntimeError("Lean GPU backend was selected but not initialized.")
            materialized = self._gpu_backend.materialize(
                candidate_layers=candidate.layers,
                include_test_metrics=self._include_test_metrics,
            )
            state = LeanState(
                candidate=candidate,
                prototypes=materialized.prototypes,
                prototype_labels=materialized.prototype_labels,
                split_metrics={
                    split_name: LeanSplitMetrics(
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

        final_train_bits = _apply_layers(
            self._bundle.train_bits,
            candidate.layers,
            bit_length=self._bundle.bit_length,
        )
        prototypes, prototype_labels = _select_prototypes(
            final_train_bits,
            self._bundle.train_labels,
            self._bundle.class_labels,
        )
        split_metrics = {
            "train": _evaluate_split(
                self._bundle.train_bits,
                self._bundle.train_labels,
                candidate.layers,
                prototypes,
                prototype_labels,
                bit_length=self._bundle.bit_length,
            ),
            "val": _evaluate_split(
                self._bundle.val_bits,
                self._bundle.val_labels,
                candidate.layers,
                prototypes,
                prototype_labels,
                bit_length=self._bundle.bit_length,
            ),
        }
        if self._include_test_metrics:
            split_metrics["test"] = _evaluate_split(
                self._bundle.test_bits,
                self._bundle.test_labels,
                candidate.layers,
                prototypes,
                prototype_labels,
                bit_length=self._bundle.bit_length,
            )
        state = LeanState(
            candidate=candidate,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            split_metrics=split_metrics,
        )
        self._cache[candidate] = state
        return state


def build_lean_initializer(bundle: LeanBundle):
    """Return the shared-loop initializer for lean candidates."""

    def initialize_candidate(
        rng: random.Random,
        index: int,
        evolution_config: EvolutionConfig,
    ) -> LeanCandidate:
        base = LeanCandidate(
            layers=tuple(
                _identity_layer()
                for _ in range(evolution_config.min_layers)
            )
        )
        if index == 0:
            return base

        candidate = base
        warmup_steps = 1 + (index % max(1, evolution_config.max_layers))
        for _ in range(warmup_steps):
            candidate = mutate_lean_candidate(
                candidate,
                rng,
                max(0.6, evolution_config.mutation_rate),
                0,
                evolution_config,
                bit_length=bundle.bit_length,
            )
        return candidate

    return initialize_candidate


def build_lean_mutator(bundle: LeanBundle):
    """Return the shared-loop mutator for lean candidates."""

    def mutate_candidate(
        parent: LeanCandidate,
        rng: random.Random,
        mutation_rate: float,
        generation: int,
        evolution_config: EvolutionConfig,
    ) -> LeanCandidate:
        return mutate_lean_candidate(
            parent,
            rng,
            mutation_rate,
            generation,
            evolution_config,
            bit_length=bundle.bit_length,
        )

    return mutate_candidate


def mutate_lean_candidate(
    parent: LeanCandidate,
    rng: random.Random,
    mutation_rate: float,
    generation: int,
    evolution_config: EvolutionConfig,
    *,
    bit_length: int,
) -> LeanCandidate:
    """Mutate a lean layer stack while staying within shared bounds."""

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

    return LeanCandidate(layers=tuple(layers))


def count_lean_layers(candidate: LeanCandidate) -> int:
    """Return the honest lean layer count used for shared tie-breaking."""

    return len(candidate.layers)


def serialize_lean_candidate(candidate: LeanCandidate) -> dict[str, object]:
    """Serialize the honest lean candidate state required for resume."""

    return {
        "layers": [
            {
                "op": layer.op,
                "shift": layer.shift,
                "mask": layer.mask,
            }
            for layer in candidate.layers
        ]
    }


def deserialize_lean_candidate(payload: Mapping[str, object]) -> LeanCandidate:
    """Restore one lean candidate from checkpoint state."""

    if not isinstance(payload, Mapping):
        raise ValueError("Lean checkpoint candidate payload must be a mapping.")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Lean checkpoint candidates must contain a non-empty `layers` list.")
    layers = tuple(
        _deserialize_lean_layer(layer_payload, path=f"candidate.layers[{index}]")
        for index, layer_payload in enumerate(raw_layers)
    )
    return LeanCandidate(layers=layers)


def _deserialize_lean_layer(payload: object, *, path: str) -> LeanLayer:
    if not isinstance(payload, Mapping):
        raise ValueError(f"`{path}` must be a mapping.")
    op = payload.get("op")
    shift = payload.get("shift")
    mask = payload.get("mask")
    if not isinstance(op, str) or not op:
        raise ValueError(f"`{path}.op` must be a non-empty string.")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ValueError(f"`{path}.shift` must be an integer.")
    if mask is not None and (isinstance(mask, bool) or not isinstance(mask, int)):
        raise ValueError(f"`{path}.mask` must be an integer or null.")
    return LeanLayer(op=op, shift=shift, mask=mask)


def _build_lean_artifact_identity(bundle: LeanBundle) -> dict[str, object]:
    return {
        "mode": "lean",
        "bundle_dir": str(bundle.bundle_dir.resolve()),
        "bundle_sha256": {
            file_name: _sha256_file(bundle.bundle_dir / file_name)
            for file_name in (_CONTRACT_NAME, *_SPLIT_FILE_NAMES.values())
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run_lean_evolution(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    evolution_config: EvolutionConfig,
    lean_config: LeanTrainingConfig | None = None,
    backend: str | None = None,
    allow_backend_fallback: bool | None = None,
    selection_spec: SelectionSpec | None = None,
    include_test_metrics: bool = True,
) -> LeanEvolutionResult:
    """Run lean through the shared evolution loop and write real artifacts."""

    bundle = load_lean_bundle(bundle_dir)
    resolved_lean_config = lean_config if lean_config is not None else LeanTrainingConfig()
    evaluator = LeanEvaluator(
        bundle,
        lean_config=resolved_lean_config,
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
        initialize_candidate=build_lean_initializer(bundle),
        mutate_candidate=build_lean_mutator(bundle),
        evaluator=evaluator,
        output_dir=output_path,
        selection_spec=resolved_selection,
        layer_counter=count_lean_layers,
        candidate_serializer=serialize_lean_candidate,
        candidate_deserializer=deserialize_lean_candidate,
        artifact_identity=_build_lean_artifact_identity(bundle),
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
            lean_config=resolved_lean_config,
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

    return LeanEvolutionResult(
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


def _apply_layers(
    rows: Sequence[int],
    layers: Sequence[LeanLayer],
    *,
    bit_length: int,
) -> tuple[int, ...]:
    return tuple(_apply_row_layers(row, layers, bit_length=bit_length) for row in rows)


def _apply_row_layers(
    row: int,
    layers: Sequence[LeanLayer],
    *,
    bit_length: int,
) -> int:
    result = row
    for layer in layers:
        result = _apply_layer(result, layer, bit_length=bit_length)
    return result


def _apply_layer(row: int, layer: LeanLayer, *, bit_length: int) -> int:
    mask_all = (1 << bit_length) - 1
    shifted = _rotate_left(row, layer.shift, bit_length=bit_length)
    if layer.op == "not":
        return (~shifted) & mask_all
    if layer.mask is None:
        raise ValueError(f"Layer `{layer.op}` requires a mask.")
    mask = layer.mask & mask_all
    if layer.op == "xor":
        return shifted ^ mask
    if layer.op == "and":
        return shifted & mask
    if layer.op == "or":
        return shifted | mask
    raise ValueError(f"Unsupported lean op `{layer.op}`.")


def _rotate_left(value: int, shift: int, *, bit_length: int) -> int:
    if bit_length < 1:
        raise ValueError("`bit_length` must be greater than or equal to 1.")
    if shift == 0:
        return value & ((1 << bit_length) - 1)
    amount = shift % bit_length
    mask_all = (1 << bit_length) - 1
    return ((value << amount) | (value >> (bit_length - amount))) & mask_all


def _select_prototypes(
    train_bits: Sequence[int],
    train_labels: Sequence[int],
    class_labels: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prototypes: list[int] = []
    prototype_labels: list[int] = []
    for class_label in class_labels:
        class_rows = [
            row
            for row, label in zip(train_bits, train_labels, strict=True)
            if label == class_label
        ]
        if not class_rows:
            raise ValueError(
                f"Train split produced no rows for class `{class_label}`."
            )
        prototypes.append(_select_medoid(class_rows))
        prototype_labels.append(class_label)
    return tuple(prototypes), tuple(prototype_labels)


def _select_medoid(rows: Sequence[int]) -> int:
    if len(rows) == 1:
        return rows[0]
    best_row = rows[0]
    best_cost: int | None = None
    for candidate in rows:
        cost = sum(_hamming_distance(candidate, other) for other in rows)
        if best_cost is None or cost < best_cost or (cost == best_cost and candidate < best_row):
            best_row = candidate
            best_cost = cost
    return best_row


def _evaluate_split(
    source_rows: Sequence[int],
    labels: Sequence[int],
    layers: Sequence[LeanLayer],
    prototypes: Sequence[int],
    prototype_labels: Sequence[int],
    *,
    bit_length: int,
) -> LeanSplitMetrics:
    final_rows = _apply_layers(source_rows, layers, bit_length=bit_length)
    predictions: list[int] = []
    margins: list[int] = []
    for row in final_rows:
        predicted_label, margin = _predict_row(
            row,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
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
    return LeanSplitMetrics(
        n_rows=len(labels),
        accuracy=accuracy,
        macro_f1=macro_f1,
        mean_margin=mean_margin,
    )


def _predict_row(
    row: int,
    *,
    prototypes: Sequence[int],
    prototype_labels: Sequence[int],
) -> tuple[int, int]:
    ranked = sorted(
        (
            (_hamming_distance(row, prototype), prototype_labels[index])
            for index, prototype in enumerate(prototypes)
        ),
        key=lambda item: (item[0], item[1]),
    )
    best_distance, best_label = ranked[0]
    margin = 0 if len(ranked) < 2 else ranked[1][0] - best_distance
    return best_label, margin


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


def _identity_layer() -> LeanLayer:
    return LeanLayer(op="xor", shift=0, mask=0)


def _random_layer(bit_length: int, rng: random.Random) -> LeanLayer:
    op = ("xor", "and", "or", "not")[rng.randrange(4)]
    shift = rng.randrange(bit_length)
    if op == "not":
        return LeanLayer(op=op, shift=shift)
    return LeanLayer(op=op, shift=shift, mask=_random_mask(bit_length, rng))


def _mutate_layer(layer: LeanLayer, *, bit_length: int, rng: random.Random) -> LeanLayer:
    choice = rng.randrange(4)
    if choice == 0:
        return _random_layer(bit_length, rng)
    if choice == 1:
        if layer.op == "not":
            return LeanLayer(op="not", shift=rng.randrange(bit_length))
        return LeanLayer(
            op=layer.op,
            shift=rng.randrange(bit_length),
            mask=layer.mask,
        )
    if choice == 2 and layer.op != "not" and layer.mask is not None:
        flip_count = max(1, bit_length // 4)
        return LeanLayer(
            op=layer.op,
            shift=layer.shift,
            mask=_flip_mask_bits(layer.mask, bit_length=bit_length, flip_count=flip_count, rng=rng),
        )
    if layer.op == "not":
        return LeanLayer(op="xor", shift=layer.shift, mask=_random_mask(bit_length, rng))
    return LeanLayer(op="not", shift=layer.shift)


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
    bundle: LeanBundle,
    state: LeanState,
    lean_config: LeanTrainingConfig,
    backend_summary: LeanBackendSummary,
    best_candidate: EvaluatedCandidate[LeanCandidate],
    evolution_config: EvolutionConfig,
    selection_spec: SelectionSpec,
) -> dict[str, object]:
    return {
        "schema_version": "3.0-lean-proof-1",
        "mode": "lean",
        "scoring_mode": "final_layer_only",
        "inputs": {
            "bundle_dir": str(bundle.bundle_dir.resolve()),
        },
        "bit_length": bundle.bit_length,
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
            "lean": {
                "backend": lean_config.backend,
                "allow_backend_fallback": lean_config.allow_backend_fallback,
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
            "prototypes": [
                _int_to_bitstring(prototype, bit_length=bundle.bit_length)
                for prototype in state.prototypes
            ],
            "prototype_labels": list(state.prototype_labels),
        },
    }


def _build_metrics_summary_payload(
    *,
    state: LeanState,
    best_candidate: EvaluatedCandidate[LeanCandidate],
    evolution_result: EvolutionRunResult[LeanCandidate],
    artifact_path: Path,
    backend_summary: LeanBackendSummary,
) -> dict[str, object]:
    return {
        "mode": "lean",
        "scoring_mode": "final_layer_only",
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
    "LeanBackendSummary",
    "LeanBundle",
    "LeanCandidate",
    "LeanEvaluator",
    "LeanEvolutionResult",
    "LeanLayer",
    "LeanSplitMetrics",
    "LeanState",
    "build_lean_initializer",
    "build_lean_mutator",
    "count_lean_layers",
    "deserialize_lean_candidate",
    "load_lean_bundle",
    "mutate_lean_candidate",
    "run_lean_evolution",
    "serialize_lean_candidate",
]
