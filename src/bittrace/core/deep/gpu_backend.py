"""CUDA-backed helpers for the deep packed-bit evaluator.

This module keeps the GPU path narrow and honest:
- only packed 64-bit, 128-bit, and 256-bit deep bundles are supported today
- rows stay packed on device as little-endian 32-bit words
- no byte-per-bit or bool-matrix expansion is used in the hot path
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from bittrace.core.config import DeepTrainingConfig


try:
    import torch
except ModuleNotFoundError:
    torch = None


_PACKED_ROW_FORMAT = "packed_int_lsb0"
_SUPPORTED_BIT_LENGTHS = frozenset({64, 128, 256})
_GPU_WORD_BITS = 32
_MASK32 = (1 << 32) - 1
_POPCOUNT_MASK_1 = 0x55555555
_POPCOUNT_MASK_2 = 0x33333333
_POPCOUNT_MASK_4 = 0x0F0F0F0F
_POPCOUNT_MULTIPLIER = 0x01010101
_MAX_EXACT_MEDOID_COMBINATIONS = 4096
_PAIRWISE_TARGET_ELEMENTS = 12_000_000


@dataclass(frozen=True, slots=True)
class DeepBackendSummary:
    """Resolved deep execution backend with honest packed-path reporting."""

    backend_requested: str
    backend_actual: str
    allow_backend_fallback: bool
    gpu_visible: bool
    gpu_visibility_details: str
    gpu_path_supported: bool
    gpu_actually_used: bool
    torch_installed: bool
    torch_version: str | None
    torch_cuda_available: bool
    cuda_device_name: str | None
    stage_backends: Mapping[str, str]
    packed_layout: str
    note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_backends", MappingProxyType(dict(self.stage_backends)))

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_requested": self.backend_requested,
            "backend_actual": self.backend_actual,
            "allow_backend_fallback": self.allow_backend_fallback,
            "gpu_visible": self.gpu_visible,
            "gpu_visibility_details": self.gpu_visibility_details,
            "gpu_path_supported": self.gpu_path_supported,
            "gpu_actually_used": self.gpu_actually_used,
            "torch_installed": self.torch_installed,
            "torch_version": self.torch_version,
            "torch_cuda_available": self.torch_cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "stage_backends": dict(self.stage_backends),
            "packed_layout": self.packed_layout,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class GpuMaterialization:
    """GPU-produced deep materialization normalized back into Python scalars."""

    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
    k_per_class: Mapping[int, int]
    embedding_bit_length: int
    split_metrics: Mapping[str, Mapping[str, float | int]]


def detect_gpu_visibility() -> tuple[bool, str]:
    """Best-effort local GPU visibility probe; visibility is not use."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False, "`nvidia-smi` not found."
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError as exc:
        return False, f"`nvidia-smi` probe failed: {exc}"
    except subprocess.TimeoutExpired:
        return False, "`nvidia-smi -L` timed out."

    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        detail = output if output else f"`nvidia-smi -L` exited with {result.returncode}."
        return False, detail
    if not output:
        return False, "`nvidia-smi -L` reported no devices."
    return True, output


def resolve_deep_backend(
    *,
    row_format: str,
    bit_length: int,
    requested_backend: str,
    allow_backend_fallback: bool,
) -> DeepBackendSummary:
    """Resolve one deep runtime backend against actual local support."""

    gpu_visible, gpu_visibility_details = detect_gpu_visibility()
    torch_installed = torch is not None
    torch_version = None if torch is None else str(torch.__version__)
    torch_cuda_available = False if torch is None else bool(torch.cuda.is_available())
    cuda_device_name = (
        None
        if torch is None or not torch_cuda_available
        else str(torch.cuda.get_device_name(0))
    )
    support_reasons: list[str] = []
    if row_format != _PACKED_ROW_FORMAT:
        support_reasons.append(
            "the deep GPU path currently requires `contract.row_format = packed_int_lsb0`"
        )
    if bit_length not in _SUPPORTED_BIT_LENGTHS:
        support_reasons.append(
            "the deep GPU path currently supports only packed 64-bit, 128-bit, "
            "and 256-bit deep bundles"
        )
    if not torch_installed:
        support_reasons.append("`torch` is not installed")
    elif not torch_cuda_available:
        support_reasons.append("`torch.cuda.is_available()` is false")

    gpu_path_supported = not support_reasons
    word_count = max(1, bit_length // _GPU_WORD_BITS)
    packed_layout = (
        f"CPU keeps one packed Python int per row. GPU uploads each {bit_length}-bit packed "
        f"row as {word_count} little-endian 32-bit words because torch 2.10 lacks the needed "
        "UInt64 shift kernels; the hot path stays word-packed and never expands to one byte "
        "or bool per bit."
    )
    gpu_stages = {
        "layer_application": "gpu",
        "residue_embedding_readout": "gpu",
        "distance_hamming": "gpu",
        "medoid_prototype_scoring": "gpu",
        "selection_evolution_loop": "cpu",
    }
    cpu_stages = {
        "layer_application": "cpu",
        "residue_embedding_readout": "cpu",
        "distance_hamming": "cpu",
        "medoid_prototype_scoring": "cpu",
        "selection_evolution_loop": "cpu",
    }
    unsupported_note = "; ".join(support_reasons) if support_reasons else ""

    if requested_backend == "gpu":
        if gpu_path_supported:
            note = "CUDA-backed packed deep evaluator selected explicitly."
            return DeepBackendSummary(
                backend_requested=requested_backend,
                backend_actual="gpu",
                allow_backend_fallback=allow_backend_fallback,
                gpu_visible=gpu_visible,
                gpu_visibility_details=gpu_visibility_details,
                gpu_path_supported=True,
                gpu_actually_used=True,
                torch_installed=torch_installed,
                torch_version=torch_version,
                torch_cuda_available=torch_cuda_available,
                cuda_device_name=cuda_device_name,
                stage_backends=gpu_stages,
                packed_layout=packed_layout,
                note=note,
            )
        if not allow_backend_fallback:
            raise ValueError(
                "`training.deep.backend: gpu` was requested, but the deep GPU path is not "
                f"available: {unsupported_note}."
            )
        note = (
            "GPU was requested for deep, but the runtime downgraded honestly to CPU because "
            f"{unsupported_note}."
        )
        return DeepBackendSummary(
            backend_requested=requested_backend,
            backend_actual="cpu",
            allow_backend_fallback=allow_backend_fallback,
            gpu_visible=gpu_visible,
            gpu_visibility_details=gpu_visibility_details,
            gpu_path_supported=False,
            gpu_actually_used=False,
            torch_installed=torch_installed,
            torch_version=torch_version,
            torch_cuda_available=torch_cuda_available,
            cuda_device_name=cuda_device_name,
            stage_backends=cpu_stages,
            packed_layout=packed_layout,
            note=note,
        )

    if requested_backend == "auto" and gpu_path_supported:
        note = "Auto selected the CUDA-backed packed deep evaluator."
        return DeepBackendSummary(
            backend_requested=requested_backend,
            backend_actual="gpu",
            allow_backend_fallback=allow_backend_fallback,
            gpu_visible=gpu_visible,
            gpu_visibility_details=gpu_visibility_details,
            gpu_path_supported=True,
            gpu_actually_used=True,
            torch_installed=torch_installed,
            torch_version=torch_version,
            torch_cuda_available=torch_cuda_available,
            cuda_device_name=cuda_device_name,
            stage_backends=gpu_stages,
            packed_layout=packed_layout,
            note=note,
        )

    if requested_backend == "auto":
        note = (
            "Auto selected CPU because the deep GPU path is unavailable: "
            f"{unsupported_note}."
        )
    else:
        note = "CPU backend selected explicitly for the deep evaluator."
    return DeepBackendSummary(
        backend_requested=requested_backend,
        backend_actual="cpu",
        allow_backend_fallback=allow_backend_fallback,
        gpu_visible=gpu_visible,
        gpu_visibility_details=gpu_visibility_details,
        gpu_path_supported=gpu_path_supported,
        gpu_actually_used=False,
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        cuda_device_name=cuda_device_name,
        stage_backends=cpu_stages,
        packed_layout=packed_layout,
        note=note,
    )


class PackedGpuDeepBackend:
    """GPU-backed deep evaluator for packed 64-bit, 128-bit, and 256-bit rows."""

    def __init__(
        self,
        *,
        bit_length: int,
        train_rows: Sequence[int],
        train_labels: Sequence[int],
        val_rows: Sequence[int],
        val_labels: Sequence[int],
        test_rows: Sequence[int],
        test_labels: Sequence[int],
        class_labels: Sequence[int],
    ) -> None:
        if torch is None:
            raise RuntimeError("GPU backend requested without torch installed.")
        if bit_length not in _SUPPORTED_BIT_LENGTHS:
            raise RuntimeError(
                "GPU backend supports only packed 64-bit, 128-bit, and 256-bit deep bundles; "
                f"received {bit_length}."
            )
        if bit_length % _GPU_WORD_BITS != 0:
            raise RuntimeError(
                "GPU backend requires bit lengths that are exact multiples of 32 bits; "
                f"received {bit_length}."
            )
        self._torch = torch
        self._bit_length = bit_length
        self._word_count = bit_length // _GPU_WORD_BITS
        self._device = torch.device("cuda")
        self._mask32 = _MASK32
        self._train_rows = self._rows_to_word_tensor(train_rows)
        self._val_rows = self._rows_to_word_tensor(val_rows)
        self._test_rows = self._rows_to_word_tensor(test_rows)
        self._train_labels = tuple(int(label) for label in train_labels)
        self._val_labels = tuple(int(label) for label in val_labels)
        self._test_labels = tuple(int(label) for label in test_labels)
        self._class_labels = tuple(int(label) for label in class_labels)
        self._class_train_indices = {
            class_label: self._torch.tensor(
                [index for index, label in enumerate(self._train_labels) if label == class_label],
                dtype=self._torch.long,
                device=self._device,
            )
            for class_label in self._class_labels
        }
        self._class_val_indices = {
            class_label: self._torch.tensor(
                [index for index, label in enumerate(self._val_labels) if label == class_label],
                dtype=self._torch.long,
                device=self._device,
            )
            for class_label in self._class_labels
        }

    def materialize(
        self,
        *,
        candidate_layers: Sequence[object],
        deep_config: DeepTrainingConfig,
        include_test_metrics: bool,
    ) -> GpuMaterialization:
        train_embeddings = self._apply_layers_as_embedding(self._train_rows, candidate_layers)
        val_embeddings = self._apply_layers_as_embedding(self._val_rows, candidate_layers)
        test_embeddings = self._apply_layers_as_embedding(self._test_rows, candidate_layers)
        prototypes, prototype_labels, k_per_class = self._select_prototypes(
            train_embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            deep_config=deep_config,
        )
        split_metrics: dict[str, Mapping[str, float | int]] = {
            "train": self._evaluate_split(
                embeddings=train_embeddings,
                labels=self._train_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            ),
            "val": self._evaluate_split(
                embeddings=val_embeddings,
                labels=self._val_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            ),
        }
        if include_test_metrics:
            split_metrics["test"] = self._evaluate_split(
                embeddings=test_embeddings,
                labels=self._test_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            )

        return GpuMaterialization(
            prototypes=tuple(self._row_words_to_int(words) for words in prototypes.to("cpu").tolist()),
            prototype_labels=tuple(int(label) for label in prototype_labels),
            k_per_class=MappingProxyType(dict(k_per_class)),
            embedding_bit_length=self._bit_length * len(candidate_layers),
            split_metrics=MappingProxyType(dict(split_metrics)),
        )

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(device=self._device)

    def _rows_to_word_tensor(self, rows: Sequence[int]) -> Any:
        payload = [
            [
                (int(row) >> (_GPU_WORD_BITS * word_index)) & self._mask32
                for word_index in range(self._word_count)
            ]
            for row in rows
        ]
        return self._torch.tensor(payload, dtype=self._torch.int64, device=self._device)

    def _apply_layers_as_embedding(self, rows: Any, layers: Sequence[object]) -> Any:
        current = rows
        outputs = []
        for layer in layers:
            current = self._apply_layer(current, layer)
            outputs.append(current)
        return self._torch.cat(outputs, dim=1)

    def _apply_layer(self, rows: Any, layer: object) -> Any:
        rotated = self._rotate_left_words(rows, int(getattr(layer, "shift")))
        op = str(getattr(layer, "op"))
        if op == "not":
            return (~rotated) & self._mask32
        if op == "rule3":
            rule = getattr(layer, "rule")
            if rule is None:
                raise ValueError("`rule3` layers require a rule value.")
            return self._apply_rule3(rotated, int(rule))

        mask = getattr(layer, "mask")
        if mask is None:
            raise ValueError(f"Layer `{op}` requires a mask.")
        mask_words = self._mask_to_words(int(mask))
        if op == "xor":
            return rotated ^ mask_words
        if op == "and":
            return rotated & mask_words
        if op == "or":
            return rotated | mask_words
        if op == "nand":
            return (~(rotated & mask_words)) & self._mask32
        if op == "set0":
            return rotated & ((~mask_words) & self._mask32)
        if op == "set1":
            return rotated | mask_words
        raise ValueError(f"Unsupported deep op `{op}`.")

    def _mask_to_words(self, mask: int) -> Any:
        return self._torch.tensor(
            [
                [
                    (int(mask) >> (_GPU_WORD_BITS * word_index)) & self._mask32
                    for word_index in range(self._word_count)
                ]
            ],
            dtype=self._torch.int64,
            device=self._device,
        )

    def _rotate_left_words(self, rows: Any, shift: int) -> Any:
        amount = shift % self._bit_length
        if amount == 0:
            return rows
        word_shift, bit_shift = divmod(amount, _GPU_WORD_BITS)
        rotated = (
            rows
            if word_shift == 0
            else self._torch.roll(rows, shifts=word_shift, dims=1)
        )
        if bit_shift == 0:
            return rotated
        spill = self._torch.roll(rotated, shifts=1, dims=1)
        return (
            (((rotated << bit_shift) & self._mask32) | (spill >> (_GPU_WORD_BITS - bit_shift)))
            & self._mask32
        )

    def _apply_rule3(self, rows: Any, rule: int) -> Any:
        left = self._rotate_left_words(rows, 1)
        center = rows
        right = self._rotate_left_words(rows, self._bit_length - 1)
        result = self._torch.zeros_like(rows)
        for pattern in range(8):
            if ((rule >> pattern) & 1) == 0:
                continue
            result |= self._pattern_term(pattern, left, center, right)
        return result & self._mask32

    def _pattern_term(self, pattern: int, left: Any, center: Any, right: Any) -> Any:
        out = self._torch.full_like(left, self._mask32)
        out &= left if pattern & 0b100 else (~left) & self._mask32
        out &= center if pattern & 0b010 else (~center) & self._mask32
        out &= right if pattern & 0b001 else (~right) & self._mask32
        return out & self._mask32

    def _select_prototypes(
        self,
        *,
        train_embeddings: Any,
        val_embeddings: Any,
        deep_config: DeepTrainingConfig,
    ) -> tuple[Any, tuple[int, ...], Mapping[int, int]]:
        prototypes = []
        prototype_labels: list[int] = []
        k_per_class: dict[int, int] = {}
        for class_label in self._class_labels:
            class_train = train_embeddings.index_select(0, self._class_train_indices[class_label])
            if class_train.shape[0] < 1:
                raise ValueError(f"Train split produced no rows for class `{class_label}`.")
            class_val = val_embeddings.index_select(0, self._class_val_indices[class_label])
            medoids = self._select_class_medoids(
                train_rows=class_train,
                val_rows=class_val,
                base_k=deep_config.k_medoids_per_class,
                adaptive_k=deep_config.adaptive_k,
                adaptive_k_candidates=deep_config.adaptive_k_candidates,
            )
            prototypes.append(medoids)
            prototype_labels.extend([class_label] * medoids.shape[0])
            k_per_class[class_label] = int(medoids.shape[0])
        return self._torch.cat(prototypes, dim=0), tuple(prototype_labels), MappingProxyType(k_per_class)

    def _select_class_medoids(
        self,
        *,
        train_rows: Any,
        val_rows: Any,
        base_k: int,
        adaptive_k: bool,
        adaptive_k_candidates: Sequence[int],
    ) -> Any:
        unique_rows, row_ints = self._stable_unique_rows(train_rows)
        if unique_rows.shape[0] < 1:
            raise ValueError("Cannot compute medoids on an empty class.")

        candidate_ks = [base_k]
        if adaptive_k:
            candidate_ks.extend(adaptive_k_candidates)
        ordered_ks = sorted(set(int(k_value) for k_value in candidate_ks))
        distance_matrix = self._pairwise_distance_matrix(unique_rows)
        eval_rows = val_rows if val_rows.shape[0] > 0 else train_rows
        best_rows = None
        best_score = None
        best_k = None
        for k_candidate in ordered_ks:
            medoids = self._compute_class_medoids(
                rows=unique_rows,
                row_ints=row_ints,
                distance_matrix=distance_matrix,
                k_medoids=k_candidate,
            )
            score = self._min_distance_total(eval_rows, medoids)
            if (
                best_score is None
                or score < best_score
                or (score == best_score and (best_k is None or int(medoids.shape[0]) < best_k))
            ):
                best_rows = medoids
                best_score = score
                best_k = int(medoids.shape[0])
        if best_rows is None:
            raise ValueError("Failed to build class medoids.")
        return best_rows

    def _compute_class_medoids(
        self,
        *,
        rows: Any,
        row_ints: Sequence[int],
        distance_matrix: Any,
        k_medoids: int,
    ) -> Any:
        row_count = int(rows.shape[0])
        k_eff = min(max(1, int(k_medoids)), row_count)
        if k_eff == row_count:
            return rows
        if k_eff == 1:
            costs = distance_matrix.sum(dim=1).to("cpu").tolist()
            best_index = self._choose_lowest_index(costs=costs, row_ints=row_ints)
            return rows.index_select(
                0,
                self._torch.tensor([best_index], dtype=self._torch.long, device=self._device),
            )
        if comb(row_count, k_eff) <= _MAX_EXACT_MEDOID_COMBINATIONS:
            return self._compute_exact_medoids(rows=rows, row_ints=row_ints, distance_matrix=distance_matrix, k_eff=k_eff)
        return self._compute_greedy_medoids(rows=rows, row_ints=row_ints, distance_matrix=distance_matrix, k_eff=k_eff)

    def _compute_exact_medoids(
        self,
        *,
        rows: Any,
        row_ints: Sequence[int],
        distance_matrix: Any,
        k_eff: int,
    ) -> Any:
        from itertools import combinations

        distance_rows = distance_matrix.to("cpu").tolist()
        best_indices = None
        best_cost = None
        best_rows = None
        for indices in combinations(range(len(row_ints)), k_eff):
            total_cost = sum(
                min(distance_rows[row_index][medoid_index] for medoid_index in indices)
                for row_index in range(len(row_ints))
            )
            medoid_rows = tuple(row_ints[index] for index in indices)
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
        return rows.index_select(
            0,
            self._torch.tensor(best_indices, dtype=self._torch.long, device=self._device),
        )

    def _compute_greedy_medoids(
        self,
        *,
        rows: Any,
        row_ints: Sequence[int],
        distance_matrix: Any,
        k_eff: int,
    ) -> Any:
        total_costs = distance_matrix.sum(dim=1)
        total_cost_values = total_costs.to("cpu").tolist()
        first_index = self._choose_lowest_index(costs=total_cost_values, row_ints=row_ints)
        medoid_indices = [first_index]

        while len(medoid_indices) < k_eff:
            remaining = [index for index in range(len(row_ints)) if index not in medoid_indices]
            min_distances = distance_matrix[:, medoid_indices].min(dim=1).values.to("cpu").tolist()
            next_index = max(
                remaining,
                key=lambda index: (
                    min_distances[index],
                    -total_cost_values[index],
                    -row_ints[index],
                ),
            )
            medoid_indices.append(next_index)

        for _ in range(4):
            sorted_medoids = sorted(medoid_indices, key=lambda index: (row_ints[index], index))
            medoid_tensor = self._torch.tensor(sorted_medoids, dtype=self._torch.long, device=self._device)
            nearest_positions = distance_matrix.index_select(1, medoid_tensor).argmin(dim=1).to("cpu").tolist()
            assignments: dict[int, list[int]] = {index: [] for index in sorted_medoids}
            for row_index, position in enumerate(nearest_positions):
                assignments[sorted_medoids[int(position)]].append(row_index)

            new_indices: list[int] = []
            for medoid_index in sorted_medoids:
                members = assignments.get(medoid_index, [])
                if not members:
                    new_indices.append(medoid_index)
                    continue
                member_tensor = self._torch.tensor(members, dtype=self._torch.long, device=self._device)
                local_costs = (
                    distance_matrix
                    .index_select(0, member_tensor)
                    .index_select(1, member_tensor)
                    .sum(dim=1)
                    .to("cpu")
                    .tolist()
                )
                best_local = self._choose_lowest_index(
                    costs=local_costs,
                    row_ints=[row_ints[index] for index in members],
                )
                selected_index = members[int(best_local)]
                if selected_index not in new_indices:
                    new_indices.append(selected_index)
            if set(new_indices) == set(medoid_indices):
                medoid_indices = new_indices
                break
            medoid_indices = self._fill_missing_medoids(
                medoid_indices=new_indices,
                target_count=k_eff,
                total_costs=total_cost_values,
                row_ints=row_ints,
            )

        medoid_indices = sorted(medoid_indices, key=lambda index: (row_ints[index], index))
        return rows.index_select(
            0,
            self._torch.tensor(medoid_indices, dtype=self._torch.long, device=self._device),
        )

    def _fill_missing_medoids(
        self,
        *,
        medoid_indices: Sequence[int],
        target_count: int,
        total_costs: Sequence[int | float],
        row_ints: Sequence[int],
    ) -> list[int]:
        filled = list(dict.fromkeys(int(index) for index in medoid_indices))
        if len(filled) >= target_count:
            return filled[:target_count]
        remaining = [index for index in range(len(row_ints)) if index not in filled]
        remaining.sort(key=lambda index: (total_costs[index], row_ints[index], index))
        for index in remaining:
            filled.append(index)
            if len(filled) == target_count:
                break
        return filled

    def _choose_lowest_index(self, *, costs: Sequence[int | float], row_ints: Sequence[int]) -> int:
        return min(
            range(len(costs)),
            key=lambda index: (costs[index], row_ints[index], index),
        )

    def _stable_unique_rows(self, rows: Any) -> tuple[Any, tuple[int, ...]]:
        unique_indices: list[int] = []
        unique_ints: list[int] = []
        seen: set[tuple[int, ...]] = set()
        for index, words in enumerate(rows.to("cpu").tolist()):
            key = tuple(int(word) for word in words)
            if key in seen:
                continue
            seen.add(key)
            unique_indices.append(index)
            unique_ints.append(self._row_words_to_int(key))
        return (
            rows.index_select(
                0,
                self._torch.tensor(unique_indices, dtype=self._torch.long, device=self._device),
            ),
            tuple(unique_ints),
        )

    def _pairwise_distance_matrix(self, rows: Any) -> Any:
        return self._distance_matrix(rows, rows)

    def _distance_matrix(self, left_rows: Any, right_rows: Any) -> Any:
        result = self._torch.empty(
            (left_rows.shape[0], right_rows.shape[0]),
            dtype=self._torch.int16,
            device=self._device,
        )
        if left_rows.shape[0] == 0 or right_rows.shape[0] == 0:
            return result
        words_per_row = max(1, int(left_rows.shape[1]))
        chunk_rows = max(
            32,
            min(
                int(left_rows.shape[0]),
                max(32, _PAIRWISE_TARGET_ELEMENTS // max(1, int(right_rows.shape[0]) * words_per_row)),
            ),
        )
        for start in range(0, int(left_rows.shape[0]), chunk_rows):
            stop = min(int(left_rows.shape[0]), start + chunk_rows)
            xor_words = left_rows[start:stop].unsqueeze(1) ^ right_rows.unsqueeze(0)
            distances = self._popcount32(xor_words).sum(dim=2)
            result[start:stop] = distances.to(self._torch.int16)
        return result

    def _popcount32(self, values: Any) -> Any:
        values = values & self._mask32
        values = values - ((values >> 1) & _POPCOUNT_MASK_1)
        values = (values & _POPCOUNT_MASK_2) + ((values >> 2) & _POPCOUNT_MASK_2)
        values = (values + (values >> 4)) & _POPCOUNT_MASK_4
        return ((values * _POPCOUNT_MULTIPLIER) >> 24) & 0xFF

    def _min_distance_total(self, rows: Any, medoids: Any) -> int:
        if rows.shape[0] == 0:
            return 0
        return int(self._distance_matrix(rows, medoids).min(dim=1).values.sum().item())

    def _evaluate_split(
        self,
        *,
        embeddings: Any,
        labels: Sequence[int],
        prototypes: Any,
        prototype_labels: Sequence[int],
    ) -> Mapping[str, float | int]:
        class_distances = []
        for class_label in self._class_labels:
            indices = [
                index
                for index, label in enumerate(prototype_labels)
                if int(label) == class_label
            ]
            if not indices:
                raise ValueError(f"No prototypes were built for class `{class_label}`.")
            prototype_tensor = self._torch.tensor(indices, dtype=self._torch.long, device=self._device)
            per_class = self._distance_matrix(
                embeddings,
                prototypes.index_select(0, prototype_tensor),
            ).min(dim=1).values
            class_distances.append(per_class)

        distance_table = self._torch.stack(class_distances, dim=1)
        prediction_positions = distance_table.argmin(dim=1).to("cpu").tolist()
        sorted_distances = self._torch.sort(distance_table, dim=1).values.to("cpu").tolist()
        predictions = [self._class_labels[int(position)] for position in prediction_positions]
        margins = [
            0 if len(row) < 2 else int(row[1]) - int(row[0])
            for row in sorted_distances
        ]
        accuracy = self._compute_accuracy(labels, predictions)
        macro_f1 = self._compute_macro_f1(labels, predictions)
        mean_margin = (
            float(sum(margins)) / float(len(margins))
            if margins
            else 0.0
        )
        return {
            "n_rows": len(labels),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mean_margin": mean_margin,
        }

    def _compute_accuracy(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        if not y_true:
            return 0.0
        matches = sum(
            1
            for expected, predicted in zip(y_true, y_pred, strict=True)
            if expected == predicted
        )
        return float(matches) / float(len(y_true))

    def _compute_macro_f1(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        labels = sorted(set(y_true).union(y_pred))
        if not labels:
            return 0.0
        f1_values = []
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
            else:
                f1_values.append((2.0 * precision * recall) / (precision + recall))
        return sum(f1_values) / float(len(f1_values))

    def _row_words_to_int(self, words: Sequence[int]) -> int:
        value = 0
        for word_index, word in enumerate(words):
            value |= int(word) << (32 * word_index)
        return value
