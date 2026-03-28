"""CUDA-backed helpers for the lean packed-bit evaluator.

This module keeps the GPU path narrow and honest:
- only packed 64-bit lean bundles are supported today
- rows stay packed on device as two little-endian 32-bit words
- no byte-per-bit or bool-matrix expansion is used in the hot path
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import shutil
import subprocess
from types import MappingProxyType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from bittrace.core.config import LeanTrainingConfig


try:
    import torch
except ModuleNotFoundError:
    torch = None


_PACKED_ROW_FORMAT = "packed_int_lsb0"
_SUPPORTED_BIT_LENGTH = 64
_MASK32 = (1 << 32) - 1
_POPCOUNT_MASK_1 = 0x55555555
_POPCOUNT_MASK_2 = 0x33333333
_POPCOUNT_MASK_4 = 0x0F0F0F0F
_POPCOUNT_MULTIPLIER = 0x01010101
_PAIRWISE_TARGET_ELEMENTS = 12_000_000


@dataclass(frozen=True, slots=True)
class LeanBackendSummary:
    """Resolved lean execution backend with honest packed-path reporting."""

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
    """GPU-produced lean materialization normalized back into Python scalars."""

    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
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


def resolve_lean_backend(
    *,
    row_format: str,
    bit_length: int,
    requested_backend: str,
    allow_backend_fallback: bool,
) -> LeanBackendSummary:
    """Resolve one lean runtime backend against actual local support."""

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
            "the lean GPU path currently requires `contract.row_format = packed_int_lsb0`"
        )
    if bit_length != _SUPPORTED_BIT_LENGTH:
        support_reasons.append(
            "the lean GPU path currently supports only packed 64-bit lean bundles"
        )
    if not torch_installed:
        support_reasons.append("`torch` is not installed")
    elif not torch_cuda_available:
        support_reasons.append("`torch.cuda.is_available()` is false")

    gpu_path_supported = not support_reasons
    packed_layout = (
        "CPU keeps one packed Python int per row. GPU uploads each 64-bit packed row as two "
        "little-endian 32-bit words because torch 2.10 lacks the needed UInt64 shift kernels; "
        "the hot path stays word-packed and never expands to one byte or bool per bit."
    )
    gpu_stages = {
        "final_layer_application": "gpu",
        "distance_hamming": "gpu",
        "medoid_prototype_scoring": "gpu",
        "final_prediction_scoring": "gpu",
        "selection_evolution_loop": "cpu",
    }
    cpu_stages = {
        "final_layer_application": "cpu",
        "distance_hamming": "cpu",
        "medoid_prototype_scoring": "cpu",
        "final_prediction_scoring": "cpu",
        "selection_evolution_loop": "cpu",
    }
    unsupported_note = "; ".join(support_reasons) if support_reasons else ""

    if requested_backend == "gpu":
        if gpu_path_supported:
            return LeanBackendSummary(
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
                note="CUDA-backed packed lean evaluator selected explicitly.",
            )
        if not allow_backend_fallback:
            raise ValueError(
                "`training.lean.backend: gpu` was requested, but the lean GPU path is not "
                f"available: {unsupported_note}."
            )
        note = (
            "GPU was requested for lean, but the runtime downgraded honestly to CPU because "
            f"{unsupported_note}."
        )
        return LeanBackendSummary(
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
        return LeanBackendSummary(
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
            note="Auto selected the CUDA-backed packed lean evaluator.",
        )

    if requested_backend == "auto":
        note = (
            "Auto selected CPU because the lean GPU path is unavailable: "
            f"{unsupported_note}."
        )
    else:
        note = "CPU backend selected explicitly for the lean evaluator."
    return LeanBackendSummary(
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


class PackedGpuLeanBackend:
    """GPU-backed lean evaluator for packed 64-bit rows."""

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
        if bit_length != _SUPPORTED_BIT_LENGTH:
            raise RuntimeError(
                f"GPU backend supports only {_SUPPORTED_BIT_LENGTH}-bit lean bundles; "
                f"received {bit_length}."
            )
        self._torch = torch
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

    def materialize(
        self,
        *,
        candidate_layers: Sequence[object],
        include_test_metrics: bool,
    ) -> GpuMaterialization:
        train_rows = self._apply_layers(self._train_rows, candidate_layers)
        val_rows = self._apply_layers(self._val_rows, candidate_layers)
        test_rows = self._apply_layers(self._test_rows, candidate_layers)
        prototypes, prototype_labels = self._select_prototypes(train_rows=train_rows)
        split_metrics: dict[str, Mapping[str, float | int]] = {
            "train": self._evaluate_split(
                rows=train_rows,
                labels=self._train_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            ),
            "val": self._evaluate_split(
                rows=val_rows,
                labels=self._val_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            ),
        }
        if include_test_metrics:
            split_metrics["test"] = self._evaluate_split(
                rows=test_rows,
                labels=self._test_labels,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
            )
        return GpuMaterialization(
            prototypes=tuple(self._row_words_to_int(words) for words in prototypes.to("cpu").tolist()),
            prototype_labels=tuple(int(label) for label in prototype_labels),
            split_metrics=MappingProxyType(dict(split_metrics)),
        )

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(device=self._device)

    def _rows_to_word_tensor(self, rows: Sequence[int]) -> Any:
        payload = [
            [row & self._mask32, (row >> 32) & self._mask32]
            for row in rows
        ]
        return self._torch.tensor(payload, dtype=self._torch.int64, device=self._device)

    def _apply_layers(self, rows: Any, layers: Sequence[object]) -> Any:
        current = rows
        for layer in layers:
            current = self._apply_layer(current, layer)
        return current

    def _apply_layer(self, rows: Any, layer: object) -> Any:
        rotated = self._rotate_left_64(rows, int(getattr(layer, "shift")))
        op = str(getattr(layer, "op"))
        if op == "not":
            return self._torch.stack(
                (((~rotated[:, 0]) & self._mask32), ((~rotated[:, 1]) & self._mask32)),
                dim=1,
            )
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
        raise ValueError(f"Unsupported lean op `{op}`.")

    def _mask_to_words(self, mask: int) -> Any:
        return self._torch.tensor(
            [[mask & self._mask32, (mask >> 32) & self._mask32]],
            dtype=self._torch.int64,
            device=self._device,
        )

    def _rotate_left_64(self, rows: Any, shift: int) -> Any:
        amount = shift % 64
        if amount == 0:
            return rows
        low = rows[:, 0]
        high = rows[:, 1]
        if amount == 32:
            return self._torch.stack((high, low), dim=1)
        if amount < 32:
            new_low = (((low << amount) & self._mask32) | (high >> (32 - amount))) & self._mask32
            new_high = (((high << amount) & self._mask32) | (low >> (32 - amount))) & self._mask32
            return self._torch.stack((new_low, new_high), dim=1)

        amount -= 32
        new_low = (((high << amount) & self._mask32) | (low >> (32 - amount))) & self._mask32
        new_high = (((low << amount) & self._mask32) | (high >> (32 - amount))) & self._mask32
        return self._torch.stack((new_low, new_high), dim=1)

    def _select_prototypes(self, *, train_rows: Any) -> tuple[Any, tuple[int, ...]]:
        prototypes = []
        prototype_labels: list[int] = []
        for class_label in self._class_labels:
            class_rows = train_rows.index_select(0, self._class_train_indices[class_label])
            if class_rows.shape[0] < 1:
                raise ValueError(f"Train split produced no rows for class `{class_label}`.")
            prototypes.append(self._select_class_medoid(class_rows))
            prototype_labels.append(class_label)
        return self._torch.cat(prototypes, dim=0), tuple(prototype_labels)

    def _select_class_medoid(self, rows: Any) -> Any:
        if int(rows.shape[0]) == 1:
            return rows
        costs = self._total_distance_by_row(rows)
        row_words = rows.to("cpu").tolist()
        row_ints = [self._row_words_to_int(words) for words in row_words]
        cost_values = costs.to("cpu").tolist()
        best_index = min(
            range(len(cost_values)),
            key=lambda index: (int(cost_values[index]), row_ints[index], index),
        )
        return rows.index_select(
            0,
            self._torch.tensor([best_index], dtype=self._torch.long, device=self._device),
        )

    def _total_distance_by_row(self, rows: Any) -> Any:
        row_count = int(rows.shape[0])
        costs = self._torch.empty(row_count, dtype=self._torch.int64, device=self._device)
        if row_count == 0:
            return costs
        words_per_row = max(1, int(rows.shape[1]))
        chunk_rows = max(
            32,
            min(
                row_count,
                max(32, _PAIRWISE_TARGET_ELEMENTS // max(1, row_count * words_per_row)),
            ),
        )
        for start in range(0, row_count, chunk_rows):
            stop = min(row_count, start + chunk_rows)
            xor_words = rows[start:stop].unsqueeze(1) ^ rows.unsqueeze(0)
            distances = self._popcount32(xor_words).sum(dim=2)
            costs[start:stop] = distances.sum(dim=1).to(self._torch.int64)
        return costs

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

    def _evaluate_split(
        self,
        *,
        rows: Any,
        labels: Sequence[int],
        prototypes: Any,
        prototype_labels: Sequence[int],
    ) -> Mapping[str, float | int]:
        if prototypes.shape[0] < 1:
            raise ValueError("Lean prediction requires at least one prototype.")
        distance_table = self._distance_matrix(rows, prototypes)
        prediction_positions = distance_table.argmin(dim=1).to("cpu").tolist()
        sorted_distances = self._torch.sort(distance_table, dim=1).values[:, :2].to("cpu").tolist()
        predictions = [int(prototype_labels[int(position)]) for position in prediction_positions]
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


__all__ = [
    "GpuMaterialization",
    "LeanBackendSummary",
    "PackedGpuLeanBackend",
    "detect_gpu_visibility",
    "resolve_lean_backend",
]
