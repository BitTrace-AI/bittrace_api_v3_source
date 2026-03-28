"""Pure-Python deterministic PCA fit/transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

from bittrace.core.frontends.types import FeatureTable, FrontendError


_EIGENVALUE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PCAProjection:
    """Fitted PCA projection that can transform same-schema feature tables."""

    input_feature_names: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    zero_std_feature_count: int

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Project a same-schema feature table into PCA component space."""

        if table.feature_names != self.input_feature_names:
            raise FrontendError(
                "PCA frontend schema mismatch. Expected feature names "
                f"{self.input_feature_names!r}, got {table.feature_names!r}."
            )

        projected_rows = []
        for row in table.rows:
            normalized = [
                (row[index] - self.means[index]) / self.scales[index]
                for index in range(len(self.input_feature_names))
            ]
            projected_rows.append(
                [
                    math.fsum(
                        normalized[feature_index] * component[feature_index]
                        for feature_index in range(len(component))
                    )
                    for component in self.components
                ]
            )

        return FeatureTable.from_rows(
            projected_rows,
            feature_names=self.output_feature_names,
            labels=table.labels,
        )

    def to_artifact(self, *, n_rows: int, n_features: int) -> dict[str, object]:
        """Render a serializable PCA fit summary."""

        cumulative: list[float] = []
        running = 0.0
        for value in self.explained_variance_ratio:
            running += value
            cumulative.append(running)

        return {
            "n_rows": n_rows,
            "n_features": n_features,
            "n_components": len(self.components),
            "component_selection": "positive_eigenvalues_or_1",
            "standardize": "zscore",
            "zero_std_feature_count": self.zero_std_feature_count,
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "explained_variance_ratio_cumulative": cumulative,
            "components": [list(component) for component in self.components],
            "projected_feature_names": list(self.output_feature_names),
        }


def fit_pca(table: FeatureTable) -> PCAProjection:
    """Fit a deterministic PCA projection on a validated feature table."""

    n_rows, n_features = table.shape
    means = tuple(
        math.fsum(row[index] for row in table.rows) / float(n_rows)
        for index in range(n_features)
    )

    centered = [
        [row[index] - means[index] for index in range(n_features)]
        for row in table.rows
    ]
    raw_scales = []
    zero_std_count = 0
    for index in range(n_features):
        variance = math.fsum(row[index] ** 2 for row in centered) / float(n_rows)
        if variance <= 0.0:
            raw_scales.append(1.0)
            zero_std_count += 1
            continue
        raw_scales.append(math.sqrt(variance))
    scales = tuple(raw_scales)

    normalized = [
        [row[index] / scales[index] for index in range(n_features)]
        for row in centered
    ]
    covariance = _covariance(normalized, n_features=n_features)
    eigenpairs = _sorted_eigenpairs(covariance)

    positive_components = sum(
        1 for eigenvalue, _ in eigenpairs if eigenvalue > _EIGENVALUE_TOLERANCE
    )
    component_count = positive_components if positive_components > 0 else 1
    kept_pairs = eigenpairs[:component_count]
    total_variance = math.fsum(max(eigenvalue, 0.0) for eigenvalue, _ in eigenpairs)
    explained_variance_ratio = tuple(
        (max(eigenvalue, 0.0) / total_variance) if total_variance > 0.0 else 0.0
        for eigenvalue, _ in kept_pairs
    )

    components = tuple(vector for _, vector in kept_pairs)
    output_feature_names = tuple(
        f"pc_{index}"
        for index in range(1, component_count + 1)
    )
    return PCAProjection(
        input_feature_names=table.feature_names,
        output_feature_names=output_feature_names,
        means=means,
        scales=scales,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        zero_std_feature_count=zero_std_count,
    )


def _covariance(
    rows: list[list[float]],
    *,
    n_features: int,
) -> list[list[float]]:
    if len(rows) <= 1:
        return [[0.0 for _ in range(n_features)] for _ in range(n_features)]

    factor = 1.0 / float(len(rows) - 1)
    covariance = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for left in range(n_features):
        for right in range(left, n_features):
            value = factor * math.fsum(row[left] * row[right] for row in rows)
            covariance[left][right] = value
            covariance[right][left] = value
    return covariance


def _sorted_eigenpairs(matrix: list[list[float]]) -> list[tuple[float, tuple[float, ...]]]:
    size = len(matrix)
    if size == 0:
        raise FrontendError("PCA requires at least one feature column.")
    if size == 1:
        return [(float(matrix[0][0]), (1.0,))]

    eigenvalues, eigenvectors = _jacobi_eigendecomposition(matrix)
    stabilized = [
        (float(eigenvalue), _stabilize_sign(vector))
        for eigenvalue, vector in zip(eigenvalues, eigenvectors, strict=True)
    ]
    stabilized.sort(key=lambda item: (-item[0], item[1]))
    return stabilized


def _stabilize_sign(vector: tuple[float, ...]) -> tuple[float, ...]:
    pivot_index = max(range(len(vector)), key=lambda index: (abs(vector[index]), -index))
    if vector[pivot_index] < 0.0:
        return tuple(-value for value in vector)
    return vector


def _jacobi_eigendecomposition(
    matrix: list[list[float]],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    size = len(matrix)
    working = [row[:] for row in matrix]
    eigenvectors = [
        [1.0 if row_index == column_index else 0.0 for column_index in range(size)]
        for row_index in range(size)
    ]

    max_iterations = max(16, size * size * 32)
    for _ in range(max_iterations):
        pivot_left, pivot_right, pivot_value = _largest_off_diagonal(working)
        if pivot_value <= _EIGENVALUE_TOLERANCE:
            break

        app = working[pivot_left][pivot_left]
        aqq = working[pivot_right][pivot_right]
        apq = working[pivot_left][pivot_right]
        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + (tau * tau))), tau)
        c = 1.0 / math.sqrt(1.0 + (t * t))
        s = t * c

        for index in range(size):
            if index in {pivot_left, pivot_right}:
                continue
            left_value = working[index][pivot_left]
            right_value = working[index][pivot_right]
            rotated_left = (c * left_value) - (s * right_value)
            rotated_right = (s * left_value) + (c * right_value)
            working[index][pivot_left] = rotated_left
            working[pivot_left][index] = rotated_left
            working[index][pivot_right] = rotated_right
            working[pivot_right][index] = rotated_right

        working[pivot_left][pivot_left] = (
            (c * c * app) - (2.0 * s * c * apq) + (s * s * aqq)
        )
        working[pivot_right][pivot_right] = (
            (s * s * app) + (2.0 * s * c * apq) + (c * c * aqq)
        )
        working[pivot_left][pivot_right] = 0.0
        working[pivot_right][pivot_left] = 0.0

        for index in range(size):
            left_vector = eigenvectors[index][pivot_left]
            right_vector = eigenvectors[index][pivot_right]
            eigenvectors[index][pivot_left] = (c * left_vector) - (s * right_vector)
            eigenvectors[index][pivot_right] = (s * left_vector) + (c * right_vector)

    eigenvalues = tuple(working[index][index] for index in range(size))
    components = tuple(
        tuple(eigenvectors[row_index][column_index] for row_index in range(size))
        for column_index in range(size)
    )
    return eigenvalues, components


def _largest_off_diagonal(matrix: list[list[float]]) -> tuple[int, int, float]:
    pivot_left = 0
    pivot_right = 1
    pivot_value = 0.0
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            candidate = abs(matrix[left][right])
            if candidate > pivot_value:
                pivot_left = left
                pivot_right = right
                pivot_value = candidate
    return pivot_left, pivot_right, pivot_value
