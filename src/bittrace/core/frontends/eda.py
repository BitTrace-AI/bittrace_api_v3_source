"""Deterministic EDA summaries for validated feature tables."""

from __future__ import annotations

from collections import Counter
import math

from bittrace.core.frontends.types import FeatureTable


def summarize_eda(table: FeatureTable) -> dict[str, object]:
    """Return a compact, deterministic EDA summary for a feature table."""

    n_rows, n_features = table.shape
    columns = tuple(tuple(row[index] for row in table.rows) for index in range(n_features))

    means = tuple(_mean(column) for column in columns)
    variances = tuple(_variance(column, mean=means[index]) for index, column in enumerate(columns))
    stds = tuple(math.sqrt(variance) for variance in variances)
    mins = tuple(min(column) for column in columns)
    maxs = tuple(max(column) for column in columns)
    ranked = sorted(range(n_features), key=lambda index: (-variances[index], index))

    payload: dict[str, object] = {
        "n_rows": n_rows,
        "n_features": n_features,
        "labels_present": table.labels is not None,
        "nan_count": 0,
        "inf_count": 0,
        "feature_stats": [
            {
                "feature_index": index,
                "feature_name": table.feature_names[index],
                "mean": means[index],
                "std": stds[index],
                "min": mins[index],
                "max": maxs[index],
                "variance": variances[index],
            }
            for index in range(n_features)
        ],
        "variance_rank": [
            {
                "rank": rank,
                "feature_index": index,
                "feature_name": table.feature_names[index],
                "variance": variances[index],
            }
            for rank, index in enumerate(ranked, start=1)
        ],
    }

    if table.labels is not None:
        counts = Counter(_label_token(label) for label in table.labels)
        payload["label_distribution"] = {
            label: counts[label]
            for label in sorted(counts)
        }

    return payload


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / float(len(values))


def _variance(values: tuple[float, ...], *, mean: float) -> float:
    return math.fsum((value - mean) ** 2 for value in values) / float(len(values))


def _label_token(value: object) -> str:
    return str(value)
