"""Frontend selection and fit/transform orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from bittrace.core.config import FrontendConfig, FrontendMode
from bittrace.core.frontends.eda import summarize_eda
from bittrace.core.frontends.pca import PCAProjection, fit_pca
from bittrace.core.frontends.types import FeatureTable, FrontendError, FrontendResult


_SUPPORTED_FRONTEND_MODES = frozenset({"none", "eda", "pca", "eda_pca"})


@dataclass(frozen=True, slots=True)
class FittedFeatureFrontend:
    """Resolved frontend state that can transform same-schema feature tables."""

    mode: FrontendMode
    input_feature_names: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    artifacts: Mapping[str, object]
    pca_projection: PCAProjection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    def transform(self, table: FeatureTable) -> FrontendResult:
        """Apply the fitted frontend to a same-schema feature table."""

        if table.feature_names != self.input_feature_names:
            raise FrontendError(
                "Frontend schema mismatch. Expected feature names "
                f"{self.input_feature_names!r}, got {table.feature_names!r}."
            )

        transformed = table
        if self.pca_projection is not None:
            transformed = self.pca_projection.transform(table)

        return FrontendResult(
            mode=self.mode,
            table=transformed,
            artifacts=self.artifacts,
        )

    def fit_transform(self, table: FeatureTable) -> FrontendResult:
        """Apply the fitted frontend to the fit table itself."""

        return self.transform(table)


@dataclass(frozen=True, slots=True)
class FeatureFrontend:
    """Small frontend selector that engines can fit once and then reuse."""

    mode: FrontendMode

    def fit(self, table: FeatureTable) -> FittedFeatureFrontend:
        """Fit the selected frontend on one feature table."""

        artifacts: dict[str, object] = {}
        projection: PCAProjection | None = None
        output_feature_names = table.feature_names

        if self.mode in {"eda", "eda_pca"}:
            artifacts["eda"] = summarize_eda(table)
        if self.mode in {"pca", "eda_pca"}:
            projection = fit_pca(table)
            artifacts["pca"] = projection.to_artifact(
                n_rows=table.shape[0],
                n_features=table.shape[1],
            )
            output_feature_names = projection.output_feature_names

        return FittedFeatureFrontend(
            mode=self.mode,
            input_feature_names=table.feature_names,
            output_feature_names=output_feature_names,
            artifacts=artifacts,
            pca_projection=projection,
        )

    def fit_transform(self, table: FeatureTable) -> FrontendResult:
        """Fit the selected frontend and immediately transform the same table."""

        return self.fit(table).fit_transform(table)


def build_frontend(frontend: FrontendMode | FrontendConfig) -> FeatureFrontend:
    """Build a frontend selector from a mode string or parsed config."""

    mode = frontend.mode if isinstance(frontend, FrontendConfig) else frontend
    if mode not in _SUPPORTED_FRONTEND_MODES:
        choices = ", ".join(sorted(_SUPPORTED_FRONTEND_MODES))
        raise FrontendError(
            f"Unsupported frontend mode `{mode}`. Supported values: {choices}."
        )
    return FeatureFrontend(mode=mode)


def apply_frontend(
    frontend: FrontendMode | FrontendConfig,
    table: FeatureTable,
) -> FrontendResult:
    """Convenience helper for one-shot fit+transform on a single table."""

    return build_frontend(frontend).fit_transform(table)
