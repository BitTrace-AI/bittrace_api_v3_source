"""Modular frontend fit/transform surface for shared BitTrace feature tables."""

from bittrace.core.frontends.eda import summarize_eda
from bittrace.core.frontends.pca import PCAProjection, fit_pca
from bittrace.core.frontends.pipeline import (
    FeatureFrontend,
    FittedFeatureFrontend,
    apply_frontend,
    build_frontend,
)
from bittrace.core.frontends.types import FeatureTable, FrontendError, FrontendResult

__all__ = [
    "FeatureFrontend",
    "FeatureTable",
    "FittedFeatureFrontend",
    "FrontendError",
    "FrontendResult",
    "PCAProjection",
    "apply_frontend",
    "build_frontend",
    "fit_pca",
    "summarize_eda",
]
