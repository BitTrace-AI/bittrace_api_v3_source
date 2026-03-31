"""Strict shared config contract for BitTrace API 3.0.

Shared contract:
- `dataset`
- `frontend`
- `encoder`
- `model`
- `training.evolution`
- `evaluation`
- `export`
- `logging`

Lean-only and deep-only boundaries:
- `training.lean` is reserved for lean-specific training knobs.
- `training.deep` holds only deep/DIM-specific knobs that are truly live in the
  implemented deep evaluator. Decorative or deferred deep fields are rejected.

Canonical placement:
- Shared evolutionary search settings live under `training.evolution`, not a
  top-level `evolution` block. This keeps one honest shared contract while
  preserving separate namespaces for lean-only and deep-only controls.

Sections without a stable shared schema yet are intentionally strict empty
blocks. They exist to lock the layout without silently accepting decorative or
deferred fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import tomllib
from typing import Literal, Self, TypeAlias

import yaml


FrontendMode: TypeAlias = Literal["none", "eda", "pca", "eda_pca"]
ModelMode: TypeAlias = Literal["lean", "deep"]

_ROOT_KEYS = frozenset(
    {
        "dataset",
        "frontend",
        "encoder",
        "model",
        "training",
        "evaluation",
        "export",
        "logging",
    }
)
_TRAINING_KEYS = frozenset({"evolution", "lean", "deep"})
_FRONTEND_MODES = frozenset({"none", "eda", "pca", "eda_pca"})
_MODEL_MODES = frozenset({"lean", "deep"})
_DEFERRED_FRONTEND_FIELDS = frozenset(
    {
        "feature_columns",
        "feature_regex",
        "keep_features",
        "n_components",
        "pca_components",
        "standardize",
        "variance_threshold",
        "whiten",
    }
)
_LIVE_LEAN_ONLY_FIELDS = frozenset(
    {
        "backend",
        "allow_backend_fallback",
    }
)
_LIVE_DEEP_ONLY_FIELDS = frozenset(
    {
        "k_medoids_per_class",
        "adaptive_k",
        "adaptive_k_candidates",
        "backend",
        "allow_backend_fallback",
    }
)
_DEFERRED_DEEP_ONLY_FIELDS = frozenset(
    {
        "reject_label",
        "reject_percentile",
        "reject_mode",
        "reject_threshold",
        "reject_margin",
    }
)
_ALL_DEEP_ONLY_FIELDS = _LIVE_DEEP_ONLY_FIELDS.union(_DEFERRED_DEEP_ONLY_FIELDS)

_ROOT_REJECTIONS = {
    "evolution": "Top-level `evolution` is not supported. Use `training.evolution`.",
    "evaluator": (
        "Top-level `evaluator` is not supported. Keep metric/reporting intent in "
        "`evaluation` and keep evaluator implementations in code."
    ),
}
_EVALUATION_REJECTIONS = {
    field: (
        f"`evaluation.{field}` is not supported. `evaluation` is reserved for "
        "metric/reporting intent only, not engine-specific search knobs."
    )
    for field in _ALL_DEEP_ONLY_FIELDS
}
_DEEP_REJECTIONS = {
    field: (
        f"`training.deep.{field}` is recognized as a deep-only knob but is "
        "deferred in this task and therefore rejected."
    )
    for field in _DEFERRED_DEEP_ONLY_FIELDS
}
_FRONTEND_REJECTIONS = {
    field: (
        f"`frontend.{field}` is not supported in API 3.0 yet. The only live "
        "frontend config field for this task is `frontend.mode`."
    )
    for field in _DEFERRED_FRONTEND_FIELDS
}


class ConfigValidationError(ValueError):
    """Raised when a config file violates the strict API 3.0 schema."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Reserved shared block. Dataset fields are deferred until stabilized."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_reserved_empty_block(raw, path="dataset")
        return cls()


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    """Optional preprocessing stage before bit encoding."""

    mode: FrontendMode

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="frontend",
            required={"mode"},
            explicit_rejections=_FRONTEND_REJECTIONS,
        )
        mode = _require_literal_str(
            raw["mode"],
            path="frontend.mode",
            allowed=_FRONTEND_MODES,
        )
        return cls(mode=mode)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Reserved encoder block for bit-layout and patch/offset-eye settings."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_reserved_empty_block(raw, path="encoder")
        return cls()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shared model selector for the engine family."""

    mode: ModelMode
    random_seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(raw, path="model", required={"mode", "random_seed"})
        mode = _require_literal_str(
            raw["mode"],
            path="model.mode",
            allowed=_MODEL_MODES,
        )
        random_seed = _require_int(raw["random_seed"], path="model.random_seed")
        return cls(mode=mode, random_seed=random_seed)


@dataclass(frozen=True, slots=True)
class EvolutionCheckpointConfig:
    """Real checkpoint/resume paths for the shared evolution loop."""

    save_path: str | None = None
    resume_from: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="training.evolution.checkpoint",
            required=frozenset(),
            optional={"save_path", "resume_from"},
        )
        return cls(
            save_path=_require_optional_non_empty_str(
                raw.get("save_path"),
                path="training.evolution.checkpoint.save_path",
            ),
            resume_from=_require_optional_non_empty_str(
                raw.get("resume_from"),
                path="training.evolution.checkpoint.resume_from",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Shared evolutionary search controls used by both engines."""

    seed: int
    generations: int
    population_size: int
    mu: int
    lam: int
    elite_count: int
    min_layers: int
    max_layers: int
    mutation_rate: float
    mutation_rate_schedule: str
    selection_mode: str
    tournament_k: int
    early_stopping_patience: int
    checkpoint: EvolutionCheckpointConfig = field(default_factory=EvolutionCheckpointConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="training.evolution",
            required={
                "seed",
                "generations",
                "population_size",
                "mu",
                "lam",
                "elite_count",
                "min_layers",
                "max_layers",
                "mutation_rate",
                "mutation_rate_schedule",
                "selection_mode",
                "tournament_k",
                "early_stopping_patience",
            },
            optional={"checkpoint"},
        )

        seed = _require_int(raw["seed"], path="training.evolution.seed")
        generations = _require_int(
            raw["generations"],
            path="training.evolution.generations",
            minimum=1,
        )
        population_size = _require_int(
            raw["population_size"],
            path="training.evolution.population_size",
            minimum=1,
        )
        mu = _require_int(raw["mu"], path="training.evolution.mu", minimum=1)
        lam = _require_int(raw["lam"], path="training.evolution.lam", minimum=1)
        elite_count = _require_int(
            raw["elite_count"],
            path="training.evolution.elite_count",
            minimum=0,
        )
        min_layers = _require_int(
            raw["min_layers"],
            path="training.evolution.min_layers",
            minimum=1,
        )
        max_layers = _require_int(
            raw["max_layers"],
            path="training.evolution.max_layers",
            minimum=1,
        )
        mutation_rate = _require_float(
            raw["mutation_rate"],
            path="training.evolution.mutation_rate",
            minimum=0.0,
            maximum=1.0,
        )
        mutation_rate_schedule = _require_non_empty_str(
            raw["mutation_rate_schedule"],
            path="training.evolution.mutation_rate_schedule",
        )
        selection_mode = _require_non_empty_str(
            raw["selection_mode"],
            path="training.evolution.selection_mode",
        )
        tournament_k = _require_int(
            raw["tournament_k"],
            path="training.evolution.tournament_k",
            minimum=1,
        )
        early_stopping_patience = _require_int(
            raw["early_stopping_patience"],
            path="training.evolution.early_stopping_patience",
            minimum=0,
        )
        checkpoint = EvolutionCheckpointConfig.from_mapping(
            _require_mapping(
                raw.get("checkpoint", {}),
                path="training.evolution.checkpoint",
            )
        )

        if min_layers > max_layers:
            raise ConfigValidationError(
                "`training.evolution.min_layers` must be less than or equal to "
                "`training.evolution.max_layers`."
            )
        if elite_count > population_size:
            raise ConfigValidationError(
                "`training.evolution.elite_count` cannot exceed "
                "`training.evolution.population_size`."
            )
        if mu > population_size:
            raise ConfigValidationError(
                "`training.evolution.mu` cannot exceed "
                "`training.evolution.population_size`."
            )
        if tournament_k > population_size:
            raise ConfigValidationError(
                "`training.evolution.tournament_k` cannot exceed "
                "`training.evolution.population_size`."
            )

        return cls(
            seed=seed,
            generations=generations,
            population_size=population_size,
            mu=mu,
            lam=lam,
            elite_count=elite_count,
            min_layers=min_layers,
            max_layers=max_layers,
            mutation_rate=mutation_rate,
            mutation_rate_schedule=mutation_rate_schedule,
            selection_mode=selection_mode,
            tournament_k=tournament_k,
            early_stopping_patience=early_stopping_patience,
            checkpoint=checkpoint,
        )


@dataclass(frozen=True, slots=True)
class LeanTrainingConfig:
    """Lean-only knobs that are genuinely live in the current evaluator."""

    backend: str = "auto"
    allow_backend_fallback: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="training.lean",
            required=frozenset(),
            optional=_LIVE_LEAN_ONLY_FIELDS,
        )
        backend = _require_literal_str(
            raw.get("backend", "auto"),
            path="training.lean.backend",
            allowed=frozenset({"auto", "cpu", "gpu"}),
        )
        allow_backend_fallback = _require_bool(
            raw.get("allow_backend_fallback", False),
            path="training.lean.allow_backend_fallback",
        )
        return cls(
            backend=backend,
            allow_backend_fallback=allow_backend_fallback,
        )


@dataclass(frozen=True, slots=True)
class DeepTrainingConfig:
    """Deep-only knobs that are genuinely live in the current evaluator."""

    k_medoids_per_class: int = 1
    adaptive_k: bool = False
    adaptive_k_candidates: tuple[int, ...] = ()
    backend: str = "auto"
    allow_backend_fallback: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="training.deep",
            required=frozenset(),
            optional=_LIVE_DEEP_ONLY_FIELDS,
            explicit_rejections=_DEEP_REJECTIONS,
        )
        k_medoids_per_class = _require_int(
            raw.get("k_medoids_per_class", 1),
            path="training.deep.k_medoids_per_class",
            minimum=1,
        )
        adaptive_k = _require_bool(
            raw.get("adaptive_k", False),
            path="training.deep.adaptive_k",
        )
        adaptive_k_candidates = _require_int_tuple(
            raw.get("adaptive_k_candidates", ()),
            path="training.deep.adaptive_k_candidates",
            minimum=1,
        )
        backend = _require_literal_str(
            raw.get("backend", "auto"),
            path="training.deep.backend",
            allowed=frozenset({"auto", "cpu", "gpu"}),
        )
        allow_backend_fallback = _require_bool(
            raw.get("allow_backend_fallback", False),
            path="training.deep.allow_backend_fallback",
        )
        if adaptive_k and not adaptive_k_candidates:
            raise ConfigValidationError(
                "`training.deep.adaptive_k_candidates` must be a non-empty list "
                "when `training.deep.adaptive_k=true`."
            )
        return cls(
            k_medoids_per_class=k_medoids_per_class,
            adaptive_k=adaptive_k,
            adaptive_k_candidates=adaptive_k_candidates,
            backend=backend,
            allow_backend_fallback=allow_backend_fallback,
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training subtree that separates shared and engine-specific controls."""

    evolution: EvolutionConfig
    lean: LeanTrainingConfig
    deep: DeepTrainingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(raw, path="training", required=_TRAINING_KEYS)
        return cls(
            evolution=EvolutionConfig.from_mapping(
                _require_mapping(raw["evolution"], path="training.evolution")
            ),
            lean=LeanTrainingConfig.from_mapping(
                _require_mapping(raw["lean"], path="training.lean")
            ),
            deep=DeepTrainingConfig.from_mapping(
                _require_mapping(raw["deep"], path="training.deep")
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Reserved reporting block. Search knobs are rejected here by design."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_reserved_empty_block(
            raw,
            path="evaluation",
            explicit_rejections=_EVALUATION_REJECTIONS,
        )
        return cls()


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Reserved export block. Export schema is deferred."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_reserved_empty_block(raw, path="export")
        return cls()


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Reserved logging block. Logging schema is deferred."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_reserved_empty_block(raw, path="logging")
        return cls()


@dataclass(frozen=True, slots=True)
class BitTraceConfig:
    """Canonical shared API 3.0 config tree."""

    dataset: DatasetConfig
    frontend: FrontendConfig
    encoder: EncoderConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    export: ExportConfig
    logging: LoggingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _expect_allowed_keys(
            raw,
            path="<root>",
            required=_ROOT_KEYS,
            explicit_rejections=_ROOT_REJECTIONS,
        )
        return cls(
            dataset=DatasetConfig.from_mapping(
                _require_mapping(raw["dataset"], path="dataset")
            ),
            frontend=FrontendConfig.from_mapping(
                _require_mapping(raw["frontend"], path="frontend")
            ),
            encoder=EncoderConfig.from_mapping(
                _require_mapping(raw["encoder"], path="encoder")
            ),
            model=ModelConfig.from_mapping(_require_mapping(raw["model"], path="model")),
            training=TrainingConfig.from_mapping(
                _require_mapping(raw["training"], path="training")
            ),
            evaluation=EvaluationConfig.from_mapping(
                _require_mapping(raw["evaluation"], path="evaluation")
            ),
            export=ExportConfig.from_mapping(
                _require_mapping(raw["export"], path="export")
            ),
            logging=LoggingConfig.from_mapping(
                _require_mapping(raw["logging"], path="logging")
            ),
        )


def parse_config(data: Mapping[str, object]) -> BitTraceConfig:
    """Validate an in-memory mapping against the strict API 3.0 schema."""

    return BitTraceConfig.from_mapping(_require_mapping(data, path="<root>"))


def load_config(path: str | Path) -> BitTraceConfig:
    """Load and validate a config file.

    Supported formats:
    - `.yaml`
    - `.yml`
    - `.json`
    - `.toml`
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    content = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(content)
    elif suffix == ".json":
        loaded = json.loads(content)
    elif suffix == ".toml":
        loaded = tomllib.loads(content)
    else:
        raise ConfigValidationError(
            f"Unsupported config format `{suffix}` for `{config_path}`. Use "
            "`.yaml`, `.yml`, `.json`, or `.toml`."
        )

    if loaded is None:
        raise ConfigValidationError(f"`{config_path}` is empty.")

    return parse_config(_require_mapping(loaded, path="<root>"))


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"`{path}` must be a mapping/object.")

    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigValidationError(
                f"`{path}` contains a non-string key `{key!r}`. Config keys must be strings."
            )
        normalized[key] = item
    return normalized


def _expect_allowed_keys(
    raw: Mapping[str, object],
    *,
    path: str,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] | None = None,
    explicit_rejections: Mapping[str, str] | None = None,
) -> None:
    allowed = set(required)
    if optional:
        allowed.update(optional)

    for key in raw:
        if key not in allowed:
            message = explicit_rejections.get(key) if explicit_rejections else None
            if message is not None:
                raise ConfigValidationError(message)
            dotted = key if path == "<root>" else f"{path}.{key}"
            raise ConfigValidationError(
                f"Unsupported key `{dotted}`. No decorative or deferred fields are accepted."
            )

    missing = sorted(required.difference(raw.keys()))
    if missing:
        rendered = ", ".join(f"`{path}.{key}`" if path != "<root>" else f"`{key}`" for key in missing)
        raise ConfigValidationError(f"Missing required config field(s): {rendered}.")


def _expect_reserved_empty_block(
    raw: Mapping[str, object],
    *,
    path: str,
    explicit_rejections: Mapping[str, str] | None = None,
) -> None:
    _expect_allowed_keys(
        raw,
        path=path,
        required=frozenset(),
        optional=frozenset(),
        explicit_rejections=explicit_rejections,
    )


def _require_int(value: object, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"`{path}` must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"`{path}` must be greater than or equal to {minimum}.")
    return value


def _require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"`{path}` must be a boolean.")
    return value


def _require_float(
    value: object,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigValidationError(f"`{path}` must be a float.")

    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ConfigValidationError(f"`{path}` must be greater than or equal to {minimum}.")
    if maximum is not None and normalized > maximum:
        raise ConfigValidationError(f"`{path}` must be less than or equal to {maximum}.")
    return normalized


def _require_non_empty_str(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"`{path}` must be a non-empty string.")
    return value


def _require_optional_non_empty_str(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, path=path)


def _require_literal_str(
    value: object,
    *,
    path: str,
    allowed: frozenset[str],
) -> str:
    normalized = _require_non_empty_str(value, path=path)
    if normalized not in allowed:
        choices = ", ".join(f"`{choice}`" for choice in sorted(allowed))
        raise ConfigValidationError(
            f"`{path}` must be one of {choices}; received `{normalized}`."
        )
    return normalized


def _require_int_tuple(
    value: object,
    *,
    path: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigValidationError(f"`{path}` must be a list of integers.")

    normalized: list[int] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        integer = _require_int(item, path=f"{path}[{index}]", minimum=minimum)
        if integer in seen:
            raise ConfigValidationError(
                f"`{path}` cannot contain duplicate value `{integer}`."
            )
        normalized.append(integer)
        seen.add(integer)
    return tuple(normalized)


__all__ = [
    "BitTraceConfig",
    "ConfigValidationError",
    "DatasetConfig",
    "DeepTrainingConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "EvolutionConfig",
    "ExportConfig",
    "FrontendConfig",
    "FrontendMode",
    "LeanTrainingConfig",
    "load_config",
    "LoggingConfig",
    "ModelConfig",
    "ModelMode",
    "parse_config",
    "TrainingConfig",
]
