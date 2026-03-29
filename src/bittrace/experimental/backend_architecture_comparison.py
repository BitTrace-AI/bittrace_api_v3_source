"""Experimental backend-architecture comparison workflows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.core.config import DeepTrainingConfig, EvolutionConfig, LeanTrainingConfig
from bittrace.core.deep.engine import (
    DeepLayer,
    _apply_layers_as_embedding,
    _predict_row as _deep_predict_row,
    run_deep_evolution,
)
from bittrace.core.evolution.loop import SelectionSpec
from bittrace.core.lean.engine import LeanLayer, run_lean_evolution
from bittrace.v3 import ContractValidationError, StageKey, WaveformDatasetBundle, WaveformDatasetRecord
from bittrace.v3.frontend_encoding import encode_frontend_record

from bittrace.source.full_binary_campaign import (
    DEFAULT_RUNS_ROOT as FULL_BINARY_DEFAULT_RUNS_ROOT,
    _load_backend_training_configs,
    _materialize_source_bundle,
    _resolve_inventory_rows,
    load_consumer_config,
)
from bittrace.source.locked_frontend import (
    LockedFrontendSpec,
    build_locked_frontend_stage_materialization,
    load_locked_frontend_spec,
)
from bittrace.source.temporal_features import load_temporal_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experimental" / "backend_architecture_comparison.yaml"
DEFAULT_RUNS_ROOT = FULL_BINARY_DEFAULT_RUNS_ROOT
_LABEL_TO_INT = {"healthy": 0, "unhealthy": 1}
_INT_TO_LABEL = {value: key for key, value in _LABEL_TO_INT.items()}
_LABEL_ORDER = ("healthy", "unhealthy")
_PACKED_BIT_LENGTH = 64
_BUNDLE_SCHEMA_VERSION = "bittrace-bearings-v3-source-backend-architecture-comparison-bundle-1"
_PLAN_SCHEMA_VERSION = "bittrace-bearings-v3-source-backend-architecture-comparison-plan-1"
_FRONT_ONLY_ARTIFACT_SCHEMA_VERSION = (
    "bittrace-bearings-v3-source-backend-architecture-front-only-artifact-1"
)
_VARIANT_SUMMARY_SCHEMA_VERSION = (
    "bittrace-bearings-v3-source-backend-architecture-variant-summary-1"
)
_RUN_SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-source-backend-architecture-summary-1"
_SUMMARY_SPLITS = frozenset({"train", "val", "test"})
_VARIANT_ORDER = ("FRONT_ONLY", "LEAN_DEEP", "LEAN_LEAN")
_VARIANT_LABELS = {
    "FRONT_ONLY": "Front Only",
    "LEAN_DEEP": "Lean-Deep",
    "LEAN_LEAN": "Lean-Lean",
}
_DECISION_PATHS = {
    "FRONT_ONLY": "shared_frontend_final_residue_train_medoids",
    "LEAN_DEEP": "shared_frontend_plus_deep_all_layer_residue_readout",
    "LEAN_LEAN": "shared_frontend_plus_second_stage_lean_final_layer_only",
}


@dataclass(frozen=True, slots=True)
class ComparisonEvaluationConfig:
    summary_metric_split: str
    separability_split: str
    latency_split: str
    latency_warmup_passes: int
    latency_timed_passes: int


@dataclass(frozen=True, slots=True)
class SearchVariantConfig:
    seed_offset: int


@dataclass(frozen=True, slots=True)
class _BundleSplit:
    split_name: str
    rows: tuple[int, ...]
    labels: tuple[int, ...]
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SharedBackendBundle:
    bundle_dir: Path
    contract_path: Path
    semantic_bit_length: int
    packed_bit_length: int
    frontend_input_id: str
    frontend_fingerprint: str
    bit_feature_names: tuple[str, ...]
    splits: Mapping[str, _BundleSplit]


@dataclass(frozen=True, slots=True)
class PreparedBackendArchitectureComparison:
    config_path: Path
    run_root: Path
    profile_name: str
    source_profile_path: Path
    source_profile_name: str
    locked_frontend: LockedFrontendSpec
    source_bundle: WaveformDatasetBundle
    shared_bundle: SharedBackendBundle
    shared_frontend_dir: Path
    shared_row_count: int
    split_counts: Mapping[str, int]
    state_counts: Mapping[str, int]
    evaluation: ComparisonEvaluationConfig
    selection_spec: SelectionSpec
    search_config: EvolutionConfig
    lean_training_config: LeanTrainingConfig
    deep_training_config: DeepTrainingConfig
    lean_deep_config: SearchVariantConfig
    lean_lean_config: SearchVariantConfig


@dataclass(frozen=True, slots=True)
class _FrontOnlyModel:
    prototypes: Mapping[int, int]
    prototype_record_ids: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class _LeanArtifactModel:
    layers: tuple[LeanLayer, ...]
    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _DeepArtifactModel:
    layers: tuple[DeepLayer, ...]
    prototypes: tuple[int, ...]
    prototype_labels: tuple[int, ...]
    class_labels: tuple[int, ...]
    embedding_bit_length: int


def load_backend_architecture_comparison_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    for key in ("profile_name", "source_profile", "backend_architecture_comparison"):
        if key not in payload:
            raise ContractValidationError(f"`{resolved_path}` is missing required top-level key `{key}`.")
    return payload


def prepare_backend_architecture_comparison(
    config_path: str | Path,
    run_root: str | Path,
) -> PreparedBackendArchitectureComparison:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = Path(run_root).resolve()
    payload = load_backend_architecture_comparison_config(resolved_config_path)
    comparison = payload["backend_architecture_comparison"]
    if not isinstance(comparison, Mapping):
        raise ContractValidationError("`backend_architecture_comparison` must be a mapping.")

    source_profile_path = _resolve_relative_path(
        config_file=resolved_config_path,
        raw_path=payload["source_profile"],
        field_name="source_profile",
    )
    source_profile = load_consumer_config(source_profile_path)
    locked_frontend = load_locked_frontend_spec(source_profile)
    if locked_frontend is None:
        raise ContractValidationError(
            "This comparison requires a source profile with `locked_frontend.enabled: true`."
        )
    if not locked_frontend.temporal_features_enabled:
        raise ContractValidationError(
            "This comparison requires a locked shared frontend with temporal features enabled."
        )

    hard_mode = source_profile.get("hard_mode", {})
    include_test_metrics_in_frontend = (
        bool(hard_mode.get("include_test_metrics_in_frontend", False))
        if isinstance(hard_mode, Mapping)
        else False
    )
    if include_test_metrics_in_frontend:
        raise ContractValidationError(
            "This comparison requires `hard_mode.include_test_metrics_in_frontend: false` in the source profile."
        )

    evaluation = _parse_evaluation_config(comparison.get("evaluation"))
    selection_spec = _selection_spec_from_mapping(
        comparison.get("selection_spec"),
        path="backend_architecture_comparison.selection_spec",
    )
    search_config = _parse_search_config(
        comparison.get("search"),
        path="backend_architecture_comparison.search",
    )
    lean_deep_config = _parse_variant_config(
        comparison.get("variants"),
        variant_key="lean_deep",
        default_seed_offset=0,
    )
    lean_lean_config = _parse_variant_config(
        comparison.get("variants"),
        variant_key="lean_lean",
        default_seed_offset=1000,
    )

    inventory_rows = _resolve_inventory_rows(source_profile)
    temporal_feature_config = load_temporal_feature_config(source_profile)
    source_bundle = _materialize_source_bundle(
        inventory_rows,
        output_dir=resolved_run_root / "_inputs" / "shared_source_bundle",
        profile_name=str(payload["profile_name"]),
        selection_name="backend_architecture_comparison_shared",
        temporal_feature_config=temporal_feature_config,
    )
    shared_frontend_dir = resolved_run_root / "01_shared_frontend"
    shared_frontend = build_locked_frontend_stage_materialization(
        stage_key=StageKey.LEAN_MAIN_SCREEN,
        stage_output_dir=shared_frontend_dir,
        source_bundle=source_bundle,
        include_test_metrics_in_frontend=False,
        locked_frontend=locked_frontend,
    )
    shared_bundle = _materialize_shared_backend_bundle(
        bundle_dir=resolved_run_root / "02_shared_backend_bundle",
        source_bundle=source_bundle,
        deep_input=shared_frontend.downstream_deep_input,
    )
    lean_training_config, deep_training_config = _load_backend_training_configs(source_profile)
    split_counts = Counter(str(row["split"]) for row in inventory_rows)
    state_counts = Counter(str(row["binary_label"]) for row in inventory_rows)
    return PreparedBackendArchitectureComparison(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        source_profile_path=source_profile_path,
        source_profile_name=str(source_profile["profile_name"]),
        locked_frontend=locked_frontend,
        source_bundle=source_bundle,
        shared_bundle=shared_bundle,
        shared_frontend_dir=shared_frontend_dir,
        shared_row_count=len(inventory_rows),
        split_counts=dict(sorted(split_counts.items())),
        state_counts=dict(sorted(state_counts.items())),
        evaluation=evaluation,
        selection_spec=selection_spec,
        search_config=search_config,
        lean_training_config=lean_training_config,
        deep_training_config=deep_training_config,
        lean_deep_config=lean_deep_config,
        lean_lean_config=lean_lean_config,
    )


def write_backend_architecture_plan(
    prepared: PreparedBackendArchitectureComparison,
) -> Path:
    plan_path = prepared.run_root / "comparison_plan.json"
    payload = {
        "schema_version": _PLAN_SCHEMA_VERSION,
        "profile_name": prepared.profile_name,
        "config_path": str(prepared.config_path),
        "run_root": str(prepared.run_root),
        "source_profile_path": str(prepared.source_profile_path),
        "source_profile_name": prepared.source_profile_name,
        "shared_dataset": {
            "row_count": prepared.shared_row_count,
            "split_counts": dict(prepared.split_counts),
            "state_counts": dict(prepared.state_counts),
        },
        "shared_frontend": {
            "locked_frontend": prepared.locked_frontend.to_dict(),
            "shared_frontend_dir": str(prepared.shared_frontend_dir.resolve()),
            "frontend_input_id": prepared.shared_bundle.frontend_input_id,
            "frontend_fingerprint": prepared.shared_bundle.frontend_fingerprint,
            "semantic_bit_length": prepared.shared_bundle.semantic_bit_length,
            "comparison_bundle_bit_length": prepared.shared_bundle.packed_bit_length,
            "bundle_contract_path": str(prepared.shared_bundle.contract_path.resolve()),
        },
        "constraints": {
            "frontend_fixed_shared": True,
            "preprocessing_changed_between_variants": False,
            "splits_changed_between_variants": False,
            "new_features_added": False,
            "persistence_used": False,
            "packed_bit_inference": True,
            "distance_metric_family": "hamming_xor_popcount",
        },
        "variants": [
            {
                "variant_id": variant_id,
                "label": _VARIANT_LABELS[variant_id],
                "decision_path": _DECISION_PATHS[variant_id],
            }
            for variant_id in _VARIANT_ORDER
        ],
        "search": {
            "selection_spec": {
                "primary_metric": prepared.selection_spec.primary_metric,
                "tiebreak_metrics": list(prepared.selection_spec.tiebreak_metrics),
            },
            "shared": _evolution_config_to_dict(prepared.search_config),
            "lean_deep_seed_offset": prepared.lean_deep_config.seed_offset,
            "lean_lean_seed_offset": prepared.lean_lean_config.seed_offset,
        },
        "evaluation": {
            "summary_metric_split": prepared.evaluation.summary_metric_split,
            "separability_split": prepared.evaluation.separability_split,
            "latency_split": prepared.evaluation.latency_split,
            "latency_warmup_passes": prepared.evaluation.latency_warmup_passes,
            "latency_timed_passes": prepared.evaluation.latency_timed_passes,
        },
        "backend_requests": {
            "lean": {
                "backend": prepared.lean_training_config.backend,
                "allow_backend_fallback": prepared.lean_training_config.allow_backend_fallback,
            },
            "deep": {
                "backend": prepared.deep_training_config.backend,
                "allow_backend_fallback": prepared.deep_training_config.allow_backend_fallback,
                "k_medoids_per_class": prepared.deep_training_config.k_medoids_per_class,
                "adaptive_k": prepared.deep_training_config.adaptive_k,
                "adaptive_k_candidates": list(prepared.deep_training_config.adaptive_k_candidates),
            },
        },
    }
    _write_json(plan_path, payload)
    return plan_path


def run_prepared_backend_architecture_comparison(
    prepared: PreparedBackendArchitectureComparison,
) -> Path:
    variant_results = [
        _run_front_only_variant(prepared),
        _run_lean_deep_variant(prepared),
        _run_lean_lean_variant(prepared),
    ]
    summary_metric_split = prepared.evaluation.summary_metric_split
    summary_rows = [
        _build_summary_row(result, summary_metric_split=summary_metric_split)
        for result in variant_results
    ]
    summary_csv_path = prepared.run_root / "summary.csv"
    _write_summary_csv(summary_csv_path, summary_rows)
    summary_md_path = prepared.run_root / "summary.md"
    summary_md_path.write_text(
        _build_summary_markdown(
            prepared,
            variant_results,
            summary_metric_split=summary_metric_split,
        ),
        encoding="utf-8",
    )
    summary_json_path = prepared.run_root / "comparison_summary.json"
    _write_json(
        summary_json_path,
        {
            "schema_version": _RUN_SUMMARY_SCHEMA_VERSION,
            "profile_name": prepared.profile_name,
            "config_path": str(prepared.config_path),
            "run_root": str(prepared.run_root),
            "source_profile_path": str(prepared.source_profile_path),
            "summary_metric_split": summary_metric_split,
            "summary_csv_path": str(summary_csv_path.resolve()),
            "summary_md_path": str(summary_md_path.resolve()),
            "variants": variant_results,
        },
    )
    return summary_json_path


def run_backend_architecture_comparison(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
) -> Path:
    resolved_run_root = (
        Path(run_root).resolve()
        if run_root is not None
        else (DEFAULT_RUNS_ROOT / Path(config_path).resolve().stem / "manual_run").resolve()
    )
    prepared = prepare_backend_architecture_comparison(config_path, resolved_run_root)
    write_backend_architecture_plan(prepared)
    return run_prepared_backend_architecture_comparison(prepared)


def _parse_evaluation_config(raw: object) -> ComparisonEvaluationConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`backend_architecture_comparison.evaluation` must be a mapping.")
    summary_metric_split = _require_split_name(
        raw.get("summary_metric_split", "test"),
        field_name="backend_architecture_comparison.evaluation.summary_metric_split",
    )
    separability_split = _require_split_name(
        raw.get("separability_split", "train"),
        field_name="backend_architecture_comparison.evaluation.separability_split",
    )
    latency_split = _require_split_name(
        raw.get("latency_split", "test"),
        field_name="backend_architecture_comparison.evaluation.latency_split",
    )
    latency_warmup_passes = _require_int(
        raw.get("latency_warmup_passes", 2),
        field_name="backend_architecture_comparison.evaluation.latency_warmup_passes",
        minimum=0,
    )
    latency_timed_passes = _require_int(
        raw.get("latency_timed_passes", 10),
        field_name="backend_architecture_comparison.evaluation.latency_timed_passes",
        minimum=1,
    )
    return ComparisonEvaluationConfig(
        summary_metric_split=summary_metric_split,
        separability_split=separability_split,
        latency_split=latency_split,
        latency_warmup_passes=latency_warmup_passes,
        latency_timed_passes=latency_timed_passes,
    )


def _parse_search_config(raw: object, *, path: str) -> EvolutionConfig:
    if not isinstance(raw, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    try:
        config = EvolutionConfig.from_mapping(raw)
    except Exception as exc:  # pragma: no cover - config validation path
        raise ContractValidationError(f"`{path}` is invalid: {exc}") from exc
    if config.checkpoint.save_path is not None or config.checkpoint.resume_from is not None:
        raise ContractValidationError(
            "This comparison forbids persistence; leave `backend_architecture_comparison.search.checkpoint` empty."
        )
    if config.max_layers > 3:
        raise ContractValidationError(
            "This comparison keeps the backend search shallow; `search.max_layers` must be <= 3."
        )
    return config


def _parse_variant_config(
    raw: object,
    *,
    variant_key: str,
    default_seed_offset: int,
) -> SearchVariantConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`backend_architecture_comparison.variants` must be a mapping.")
    variant_raw = raw.get(variant_key, {})
    if variant_raw is None:
        variant_raw = {}
    if not isinstance(variant_raw, Mapping):
        raise ContractValidationError(
            f"`backend_architecture_comparison.variants.{variant_key}` must be a mapping."
        )
    seed_offset = _require_int(
        variant_raw.get("seed_offset", default_seed_offset),
        field_name=f"backend_architecture_comparison.variants.{variant_key}.seed_offset",
        minimum=0,
    )
    return SearchVariantConfig(seed_offset=seed_offset)


def _selection_spec_from_mapping(raw: object, *, path: str) -> SelectionSpec:
    if raw in ({}, None):
        return SelectionSpec(primary_metric="fitness", tiebreak_metrics=("accuracy", "mean_margin"))
    if not isinstance(raw, Mapping):
        raise ContractValidationError(f"`{path}` must be a mapping.")
    primary_metric = raw.get("primary_metric", "fitness")
    tiebreak_metrics = raw.get("tiebreak_metrics", ())
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ContractValidationError(f"`{path}.primary_metric` must be a non-empty string.")
    if not isinstance(tiebreak_metrics, Sequence) or isinstance(
        tiebreak_metrics,
        (str, bytes, bytearray),
    ):
        raise ContractValidationError(f"`{path}.tiebreak_metrics` must be a sequence of strings.")
    normalized_tiebreak_metrics = []
    for index, metric_name in enumerate(tiebreak_metrics):
        if not isinstance(metric_name, str) or not metric_name:
            raise ContractValidationError(
                f"`{path}.tiebreak_metrics[{index}]` must be a non-empty string."
            )
        normalized_tiebreak_metrics.append(metric_name)
    try:
        return SelectionSpec(
            primary_metric=primary_metric,
            tiebreak_metrics=tuple(normalized_tiebreak_metrics),
        )
    except ValueError as exc:
        raise ContractValidationError(f"`{path}` is invalid: {exc}") from exc


def _materialize_shared_backend_bundle(
    *,
    bundle_dir: Path,
    source_bundle: WaveformDatasetBundle,
    deep_input: Any,
) -> SharedBackendBundle:
    contract_payload = _load_json(Path(deep_input.bundle_contract_path))
    handoff_payload = _load_json(Path(deep_input.handoff_manifest_path))
    raw_records = handoff_payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ContractValidationError(
            "Shared backend comparison requires non-empty canonical `records` in the locked-frontend handoff."
        )
    split_payloads: dict[str, dict[str, list[object]]] = {
        split_name: {"X_packed": [], "y": [], "source_record_ids": []}
        for split_name in _SUMMARY_SPLITS
    }
    feature_names: tuple[str, ...] | None = None
    encoded_bit_length: int | None = None
    locked_frontend = contract_payload.get("locked_frontend")
    original_frontend_bit_length = _read_positive_int(
        _mapping_get_optional_int(contract_payload, "frontend_encoding", "bit_length"),
        default=_read_positive_int(
            _mapping_get_optional_int(locked_frontend, "bit_length"),
            default=None,
        ),
    )
    frontend_input_id = str(deep_input.frontend_input_id)
    frontend_fingerprint = str(deep_input.frontend_fingerprint)
    for raw_record in raw_records:
        record = _coerce_waveform_record(raw_record)
        split_name = str(record.split).lower()
        if split_name not in split_payloads:
            raise ContractValidationError(
                f"Backend comparison bundle encountered unsupported split `{record.split}`."
            )
        state_label = str(record.state_label).lower()
        if state_label not in _LABEL_TO_INT:
            raise ContractValidationError(
                "Backend comparison bundle supports only `healthy` and `unhealthy` labels."
            )
        encoded = encode_frontend_record(
            record,
            dataset_id=source_bundle.dataset_id,
            adapter_profile_id=source_bundle.adapter_profile_id,
            frontend_input_id=frontend_input_id,
            frontend_fingerprint=frontend_fingerprint,
            contract_payload=contract_payload,
        )
        if feature_names is None:
            feature_names = tuple(encoded.bit_feature_names)
            encoded_bit_length = int(encoded.bit_length)
        split_payloads[split_name]["X_packed"].append(int(encoded.packed_row_int))
        split_payloads[split_name]["y"].append(_LABEL_TO_INT[state_label])
        split_payloads[split_name]["source_record_ids"].append(str(record.source_record_id))
    semantic_bit_length = _resolve_semantic_bit_length(
        original_frontend_bit_length,
        encoded_bit_length,
    )
    padded_feature_names = _materialize_packed64_feature_names(
        feature_names,
        semantic_bit_length=semantic_bit_length,
    )
    split_refs: dict[str, _BundleSplit] = {}
    for split_name, split_payload in split_payloads.items():
        if not split_payload["X_packed"]:
            raise ContractValidationError(
                f"Backend comparison bundle is missing rows for `{split_name}`."
            )
        _write_json(bundle_dir / f"{split_name}_bits.json", split_payload)
        split_refs[split_name] = _BundleSplit(
            split_name=split_name,
            rows=tuple(int(value) for value in split_payload["X_packed"]),
            labels=tuple(int(value) for value in split_payload["y"]),
            source_record_ids=tuple(str(value) for value in split_payload["source_record_ids"]),
        )
    contract_path = bundle_dir / "contract.json"
    _write_json(
        contract_path,
        {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "row_format": "packed_int_lsb0",
            "bit_length": _PACKED_BIT_LENGTH,
            "feature_names": list(padded_feature_names),
            "frontend_input_id": frontend_input_id,
            "frontend_fingerprint": frontend_fingerprint,
            "label_mapping": dict(_LABEL_TO_INT),
            "bundle_materialization": {
                "semantic_bit_length": semantic_bit_length,
                "comparison_bundle_bit_length": _PACKED_BIT_LENGTH,
                "padding_rule": "zero_fill_high_bits",
                "materialization_reason": (
                    "shared_locked_frontend_backend_comparison_gpu_compatible_packed64"
                ),
                "shared_frontend_preserved": True,
            },
            "split_semantics": {
                "train": "backend fitting/search only",
                "val": "variant comparison and winner selection only",
                "test": "final comparison reporting only",
            },
            "locked_frontend": dict(locked_frontend) if isinstance(locked_frontend, Mapping) else None,
        },
    )
    return SharedBackendBundle(
        bundle_dir=bundle_dir,
        contract_path=contract_path,
        semantic_bit_length=semantic_bit_length,
        packed_bit_length=_PACKED_BIT_LENGTH,
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=frontend_fingerprint,
        bit_feature_names=padded_feature_names,
        splits=split_refs,
    )


def _run_front_only_variant(
    prepared: PreparedBackendArchitectureComparison,
) -> dict[str, object]:
    variant_id = "FRONT_ONLY"
    variant_dir = prepared.run_root / "variants" / "front_only"
    train_split = prepared.shared_bundle.splits["train"]
    prototypes: dict[int, int] = {}
    prototype_record_ids: dict[int, str] = {}
    for label_name in _LABEL_ORDER:
        label_int = _LABEL_TO_INT[label_name]
        class_rows = [
            (row, source_record_id)
            for row, label, source_record_id in zip(
                train_split.rows,
                train_split.labels,
                train_split.source_record_ids,
                strict=True,
            )
            if label == label_int
        ]
        medoid_value = _select_single_medoid(tuple(row for row, _ in class_rows))
        prototypes[label_int] = medoid_value
        prototype_record_ids[label_int] = next(
            source_record_id for row, source_record_id in class_rows if row == medoid_value
        )
    model = _FrontOnlyModel(
        prototypes=prototypes,
        prototype_record_ids=prototype_record_ids,
    )
    artifact_path = variant_dir / "front_only_artifact.json"
    _write_json(
        artifact_path,
        {
            "schema_version": _FRONT_ONLY_ARTIFACT_SCHEMA_VERSION,
            "variant_id": variant_id,
            "scoring_mode": "front_handoff_direct_class_medoids",
            "frontend_input_id": prepared.shared_bundle.frontend_input_id,
            "frontend_fingerprint": prepared.shared_bundle.frontend_fingerprint,
            "semantic_bit_length": prepared.shared_bundle.semantic_bit_length,
            "packed_bit_length": prepared.shared_bundle.packed_bit_length,
            "model": {
                "prototypes": {
                    _INT_TO_LABEL[label_int]: {
                        "packed_row_int": value,
                        "packed_row_hex": _hex_row(value, bit_length=prepared.shared_bundle.packed_bit_length),
                        "source_record_id": prototype_record_ids[label_int],
                    }
                    for label_int, value in sorted(prototypes.items())
                }
            },
        },
    )
    split_metrics = {
        split_name: _evaluate_front_only_split(prepared.shared_bundle.splits[split_name], model)
        for split_name in ("train", "val", "test")
    }
    separability = _compute_separability(
        split_name=prepared.evaluation.separability_split,
        rows=prepared.shared_bundle.splits[prepared.evaluation.separability_split].rows,
        labels=prepared.shared_bundle.splits[prepared.evaluation.separability_split].labels,
        bit_length=prepared.shared_bundle.packed_bit_length,
    )
    latency = _measure_latency(
        rows=prepared.shared_bundle.splits[prepared.evaluation.latency_split].rows,
        predict_many=lambda rows: _predict_front_only_rows(rows, model),
        split_name=prepared.evaluation.latency_split,
        warmup_passes=prepared.evaluation.latency_warmup_passes,
        timed_passes=prepared.evaluation.latency_timed_passes,
    )
    return _write_variant_summary(
        prepared,
        variant_id=variant_id,
        variant_dir=variant_dir,
        artifact_path=artifact_path,
        split_metrics=split_metrics,
        separability=separability,
        latency=latency,
        model_details={
            "decision_path": _DECISION_PATHS[variant_id],
            "prototypes": {
                _INT_TO_LABEL[label_int]: {
                    "packed_row_int": value,
                    "packed_row_hex": _hex_row(value, bit_length=prepared.shared_bundle.packed_bit_length),
                    "source_record_id": prototype_record_ids[label_int],
                }
                for label_int, value in sorted(prototypes.items())
            },
        },
        native_paths={},
    )


def _run_lean_deep_variant(
    prepared: PreparedBackendArchitectureComparison,
) -> dict[str, object]:
    variant_id = "LEAN_DEEP"
    variant_dir = prepared.run_root / "variants" / "lean_deep"
    engine_dir = variant_dir / "engine_run"
    run_result = run_deep_evolution(
        prepared.shared_bundle.bundle_dir,
        engine_dir,
        evolution_config=_offset_evolution_seed(
            prepared.search_config,
            seed_offset=prepared.lean_deep_config.seed_offset,
        ),
        deep_config=prepared.deep_training_config,
        backend=prepared.deep_training_config.backend,
        allow_backend_fallback=prepared.deep_training_config.allow_backend_fallback,
        selection_spec=prepared.selection_spec,
        include_test_metrics=False,
    )
    model = _load_deep_artifact_model(run_result.artifact_path)
    split_metrics = {
        split_name: _evaluate_deep_split(
            prepared.shared_bundle.splits[split_name],
            model,
        )
        for split_name in ("train", "val", "test")
    }
    separability_rows = _apply_layers_as_embedding(
        prepared.shared_bundle.splits[prepared.evaluation.separability_split].rows,
        model.layers,
        bit_length=prepared.shared_bundle.packed_bit_length,
    )
    separability = _compute_separability(
        split_name=prepared.evaluation.separability_split,
        rows=separability_rows,
        labels=prepared.shared_bundle.splits[prepared.evaluation.separability_split].labels,
        bit_length=model.embedding_bit_length,
    )
    latency = _measure_latency(
        rows=prepared.shared_bundle.splits[prepared.evaluation.latency_split].rows,
        predict_many=lambda rows: _predict_deep_rows(rows, model),
        split_name=prepared.evaluation.latency_split,
        warmup_passes=prepared.evaluation.latency_warmup_passes,
        timed_passes=prepared.evaluation.latency_timed_passes,
    )
    return _write_variant_summary(
        prepared,
        variant_id=variant_id,
        variant_dir=variant_dir,
        artifact_path=run_result.artifact_path,
        split_metrics=split_metrics,
        separability=separability,
        latency=latency,
        model_details={
            "decision_path": _DECISION_PATHS[variant_id],
            "scoring_mode": "all_layer_residue_readout",
            "embedding_bit_length": model.embedding_bit_length,
            "layer_count": len(model.layers),
        },
        native_paths={
            "artifact_path": str(run_result.artifact_path.resolve()),
            "metrics_summary_path": str(run_result.metrics_summary_path.resolve()),
            "history_json_path": str((engine_dir / "history.json").resolve()),
            "history_csv_path": str((engine_dir / "history.csv").resolve()),
        },
    )


def _run_lean_lean_variant(
    prepared: PreparedBackendArchitectureComparison,
) -> dict[str, object]:
    variant_id = "LEAN_LEAN"
    variant_dir = prepared.run_root / "variants" / "lean_lean"
    engine_dir = variant_dir / "engine_run"
    run_result = run_lean_evolution(
        prepared.shared_bundle.bundle_dir,
        engine_dir,
        evolution_config=_offset_evolution_seed(
            prepared.search_config,
            seed_offset=prepared.lean_lean_config.seed_offset,
        ),
        lean_config=prepared.lean_training_config,
        backend=prepared.lean_training_config.backend,
        allow_backend_fallback=prepared.lean_training_config.allow_backend_fallback,
        selection_spec=prepared.selection_spec,
        include_test_metrics=False,
    )
    model = _load_lean_artifact_model(run_result.artifact_path)
    split_metrics = {
        split_name: _evaluate_lean_split(
            prepared.shared_bundle.splits[split_name],
            model,
            bit_length=prepared.shared_bundle.packed_bit_length,
        )
        for split_name in ("train", "val", "test")
    }
    separability_rows = _apply_lean_layers(
        prepared.shared_bundle.splits[prepared.evaluation.separability_split].rows,
        model.layers,
        bit_length=prepared.shared_bundle.packed_bit_length,
    )
    separability = _compute_separability(
        split_name=prepared.evaluation.separability_split,
        rows=separability_rows,
        labels=prepared.shared_bundle.splits[prepared.evaluation.separability_split].labels,
        bit_length=prepared.shared_bundle.packed_bit_length,
    )
    latency = _measure_latency(
        rows=prepared.shared_bundle.splits[prepared.evaluation.latency_split].rows,
        predict_many=lambda rows: _predict_lean_rows(
            rows,
            model,
            bit_length=prepared.shared_bundle.packed_bit_length,
        ),
        split_name=prepared.evaluation.latency_split,
        warmup_passes=prepared.evaluation.latency_warmup_passes,
        timed_passes=prepared.evaluation.latency_timed_passes,
    )
    return _write_variant_summary(
        prepared,
        variant_id=variant_id,
        variant_dir=variant_dir,
        artifact_path=run_result.artifact_path,
        split_metrics=split_metrics,
        separability=separability,
        latency=latency,
        model_details={
            "decision_path": _DECISION_PATHS[variant_id],
            "scoring_mode": "final_layer_only",
            "layer_count": len(model.layers),
        },
        native_paths={
            "artifact_path": str(run_result.artifact_path.resolve()),
            "metrics_summary_path": str(run_result.metrics_summary_path.resolve()),
            "history_json_path": str((engine_dir / "history.json").resolve()),
            "history_csv_path": str((engine_dir / "history.csv").resolve()),
        },
    )


def _write_variant_summary(
    prepared: PreparedBackendArchitectureComparison,
    *,
    variant_id: str,
    variant_dir: Path,
    artifact_path: Path,
    split_metrics: Mapping[str, Mapping[str, object]],
    separability: Mapping[str, object],
    latency: Mapping[str, object],
    model_details: Mapping[str, object],
    native_paths: Mapping[str, object],
) -> dict[str, object]:
    summary = {
        "schema_version": _VARIANT_SUMMARY_SCHEMA_VERSION,
        "variant_id": variant_id,
        "label": _VARIANT_LABELS[variant_id],
        "decision_path": _DECISION_PATHS[variant_id],
        "source_profile_path": str(prepared.source_profile_path),
        "shared_frontend": {
            "locked_frontend": prepared.locked_frontend.to_dict(),
            "frontend_input_id": prepared.shared_bundle.frontend_input_id,
            "frontend_fingerprint": prepared.shared_bundle.frontend_fingerprint,
            "semantic_bit_length": prepared.shared_bundle.semantic_bit_length,
            "comparison_bundle_bit_length": prepared.shared_bundle.packed_bit_length,
            "shared_frontend_dir": str(prepared.shared_frontend_dir.resolve()),
        },
        "shared_bundle": {
            "bundle_dir": str(prepared.shared_bundle.bundle_dir.resolve()),
            "bundle_contract_path": str(prepared.shared_bundle.contract_path.resolve()),
        },
        "artifact_path": str(artifact_path.resolve()),
        "model_size_bytes": artifact_path.stat().st_size,
        "metrics": {
            split_name: dict(split_metrics[split_name])
            for split_name in ("train", "val", "test")
        },
        "latency": dict(latency),
        "separability": dict(separability),
        "model_details": dict(model_details),
        "native_paths": dict(native_paths),
    }
    summary_path = variant_dir / "variant_summary.json"
    _write_json(summary_path, summary)
    return {
        **summary,
        "variant_summary_path": str(summary_path.resolve()),
    }


def _build_summary_row(
    variant_result: Mapping[str, object],
    *,
    summary_metric_split: str,
) -> dict[str, object]:
    metrics = variant_result["metrics"]
    summary_metrics = metrics[summary_metric_split]
    return {
        "variant_id": variant_result["variant_id"],
        "label": variant_result["label"],
        "decision_path": variant_result["decision_path"],
        "train_accuracy": metrics["train"]["accuracy"],
        "val_accuracy": metrics["val"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
        "val_healthy_to_unhealthy_fpr": metrics["val"]["healthy_to_unhealthy_fpr"],
        "test_healthy_to_unhealthy_fpr": metrics["test"]["healthy_to_unhealthy_fpr"],
        "val_false_negative_rate": metrics["val"]["false_negative_rate"],
        "test_false_negative_rate": metrics["test"]["false_negative_rate"],
        "val_unhealthy_precision": metrics["val"]["unhealthy_precision"],
        "test_unhealthy_precision": metrics["test"]["unhealthy_precision"],
        "val_unhealthy_recall": metrics["val"]["unhealthy_recall"],
        "test_unhealthy_recall": metrics["test"]["unhealthy_recall"],
        "val_unhealthy_f1": metrics["val"]["unhealthy_f1"],
        "test_unhealthy_f1": metrics["test"]["unhealthy_f1"],
        "val_macro_f1": metrics["val"]["macro_f1"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "summary_metric_split": summary_metric_split,
        "summary_split_accuracy": summary_metrics["accuracy"],
        "summary_split_healthy_to_unhealthy_fpr": summary_metrics["healthy_to_unhealthy_fpr"],
        "summary_split_unhealthy_f1": summary_metrics["unhealthy_f1"],
        "summary_split_macro_f1": summary_metrics["macro_f1"],
        "inference_latency_ms_per_sample": variant_result["latency"]["per_sample_ms"],
        "model_size_bytes": variant_result["model_size_bytes"],
        "residue_separability_split": variant_result["separability"]["split"],
        "intra_class_distance_mean": variant_result["separability"]["intra_class_distance_mean"],
        "inter_class_distance_mean": variant_result["separability"]["inter_class_distance_mean"],
        "separation_margin": variant_result["separability"]["separation_margin"],
        "train_confusion_matrix_json": json.dumps(metrics["train"]["confusion_matrix"], sort_keys=True),
        "val_confusion_matrix_json": json.dumps(metrics["val"]["confusion_matrix"], sort_keys=True),
        "test_confusion_matrix_json": json.dumps(metrics["test"]["confusion_matrix"], sort_keys=True),
    }


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ContractValidationError("Summary CSV requires at least one row.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_summary_markdown(
    prepared: PreparedBackendArchitectureComparison,
    variant_results: Sequence[Mapping[str, object]],
    *,
    summary_metric_split: str,
) -> str:
    by_id = {str(result["variant_id"]): result for result in variant_results}
    front_only = by_id["FRONT_ONLY"]
    lean_deep = by_id["LEAN_DEEP"]
    lean_lean = by_id["LEAN_LEAN"]
    edge_sorted = sorted(
        variant_results,
        key=lambda result: (
            float(result["metrics"]["val"]["healthy_to_unhealthy_fpr"]),
            -float(result["metrics"]["val"]["unhealthy_precision"]),
            -float(result["metrics"]["val"]["unhealthy_recall"]),
            -float(result["metrics"]["val"]["unhealthy_f1"]),
            -float(result["metrics"]["val"]["macro_f1"]),
            int(result["model_size_bytes"]),
            float(result["latency"]["per_sample_ms"]),
            _VARIANT_ORDER.index(str(result["variant_id"])),
        ),
    )
    edge_candidate = edge_sorted[0]
    lines = [
        "# Backend Architecture Comparison",
        "",
        f"Source profile: `{prepared.source_profile_path}`",
        f"Shared locked frontend: `{prepared.locked_frontend.regime_id}` ({prepared.locked_frontend.bit_length} semantic bits, packed to 64 for backend compatibility only)",
        "",
        "## Comparison Table",
        "",
        "| Variant | Val Acc | Test Acc | Val FPR | Test FPR | Test Unhealthy F1 | Test Macro F1 | Size (bytes) | Latency (ms/sample) | Separation Margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_id in _VARIANT_ORDER:
        result = by_id[variant_id]
        lines.append(
            "| {label} | {val_acc:.6f} | {test_acc:.6f} | {val_fpr:.6f} | {test_fpr:.6f} | {test_f1:.6f} | {test_macro:.6f} | {size_bytes} | {latency:.6f} | {sep:.6f} |".format(
                label=result["label"],
                val_acc=float(result["metrics"]["val"]["accuracy"]),
                test_acc=float(result["metrics"]["test"]["accuracy"]),
                val_fpr=float(result["metrics"]["val"]["healthy_to_unhealthy_fpr"]),
                test_fpr=float(result["metrics"]["test"]["healthy_to_unhealthy_fpr"]),
                test_f1=float(result["metrics"]["test"]["unhealthy_f1"]),
                test_macro=float(result["metrics"]["test"]["macro_f1"]),
                size_bytes=int(result["model_size_bytes"]),
                latency=float(result["latency"]["per_sample_ms"]),
                sep=float(result["separability"]["separation_margin"]),
            )
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"- Deep after the shared front Lean {_deep_effect_text(front_only, lean_deep, split=summary_metric_split)} on the `{summary_metric_split}` summary split relative to `FRONT_ONLY`.",
            f"- Lean-Lean {_lean_vs_deep_text(lean_lean, lean_deep)} residue separation versus Lean-Deep when comparing train-space separation margin and final split metrics together.",
            f"- The front Lean residue alone {_front_only_strength_text(front_only, edge_candidate, split=summary_metric_split)} under this controlled comparison.",
            f"- Best edge deployment candidate: `{edge_candidate['variant_id']}` based on validation-led ranking with model size and latency as downstream tie-breakers.",
            "",
            "## Notes",
            "",
            "- All variants consume the same shared locked frontend handoff and the same packed-64 backend comparison bundle.",
            "- The only architecture changes are the backend decision path: direct front residue, deep all-layer residue readout, or second-stage lean final-layer readout.",
            "- Confusion matrices and full split metrics are preserved in each variant folder under `variants/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _helps(
    baseline: Mapping[str, object],
    subject: Mapping[str, object],
    *,
    split: str,
) -> bool:
    baseline_metrics = baseline["metrics"][split]
    subject_metrics = subject["metrics"][split]
    return (
        float(subject_metrics["healthy_to_unhealthy_fpr"]) < float(baseline_metrics["healthy_to_unhealthy_fpr"])
        or float(subject_metrics["unhealthy_f1"]) > float(baseline_metrics["unhealthy_f1"]) + 0.01
        or float(subject_metrics["macro_f1"]) > float(baseline_metrics["macro_f1"]) + 0.01
    )


def _deep_effect_text(
    baseline: Mapping[str, object],
    subject: Mapping[str, object],
    *,
    split: str,
) -> str:
    return "helps" if _helps(baseline, subject, split=split) else "hurts or is negligible"


def _improves_vs_deep(
    lean_lean: Mapping[str, object],
    lean_deep: Mapping[str, object],
) -> bool:
    return (
        float(lean_lean["separability"]["separation_margin"])
        > float(lean_deep["separability"]["separation_margin"]) + 0.01
        and float(lean_lean["metrics"]["test"]["macro_f1"])
        >= float(lean_deep["metrics"]["test"]["macro_f1"]) - 0.01
    )


def _lean_vs_deep_text(
    lean_lean: Mapping[str, object],
    lean_deep: Mapping[str, object],
) -> str:
    return "improves" if _improves_vs_deep(lean_lean, lean_deep) else "does not improve"


def _front_only_is_strong(
    front_only: Mapping[str, object],
    edge_candidate: Mapping[str, object],
    *,
    split: str,
) -> bool:
    front_metrics = front_only["metrics"][split]
    edge_metrics = edge_candidate["metrics"][split]
    return (
        float(front_metrics["healthy_to_unhealthy_fpr"])
        <= float(edge_metrics["healthy_to_unhealthy_fpr"]) + 0.01
        and float(front_metrics["unhealthy_f1"]) + 0.02 >= float(edge_metrics["unhealthy_f1"])
        and float(front_metrics["macro_f1"]) + 0.02 >= float(edge_metrics["macro_f1"])
    )


def _front_only_strength_text(
    front_only: Mapping[str, object],
    edge_candidate: Mapping[str, object],
    *,
    split: str,
) -> str:
    return (
        "looks strong enough that extra backend complexity is mostly noise"
        if _front_only_is_strong(front_only, edge_candidate, split=split)
        else "does not look fully sufficient on its own"
    )


def _evaluate_front_only_split(
    split: _BundleSplit,
    model: _FrontOnlyModel,
) -> dict[str, object]:
    predictions, margins = _predict_front_only_rows(split.rows, model)
    return _binary_metrics(
        split_name=split.split_name,
        labels=split.labels,
        predictions=predictions,
        margins=margins,
    )


def _predict_front_only_rows(
    rows: Sequence[int],
    model: _FrontOnlyModel,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_labels = tuple(_LABEL_TO_INT[label_name] for label_name in _LABEL_ORDER)
    predictions: list[int] = []
    margins: list[int] = []
    for row in rows:
        distances = [
            (label_int, _bit_distance(row, model.prototypes[label_int]))
            for label_int in ordered_labels
        ]
        best_label, best_distance = min(
            distances,
            key=lambda item: (item[1], ordered_labels.index(item[0])),
        )
        ranked = sorted(distances, key=lambda item: (item[1], ordered_labels.index(item[0])))
        margin = 0 if len(ranked) < 2 else int(ranked[1][1] - ranked[0][1])
        predictions.append(int(best_label))
        margins.append(margin)
    return tuple(predictions), tuple(margins)


def _evaluate_lean_split(
    split: _BundleSplit,
    model: _LeanArtifactModel,
    *,
    bit_length: int,
) -> dict[str, object]:
    predictions, margins = _predict_lean_rows(split.rows, model, bit_length=bit_length)
    return _binary_metrics(
        split_name=split.split_name,
        labels=split.labels,
        predictions=predictions,
        margins=margins,
    )


def _predict_lean_rows(
    rows: Sequence[int],
    model: _LeanArtifactModel,
    *,
    bit_length: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    final_rows = _apply_lean_layers(rows, model.layers, bit_length=bit_length)
    predictions: list[int] = []
    margins: list[int] = []
    for row in final_rows:
        ranked = sorted(
            (
                (_bit_distance(row, prototype), model.prototype_labels[index])
                for index, prototype in enumerate(model.prototypes)
            ),
            key=lambda item: (item[0], item[1]),
        )
        best_distance, best_label = ranked[0]
        margin = 0 if len(ranked) < 2 else int(ranked[1][0] - best_distance)
        predictions.append(int(best_label))
        margins.append(margin)
    return tuple(predictions), tuple(margins)


def _apply_lean_layers(
    rows: Sequence[int],
    layers: Sequence[LeanLayer],
    *,
    bit_length: int,
) -> tuple[int, ...]:
    return tuple(_apply_lean_row_layers(row, layers, bit_length=bit_length) for row in rows)


def _apply_lean_row_layers(
    row: int,
    layers: Sequence[LeanLayer],
    *,
    bit_length: int,
) -> int:
    result = row
    mask_all = (1 << bit_length) - 1
    for layer in layers:
        shifted = _rotate_left(result, layer.shift, bit_length=bit_length)
        if layer.op == "not":
            result = (~shifted) & mask_all
            continue
        if layer.mask is None:
            raise ContractValidationError("Lean artifact layers require a mask for non-`not` ops.")
        mask = layer.mask & mask_all
        if layer.op == "xor":
            result = shifted ^ mask
        elif layer.op == "and":
            result = shifted & mask
        elif layer.op == "or":
            result = shifted | mask
        else:
            raise ContractValidationError(f"Unsupported lean artifact op `{layer.op}`.")
    return result


def _evaluate_deep_split(
    split: _BundleSplit,
    model: _DeepArtifactModel,
) -> dict[str, object]:
    predictions, margins = _predict_deep_rows(split.rows, model)
    return _binary_metrics(
        split_name=split.split_name,
        labels=split.labels,
        predictions=predictions,
        margins=margins,
    )


def _predict_deep_rows(
    rows: Sequence[int],
    model: _DeepArtifactModel,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    embeddings = _apply_layers_as_embedding(
        rows,
        model.layers,
        bit_length=_PACKED_BIT_LENGTH,
    )
    predictions: list[int] = []
    margins: list[int] = []
    for embedding in embeddings:
        predicted_label, margin = _deep_predict_row(
            embedding,
            prototypes=model.prototypes,
            prototype_labels=model.prototype_labels,
            class_labels=model.class_labels,
        )
        predictions.append(int(predicted_label))
        margins.append(int(margin))
    return tuple(predictions), tuple(margins)


def _measure_latency(
    *,
    rows: Sequence[int],
    predict_many,
    split_name: str,
    warmup_passes: int,
    timed_passes: int,
) -> dict[str, object]:
    materialized_rows = tuple(int(row) for row in rows)
    for _ in range(warmup_passes):
        predict_many(materialized_rows)
    start = perf_counter()
    total_rows = 0
    for _ in range(timed_passes):
        predict_many(materialized_rows)
        total_rows += len(materialized_rows)
    elapsed_seconds = perf_counter() - start
    per_sample_seconds = elapsed_seconds / float(max(1, total_rows))
    return {
        "split": split_name,
        "warmup_passes": warmup_passes,
        "timed_passes": timed_passes,
        "timed_row_count": total_rows,
        "total_elapsed_seconds": round(elapsed_seconds, 9),
        "per_sample_seconds": round(per_sample_seconds, 9),
        "per_sample_ms": round(per_sample_seconds * 1000.0, 9),
    }


def _compute_separability(
    *,
    split_name: str,
    rows: Sequence[int],
    labels: Sequence[int],
    bit_length: int,
) -> dict[str, object]:
    within_distances: list[float] = []
    between_distances: list[float] = []
    indexed = list(zip(rows, labels, strict=True))
    for left_index in range(len(indexed)):
        left_row, left_label = indexed[left_index]
        for right_index in range(left_index + 1, len(indexed)):
            right_row, right_label = indexed[right_index]
            distance = _bit_distance(left_row, right_row) / float(bit_length)
            if left_label == right_label:
                within_distances.append(distance)
            else:
                between_distances.append(distance)
    intra = sum(within_distances) / float(len(within_distances)) if within_distances else 0.0
    inter = sum(between_distances) / float(len(between_distances)) if between_distances else 0.0
    return {
        "split": split_name,
        "distance_metric": "hamming_xor_popcount",
        "bit_length": bit_length,
        "intra_class_distance_mean": round(intra, 6),
        "inter_class_distance_mean": round(inter, 6),
        "separation_margin": round(inter - intra, 6),
    }


def _load_lean_artifact_model(path: Path) -> _LeanArtifactModel:
    payload = _load_json(path)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ContractValidationError(f"`{path}` is missing a lean `model` mapping.")
    return _LeanArtifactModel(
        layers=tuple(_lean_layer_from_payload(layer) for layer in model.get("layers", ())),
        prototypes=tuple(_lsb0_bitstring_to_int(str(value)) for value in model.get("prototypes", ())),
        prototype_labels=tuple(int(value) for value in model.get("prototype_labels", ())),
    )


def _load_deep_artifact_model(path: Path) -> _DeepArtifactModel:
    payload = _load_json(path)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ContractValidationError(f"`{path}` is missing a deep `model` mapping.")
    embedding_bit_length = model.get("embedding_bit_length", payload.get("embedding_bit_length"))
    if isinstance(embedding_bit_length, bool) or not isinstance(embedding_bit_length, int) or embedding_bit_length < 1:
        raise ContractValidationError(f"`{path}` is missing a positive deep `embedding_bit_length`.")
    return _DeepArtifactModel(
        layers=tuple(_deep_layer_from_payload(layer) for layer in model.get("layers", ())),
        prototypes=tuple(_lsb0_bitstring_to_int(str(value)) for value in model.get("prototypes", ())),
        prototype_labels=tuple(int(value) for value in model.get("prototype_labels", ())),
        class_labels=tuple(int(value) for value in payload.get("class_labels", ())),
        embedding_bit_length=int(embedding_bit_length),
    )


def _lean_layer_from_payload(payload: object) -> LeanLayer:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Lean artifact `model.layers` entries must be mappings.")
    op = payload.get("op")
    shift = payload.get("shift", 0)
    mask_bits = payload.get("mask_bits")
    if not isinstance(op, str) or not op:
        raise ContractValidationError("Lean artifact layers require non-empty `op` values.")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ContractValidationError("Lean artifact layer `shift` values must be integers.")
    mask = None if mask_bits is None else _lsb0_bitstring_to_int(str(mask_bits))
    return LeanLayer(op=op, shift=shift, mask=mask)


def _deep_layer_from_payload(payload: object) -> DeepLayer:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Deep artifact `model.layers` entries must be mappings.")
    op = payload.get("op")
    shift = payload.get("shift", 0)
    mask_bits = payload.get("mask_bits")
    rule = payload.get("rule")
    if not isinstance(op, str) or not op:
        raise ContractValidationError("Deep artifact layers require non-empty `op` values.")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ContractValidationError("Deep artifact layer `shift` values must be integers.")
    mask = None if mask_bits is None else _lsb0_bitstring_to_int(str(mask_bits))
    if rule is not None and (isinstance(rule, bool) or not isinstance(rule, int)):
        raise ContractValidationError("Deep artifact layer `rule` values must be integers.")
    return DeepLayer(op=op, shift=shift, mask=mask, rule=rule)


def _binary_metrics(
    *,
    split_name: str,
    labels: Sequence[int],
    predictions: Sequence[int],
    margins: Sequence[int],
) -> dict[str, object]:
    unhealthy_label = _LABEL_TO_INT["unhealthy"]
    healthy_label = _LABEL_TO_INT["healthy"]
    tp = sum(
        1 for actual, predicted in zip(labels, predictions, strict=True)
        if actual == unhealthy_label and predicted == unhealthy_label
    )
    fp = sum(
        1 for actual, predicted in zip(labels, predictions, strict=True)
        if actual == healthy_label and predicted == unhealthy_label
    )
    tn = sum(
        1 for actual, predicted in zip(labels, predictions, strict=True)
        if actual == healthy_label and predicted == healthy_label
    )
    fn = sum(
        1 for actual, predicted in zip(labels, predictions, strict=True)
        if actual == unhealthy_label and predicted == healthy_label
    )
    total = len(labels)
    unhealthy_precision = tp / max(1, tp + fp)
    unhealthy_recall = tp / max(1, tp + fn)
    unhealthy_f1 = _f1(unhealthy_precision, unhealthy_recall)
    healthy_precision = tn / max(1, tn + fn)
    healthy_recall = tn / max(1, tn + fp)
    healthy_f1 = _f1(healthy_precision, healthy_recall)
    macro_f1 = (healthy_f1 + unhealthy_f1) / 2.0
    accuracy = (tp + tn) / max(1, total)
    mean_margin = sum(margins) / max(1, len(margins))
    healthy_to_unhealthy_fpr = fp / max(1, fp + tn)
    false_negative_rate = fn / max(1, fn + tp)
    return {
        "split": split_name,
        "n_rows": total,
        "accuracy": round(accuracy, 6),
        "mean_margin": round(mean_margin, 6),
        "confusion_matrix": {
            "labels": ["healthy", "unhealthy"],
            "matrix": [[tn, fp], [fn, tp]],
            "counts": {
                "healthy": {"healthy": tn, "unhealthy": fp},
                "unhealthy": {"healthy": fn, "unhealthy": tp},
            },
        },
        "healthy_to_unhealthy_fpr": round(healthy_to_unhealthy_fpr, 6),
        "false_negative_rate": round(false_negative_rate, 6),
        "unhealthy_miss_rate": round(false_negative_rate, 6),
        "unhealthy_precision": round(unhealthy_precision, 6),
        "unhealthy_recall": round(unhealthy_recall, 6),
        "unhealthy_f1": round(unhealthy_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _offset_evolution_seed(
    config: EvolutionConfig,
    *,
    seed_offset: int,
) -> EvolutionConfig:
    return EvolutionConfig(
        seed=int(config.seed) + int(seed_offset),
        generations=config.generations,
        population_size=config.population_size,
        mu=config.mu,
        lam=config.lam,
        elite_count=config.elite_count,
        min_layers=config.min_layers,
        max_layers=config.max_layers,
        mutation_rate=config.mutation_rate,
        mutation_rate_schedule=config.mutation_rate_schedule,
        selection_mode=config.selection_mode,
        tournament_k=config.tournament_k,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint=config.checkpoint,
    )


def _evolution_config_to_dict(config: EvolutionConfig) -> dict[str, object]:
    return {
        "seed": config.seed,
        "generations": config.generations,
        "population_size": config.population_size,
        "mu": config.mu,
        "lam": config.lam,
        "elite_count": config.elite_count,
        "min_layers": config.min_layers,
        "max_layers": config.max_layers,
        "mutation_rate": config.mutation_rate,
        "mutation_rate_schedule": config.mutation_rate_schedule,
        "selection_mode": config.selection_mode,
        "tournament_k": config.tournament_k,
        "early_stopping_patience": config.early_stopping_patience,
        "checkpoint": {},
    }


def _resolve_relative_path(
    *,
    config_file: Path,
    raw_path: object,
    field_name: str,
) -> Path:
    if not isinstance(raw_path, str) or raw_path.strip() == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty path string.")
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (config_file.parent / candidate).resolve()
    if direct.exists():
        return direct
    return (PROJECT_ROOT / candidate).resolve()


def _coerce_waveform_record(
    record: WaveformDatasetRecord | Mapping[str, object],
) -> WaveformDatasetRecord:
    if isinstance(record, WaveformDatasetRecord):
        return record
    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Backend comparison record coercion requires a `WaveformDatasetRecord` or JSON-object mapping."
        )
    if "context" in record:
        context = record.get("context", {})
        if context is None:
            context = {}
        if not isinstance(context, Mapping):
            raise ContractValidationError("Backend comparison waveform record `context` must be a mapping.")
        label_metadata = record.get("label_metadata", {})
        if label_metadata is None:
            label_metadata = {}
        lineage_metadata = record.get("lineage_metadata", {})
        if lineage_metadata is None:
            lineage_metadata = {}
        return WaveformDatasetRecord(
            source_record_id=record.get("source_record_id"),
            split=record.get("split"),
            state_label=record.get("state_label"),
            waveforms=record.get("waveforms", {}),
            label_metadata=label_metadata,
            sampling_hz=context.get("sampling_hz"),
            rpm=context.get("rpm"),
            operating_condition=context.get("operating_condition"),
            context_metadata=context.get("metadata", {}),
            lineage_metadata=lineage_metadata,
        )
    return WaveformDatasetRecord.from_dict(record)


def _materialize_packed64_feature_names(
    feature_names: tuple[str, ...] | None,
    *,
    semantic_bit_length: int,
) -> tuple[str, ...]:
    if semantic_bit_length < 0 or semantic_bit_length > _PACKED_BIT_LENGTH:
        raise ContractValidationError(
            f"Shared backend comparison requires 0 <= semantic bits <= {_PACKED_BIT_LENGTH}; received {semantic_bit_length}."
        )
    base_names = (
        tuple(feature_names)
        if feature_names is not None
        else tuple(f"bit_{index}" for index in range(semantic_bit_length))
    )
    if len(base_names) != semantic_bit_length:
        raise ContractValidationError(
            "Shared backend comparison requires one feature name per semantic frontend bit."
        )
    return base_names + tuple(
        f"padding_zero_bit_{index}"
        for index in range(semantic_bit_length, _PACKED_BIT_LENGTH)
    )


def _resolve_semantic_bit_length(
    original_frontend_bit_length: int | None,
    encoded_bit_length: int | None,
) -> int:
    semantic_bit_length = (
        original_frontend_bit_length
        if original_frontend_bit_length is not None
        else encoded_bit_length
    )
    if semantic_bit_length is None:
        raise ContractValidationError(
            "Shared backend comparison could not determine the frontend semantic bit length."
        )
    if semantic_bit_length > _PACKED_BIT_LENGTH:
        raise ContractValidationError(
            f"Shared backend comparison received {semantic_bit_length} semantic bits; maximum supported is {_PACKED_BIT_LENGTH}."
        )
    return semantic_bit_length


def _mapping_get_optional_int(payload: object, *path: str) -> int | None:
    current = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _read_positive_int(current, default=None)


def _read_positive_int(value: object, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError("Expected a positive integer.")
    return int(value)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _lsb0_bitstring_to_int(bitstring: str) -> int:
    value = 0
    for index, bit in enumerate(bitstring):
        if bit == "1":
            value |= 1 << index
        elif bit != "0":
            raise ContractValidationError("Bitstrings must contain only `0` and `1`.")
    return value


def _rotate_left(value: int, shift: int, *, bit_length: int) -> int:
    if bit_length < 1:
        raise ContractValidationError("`bit_length` must be >= 1.")
    if shift == 0:
        return value & ((1 << bit_length) - 1)
    amount = shift % bit_length
    mask_all = (1 << bit_length) - 1
    return ((value << amount) | (value >> (bit_length - amount))) & mask_all


def _bit_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _select_single_medoid(rows: Sequence[int]) -> int:
    if not rows:
        raise ContractValidationError("Medoid selection requires at least one row.")
    if len(rows) == 1:
        return rows[0]
    best_row = rows[0]
    best_cost: int | None = None
    for candidate in rows:
        cost = sum(_bit_distance(candidate, other) for other in rows)
        if best_cost is None or cost < best_cost or (cost == best_cost and candidate < best_row):
            best_row = candidate
            best_cost = cost
    return best_row


def _hex_row(value: int, *, bit_length: int) -> str:
    hex_width = max(1, (bit_length + 3) // 4)
    return f"{value:0{hex_width}x}"


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return value


def _require_split_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value not in _SUMMARY_SPLITS:
        allowed = ", ".join(sorted(_SUMMARY_SPLITS))
        raise ContractValidationError(f"`{field_name}` must be one of: {allowed}.")
    return value


__all__ = [
    "DEFAULT_RUNS_ROOT",
    "PreparedBackendArchitectureComparison",
    "SearchVariantConfig",
    "_build_summary_row",
    "_evolution_config_to_dict",
    "_load_lean_artifact_model",
    "_materialize_shared_backend_bundle",
    "_parse_evaluation_config",
    "_parse_search_config",
    "_predict_lean_rows",
    "_resolve_relative_path",
    "_run_lean_lean_variant",
    "_selection_spec_from_mapping",
    "_write_json",
    "_write_summary_csv",
]
