"""Controlled consumer-side frontend capacity check for binary bearing runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.v3 import ContractValidationError, WaveformDatasetBundle, WaveformDatasetRecord
from bittrace.v3.artifacts import compute_json_sha256
from bittrace.v3.frontend_encoding import (
    LEGACY_HASH_FRONTEND_KIND,
    PACKED_ROW_FORMAT,
    TEMPORAL_THRESHOLD_FRONTEND_KIND,
    build_legacy_hash_frontend_summary,
    build_temporal_frontend_encoding,
    encode_frontend_record,
)

from bittrace.source.full_binary_campaign import (
    DEFAULT_RUNS_ROOT as FULL_BINARY_DEFAULT_RUNS_ROOT,
    _build_inventory_rows,
    _materialize_source_bundle,
    _sanitize_identifier,
    _validate_inventory_rows,
    load_consumer_config,
)
from bittrace.source.temporal_features import (
    TemporalFeatureConfig,
    build_temporal_feature_payload,
    load_temporal_feature_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experimental" / "frontend_capacity_check.yaml"
DEFAULT_RUNS_ROOT = FULL_BINARY_DEFAULT_RUNS_ROOT
_PACKED_BUNDLE_SCHEMA_VERSION = "bittrace-bearings-v3-1-frontend-capacity-packed-bundle-1"
_PLAN_SCHEMA_VERSION = "bittrace-bearings-v3-1-frontend-capacity-plan-1"
_SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-1-frontend-capacity-summary-1"
_REGIME_REPORT_SCHEMA_VERSION = "bittrace-bearings-v3-1-frontend-capacity-candidate-report-1"
_ENCODED_BUNDLE_SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-1-encoded-bundle-summary-1"
_DOWNSTREAM_EVALUATOR_SCHEMA_VERSION = "bittrace-bearings-v3-1-fixed-evaluator-summary-1"
_LABEL_TO_INT = {"healthy": 0, "unhealthy": 1}
_INT_TO_LABEL = {value: key for key, value in _LABEL_TO_INT.items()}
_LABEL_ORDER = ("healthy", "unhealthy")
_TEMPORAL_THRESHOLD_STRATEGY = "train_quantiles_v1"
_SUPPORTED_ENCODING_REGIMES = {"legacy_hash", "temporal_threshold"}
_FRONTEND_PROXY_METRICS_BASIS = {
    "healthy_unhealthy_margin": "one_minus_validation_false_positive_rate_from_train_medoids",
    "inter_class_separation": "train_class_medoid_hamming_distance_fraction",
    "intra_class_compactness": "one_minus_train_within_class_distance_fraction",
    "bit_balance": "mean_train_bit_balance_score",
    "bit_stability": "one_minus_cross_split_bit_rate_drift",
}
_DOWNSTREAM_RANKING_POLICY = {
    "comparison_type": "lexicographic_mixed_direction",
    "metric_order": [
        {
            "metric": "healthy_to_unhealthy_fpr",
            "priority": "primary",
            "direction": "asc",
            "split": "val",
        },
        {
            "metric": "unhealthy_precision",
            "priority": "secondary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "unhealthy_recall",
            "priority": "tertiary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "unhealthy_f1",
            "priority": "quaternary",
            "direction": "desc",
            "split": "val",
        },
        {
            "metric": "macro_f1",
            "priority": "quinary",
            "direction": "desc",
            "split": "val",
        },
    ],
    "ranking_mode": "frontend_capacity_fixed_downstream",
    "selection_split": "val",
    "selection_basis": "fixed_consumer_side_hamming_medoids",
    "test_split_note": "Test metrics are final-comparison only and are not used to rank regimes.",
}


@dataclass(frozen=True, slots=True)
class FrontendCapacityRegime:
    regime_id: str
    candidate_id: str
    label: str
    encoding_regime: str
    temporal_features_enabled: bool
    bit_length: int
    threshold_strategy: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "regime_id": self.regime_id,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "encoding_regime": self.encoding_regime,
            "temporal_features_enabled": self.temporal_features_enabled,
            "bit_length": self.bit_length,
            "threshold_strategy": self.threshold_strategy,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class FixedEvaluatorConfig:
    kind: str
    selected_k_per_class: int
    distance_metric: str
    tie_break_label_order: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "selected_k_per_class": self.selected_k_per_class,
            "distance_metric": self.distance_metric,
            "tie_break_label_order": list(self.tie_break_label_order),
            "search_free": True,
            "selection_split": "val",
            "final_reporting_split": "test",
        }


@dataclass(frozen=True, slots=True)
class AssessmentThresholds:
    healthy_to_unhealthy_fpr: float
    unhealthy_recall: float
    unhealthy_f1: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "healthy_to_unhealthy_fpr": round(self.healthy_to_unhealthy_fpr, 6),
            "unhealthy_recall": round(self.unhealthy_recall, 6),
            "unhealthy_f1": round(self.unhealthy_f1, 6),
            "macro_f1": round(self.macro_f1, 6),
        }


@dataclass(frozen=True, slots=True)
class PreparedFrontendCapacityCheck:
    config_path: Path
    run_root: Path
    profile_name: str
    source_profile_path: Path
    source_profile_name: str
    plain_source_bundle: WaveformDatasetBundle
    temporal_source_bundle: WaveformDatasetBundle | None
    regimes: tuple[FrontendCapacityRegime, ...]
    fixed_evaluator: FixedEvaluatorConfig
    reference_regime_id: str
    assessment_thresholds: AssessmentThresholds
    include_test_metrics_in_frontend: bool
    shared_row_count: int
    split_counts: Mapping[str, int]
    state_counts: Mapping[str, int]
    temporal_compatibility_drop_count: int
    temporal_compatibility_dropped_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PackedSplit:
    split_name: str
    rows: tuple[int, ...]
    labels: tuple[int, ...]
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PackedBundle:
    bundle_dir: Path
    contract_path: Path
    bit_length: int
    frontend_kind: str
    frontend_input_id: str
    frontend_fingerprint: str
    bit_feature_names: tuple[str, ...]
    splits: Mapping[str, _PackedSplit]


def load_frontend_capacity_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    for key in ("profile_name", "source_profile", "frontend_capacity_check"):
        if key not in payload:
            raise ContractValidationError(f"`{resolved_path}` is missing required top-level key `{key}`.")
    return payload


def prepare_frontend_capacity_check(
    config_path: str | Path,
    run_root: str | Path,
) -> PreparedFrontendCapacityCheck:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = Path(run_root).resolve()
    payload = load_frontend_capacity_config(resolved_config_path)
    capacity = payload["frontend_capacity_check"]
    if not isinstance(capacity, Mapping):
        raise ContractValidationError("`frontend_capacity_check` must be a mapping.")

    source_profile_path = _resolve_relative_path(
        capacity_file=resolved_config_path,
        raw_path=payload["source_profile"],
        field_name="source_profile",
    )
    source_profile = load_consumer_config(source_profile_path)
    source_rows = _build_inventory_rows(source_profile)
    source_temporal_config = load_temporal_feature_config(source_profile)
    regimes = _parse_frontend_regimes(capacity.get("regimes"))
    fixed_evaluator = _parse_fixed_evaluator(capacity.get("fixed_evaluator"))
    assessment_thresholds = _parse_assessment_thresholds(capacity.get("assessment"))
    include_test_metrics_in_frontend = _parse_include_test_metrics(capacity.get("include_test_metrics_in_frontend", False))
    reference_regime_id = _parse_reference_regime_id(
        capacity.get("reference_regime_id"),
        regimes=regimes,
    )

    shared_rows, dropped_ids = _resolve_shared_rows(
        source_rows,
        regimes=regimes,
        temporal_feature_config=source_temporal_config,
    )
    split_counts = Counter(str(row["split"]) for row in shared_rows)
    state_counts = Counter(str(row["binary_label"]) for row in shared_rows)
    disabled_temporal_config = TemporalFeatureConfig(enabled=False)

    plain_source_bundle = _materialize_source_bundle(
        shared_rows,
        output_dir=resolved_run_root / "_inputs" / "shared_plain_source_bundle",
        profile_name=str(payload["profile_name"]),
        selection_name="frontend_capacity_check_shared_plain",
        temporal_feature_config=disabled_temporal_config,
    )
    temporal_source_bundle = None
    if any(regime.temporal_features_enabled for regime in regimes):
        temporal_source_bundle = _materialize_source_bundle(
            shared_rows,
            output_dir=resolved_run_root / "_inputs" / "shared_temporal_source_bundle",
            profile_name=str(payload["profile_name"]),
            selection_name="frontend_capacity_check_shared_temporal",
            temporal_feature_config=source_temporal_config,
        )

    prepared = PreparedFrontendCapacityCheck(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        source_profile_path=source_profile_path,
        source_profile_name=str(source_profile["profile_name"]),
        plain_source_bundle=plain_source_bundle,
        temporal_source_bundle=temporal_source_bundle,
        regimes=regimes,
        fixed_evaluator=fixed_evaluator,
        reference_regime_id=reference_regime_id,
        assessment_thresholds=assessment_thresholds,
        include_test_metrics_in_frontend=include_test_metrics_in_frontend,
        shared_row_count=len(shared_rows),
        split_counts=dict(sorted(split_counts.items())),
        state_counts=dict(sorted(state_counts.items())),
        temporal_compatibility_drop_count=len(dropped_ids),
        temporal_compatibility_dropped_record_ids=tuple(dropped_ids),
    )
    write_frontend_capacity_plan(prepared)
    return prepared


def write_frontend_capacity_plan(prepared: PreparedFrontendCapacityCheck) -> Path:
    path = prepared.run_root / "frontend_capacity_plan.json"
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
            "temporal_compatibility_drop_count": prepared.temporal_compatibility_drop_count,
            "temporal_compatibility_dropped_record_ids": list(
                prepared.temporal_compatibility_dropped_record_ids
            ),
            "discipline": {
                "train": "fit only",
                "val": "ranking and interim comparison only",
                "test": "final comparative reporting only",
            },
        },
        "comparison_dimensions": {
            "bit_length_values": sorted({regime.bit_length for regime in prepared.regimes}),
            "temporal_features_enabled_values": sorted(
                {regime.temporal_features_enabled for regime in prepared.regimes}
            ),
            "encoding_regimes": sorted({regime.encoding_regime for regime in prepared.regimes}),
        },
        "regimes": [regime.to_dict() for regime in prepared.regimes],
        "fixed_evaluator": prepared.fixed_evaluator.to_dict(),
        "reference_regime_id": prepared.reference_regime_id,
        "assessment_thresholds": prepared.assessment_thresholds.to_dict(),
        "ranking_policy": dict(_DOWNSTREAM_RANKING_POLICY),
    }
    _write_json(path, payload)
    return path


def run_frontend_capacity_check(
    config_path: str | Path,
    run_root: str | Path,
) -> Path:
    prepared = prepare_frontend_capacity_check(config_path, run_root)
    return run_prepared_frontend_capacity_check(prepared)


def run_prepared_frontend_capacity_check(
    prepared: PreparedFrontendCapacityCheck,
) -> Path:
    regime_summaries = [
        _run_regime(prepared, regime, order=index)
        for index, regime in enumerate(prepared.regimes, start=1)
    ]
    summary_payload = _build_run_summary(prepared, regime_summaries)
    summary_path = prepared.run_root / "frontend_capacity_summary.json"
    _write_json(summary_path, summary_payload)
    return summary_path


def _resolve_relative_path(
    *,
    capacity_file: Path,
    raw_path: object,
    field_name: str,
) -> Path:
    if not isinstance(raw_path, str) or raw_path.strip() == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty path string.")
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (capacity_file.parent / candidate).resolve()
    if direct.exists():
        return direct
    return (PROJECT_ROOT / candidate).resolve()


def _parse_frontend_regimes(raw: object) -> tuple[FrontendCapacityRegime, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractValidationError("`frontend_capacity_check.regimes` must be a sequence.")
    if len(raw) < 2:
        raise ContractValidationError(
            "`frontend_capacity_check.regimes` must include at least two bounded comparison regimes."
        )
    if len(raw) > 8:
        raise ContractValidationError(
            "`frontend_capacity_check.regimes` must stay bounded; use at most 8 regimes in this slice."
        )
    regimes: list[FrontendCapacityRegime] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ContractValidationError(
                f"`frontend_capacity_check.regimes[{index - 1}]` must be a mapping."
            )
        regime_id = _require_identifier(
            item.get("regime_id"),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].regime_id",
        )
        if regime_id in seen_ids:
            raise ContractValidationError(f"Duplicate frontend regime id `{regime_id}`.")
        seen_ids.add(regime_id)
        label = _require_non_empty_string(
            item.get("label", regime_id),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].label",
        )
        encoding_regime = _require_non_empty_string(
            item.get("encoding_regime"),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].encoding_regime",
        ).strip()
        if encoding_regime not in _SUPPORTED_ENCODING_REGIMES:
            allowed = ", ".join(sorted(_SUPPORTED_ENCODING_REGIMES))
            raise ContractValidationError(
                f"`frontend_capacity_check.regimes[{index - 1}].encoding_regime` must be one of: {allowed}."
            )
        temporal_features_enabled = _require_bool(
            item.get("temporal_features_enabled"),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].temporal_features_enabled",
        )
        bit_length = _require_int(
            item.get("bit_length"),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].bit_length",
            minimum=1,
        )
        threshold_strategy = item.get("threshold_strategy")
        if threshold_strategy is not None:
            threshold_strategy = _require_non_empty_string(
                threshold_strategy,
                field_name=f"frontend_capacity_check.regimes[{index - 1}].threshold_strategy",
            )
        notes = _normalize_notes(
            item.get("notes", ()),
            field_name=f"frontend_capacity_check.regimes[{index - 1}].notes",
        )
        if encoding_regime == "legacy_hash":
            if temporal_features_enabled:
                raise ContractValidationError(
                    f"`{regime_id}` cannot enable temporal features with `encoding_regime=legacy_hash`."
                )
            if bit_length != 64:
                raise ContractValidationError(
                    f"`{regime_id}` must use `bit_length: 64` for the legacy hash frontend."
                )
            if threshold_strategy not in (None, "legacy_hash"):
                raise ContractValidationError(
                    f"`{regime_id}` cannot set a temporal threshold strategy for the legacy hash frontend."
                )
            threshold_strategy = None
        else:
            if not temporal_features_enabled:
                raise ContractValidationError(
                    f"`{regime_id}` must enable temporal features for `encoding_regime=temporal_threshold`."
                )
            threshold_strategy = threshold_strategy or _TEMPORAL_THRESHOLD_STRATEGY
            if threshold_strategy != _TEMPORAL_THRESHOLD_STRATEGY:
                raise ContractValidationError(
                    "This slice supports only `threshold_strategy: train_quantiles_v1` for temporal thresholds."
                )
        regimes.append(
            FrontendCapacityRegime(
                regime_id=regime_id,
                candidate_id=f"frontend-capacity-{index:02d}-{regime_id}",
                label=label,
                encoding_regime=encoding_regime,
                temporal_features_enabled=temporal_features_enabled,
                bit_length=bit_length,
                threshold_strategy=threshold_strategy,
                notes=notes,
            )
        )
    return tuple(regimes)


def _parse_fixed_evaluator(raw: object) -> FixedEvaluatorConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`frontend_capacity_check.fixed_evaluator` must be a mapping.")
    kind = _require_non_empty_string(
        raw.get("kind", "train_class_medoids"),
        field_name="frontend_capacity_check.fixed_evaluator.kind",
    )
    if kind != "train_class_medoids":
        raise ContractValidationError(
            "This slice supports only `fixed_evaluator.kind: train_class_medoids`."
        )
    selected_k_per_class = _require_int(
        raw.get("selected_k_per_class", 1),
        field_name="frontend_capacity_check.fixed_evaluator.selected_k_per_class",
        minimum=1,
    )
    if selected_k_per_class != 1:
        raise ContractValidationError(
            "This slice keeps the evaluator fixed and minimal; only `selected_k_per_class: 1` is supported."
        )
    tie_break_label_order = raw.get("tie_break_label_order", list(_LABEL_ORDER))
    if not isinstance(tie_break_label_order, Sequence) or isinstance(
        tie_break_label_order,
        (str, bytes, bytearray),
    ):
        raise ContractValidationError(
            "`frontend_capacity_check.fixed_evaluator.tie_break_label_order` must be a sequence."
        )
    normalized_order = tuple(str(value) for value in tie_break_label_order)
    if normalized_order != _LABEL_ORDER:
        raise ContractValidationError(
            "This slice fixes tie-break order to `['healthy', 'unhealthy']` for deterministic conservative evaluation."
        )
    return FixedEvaluatorConfig(
        kind=kind,
        selected_k_per_class=selected_k_per_class,
        distance_metric="hamming",
        tie_break_label_order=normalized_order,
    )


def _parse_assessment_thresholds(raw: object) -> AssessmentThresholds:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`frontend_capacity_check.assessment` must be a mapping.")
    meaningful_delta = raw.get("meaningful_delta", {})
    if meaningful_delta is None:
        meaningful_delta = {}
    if not isinstance(meaningful_delta, Mapping):
        raise ContractValidationError(
            "`frontend_capacity_check.assessment.meaningful_delta` must be a mapping."
        )
    return AssessmentThresholds(
        healthy_to_unhealthy_fpr=_require_probability_threshold(
            meaningful_delta.get("healthy_to_unhealthy_fpr", 0.05),
            field_name="frontend_capacity_check.assessment.meaningful_delta.healthy_to_unhealthy_fpr",
        ),
        unhealthy_recall=_require_probability_threshold(
            meaningful_delta.get("unhealthy_recall", 0.05),
            field_name="frontend_capacity_check.assessment.meaningful_delta.unhealthy_recall",
        ),
        unhealthy_f1=_require_probability_threshold(
            meaningful_delta.get("unhealthy_f1", 0.05),
            field_name="frontend_capacity_check.assessment.meaningful_delta.unhealthy_f1",
        ),
        macro_f1=_require_probability_threshold(
            meaningful_delta.get("macro_f1", 0.05),
            field_name="frontend_capacity_check.assessment.meaningful_delta.macro_f1",
        ),
    )


def _parse_reference_regime_id(
    raw: object,
    *,
    regimes: Sequence[FrontendCapacityRegime],
) -> str:
    regime_ids = {regime.regime_id for regime in regimes}
    if raw is None:
        return regimes[0].regime_id
    value = _require_identifier(
        raw,
        field_name="frontend_capacity_check.reference_regime_id",
    )
    if value not in regime_ids:
        allowed = ", ".join(sorted(regime_ids))
        raise ContractValidationError(
            f"`frontend_capacity_check.reference_regime_id` must reference one of: {allowed}."
        )
    return value


def _parse_include_test_metrics(raw: object) -> bool:
    return _require_bool(
        raw,
        field_name="frontend_capacity_check.include_test_metrics_in_frontend",
    )


def _resolve_shared_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    regimes: Sequence[FrontendCapacityRegime],
    temporal_feature_config: TemporalFeatureConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        raise ContractValidationError("The source profile resolved zero usable rows.")
    requires_temporal = any(regime.temporal_features_enabled for regime in regimes)
    if not requires_temporal:
        shared_rows = [dict(row) for row in rows]
        _validate_inventory_rows(shared_rows)
        return shared_rows, []
    if not temporal_feature_config.enabled:
        raise ContractValidationError(
            "At least one frontend capacity regime requires temporal features, but the source profile does not enable them."
        )
    shared_rows: list[dict[str, Any]] = []
    dropped_record_ids: list[str] = []
    for row in rows:
        row_copy = dict(row)
        try:
            build_temporal_feature_payload(row_copy["path"], config=temporal_feature_config)
        except ContractValidationError:
            dropped_record_ids.append(_row_source_record_id(row_copy))
            continue
        shared_rows.append(row_copy)
    _validate_inventory_rows(shared_rows)
    return shared_rows, dropped_record_ids


def _run_regime(
    prepared: PreparedFrontendCapacityCheck,
    regime: FrontendCapacityRegime,
    *,
    order: int,
) -> dict[str, object]:
    regime_dir = prepared.run_root / "regimes" / regime.regime_id
    source_bundle = (
        prepared.temporal_source_bundle
        if regime.temporal_features_enabled
        else prepared.plain_source_bundle
    )
    if source_bundle is None:  # pragma: no cover - guarded by prepare
        raise ContractValidationError(f"Regime `{regime.regime_id}` requires a missing temporal source bundle.")
    frontend_materialization = _materialize_regime_frontend(
        source_bundle,
        regime=regime,
        regime_dir=regime_dir,
        include_test_metrics_in_frontend=prepared.include_test_metrics_in_frontend,
    )
    packed_bundle = _materialize_packed_bundle(
        regime_dir=regime_dir,
        source_bundle=source_bundle,
        deep_input=frontend_materialization["downstream_deep_input"],
        frontend_encoding=frontend_materialization["frontend_encoding"],
        frontend_kind=str(frontend_materialization["frontend_kind"]),
    )
    encoded_bundle_summary = _summarize_packed_bundle(packed_bundle)
    encoded_bundle_summary_path = regime_dir / "encoded_bundle_summary.json"
    _write_json(
        encoded_bundle_summary_path,
        {
            "schema_version": _ENCODED_BUNDLE_SUMMARY_SCHEMA_VERSION,
            "regime_id": regime.regime_id,
            "candidate_id": regime.candidate_id,
            **encoded_bundle_summary,
        },
    )
    downstream_summary = _run_fixed_evaluator(
        packed_bundle,
        evaluator_config=prepared.fixed_evaluator,
    )
    downstream_summary_path = regime_dir / "downstream_evaluator_summary.json"
    _write_json(
        downstream_summary_path,
        {
            "schema_version": _DOWNSTREAM_EVALUATOR_SCHEMA_VERSION,
            "regime_id": regime.regime_id,
            "candidate_id": regime.candidate_id,
            **downstream_summary,
        },
    )
    regime_report_path = regime_dir / "bt3.frontend_candidate_report.json"
    regime_report = {
        "schema_version": _REGIME_REPORT_SCHEMA_VERSION,
        "regime_id": regime.regime_id,
        "candidate_id": regime.candidate_id,
        "candidate_order": order,
        "comparison_dimensions": regime.to_dict(),
        "split_discipline": {
            "train": "encoder fit statistics and fixed evaluator prototypes only",
            "val": "regime ranking and interim comparison only",
            "test": "final comparative reporting only",
        },
        "frontend_proxy_metrics": dict(frontend_materialization["frontend_proxy_metrics"]),
        "frontend_proxy_metrics_basis": dict(_FRONTEND_PROXY_METRICS_BASIS),
        "frontend_materialization": {
            "materialization_mode": frontend_materialization["materialization_mode"],
            "frontend_kind": frontend_materialization["frontend_kind"],
            "frontend_input_id": frontend_materialization["frontend_input_id"],
            "frontend_fingerprint": frontend_materialization["frontend_fingerprint"],
            "frontend_config_fingerprint": frontend_materialization["frontend_config_fingerprint"],
            "source_bundle": frontend_materialization["source_bundle"],
            "frontend_encoding": frontend_materialization["frontend_encoding"],
            "frontend_summary": frontend_materialization["frontend_summary"],
        },
        "downstream_deep_input": frontend_materialization["downstream_deep_input"].to_dict(),
        "encoded_bundle_summary_path": str(encoded_bundle_summary_path.resolve()),
        "downstream_evaluator_summary_path": str(downstream_summary_path.resolve()),
        "downstream_validation_metrics": dict(downstream_summary["val_metrics"]),
        "downstream_test_metrics": dict(downstream_summary["test_metrics"]),
    }
    _write_json(regime_report_path, regime_report)
    return {
        "regime_id": regime.regime_id,
        "candidate_id": regime.candidate_id,
        "candidate_order": order,
        "comparison_dimensions": regime.to_dict(),
        "frontend_proxy_metrics": dict(frontend_materialization["frontend_proxy_metrics"]),
        "frontend_materialization_mode": frontend_materialization["materialization_mode"],
        "frontend_kind": frontend_materialization["frontend_kind"],
        "frontend_input_id": frontend_materialization["frontend_input_id"],
        "frontend_fingerprint": frontend_materialization["frontend_fingerprint"],
        "frontend_config_fingerprint": frontend_materialization["frontend_config_fingerprint"],
        "frontend_encoding": frontend_materialization["frontend_encoding"],
        "frontend_summary": frontend_materialization["frontend_summary"],
        "encoded_bundle_summary": encoded_bundle_summary,
        "downstream_evaluator": {
            "config": prepared.fixed_evaluator.to_dict(),
            "summary_path": str(downstream_summary_path.resolve()),
            **downstream_summary,
        },
        "regime_report_path": str(regime_report_path.resolve()),
        "effective_deep_input_dir": str(
            Path(frontend_materialization["downstream_deep_input"].bundle_dir).resolve()
        ),
        "packed_bundle_dir": str(packed_bundle.bundle_dir.resolve()),
        "ranking_metrics": dict(downstream_summary["val_ranking_metrics"]),
    }


def _coerce_frontend_capacity_waveform_record(
    record: WaveformDatasetRecord | Mapping[str, object],
) -> WaveformDatasetRecord:
    if isinstance(record, WaveformDatasetRecord):
        return record
    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Frontend capacity record coercion requires a `WaveformDatasetRecord` or JSON-object mapping."
        )
    if "context" in record:
        context = record.get("context", {})
        if context is None:
            context = {}
        if not isinstance(context, Mapping):
            raise ContractValidationError("Frontend capacity waveform record `context` must be a mapping.")
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


def _materialize_regime_frontend(
    source_bundle: WaveformDatasetBundle,
    *,
    regime: FrontendCapacityRegime,
    regime_dir: Path,
    include_test_metrics_in_frontend: bool,
) -> dict[str, object]:
    frontend_input_id = f"frontend-capacity-{regime.regime_id}-{source_bundle.bundle_fingerprint[:12]}"
    frontend_encoding: dict[str, object] | None = None
    frontend_kind: str
    frontend_summary: dict[str, object]
    legacy_records: tuple[WaveformDatasetRecord, ...] | None = None
    legacy_record_payloads: tuple[dict[str, object], ...] | None = None
    materialization_mode: str
    if regime.encoding_regime == "legacy_hash":
        legacy_records = tuple(
            _coerce_frontend_capacity_waveform_record(record)
            for record in source_bundle.canonical_records
        )
        legacy_record_payloads = tuple(record.to_dict() for record in legacy_records)
        frontend_kind = LEGACY_HASH_FRONTEND_KIND
        frontend_summary = build_legacy_hash_frontend_summary(
            legacy_record_payloads,
            dataset_id=source_bundle.dataset_id,
            adapter_profile_id=source_bundle.adapter_profile_id,
            frontend_input_id=frontend_input_id,
            frontend_fingerprint="legacy-hash-bootstrap",
            bit_length=64,
        )
        materialization_mode = "legacy_hash_waveform_bundle"
    else:
        temporal_frontend = build_temporal_frontend_encoding(
            source_bundle.canonical_records,
            bit_length=regime.bit_length,
        )
        frontend_encoding = dict(temporal_frontend["encoder"])
        frontend_summary = dict(temporal_frontend["summary"])
        frontend_kind = str(frontend_encoding["frontend_kind"])
        materialization_mode = "temporal_threshold_waveform_bundle"

    frontend_config_fingerprint = compute_json_sha256(
        {
            "regime": regime.to_dict(),
            "materialization_mode": materialization_mode,
            "source_bundle_fingerprint": source_bundle.bundle_fingerprint,
            "include_test_metrics_in_frontend": include_test_metrics_in_frontend,
            "frontend_encoding_fingerprint": (
                frontend_encoding.get("encoder_fingerprint")
                if frontend_encoding is not None
                else None
            ),
        }
    )
    frontend_fingerprint = compute_json_sha256(
        {
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "frontend_input_id": frontend_input_id,
            "source_bundle_fingerprint": source_bundle.bundle_fingerprint,
            "materialization_mode": materialization_mode,
            "frontend_kind": frontend_kind,
            "regime_id": regime.regime_id,
            "frontend_summary_ranking_metrics": frontend_summary.get("ranking_metrics"),
        }
    )
    if regime.encoding_regime == "legacy_hash":
        assert legacy_record_payloads is not None
        frontend_summary = build_legacy_hash_frontend_summary(
            legacy_record_payloads,
            dataset_id=source_bundle.dataset_id,
            adapter_profile_id=source_bundle.adapter_profile_id,
            frontend_input_id=frontend_input_id,
            frontend_fingerprint=frontend_fingerprint,
            bit_length=64,
        )
        frontend_encoding = {
            "frontend_kind": LEGACY_HASH_FRONTEND_KIND,
            "bit_length": 64,
            "row_format": PACKED_ROW_FORMAT,
            "bit_feature_names": [f"bit_{index}" for index in range(64)],
            "encoder_fingerprint": frontend_config_fingerprint,
        }

    deep_input = source_bundle.materialize_deep_input_ref(
        output_dir=regime_dir / "effective_deep_input",
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=frontend_fingerprint,
        include_test_metrics=include_test_metrics_in_frontend,
        extra_handoff_fields=(
            {"frontend_encoding": frontend_encoding}
            if frontend_encoding is not None and regime.encoding_regime != "legacy_hash"
            else None
        ),
        extra_contract_fields=(
            {"frontend_encoding": frontend_encoding}
            if frontend_encoding is not None and regime.encoding_regime != "legacy_hash"
            else None
        ),
    )
    return {
        "materialization_mode": materialization_mode,
        "frontend_kind": frontend_kind,
        "frontend_input_id": frontend_input_id,
        "frontend_fingerprint": frontend_fingerprint,
        "frontend_config_fingerprint": frontend_config_fingerprint,
        "frontend_encoding": frontend_encoding,
        "frontend_summary": frontend_summary,
        "frontend_proxy_metrics": dict(frontend_summary["ranking_metrics"]),
        "downstream_deep_input": deep_input,
        "source_bundle": {
            "bundle_dir": str(source_bundle.bundle_dir.resolve()),
            "bundle_contract_path": str(source_bundle.bundle_contract_path.resolve()),
            "bundle_fingerprint": source_bundle.bundle_fingerprint,
            "source_handoff_manifest_path": str(source_bundle.handoff_manifest_path.resolve()),
        },
    }


def _materialize_packed_bundle(
    *,
    regime_dir: Path,
    source_bundle: WaveformDatasetBundle,
    deep_input,
    frontend_encoding: Mapping[str, object] | None,
    frontend_kind: str,
) -> _PackedBundle:
    bundle_dir = regime_dir / "packed_bundle"
    contract_payload = _load_json(Path(deep_input.bundle_contract_path))
    handoff_payload = _load_json(Path(deep_input.handoff_manifest_path))
    split_payloads = {
        split_name: {"X_packed": [], "y": [], "source_record_ids": []}
        for split_name in ("train", "val", "test")
    }
    feature_names: tuple[str, ...] | None = None
    bit_length: int | None = None
    for record_payload in handoff_payload["records"]:
        record = _coerce_frontend_capacity_waveform_record(record_payload)
        split_name = record.split.lower()
        if split_name not in split_payloads:
            raise ContractValidationError(
                f"Frontend capacity packed bundle encountered unsupported split `{record.split}`."
            )
        state_label = record.state_label.lower()
        if state_label not in _LABEL_TO_INT:
            raise ContractValidationError(
                "Frontend capacity packed bundle supports only `healthy` and `unhealthy` labels."
            )
        encoded = encode_frontend_record(
            record,
            dataset_id=source_bundle.dataset_id,
            adapter_profile_id=source_bundle.adapter_profile_id,
            frontend_input_id=str(deep_input.frontend_input_id),
            frontend_fingerprint=str(deep_input.frontend_fingerprint),
            contract_payload=contract_payload,
        )
        if feature_names is None:
            feature_names = tuple(encoded.bit_feature_names)
            bit_length = int(encoded.bit_length)
        split_payloads[split_name]["X_packed"].append(int(encoded.packed_row_int))
        split_payloads[split_name]["y"].append(_LABEL_TO_INT[state_label])
        split_payloads[split_name]["source_record_ids"].append(record.source_record_id)
    split_refs: dict[str, _PackedSplit] = {}
    for split_name, payload in split_payloads.items():
        if not payload["X_packed"]:
            raise ContractValidationError(
                f"Frontend capacity packed bundle is missing rows for `{split_name}`."
            )
        _write_json(bundle_dir / f"{split_name}_bits.json", payload)
        split_refs[split_name] = _PackedSplit(
            split_name=split_name,
            rows=tuple(int(value) for value in payload["X_packed"]),
            labels=tuple(int(value) for value in payload["y"]),
            source_record_ids=tuple(str(value) for value in payload["source_record_ids"]),
        )
    packed_contract = {
        "schema_version": _PACKED_BUNDLE_SCHEMA_VERSION,
        "row_format": PACKED_ROW_FORMAT,
        "frontend_kind": frontend_kind,
        "bit_length": bit_length if bit_length is not None else 64,
        "feature_names": list(feature_names) if feature_names is not None else [],
        "frontend_input_id": str(deep_input.frontend_input_id),
        "frontend_fingerprint": str(deep_input.frontend_fingerprint),
        "label_mapping": dict(_LABEL_TO_INT),
        "split_semantics": {
            "train": "fixed evaluator fitting only",
            "val": "comparative ranking only",
            "test": "final comparative reporting only",
        },
        "source_bundle_contract_path": str(Path(deep_input.source_bundle_contract_path).resolve()),
        "effective_deep_input_contract_path": str(Path(deep_input.bundle_contract_path).resolve()),
        "frontend_encoding": (
            dict(frontend_encoding) if frontend_encoding is not None else None
        ),
    }
    contract_path = bundle_dir / "contract.json"
    _write_json(contract_path, packed_contract)
    return _PackedBundle(
        bundle_dir=bundle_dir,
        contract_path=contract_path,
        bit_length=int(packed_contract["bit_length"]),
        frontend_kind=frontend_kind,
        frontend_input_id=str(deep_input.frontend_input_id),
        frontend_fingerprint=str(deep_input.frontend_fingerprint),
        bit_feature_names=tuple(str(value) for value in packed_contract["feature_names"]),
        splits=split_refs,
    )


def _summarize_packed_bundle(bundle: _PackedBundle) -> dict[str, object]:
    split_counts = {
        split_name: len(split.rows)
        for split_name, split in bundle.splits.items()
    }
    state_counts = Counter(
        _INT_TO_LABEL[label]
        for split in bundle.splits.values()
        for label in split.labels
    )
    train_split = bundle.splits["train"]
    all_rows = tuple(
        row
        for split in bundle.splits.values()
        for row in split.rows
    )
    split_unique_row_rate = {
        split_name: round(_unique_row_rate(split.rows), 6)
        for split_name, split in bundle.splits.items()
    }
    train_label_groups = {
        label_name: tuple(
            row
            for row, label in zip(train_split.rows, train_split.labels, strict=True)
            if label == _LABEL_TO_INT[label_name]
        )
        for label_name in _LABEL_ORDER
    }
    between_pairs = [
        _hamming_fraction(left, right, bit_length=bundle.bit_length)
        for left in train_label_groups["healthy"]
        for right in train_label_groups["unhealthy"]
    ]
    within_pairs = {
        label_name: _pairwise_hamming_mean(
            rows,
            bit_length=bundle.bit_length,
        )
        for label_name, rows in train_label_groups.items()
    }
    return {
        "packed_bundle_contract_path": str(bundle.contract_path.resolve()),
        "packed_bundle_dir": str(bundle.bundle_dir.resolve()),
        "frontend_kind": bundle.frontend_kind,
        "bit_length": bundle.bit_length,
        "bit_feature_count": len(bundle.bit_feature_names),
        "bit_feature_names": list(bundle.bit_feature_names),
        "split_counts": dict(sorted(split_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "bundle_diversity": {
            "overall_unique_row_rate": round(_unique_row_rate(all_rows), 6),
            "split_unique_row_rate": split_unique_row_rate,
        },
        "bit_statistics": {
            "bit_balance": round(_bit_balance(train_split.rows, bit_length=bundle.bit_length), 6),
            "bit_stability": round(
                _bit_stability(
                    {
                        split_name: split.rows
                        for split_name, split in bundle.splits.items()
                    },
                    bit_length=bundle.bit_length,
                ),
                6,
            ),
        },
        "hamming_spread": {
            "train_pairwise_mean_fraction": round(
                _pairwise_hamming_mean(train_split.rows, bit_length=bundle.bit_length),
                6,
            ),
            "train_within_class_mean_fraction": {
                label_name: round(value, 6)
                for label_name, value in within_pairs.items()
            },
            "train_between_class_mean_fraction": round(
                sum(between_pairs) / float(len(between_pairs)) if between_pairs else 0.0,
                6,
            ),
        },
    }


def _run_fixed_evaluator(
    bundle: _PackedBundle,
    *,
    evaluator_config: FixedEvaluatorConfig,
) -> dict[str, object]:
    train_split = bundle.splits["train"]
    class_prototypes: dict[int, int] = {}
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
        if not class_rows:
            raise ContractValidationError(
                f"Fixed evaluator could not find any `{label_name}` train rows."
            )
        medoid_value = _select_single_medoid(
            tuple(row for row, _ in class_rows),
            bit_length=bundle.bit_length,
        )
        class_prototypes[label_int] = medoid_value
        prototype_record_ids[label_int] = next(
            source_record_id
            for row, source_record_id in class_rows
            if row == medoid_value
        )

    train_predictions, train_margins = _predict_split(
        train_split.rows,
        prototypes=class_prototypes,
        bit_length=bundle.bit_length,
    )
    val_predictions, val_margins = _predict_split(
        bundle.splits["val"].rows,
        prototypes=class_prototypes,
        bit_length=bundle.bit_length,
    )
    test_predictions, test_margins = _predict_split(
        bundle.splits["test"].rows,
        prototypes=class_prototypes,
        bit_length=bundle.bit_length,
    )
    train_metrics = _binary_metrics(
        split_name="train",
        labels=train_split.labels,
        predictions=train_predictions,
        margins=train_margins,
    )
    val_metrics = _binary_metrics(
        split_name="val",
        labels=bundle.splits["val"].labels,
        predictions=val_predictions,
        margins=val_margins,
    )
    test_metrics = _binary_metrics(
        split_name="test",
        labels=bundle.splits["test"].labels,
        predictions=test_predictions,
        margins=test_margins,
    )
    val_ranking_metrics = _ranking_metrics_from_split_metrics(val_metrics)
    return {
        "fixed_evaluator": evaluator_config.to_dict(),
        "ranking_policy": dict(_DOWNSTREAM_RANKING_POLICY),
        "prototypes": {
            _INT_TO_LABEL[label_int]: {
                "packed_row_hex": _hex_row(value, bit_length=bundle.bit_length),
                "packed_row_int": value,
                "source_record_id": prototype_record_ids[label_int],
            }
            for label_int, value in sorted(class_prototypes.items())
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_ranking_metrics": val_ranking_metrics,
    }


def _predict_split(
    rows: Sequence[int],
    *,
    prototypes: Mapping[int, int],
    bit_length: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_labels = tuple(_LABEL_TO_INT[label_name] for label_name in _LABEL_ORDER)
    predictions: list[int] = []
    margins: list[int] = []
    for row in rows:
        distances = [
            (
                label_int,
                _bit_distance(row, prototypes[label_int]),
            )
            for label_int in ordered_labels
        ]
        best_label, best_distance = min(
            distances,
            key=lambda item: (
                item[1],
                ordered_labels.index(item[0]),
            ),
        )
        sorted_distances = sorted(distances, key=lambda item: (item[1], ordered_labels.index(item[0])))
        margin = 0
        if len(sorted_distances) >= 2:
            margin = int(sorted_distances[1][1] - sorted_distances[0][1])
        predictions.append(int(best_label))
        margins.append(margin)
    return tuple(predictions), tuple(margins)


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
        1
        for actual, predicted in zip(labels, predictions, strict=True)
        if actual == unhealthy_label and predicted == unhealthy_label
    )
    fp = sum(
        1
        for actual, predicted in zip(labels, predictions, strict=True)
        if actual == healthy_label and predicted == unhealthy_label
    )
    tn = sum(
        1
        for actual, predicted in zip(labels, predictions, strict=True)
        if actual == healthy_label and predicted == healthy_label
    )
    fn = sum(
        1
        for actual, predicted in zip(labels, predictions, strict=True)
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
        "unhealthy_precision": round(unhealthy_precision, 6),
        "unhealthy_recall": round(unhealthy_recall, 6),
        "unhealthy_f1": round(unhealthy_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _build_run_summary(
    prepared: PreparedFrontendCapacityCheck,
    regime_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ranked_regimes = sorted(
        regime_summaries,
        key=lambda summary: _ranking_sort_key(
            summary["ranking_metrics"],
            candidate_order=int(summary["candidate_order"]),
        ),
    )
    ranking_table = [
        {
            "rank": rank,
            "regime_id": str(summary["regime_id"]),
            "candidate_id": str(summary["candidate_id"]),
            "validation_ranking_metrics": dict(summary["ranking_metrics"]),
            "validation_metrics": dict(summary["downstream_evaluator"]["val_metrics"]),
            "test_metrics": dict(summary["downstream_evaluator"]["test_metrics"]),
        }
        for rank, summary in enumerate(ranked_regimes, start=1)
    ]
    best_regime = ranked_regimes[0]
    reference_regime = next(
        summary
        for summary in regime_summaries
        if str(summary["regime_id"]) == prepared.reference_regime_id
    )
    assessment = _assess_frontend_capacity(
        best_regime=best_regime,
        reference_regime=reference_regime,
        all_regimes=regime_summaries,
        thresholds=prepared.assessment_thresholds,
    )
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "profile_name": prepared.profile_name,
        "config_path": str(prepared.config_path),
        "run_root": str(prepared.run_root),
        "source_profile_path": str(prepared.source_profile_path),
        "source_profile_name": prepared.source_profile_name,
        "consumer_side_only": True,
        "shared_dataset": {
            "row_count": prepared.shared_row_count,
            "split_counts": dict(prepared.split_counts),
            "state_counts": dict(prepared.state_counts),
            "temporal_compatibility_drop_count": prepared.temporal_compatibility_drop_count,
            "temporal_compatibility_dropped_record_ids": list(
                prepared.temporal_compatibility_dropped_record_ids
            ),
            "discipline": {
                "train": "fit only",
                "val": "ranking and interim comparison only",
                "test": "final comparative reporting only",
            },
        },
        "comparison_dimensions": {
            "bit_length_values": sorted({int(summary["comparison_dimensions"]["bit_length"]) for summary in regime_summaries}),
            "temporal_features_enabled_values": sorted(
                {
                    bool(summary["comparison_dimensions"]["temporal_features_enabled"])
                    for summary in regime_summaries
                }
            ),
            "encoding_regimes": sorted(
                {
                    str(summary["comparison_dimensions"]["encoding_regime"])
                    for summary in regime_summaries
                }
            ),
        },
        "fixed_evaluator": prepared.fixed_evaluator.to_dict(),
        "ranking_policy": dict(_DOWNSTREAM_RANKING_POLICY),
        "reference_regime_id": prepared.reference_regime_id,
        "best_validation_regime_id": str(best_regime["regime_id"]),
        "validation_ranking": ranking_table,
        "frontend_capacity_assessment": assessment,
        "regimes": list(ranked_regimes),
    }


def _assess_frontend_capacity(
    *,
    best_regime: Mapping[str, object],
    reference_regime: Mapping[str, object],
    all_regimes: Sequence[Mapping[str, object]],
    thresholds: AssessmentThresholds,
) -> dict[str, object]:
    best_val = best_regime["downstream_evaluator"]["val_metrics"]
    reference_val = reference_regime["downstream_evaluator"]["val_metrics"]
    best_test = best_regime["downstream_evaluator"]["test_metrics"]
    reference_test = reference_regime["downstream_evaluator"]["test_metrics"]
    delta_vs_reference_val = _metric_delta(best_val, reference_val)
    delta_vs_reference_test = _metric_delta(best_test, reference_test)
    val_spread = _spread_across_regimes(
        [summary["downstream_evaluator"]["val_metrics"] for summary in all_regimes]
    )
    meaningful_val_gain = (
        (
            float(reference_val["healthy_to_unhealthy_fpr"])
            - float(best_val["healthy_to_unhealthy_fpr"])
        )
        >= thresholds.healthy_to_unhealthy_fpr
        or (
            float(best_val["unhealthy_recall"]) - float(reference_val["unhealthy_recall"])
        )
        >= thresholds.unhealthy_recall
        or (
            float(best_val["unhealthy_f1"]) - float(reference_val["unhealthy_f1"])
        )
        >= thresholds.unhealthy_f1
        or (
            float(best_val["macro_f1"]) - float(reference_val["macro_f1"])
        )
        >= thresholds.macro_f1
    )
    test_not_contradictory = (
        float(best_test["healthy_to_unhealthy_fpr"])
        <= float(reference_test["healthy_to_unhealthy_fpr"]) + thresholds.healthy_to_unhealthy_fpr
        and float(best_test["macro_f1"]) + thresholds.macro_f1
        >= float(reference_test["macro_f1"])
        and float(best_test["unhealthy_f1"]) + thresholds.unhealthy_f1
        >= float(reference_test["unhealthy_f1"])
    )
    if (
        str(best_regime["regime_id"]) != str(reference_regime["regime_id"])
        and meaningful_val_gain
        and test_not_contradictory
    ):
        status = "likely_limiting"
        reasons = [
            "A frontend-only regime change produced a meaningful validation gain under a fixed downstream evaluator.",
            "The final test comparison does not contradict the validation-led ranking.",
        ]
    elif (
        val_spread["healthy_to_unhealthy_fpr"] < thresholds.healthy_to_unhealthy_fpr
        and val_spread["unhealthy_recall"] < thresholds.unhealthy_recall
        and val_spread["unhealthy_f1"] < thresholds.unhealthy_f1
        and val_spread["macro_f1"] < thresholds.macro_f1
    ):
        status = "not_obviously_limiting"
        reasons = [
            "Validation metrics stayed within the configured small-spread thresholds across the bounded regime sweep."
        ]
    else:
        status = "inconclusive"
        reasons = [
            "Frontend regimes moved the fixed-evaluator metrics, but the spread did not clear the configured confidence rule cleanly."
        ]
    return {
        "status": status,
        "reference_regime_id": str(reference_regime["regime_id"]),
        "best_validation_regime_id": str(best_regime["regime_id"]),
        "assessment_thresholds": thresholds.to_dict(),
        "validation_delta_vs_reference": delta_vs_reference_val,
        "test_delta_vs_reference": delta_vs_reference_test,
        "validation_spread_across_regimes": val_spread,
        "reasons": reasons,
        "selection_note": "Regimes are ranked on validation only; test metrics are included for final comparative reporting only.",
    }


def _spread_across_regimes(metrics_list: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        "healthy_to_unhealthy_fpr": round(
            max(float(metrics["healthy_to_unhealthy_fpr"]) for metrics in metrics_list)
            - min(float(metrics["healthy_to_unhealthy_fpr"]) for metrics in metrics_list),
            6,
        ),
        "unhealthy_recall": round(
            max(float(metrics["unhealthy_recall"]) for metrics in metrics_list)
            - min(float(metrics["unhealthy_recall"]) for metrics in metrics_list),
            6,
        ),
        "unhealthy_f1": round(
            max(float(metrics["unhealthy_f1"]) for metrics in metrics_list)
            - min(float(metrics["unhealthy_f1"]) for metrics in metrics_list),
            6,
        ),
        "macro_f1": round(
            max(float(metrics["macro_f1"]) for metrics in metrics_list)
            - min(float(metrics["macro_f1"]) for metrics in metrics_list),
            6,
        ),
    }


def _metric_delta(
    subject: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, float]:
    return {
        "healthy_to_unhealthy_fpr": round(
            float(subject["healthy_to_unhealthy_fpr"])
            - float(baseline["healthy_to_unhealthy_fpr"]),
            6,
        ),
        "unhealthy_precision": round(
            float(subject["unhealthy_precision"])
            - float(baseline["unhealthy_precision"]),
            6,
        ),
        "unhealthy_recall": round(
            float(subject["unhealthy_recall"])
            - float(baseline["unhealthy_recall"]),
            6,
        ),
        "unhealthy_f1": round(
            float(subject["unhealthy_f1"])
            - float(baseline["unhealthy_f1"]),
            6,
        ),
        "macro_f1": round(
            float(subject["macro_f1"])
            - float(baseline["macro_f1"]),
            6,
        ),
    }


def _ranking_metrics_from_split_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        "healthy_to_unhealthy_fpr": float(metrics["healthy_to_unhealthy_fpr"]),
        "unhealthy_precision": float(metrics["unhealthy_precision"]),
        "unhealthy_recall": float(metrics["unhealthy_recall"]),
        "unhealthy_f1": float(metrics["unhealthy_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _ranking_sort_key(
    metrics: Mapping[str, object],
    *,
    candidate_order: int,
) -> tuple[float, float, float, float, float, int]:
    return (
        float(metrics["healthy_to_unhealthy_fpr"]),
        -float(metrics["unhealthy_precision"]),
        -float(metrics["unhealthy_recall"]),
        -float(metrics["unhealthy_f1"]),
        -float(metrics["macro_f1"]),
        candidate_order,
    )


def _row_source_record_id(row: Mapping[str, object]) -> str:
    return f"{row['condition']}__{row['bearing_id']}__r{int(row['recording']):02d}"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON object.")
    return payload


def _bit_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _hamming_fraction(left: int, right: int, *, bit_length: int) -> float:
    return _bit_distance(left, right) / float(bit_length)


def _pairwise_hamming_mean(rows: Sequence[int], *, bit_length: int) -> float:
    if len(rows) < 2:
        return 0.0
    distances = [
        _hamming_fraction(rows[left_index], rows[right_index], bit_length=bit_length)
        for left_index in range(len(rows))
        for right_index in range(left_index + 1, len(rows))
    ]
    return sum(distances) / float(len(distances)) if distances else 0.0


def _bit_balance(rows: Sequence[int], *, bit_length: int) -> float:
    if not rows:
        return 0.0
    total = float(len(rows))
    scores: list[float] = []
    for bit_index in range(bit_length):
        ones = sum(1 for row in rows if row & (1 << bit_index))
        rate = float(ones) / total
        scores.append(1.0 - (abs(rate - 0.5) / 0.5))
    return max(0.0, min(1.0, sum(scores) / float(len(scores))))


def _bit_stability(split_rows: Mapping[str, Sequence[int]], *, bit_length: int) -> float:
    active_groups = [tuple(rows) for rows in split_rows.values() if rows]
    if len(active_groups) < 2:
        return 1.0
    stabilities: list[float] = []
    for bit_index in range(bit_length):
        rates = []
        for rows in active_groups:
            ones = sum(1 for row in rows if row & (1 << bit_index))
            rates.append(float(ones) / float(len(rows)))
        stabilities.append(1.0 - (max(rates) - min(rates)))
    return max(0.0, min(1.0, sum(stabilities) / float(len(stabilities))))


def _unique_row_rate(rows: Sequence[int]) -> float:
    if not rows:
        return 0.0
    return len(set(rows)) / float(len(rows))


def _select_single_medoid(rows: Sequence[int], *, bit_length: int) -> int:
    if not rows:
        raise ContractValidationError("Medoid selection requires at least one row.")
    return min(
        rows,
        key=lambda row: (
            sum(_bit_distance(row, other) for other in rows),
            _hex_row(row, bit_length=bit_length),
        ),
    )


def _hex_row(value: int, *, bit_length: int) -> str:
    hex_width = max(1, (bit_length + 3) // 4)
    return f"{value:0{hex_width}x}"


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractValidationError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _require_identifier(value: object, *, field_name: str) -> str:
    text = _require_non_empty_string(value, field_name=field_name)
    if re.search(r"[^A-Za-z0-9_.-]", text):
        raise ContractValidationError(
            f"`{field_name}` must contain only letters, numbers, `.`, `_`, or `-`."
        )
    return text


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a boolean.")
    return value


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return value


def _require_probability_threshold(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"`{field_name}` must be a float in [0.0, 1.0].")
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ContractValidationError(f"`{field_name}` must be a float in [0.0, 1.0].")
    return normalized


def _normalize_notes(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of strings.")
    notes: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item.strip() == "":
            raise ContractValidationError(f"`{field_name}[{index}]` must be a non-empty string.")
        notes.append(item.strip())
    return tuple(notes)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RUNS_ROOT",
    "PreparedFrontendCapacityCheck",
    "load_frontend_capacity_config",
    "prepare_frontend_capacity_check",
    "run_prepared_frontend_capacity_check",
    "run_frontend_capacity_check",
    "write_frontend_capacity_plan",
]
