"""Consumer-side locked frontend helpers for hard-mode binary experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from bittrace.v3 import (
    ArtifactRef,
    ContractValidationError,
    FrontendPromotionCandidate,
    PromotedFrontendWinner,
    StageKey,
    WaveformDatasetBundle,
)
from bittrace.v3.artifacts import compute_file_sha256, compute_json_sha256
from bittrace.v3.frontend_encoding import build_temporal_frontend_encoding
from bittrace.v3.frontend_stage import FRONTEND_PROMOTION_ARTIFACT_NAME


_BEST_FRONTEND_ARTIFACT_NAME = "bt3.best_frontend.json"
_FRONTEND_CANDIDATE_REPORT_ARTIFACT_NAME = "bt3.frontend_candidate_report.json"
_LOCKED_FRONTEND_REFERENCE_SUMMARY_ARTIFACT_NAME = "bt3.locked_frontend_reference_summary.json"
_BEST_FRONTEND_KIND = "bittrace_bearings_v3_source_locked_frontend_artifact"
_BEST_FRONTEND_SCHEMA_VERSION = "bittrace-bearings-v3-source-locked-frontend-artifact-1"
_FRONTEND_CANDIDATE_REPORT_KIND = "bittrace_bearings_v3_source_locked_frontend_candidate_report"
_FRONTEND_CANDIDATE_REPORT_SCHEMA_VERSION = (
    "bittrace-bearings-v3-source-locked-frontend-candidate-report-1"
)
_LOCKED_FRONTEND_REFERENCE_SUMMARY_KIND = (
    "bittrace_bearings_v3_source_locked_frontend_reference_summary"
)
_LOCKED_FRONTEND_REFERENCE_SUMMARY_SCHEMA_VERSION = (
    "bittrace-bearings-v3-source-locked-frontend-reference-summary-1"
)
_TEMPORAL_THRESHOLD_STRATEGY = "train_quantiles_v1"
_LOCKED_FRONTEND_RANKING_POLICY = {
    "ranking_mode": "encoder_proxy_quality",
    "selection_mode": "consumer_locked_single_candidate",
    "selection_basis": "shipping_lane_frozen_frontend_reference",
    "selection_split": "val",
    "frontend_sweep_enabled": False,
    "selection_note": (
        "A single frozen frontend candidate is injected into the canonical frontend "
        "stage for a retained supported/reference lane rather than as a universal "
        "BitTrace frontend identity."
    ),
}


@dataclass(frozen=True, slots=True)
class LockedFrontendSpec:
    regime_id: str
    label: str
    encoding_regime: str
    temporal_features_enabled: bool
    threshold_strategy: str
    bit_length: int
    selection_source: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "regime_id": self.regime_id,
            "label": self.label,
            "encoding_regime": self.encoding_regime,
            "temporal_features_enabled": self.temporal_features_enabled,
            "threshold_strategy": self.threshold_strategy,
            "bit_length": self.bit_length,
            "selection_source": self.selection_source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class LockedFrontendStageMaterialization:
    promoted_candidate: FrontendPromotionCandidate
    downstream_deep_input: object
    ranking_policy: Mapping[str, object]
    stage_notes: tuple[str, ...]


def load_locked_frontend_spec(profile: Mapping[str, Any]) -> LockedFrontendSpec | None:
    raw = profile.get("locked_frontend")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ContractValidationError("`locked_frontend` must be a mapping when present.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ContractValidationError("`locked_frontend.enabled` must be a boolean.")
    if not enabled:
        return None

    regime_id = _require_non_empty_string(
        raw.get("regime_id"),
        field_name="locked_frontend.regime_id",
    )
    label = _require_non_empty_string(
        raw.get("label"),
        field_name="locked_frontend.label",
    )
    encoding_regime = _require_non_empty_string(
        raw.get("encoding_regime"),
        field_name="locked_frontend.encoding_regime",
    )
    temporal_features_enabled = _require_bool(
        raw.get("temporal_features_enabled"),
        field_name="locked_frontend.temporal_features_enabled",
    )
    threshold_strategy = _require_non_empty_string(
        raw.get("threshold_strategy"),
        field_name="locked_frontend.threshold_strategy",
    )
    bit_length = _require_int(
        raw.get("bit_length"),
        field_name="locked_frontend.bit_length",
        minimum=1,
    )
    selection_source = _require_non_empty_string(
        raw.get("selection_source", "legacy_paderborn_reference.temporal_threshold_36"),
        field_name="locked_frontend.selection_source",
    )
    notes = _require_string_sequence(
        raw.get("notes", ()),
        field_name="locked_frontend.notes",
    )

    if regime_id != "temporal_threshold_36":
        raise ContractValidationError(
            "This locked-frontend slice supports only `locked_frontend.regime_id: temporal_threshold_36`."
        )
    if encoding_regime != "temporal_threshold":
        raise ContractValidationError(
            "This locked-frontend slice supports only `locked_frontend.encoding_regime: temporal_threshold`."
        )
    if not temporal_features_enabled:
        raise ContractValidationError(
            "Locked temporal threshold frontend requires `locked_frontend.temporal_features_enabled: true`."
        )
    if threshold_strategy != _TEMPORAL_THRESHOLD_STRATEGY:
        raise ContractValidationError(
            "Locked temporal threshold frontend requires `threshold_strategy: train_quantiles_v1`."
        )
    if bit_length != 36:
        raise ContractValidationError(
            "This locked-frontend slice supports only `locked_frontend.bit_length: 36`."
        )

    return LockedFrontendSpec(
        regime_id=regime_id,
        label=label,
        encoding_regime=encoding_regime,
        temporal_features_enabled=temporal_features_enabled,
        threshold_strategy=threshold_strategy,
        bit_length=bit_length,
        selection_source=selection_source,
        notes=notes,
    )


def build_locked_frontend_stage_materialization(
    *,
    stage_key: StageKey,
    stage_output_dir: Path,
    source_bundle: WaveformDatasetBundle,
    include_test_metrics_in_frontend: bool,
    locked_frontend: LockedFrontendSpec,
) -> LockedFrontendStageMaterialization:
    temporal_frontend = build_temporal_frontend_encoding(
        source_bundle.canonical_records,
        bit_length=locked_frontend.bit_length,
    )
    frontend_encoding = dict(temporal_frontend["encoder"])
    frontend_summary = dict(temporal_frontend["summary"])
    materialization_mode = "consumer_locked_temporal_threshold_waveform_bundle"
    candidate_id = (
        f"frontend-{stage_key.value}-locked-{locked_frontend.regime_id}-"
        f"{source_bundle.bundle_fingerprint[:12]}"
    )
    frontend_input_id = candidate_id
    frontend_config_fingerprint = compute_json_sha256(
        {
            "locked_frontend": locked_frontend.to_dict(),
            "materialization_mode": materialization_mode,
            "source_bundle_fingerprint": source_bundle.bundle_fingerprint,
            "include_test_metrics_in_frontend": include_test_metrics_in_frontend,
            "stage_key": stage_key.value,
            "frontend_encoder_fingerprint": frontend_encoding["encoder_fingerprint"],
        }
    )
    frontend_fingerprint = compute_json_sha256(
        {
            "candidate_id": candidate_id,
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "frontend_input_id": frontend_input_id,
            "frontend_kind": frontend_encoding["frontend_kind"],
            "locked_frontend_regime_id": locked_frontend.regime_id,
            "source_bundle_fingerprint": source_bundle.bundle_fingerprint,
            "stage_key": stage_key.value,
            "validation_proxy_metrics": frontend_summary["ranking_metrics"],
        }
    )
    promotion_artifact_path = stage_output_dir / FRONTEND_PROMOTION_ARTIFACT_NAME
    deep_input = source_bundle.materialize_deep_input_ref(
        output_dir=stage_output_dir / "effective_deep_input",
        frontend_input_id=frontend_input_id,
        frontend_fingerprint=frontend_fingerprint,
        include_test_metrics=include_test_metrics_in_frontend,
        source_promotion_artifact_ref=ArtifactRef(
            kind=PromotedFrontendWinner.KIND,
            schema_version=PromotedFrontendWinner.SCHEMA_VERSION,
            path=str(promotion_artifact_path.resolve()),
        ),
        extra_handoff_fields={
            "frontend_encoding": frontend_encoding,
            "locked_frontend": locked_frontend.to_dict(),
        },
        extra_contract_fields={
            "frontend_encoding": frontend_encoding,
            "locked_frontend": locked_frontend.to_dict(),
        },
    )

    reference_summary = _build_reference_summary(
        stage_output_dir=stage_output_dir,
        source_bundle=source_bundle,
        deep_input=deep_input,
        frontend_encoding=frontend_encoding,
        frontend_fingerprint=frontend_fingerprint,
        locked_frontend=locked_frontend,
    )
    reference_summary_ref = _write_json_artifact(
        stage_output_dir / _LOCKED_FRONTEND_REFERENCE_SUMMARY_ARTIFACT_NAME,
        kind=_LOCKED_FRONTEND_REFERENCE_SUMMARY_KIND,
        schema_version=_LOCKED_FRONTEND_REFERENCE_SUMMARY_SCHEMA_VERSION,
        payload=reference_summary,
    )
    best_frontend_ref = _write_json_artifact(
        stage_output_dir / _BEST_FRONTEND_ARTIFACT_NAME,
        kind=_BEST_FRONTEND_KIND,
        schema_version=_BEST_FRONTEND_SCHEMA_VERSION,
        payload={
            "materialization_mode": materialization_mode,
            "stage_key": stage_key.value,
            "candidate_id": candidate_id,
            "frontend_input_id": frontend_input_id,
            "promoted_frontend_fingerprint": frontend_fingerprint,
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "include_test_metrics": include_test_metrics_in_frontend,
            "source_bundle": {
                "bundle_dir": str(source_bundle.bundle_dir.resolve()),
                "bundle_contract_path": str(source_bundle.bundle_contract_path.resolve()),
                "bundle_fingerprint": source_bundle.bundle_fingerprint,
                "source_handoff_manifest_path": str(source_bundle.handoff_manifest_path.resolve()),
            },
            "frontend_encoding": frontend_encoding,
            "temporal_frontend_summary": frontend_summary,
            "locked_frontend": locked_frontend.to_dict(),
            "selection_reference_summary_ref": reference_summary_ref.to_dict(),
            "frontend_sweep_enabled": False,
        },
    )
    candidate_report_ref = _write_json_artifact(
        stage_output_dir / _FRONTEND_CANDIDATE_REPORT_ARTIFACT_NAME,
        kind=_FRONTEND_CANDIDATE_REPORT_KIND,
        schema_version=_FRONTEND_CANDIDATE_REPORT_SCHEMA_VERSION,
        payload={
            "materialization_mode": materialization_mode,
            "stage_key": stage_key.value,
            "candidate_id": candidate_id,
            "candidate_order": 1,
            "frontend_input_id": frontend_input_id,
            "promoted_frontend_fingerprint": frontend_fingerprint,
            "frontend_config_fingerprint": frontend_config_fingerprint,
            "ranking_metrics": dict(frontend_summary["ranking_metrics"]),
            "frontend_encoding": frontend_encoding,
            "temporal_frontend_summary": frontend_summary,
            "locked_frontend": locked_frontend.to_dict(),
            "selection_reference_summary_ref": reference_summary_ref.to_dict(),
            "best_frontend_artifact_ref": best_frontend_ref.to_dict(),
            "downstream_deep_input": deep_input.to_dict(),
            "frontend_sweep_enabled": False,
        },
    )
    promoted_candidate = FrontendPromotionCandidate(
        candidate_id=candidate_id,
        candidate_order=1,
        promoted_frontend_fingerprint=frontend_fingerprint,
        ranking_eligible=True,
        ranking_metrics=dict(frontend_summary["ranking_metrics"]),
        candidate_report_ref=candidate_report_ref,
        best_frontend_artifact_ref=best_frontend_ref,
        frontend_config_fingerprint=frontend_config_fingerprint,
    )
    ranking_policy = {
        **dict(_LOCKED_FRONTEND_RANKING_POLICY),
        "selection_source": locked_frontend.selection_source,
        "locked_frontend": locked_frontend.to_dict(),
    }
    return LockedFrontendStageMaterialization(
        promoted_candidate=promoted_candidate,
        downstream_deep_input=deep_input,
        ranking_policy=ranking_policy,
        stage_notes=(
            "frontend_lock=true",
            f"frontend_lock_regime={locked_frontend.regime_id}",
            f"frontend_lock_bit_length={locked_frontend.bit_length}",
            "frontend_sweep=false",
            f"frontend_lock_selection_source={locked_frontend.selection_source}",
        ),
    )


def _build_reference_summary(
    *,
    stage_output_dir: Path,
    source_bundle: WaveformDatasetBundle,
    deep_input: object,
    frontend_encoding: Mapping[str, object],
    frontend_fingerprint: str,
    locked_frontend: LockedFrontendSpec,
) -> dict[str, object]:
    return {
        "locked_frontend": locked_frontend.to_dict(),
        "frozen_reference_note": (
            "The source lane records the locked frontend materialization directly and does not "
            "ship the frontend-sweep or architecture-comparison helpers."
        ),
        "frontend_fingerprint": frontend_fingerprint,
        "frontend_encoding": dict(frontend_encoding),
        "source_bundle": {
            "bundle_dir": str(source_bundle.bundle_dir.resolve()),
            "bundle_contract_path": str(source_bundle.bundle_contract_path.resolve()),
            "bundle_fingerprint": source_bundle.bundle_fingerprint,
            "handoff_manifest_path": str(source_bundle.handoff_manifest_path.resolve()),
        },
        "downstream_deep_input": (
            deep_input.to_dict() if hasattr(deep_input, "to_dict") else {"repr": repr(deep_input)}
        ),
        "ranking_policy": dict(_LOCKED_FRONTEND_RANKING_POLICY),
        "split_discipline": {
            "train": "fit only",
            "val": "winner traceability only",
            "test": "final reporting only",
        },
    }


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"`{field_name}` must be a boolean.")
    return value


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractValidationError(f"`{field_name}` must be an integer >= {minimum}.")
    return value


def _require_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of strings.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(
            _require_non_empty_string(
                item,
                field_name=f"{field_name}[{index}]",
            )
        )
    return tuple(normalized)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_artifact(
    path: Path,
    *,
    kind: str,
    schema_version: str,
    payload: Mapping[str, object],
) -> ArtifactRef:
    artifact_payload = {
        "kind": kind,
        "schema_version": schema_version,
        **dict(payload),
    }
    _write_json(path, artifact_payload)
    return ArtifactRef(
        kind=kind,
        schema_version=schema_version,
        path=str(path.resolve()),
        sha256=compute_file_sha256(path),
    )


__all__ = [
    "LockedFrontendSpec",
    "LockedFrontendStageMaterialization",
    "build_locked_frontend_stage_materialization",
    "load_locked_frontend_spec",
]
