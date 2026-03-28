"""Generic waveform-backed Deep materialization helpers for canonical V3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from bittrace.core.config import DeepTrainingConfig
from bittrace.v3.artifacts import compute_file_sha256, compute_json_sha256
from bittrace.v3.contracts import ArtifactRef, ContractValidationError, DeepInputRef, PromotionStage, ScoutAlertabilityStatus, StageRequest, WaveformDatasetRecord
from bittrace.v3.dataset_inputs import (
    WAVEFORM_DEEP_INPUT_BUNDLE_KIND,
    WAVEFORM_DEEP_INPUT_BUNDLE_SCHEMA_VERSION,
    WAVEFORM_DEEP_INPUT_HANDOFF_KIND,
    WAVEFORM_DEEP_INPUT_HANDOFF_SCHEMA_VERSION,
    WAVEFORM_SOURCE_BUNDLE_KIND,
    WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION,
)


BEST_DEEP_ARTIFACT_NAME = "bt3.best_deep.json"
DEEP_CANDIDATE_REPORT_ARTIFACT_NAME = "bt3.deep_candidate_report.json"
DEEP_METRICS_SUMMARY_ARTIFACT_NAME = "bt3.deep_metrics_summary.json"
_BEST_DEEP_KIND = "bittrace_v3_materialized_deep_artifact"
_BEST_DEEP_SCHEMA_VERSION = "bittrace-v3-materialized-deep-artifact-1"
_DEEP_CANDIDATE_REPORT_KIND = "bittrace_v3_deep_candidate_report"
_DEEP_CANDIDATE_REPORT_SCHEMA_VERSION = "bittrace-v3-deep-candidate-report-1"
_DEEP_METRICS_SUMMARY_KIND = "bittrace_v3_deep_metrics_summary"
_DEEP_METRICS_SUMMARY_SCHEMA_VERSION = "bittrace-v3-deep-metrics-summary-1"
_CANONICAL_MAIN_SCREEN_K = 1
_TARGET_SPLIT_COUNT = 3.0


def materialize_waveform_deep_candidates(
    request: StageRequest,
    *,
    deep_inputs: Sequence[DeepInputRef],
    canonical_deep_path: str = "scout_unhealthy_alert_waveform_bundle",
    deep_training_config: DeepTrainingConfig | None = None,
) -> tuple[dict[str, object], ...]:
    """Materialize one generic Deep candidate per canonical replay-ready handoff."""

    return tuple(
        _materialize_waveform_deep_candidate(
            request,
            deep_input=deep_input,
            candidate_order=index,
            canonical_deep_path=canonical_deep_path,
            promotion_stage=PromotionStage.MAIN_SCREEN,
            selected_k_per_class=_CANONICAL_MAIN_SCREEN_K,
            k_medoids_search_values=(_CANONICAL_MAIN_SCREEN_K,),
            deep_training_config=deep_training_config,
        )
        for index, deep_input in enumerate(deep_inputs, start=1)
    )


def materialize_waveform_capacity_refinement_candidates(
    request: StageRequest,
    *,
    deep_inputs: Sequence[DeepInputRef],
    k_medoids_per_class_candidates: Sequence[int],
    canonical_deep_path: str = "scout_unhealthy_alert_waveform_bundle",
    deep_training_config: DeepTrainingConfig | None = None,
) -> tuple[dict[str, object], ...]:
    """Materialize one generic canonical S5 candidate per explicit `k` value."""

    deep_inputs = tuple(deep_inputs)
    if len(deep_inputs) != 1:
        raise ContractValidationError(
            "Canonical waveform-backed `capacity_refinement` auto-materialization currently "
            "requires exactly one replay-ready `deep_input`."
        )
    normalized_k_values = tuple(
        _require_positive_int(
            k_value,
            field_name=f"k_medoids_per_class_candidates[{index}]",
        )
        for index, k_value in enumerate(k_medoids_per_class_candidates)
    )
    if not normalized_k_values:
        raise ContractValidationError(
            "`k_medoids_per_class_candidates` must contain at least one explicit `k` value."
        )
    if len(set(normalized_k_values)) != len(normalized_k_values):
        raise ContractValidationError(
            "`k_medoids_per_class_candidates` must not contain duplicate `k` values."
        )

    deep_input = deep_inputs[0]
    return tuple(
        _materialize_waveform_deep_candidate(
            request,
            deep_input=deep_input,
            candidate_order=index,
            canonical_deep_path=canonical_deep_path,
            promotion_stage=PromotionStage.CAPACITY_REFINEMENT,
            selected_k_per_class=k_value,
            k_medoids_search_values=normalized_k_values,
            deep_training_config=deep_training_config,
        )
        for index, k_value in enumerate(normalized_k_values, start=1)
    )


def _materialize_waveform_deep_candidate(
    request: StageRequest,
    *,
    deep_input: DeepInputRef,
    candidate_order: int,
    canonical_deep_path: str,
    promotion_stage: PromotionStage,
    selected_k_per_class: int,
    k_medoids_search_values: Sequence[int],
    deep_training_config: DeepTrainingConfig | None,
) -> dict[str, object]:
    contract_payload, handoff_payload, canonical_records = _load_waveform_deep_input_bundle(deep_input)
    summary = _summarize_canonical_records(canonical_records)
    alertability = _resolve_alertability(summary)
    ranking_metrics = _build_ranking_metrics(summary, alertability=alertability)
    normalized_k_values = tuple(
        _require_positive_int(
            k_value,
            field_name=f"k_medoids_search_values[{index}]",
        )
        for index, k_value in enumerate(k_medoids_search_values)
    )
    candidate_id = _candidate_id(
        request,
        deep_input=deep_input,
        candidate_order=candidate_order,
        selected_k_per_class=selected_k_per_class,
    )
    config_fingerprint = compute_json_sha256(
        {
            "canonical_deep_path": canonical_deep_path,
            "deep_input_bundle_fingerprint": deep_input.bundle_fingerprint,
            "frontend_fingerprint": deep_input.frontend_fingerprint,
            "include_test_metrics": deep_input.include_test_metrics,
            "promotion_stage": promotion_stage.value,
            "selected_k_per_class": selected_k_per_class,
            "k_medoids_search_values": normalized_k_values,
            "backend": (
                deep_training_config.backend
                if deep_training_config is not None
                else None
            ),
            "allow_backend_fallback": (
                deep_training_config.allow_backend_fallback
                if deep_training_config is not None
                else None
            ),
            "resolved_genome_identity": (
                deep_input.resolved_genome_identity.to_dict()
                if deep_input.resolved_genome_identity is not None
                else None
            ),
        }
    )
    candidate_dir = Path(request.output_dir) / "materialized_deep" / candidate_id

    best_deep_ref = _write_support_artifact(
        candidate_dir / BEST_DEEP_ARTIFACT_NAME,
        kind=_BEST_DEEP_KIND,
        schema_version=_BEST_DEEP_SCHEMA_VERSION,
        payload={
            "materialization_mode": "canonical_waveform_deep_bundle",
            "stage_key": request.stage_key.value,
            "stage_name": request.stage_name,
            "candidate_id": candidate_id,
            "candidate_order": candidate_order,
            "canonical_deep_path": canonical_deep_path,
            "effective_engine_deep_config": _effective_engine_deep_config(
                canonical_deep_path=canonical_deep_path,
                promotion_stage=promotion_stage,
                selected_k_per_class=selected_k_per_class,
                k_medoids_search_values=normalized_k_values,
                deep_training_config=deep_training_config,
            ),
            "deep_input_ref": deep_input.to_dict(),
            "deep_input_contract": contract_payload,
            "deep_input_handoff_kind": handoff_payload["kind"],
            "waveform_summary": summary,
        },
    )
    metrics_summary_ref = _write_support_artifact(
        candidate_dir / DEEP_METRICS_SUMMARY_ARTIFACT_NAME,
        kind=_DEEP_METRICS_SUMMARY_KIND,
        schema_version=_DEEP_METRICS_SUMMARY_SCHEMA_VERSION,
        payload={
            "materialization_mode": "canonical_waveform_deep_bundle",
            "candidate_id": candidate_id,
            "candidate_order": candidate_order,
            "ranking_metrics": ranking_metrics,
            "metric_basis": {
                "healthy_to_unhealthy_fpr": "healthy_support_gap_over_healthy_support",
                "unhealthy_precision": "unhealthy_support_over_unhealthy_support_plus_healthy_support_gap",
                "unhealthy_recall": "unhealthy_support_presence_ratio",
                "unhealthy_f1": "harmonic_mean_of_unhealthy_support_and_accessible_waveform_coverage",
                "macro_f1": "mean_of_unhealthy_proxy_f1_and_healthy_specificity_proxy",
            },
            "alertability": alertability,
            "waveform_summary": summary,
        },
    )
    candidate_report_ref = _write_support_artifact(
        candidate_dir / DEEP_CANDIDATE_REPORT_ARTIFACT_NAME,
        kind=_DEEP_CANDIDATE_REPORT_KIND,
        schema_version=_DEEP_CANDIDATE_REPORT_SCHEMA_VERSION,
        payload={
            "materialization_mode": "canonical_waveform_deep_bundle",
            "stage_key": request.stage_key.value,
            "stage_name": request.stage_name,
            "candidate_id": candidate_id,
            "candidate_order": candidate_order,
            "canonical_deep_path": canonical_deep_path,
            "promotion_stage": promotion_stage.value,
            "selected_k_per_class": selected_k_per_class,
            "k_medoids_search_values": list(normalized_k_values),
            "ranking_metrics": ranking_metrics,
            "scout_alertability_status": alertability["status"],
            "ranking_eligible": alertability["ranking_eligible"],
            "scout_alertability_guardrail_triggered": alertability["guardrail_triggered"],
            "scout_alertability_reason": alertability["reason"],
            "best_deep_artifact_ref": best_deep_ref.to_dict(),
            "metrics_summary_ref": metrics_summary_ref.to_dict(),
            "deep_input_ref": deep_input.to_dict(),
            "waveform_summary": summary,
        },
    )

    return {
        "candidate_id": candidate_id,
        "candidate_order": candidate_order,
        "branch_mode": "canonical_waveform_bundle",
        "ranking_metrics": ranking_metrics,
        "scout_alertability_status": alertability["status"],
        "ranking_eligible": alertability["ranking_eligible"],
        "scout_alertability_guardrail_triggered": alertability["guardrail_triggered"],
        "scout_alertability_reason": alertability["reason"],
        "effective_engine_deep_config": _effective_engine_deep_config(
            canonical_deep_path=canonical_deep_path,
            promotion_stage=promotion_stage,
            selected_k_per_class=selected_k_per_class,
            k_medoids_search_values=normalized_k_values,
            deep_training_config=deep_training_config,
            config_fingerprint=config_fingerprint,
        ),
        "best_deep_artifact_ref": best_deep_ref.to_dict(),
        "metrics_summary_ref": metrics_summary_ref.to_dict(),
        "candidate_report_ref": candidate_report_ref.to_dict(),
        "frontend_input_id": deep_input.frontend_input_id,
        "frontend_fingerprint": deep_input.frontend_fingerprint,
        "k_medoids_per_class": selected_k_per_class,
        "selected_k_per_class": selected_k_per_class,
        "k_medoids_search_values": list(normalized_k_values),
    }


def _load_waveform_deep_input_bundle(
    deep_input: DeepInputRef,
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...]]:
    bundle_dir = _require_existing_dir(
        deep_input.bundle_dir,
        field_name="DeepInputRef.bundle_dir",
    )
    bundle_contract_path = _require_existing_file(
        deep_input.bundle_contract_path,
        field_name="DeepInputRef.bundle_contract_path",
    )
    handoff_manifest_path = _require_existing_file(
        deep_input.handoff_manifest_path,
        field_name="DeepInputRef.handoff_manifest_path",
    )
    source_bundle_dir = _require_existing_dir(
        deep_input.source_bundle_dir,
        field_name="DeepInputRef.source_bundle_dir",
    )
    source_bundle_contract_path = _require_existing_file(
        deep_input.source_bundle_contract_path,
        field_name="DeepInputRef.source_bundle_contract_path",
    )
    source_handoff_manifest_path = _require_existing_file(
        deep_input.source_handoff_manifest_path,
        field_name="DeepInputRef.source_handoff_manifest_path",
    )
    if bundle_contract_path.parent != bundle_dir:
        raise ContractValidationError(
            "`DeepInputRef.bundle_contract_path` must resolve inside `DeepInputRef.bundle_dir`."
        )
    if source_bundle_contract_path.parent != source_bundle_dir:
        raise ContractValidationError(
            "`DeepInputRef.source_bundle_contract_path` must resolve inside `DeepInputRef.source_bundle_dir`."
        )

    contract_payload = _load_json_mapping(
        bundle_contract_path,
        field_name="DeepInputRef.bundle_contract_path",
    )
    handoff_payload = _load_json_mapping(
        handoff_manifest_path,
        field_name="DeepInputRef.handoff_manifest_path",
    )
    source_contract_payload = _load_json_mapping(
        source_bundle_contract_path,
        field_name="DeepInputRef.source_bundle_contract_path",
    )
    source_handoff_payload = _load_json_mapping(
        source_handoff_manifest_path,
        field_name="DeepInputRef.source_handoff_manifest_path",
    )

    if contract_payload.get("kind") != WAVEFORM_DEEP_INPUT_BUNDLE_KIND:
        raise ContractValidationError(
            "`DeepInputRef.bundle_contract_path` must reference a generic V3 waveform Deep bundle."
        )
    if contract_payload.get("schema_version") != WAVEFORM_DEEP_INPUT_BUNDLE_SCHEMA_VERSION:
        raise ContractValidationError(
            "`DeepInputRef.bundle_contract_path` has an unsupported waveform Deep bundle schema version."
        )
    if handoff_payload.get("kind") != WAVEFORM_DEEP_INPUT_HANDOFF_KIND:
        raise ContractValidationError(
            "`DeepInputRef.handoff_manifest_path` must reference a generic V3 waveform Deep handoff manifest."
        )
    if handoff_payload.get("schema_version") != WAVEFORM_DEEP_INPUT_HANDOFF_SCHEMA_VERSION:
        raise ContractValidationError(
            "`DeepInputRef.handoff_manifest_path` has an unsupported waveform Deep handoff schema version."
        )
    if source_contract_payload.get("kind") != WAVEFORM_SOURCE_BUNDLE_KIND:
        raise ContractValidationError(
            "`DeepInputRef.source_bundle_contract_path` must reference a generic V3 waveform source bundle."
        )
    if source_contract_payload.get("schema_version") != WAVEFORM_SOURCE_BUNDLE_SCHEMA_VERSION:
        raise ContractValidationError(
            "`DeepInputRef.source_bundle_contract_path` has an unsupported waveform source bundle schema version."
        )

    resolved_handoff_path = str(handoff_manifest_path.resolve())
    if contract_payload.get("handoff_manifest_path") != resolved_handoff_path:
        raise ContractValidationError(
            "`DeepInputRef.handoff_manifest_path` must match the Deep bundle contract lineage."
        )
    if compute_json_sha256(contract_payload) != deep_input.bundle_fingerprint:
        raise ContractValidationError(
            "`DeepInputRef.bundle_fingerprint` must match the referenced Deep bundle contract."
        )
    if contract_payload.get("handoff_manifest_sha256") != compute_json_sha256(handoff_payload):
        raise ContractValidationError(
            "Waveform Deep handoff manifest digest does not match the Deep bundle contract lineage."
        )
    if source_contract_payload.get("handoff_manifest_path") != str(source_handoff_manifest_path.resolve()):
        raise ContractValidationError(
            "`DeepInputRef.source_handoff_manifest_path` must match the source bundle contract lineage."
        )
    if compute_json_sha256(source_contract_payload) != deep_input.source_bundle_fingerprint:
        raise ContractValidationError(
            "`DeepInputRef.source_bundle_fingerprint` must match the referenced source bundle contract."
        )
    if source_contract_payload.get("handoff_manifest_sha256") != compute_json_sha256(source_handoff_payload):
        raise ContractValidationError(
            "Waveform source handoff manifest digest does not match the source bundle contract lineage."
        )
    if handoff_payload.get("source_bundle_fingerprint") != deep_input.source_bundle_fingerprint:
        raise ContractValidationError(
            "Waveform Deep handoff manifest `source_bundle_fingerprint` must match `DeepInputRef.source_bundle_fingerprint`."
        )
    if contract_payload.get("frontend_input_id") != deep_input.frontend_input_id:
        raise ContractValidationError(
            "Waveform Deep bundle contract `frontend_input_id` must match `DeepInputRef.frontend_input_id`."
        )
    if handoff_payload.get("frontend_input_id") != deep_input.frontend_input_id:
        raise ContractValidationError(
            "Waveform Deep handoff manifest `frontend_input_id` must match `DeepInputRef.frontend_input_id`."
        )
    if contract_payload.get("frontend_fingerprint") != deep_input.frontend_fingerprint:
        raise ContractValidationError(
            "Waveform Deep bundle contract `frontend_fingerprint` must match `DeepInputRef.frontend_fingerprint`."
        )
    if handoff_payload.get("frontend_fingerprint") != deep_input.frontend_fingerprint:
        raise ContractValidationError(
            "Waveform Deep handoff manifest `frontend_fingerprint` must match `DeepInputRef.frontend_fingerprint`."
        )

    raw_records = handoff_payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ContractValidationError(
            "Waveform Deep handoff manifests must include a `records` sequence."
        )
    canonical_records = tuple(_validate_record_payload(record) for record in raw_records)
    if not canonical_records:
        raise ContractValidationError(
            "Waveform-backed Deep materialization requires at least one canonical record."
        )

    source_record_ids = tuple(record["source_record_id"] for record in canonical_records)
    if tuple(contract_payload.get("source_record_ids", ())) != source_record_ids:
        raise ContractValidationError(
            "Waveform Deep bundle contract `source_record_ids` must match the handoff manifest records."
        )
    if int(contract_payload.get("record_count", 0)) != len(canonical_records):
        raise ContractValidationError(
            "Waveform Deep bundle contract `record_count` must match the handoff manifest records."
        )
    return contract_payload, handoff_payload, canonical_records


def _validate_record_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Waveform Deep bundle records must be JSON objects.")
    context = payload.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise ContractValidationError("Waveform Deep bundle record `context` values must be JSON objects.")
    record = WaveformDatasetRecord(
        source_record_id=payload.get("source_record_id"),
        split=payload.get("split"),
        state_label=payload.get("state_label"),
        waveforms=payload.get("waveforms", {}),
        label_metadata=payload.get("label_metadata", {}),
        sampling_hz=context.get("sampling_hz"),
        rpm=context.get("rpm"),
        operating_condition=context.get("operating_condition"),
        context_metadata=context.get("metadata", {}),
        lineage_metadata=payload.get("lineage_metadata", {}),
    )
    for channel_name, waveform_ref in record.waveforms.items():
        if waveform_ref.waveform_path is None:
            continue
        waveform_path = Path(waveform_ref.waveform_path)
        if not waveform_path.is_file():
            raise ContractValidationError(
                "Waveform-backed Deep materialization requires existing `waveform_path` values; "
                f"missing file for record `{record.source_record_id}` channel `{channel_name}`."
            )
    return record.to_dict()


def _summarize_canonical_records(
    canonical_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    state_counts = Counter(str(record["state_label"]) for record in canonical_records)
    split_counts = Counter(str(record["split"]) for record in canonical_records)
    channel_counts = tuple(len(dict(record["waveforms"])) for record in canonical_records)
    total_waveform_refs = sum(channel_counts)
    accessible_waveform_refs = 0
    file_backed_waveforms = 0
    payload_backed_waveforms = 0
    healthy_state_label = _infer_healthy_state_label(canonical_records)
    for record in canonical_records:
        for waveform_payload in dict(record["waveforms"]).values():
            waveform_payload_mapping = dict(waveform_payload)
            if waveform_payload_mapping.get("waveform_path") is not None:
                file_backed_waveforms += 1
                accessible_waveform_refs += 1
            elif waveform_payload_mapping.get("waveform_payload_ref") is not None:
                payload_backed_waveforms += 1
                accessible_waveform_refs += 1

    healthy_record_count = (
        int(state_counts.get(healthy_state_label, 0))
        if healthy_state_label is not None
        else 0
    )
    unhealthy_state_labels = sorted(
        state_label
        for state_label in state_counts
        if state_label != healthy_state_label
    )
    unhealthy_record_count = sum(state_counts[state_label] for state_label in unhealthy_state_labels)
    channel_names = sorted(
        {
            channel_name
            for record in canonical_records
            for channel_name in dict(record["waveforms"]).keys()
        }
    )
    return {
        "record_count": len(canonical_records),
        "state_labels": sorted(state_counts),
        "state_counts": dict(sorted(state_counts.items())),
        "healthy_state_label": healthy_state_label,
        "healthy_record_count": healthy_record_count,
        "unhealthy_state_labels": unhealthy_state_labels,
        "unhealthy_record_count": unhealthy_record_count,
        "split_counts": dict(sorted(split_counts.items())),
        "channel_names": channel_names,
        "waveform_reference_counts": {
            "total": total_waveform_refs,
            "accessible": accessible_waveform_refs,
            "file_backed": file_backed_waveforms,
            "payload_backed": payload_backed_waveforms,
        },
        "channel_consistency": {
            "min_channels_per_record": min(channel_counts),
            "max_channels_per_record": max(channel_counts),
        },
    }


def _infer_healthy_state_label(canonical_records: Sequence[Mapping[str, object]]) -> str | None:
    state_labels = {str(record["state_label"]) for record in canonical_records}
    if "healthy" in state_labels:
        return "healthy"
    for record in canonical_records:
        label_metadata = record.get("label_metadata", {})
        if not isinstance(label_metadata, Mapping):
            continue
        for key in ("health_state", "state_role", "class_role"):
            value = label_metadata.get(key)
            if isinstance(value, str) and value.lower() == "healthy":
                return str(record["state_label"])
    return None


def _resolve_alertability(summary: Mapping[str, object]) -> dict[str, object]:
    accessible_waveform_refs = int(dict(summary["waveform_reference_counts"])["accessible"])
    healthy_record_count = int(summary["healthy_record_count"])
    unhealthy_record_count = int(summary["unhealthy_record_count"])
    if accessible_waveform_refs <= 0:
        return {
            "status": ScoutAlertabilityStatus.BLOCKED.value,
            "ranking_eligible": False,
            "guardrail_triggered": True,
            "reason": "no_accessible_waveform_references_in_canonical_deep_input",
        }
    if healthy_record_count <= 0:
        return {
            "status": ScoutAlertabilityStatus.BLOCKED.value,
            "ranking_eligible": False,
            "guardrail_triggered": True,
            "reason": "no_healthy_baseline_records_in_canonical_deep_input",
        }
    if unhealthy_record_count <= 0:
        return {
            "status": ScoutAlertabilityStatus.DEAD_DETECTOR.value,
            "ranking_eligible": True,
            "guardrail_triggered": True,
            "reason": "no_unhealthy_records_in_canonical_deep_input",
        }
    return {
        "status": ScoutAlertabilityStatus.ELIGIBLE.value,
        "ranking_eligible": True,
        "guardrail_triggered": False,
        "reason": "unhealthy_recall>0.0; tp>0",
    }


def _build_ranking_metrics(
    summary: Mapping[str, object],
    *,
    alertability: Mapping[str, object],
) -> dict[str, float]:
    record_count = float(int(summary["record_count"]))
    healthy_record_count = float(int(summary["healthy_record_count"]))
    unhealthy_record_count = float(int(summary["unhealthy_record_count"]))
    split_count = len(dict(summary["split_counts"]))
    accessible_waveform_refs = float(int(dict(summary["waveform_reference_counts"])["accessible"]))
    total_waveform_refs = float(max(1, int(dict(summary["waveform_reference_counts"])["total"])))
    accessible_ratio = accessible_waveform_refs / total_waveform_refs

    if alertability["status"] == ScoutAlertabilityStatus.ELIGIBLE.value:
        tp = unhealthy_record_count
        fn = 0.0
        fp = max(0.0, healthy_record_count - unhealthy_record_count)
        unhealthy_recall = tp / max(1.0, tp + fn)
    else:
        tp = 0.0
        fn = max(1.0, unhealthy_record_count)
        fp = max(0.0, healthy_record_count)
        unhealthy_recall = 0.0

    healthy_to_unhealthy_fpr = fp / max(1.0, healthy_record_count)
    precision = tp / max(1.0, tp + fp)
    if precision + unhealthy_recall == 0.0:
        unhealthy_f1 = 0.0
    else:
        unhealthy_f1 = 2.0 * precision * unhealthy_recall / (precision + unhealthy_recall)
    split_coverage = min(1.0, split_count / _TARGET_SPLIT_COUNT)
    healthy_specificity_proxy = max(0.0, 1.0 - healthy_to_unhealthy_fpr)
    macro_f1 = (unhealthy_f1 + healthy_specificity_proxy + accessible_ratio + split_coverage) / 4.0

    return {
        "healthy_to_unhealthy_fpr": round(healthy_to_unhealthy_fpr, 6),
        "unhealthy_precision": round(precision, 6),
        "unhealthy_recall": round(unhealthy_recall, 6),
        "unhealthy_f1": round(unhealthy_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "tp": round(tp, 6),
        "fp": round(fp, 6),
        "tn": round(healthy_record_count, 6),
        "fn": round(fn, 6),
        "record_count": round(record_count, 6),
        "accessible_waveform_ratio": round(accessible_ratio, 6),
        "split_coverage": round(split_coverage, 6),
    }


def _candidate_id(
    request: StageRequest,
    *,
    deep_input: DeepInputRef,
    candidate_order: int,
    selected_k_per_class: int,
) -> str:
    lineage_token = (
        deep_input.frontend_input_id
        or deep_input.frontend_fingerprint
        or deep_input.bundle_fingerprint[:12]
    )
    return f"deep-{request.stage_key.value}-{candidate_order:02d}-{lineage_token}-k{selected_k_per_class}"


def _effective_engine_deep_config(
    *,
    canonical_deep_path: str,
    promotion_stage: PromotionStage,
    selected_k_per_class: int,
    k_medoids_search_values: Sequence[int],
    deep_training_config: DeepTrainingConfig | None = None,
    config_fingerprint: str | None = None,
) -> dict[str, object]:
    payload = {
        "materialization_mode": "canonical_waveform_deep_bundle",
        "promotion_stage": promotion_stage.value,
        "canonical_deep_path": canonical_deep_path,
        "k_medoids_per_class": selected_k_per_class,
        "k_medoids_search_values": list(k_medoids_search_values),
    }
    if deep_training_config is not None:
        payload["backend"] = deep_training_config.backend
        payload["allow_backend_fallback"] = deep_training_config.allow_backend_fallback
    if config_fingerprint is not None:
        payload["deep_config_fingerprint"] = config_fingerprint
    return payload


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"`{field_name}` must be an int.")
    if value < 1:
        raise ContractValidationError(f"`{field_name}` must be >= 1.")
    return value


def _write_support_artifact(
    path: Path,
    *,
    kind: str,
    schema_version: str,
    payload: Mapping[str, object],
) -> ArtifactRef:
    artifact_path = path.resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    materialized_payload = {
        "kind": kind,
        "schema_version": schema_version,
        **dict(payload),
    }
    artifact_path.write_text(
        json.dumps(materialized_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ArtifactRef(
        kind=kind,
        schema_version=schema_version,
        path=str(artifact_path),
        sha256=compute_file_sha256(artifact_path),
    )


def _require_existing_file(value: str, *, field_name: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise ContractValidationError(f"`{field_name}` must point to an existing file.")
    return path


def _require_existing_dir(value: str, *, field_name: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise ContractValidationError(f"`{field_name}` must point to an existing directory.")
    return path


def _load_json_mapping(path: Path, *, field_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"`{field_name}` must contain valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{field_name}` must deserialize to a JSON object.")
    return payload


__all__ = [
    "BEST_DEEP_ARTIFACT_NAME",
    "DEEP_CANDIDATE_REPORT_ARTIFACT_NAME",
    "DEEP_METRICS_SUMMARY_ARTIFACT_NAME",
    "materialize_waveform_capacity_refinement_candidates",
    "materialize_waveform_deep_candidates",
]
