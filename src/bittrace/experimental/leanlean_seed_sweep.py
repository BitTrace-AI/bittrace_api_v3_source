"""Deterministic multi-seed sweep wrapper for the Lean-Lean deployment candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required in this venv. Install with: pip install pyyaml") from exc

from bittrace.v3 import ContractValidationError

from .backend_architecture_comparison import _write_json, _write_summary_csv, _resolve_relative_path
from bittrace.source.leanlean_deployment_candidate import (
    DEFAULT_RUNS_ROOT,
    prepare_leanlean_deployment_candidate,
    run_prepared_leanlean_deployment_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experimental" / "leanlean_seed_sweep.yaml"
SUMMARY_ARTIFACT_NAME = "leanlean_seed_sweep_summary.json"
PLAN_ARTIFACT_NAME = "leanlean_seed_sweep_plan.json"
SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-1-leanlean-seed-sweep-summary-1"
PLAN_SCHEMA_VERSION = "bittrace-bearings-v3-1-leanlean-seed-sweep-plan-1"
_AGGREGATE_METRICS = (
    "accuracy",
    "healthy_to_unhealthy_fpr",
    "unhealthy_precision",
    "unhealthy_recall",
    "unhealthy_f1",
    "macro_f1",
)


@dataclass(frozen=True, slots=True)
class PreparedLeanLeanSeedSweep:
    config_path: Path
    run_root: Path
    profile_name: str
    deployment_candidate_config_path: Path
    seeds: tuple[int, ...]
    notes: tuple[str, ...]


def load_leanlean_seed_sweep_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{resolved_path}` must deserialize to a YAML mapping.")
    for key in ("profile_name", "deployment_candidate_config", "seeds"):
        if key not in payload:
            raise ContractValidationError(f"`{resolved_path}` is missing required top-level key `{key}`.")
    return payload


def prepare_leanlean_seed_sweep(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
) -> PreparedLeanLeanSeedSweep:
    resolved_config_path = Path(config_path).resolve()
    resolved_run_root = (
        Path(run_root).resolve()
        if run_root is not None
        else (DEFAULT_RUNS_ROOT / resolved_config_path.stem / "manual_run").resolve()
    )
    payload = load_leanlean_seed_sweep_config(resolved_config_path)
    seeds = _normalize_seeds(payload.get("seeds"))
    deployment_candidate_config_path = _resolve_relative_path(
        config_file=resolved_config_path,
        raw_path=payload["deployment_candidate_config"],
        field_name="deployment_candidate_config",
    )
    return PreparedLeanLeanSeedSweep(
        config_path=resolved_config_path,
        run_root=resolved_run_root,
        profile_name=str(payload["profile_name"]),
        deployment_candidate_config_path=deployment_candidate_config_path,
        seeds=seeds,
        notes=_require_string_sequence(payload.get("notes", ()), field_name="notes"),
    )


def write_leanlean_seed_sweep_plan(
    prepared: PreparedLeanLeanSeedSweep,
) -> Path:
    plan_path = prepared.run_root / PLAN_ARTIFACT_NAME
    _write_json(
        plan_path,
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "profile_name": prepared.profile_name,
            "config_path": str(prepared.config_path),
            "run_root": str(prepared.run_root),
            "deployment_candidate_config_path": str(prepared.deployment_candidate_config_path),
            "seeds": list(prepared.seeds),
            "per_seed_run_roots": {
                str(seed): str((prepared.run_root / "seeds" / f"seed_{seed}").resolve())
                for seed in prepared.seeds
            },
            "artifacts": {
                "summary_json_path": str((prepared.run_root / SUMMARY_ARTIFACT_NAME).resolve()),
                "summary_csv_path": str((prepared.run_root / "summary.csv").resolve()),
                "summary_md_path": str((prepared.run_root / "summary.md").resolve()),
            },
            "notes": list(prepared.notes),
        },
    )
    return plan_path


def run_prepared_leanlean_seed_sweep(
    prepared: PreparedLeanLeanSeedSweep,
) -> Path:
    plan_path = write_leanlean_seed_sweep_plan(prepared)
    per_seed_results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed in prepared.seeds:
        seed_run_root = prepared.run_root / "seeds" / f"seed_{seed}"
        candidate = prepare_leanlean_deployment_candidate(
            config_path=prepared.deployment_candidate_config_path,
            run_root=seed_run_root,
            search_seed=seed,
        )
        summary_path = run_prepared_leanlean_deployment_candidate(candidate)
        summary_payload = _load_json(summary_path)
        variant = _require_mapping(summary_payload.get("variant"), field_name="variant")
        test_metrics = _require_mapping(
            _require_mapping(variant.get("metrics"), field_name="variant.metrics").get("test"),
            field_name="variant.metrics.test",
        )
        latency = _require_mapping(variant.get("latency"), field_name="variant.latency")
        per_seed_results.append(
            {
                "seed": seed,
                "run_root": str(seed_run_root.resolve()),
                "summary_path": str(summary_path.resolve()),
                "variant_summary_path": str(variant.get("variant_summary_path", "")),
                "test_metrics": dict(test_metrics),
                "model_size_bytes": int(variant.get("model_size_bytes", 0)),
                "latency_ms_per_sample": float(latency.get("per_sample_ms", 0.0)),
            }
        )
        summary_rows.append(
            {
                "seed": seed,
                "run_root": str(seed_run_root.resolve()),
                "summary_path": str(summary_path.resolve()),
                "test_accuracy": float(test_metrics["accuracy"]),
                "healthy_to_unhealthy_fpr": float(test_metrics["healthy_to_unhealthy_fpr"]),
                "unhealthy_precision": float(test_metrics["unhealthy_precision"]),
                "unhealthy_recall": float(test_metrics["unhealthy_recall"]),
                "unhealthy_f1": float(test_metrics["unhealthy_f1"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "model_size_bytes": int(variant.get("model_size_bytes", 0)),
                "latency_ms_per_sample": float(latency.get("per_sample_ms", 0.0)),
            }
        )

    summary_csv_path = prepared.run_root / "summary.csv"
    _write_summary_csv(summary_csv_path, tuple(summary_rows))
    aggregates = _aggregate_results(per_seed_results)
    stability_comments = _build_stability_comments(aggregates)
    summary_md_path = prepared.run_root / "summary.md"
    summary_md_path.write_text(
        _build_summary_markdown(prepared, summary_rows, aggregates, stability_comments),
        encoding="utf-8",
    )
    summary_json_path = prepared.run_root / SUMMARY_ARTIFACT_NAME
    _write_json(
        summary_json_path,
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "profile_name": prepared.profile_name,
            "config_path": str(prepared.config_path),
            "run_root": str(prepared.run_root),
            "plan_path": str(plan_path.resolve()),
            "deployment_candidate_config_path": str(prepared.deployment_candidate_config_path),
            "seeds": list(prepared.seeds),
            "summary_csv_path": str(summary_csv_path.resolve()),
            "summary_md_path": str(summary_md_path.resolve()),
            "per_seed_results": per_seed_results,
            "aggregates": aggregates,
            "stability_comments": stability_comments,
        },
    )
    return summary_json_path


def run_leanlean_seed_sweep(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_root: str | Path | None = None,
) -> Path:
    prepared = prepare_leanlean_seed_sweep(config_path=config_path, run_root=run_root)
    return run_prepared_leanlean_seed_sweep(prepared)


def _aggregate_results(per_seed_results: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    if not per_seed_results:
        raise ContractValidationError("Seed sweep requires at least one per-seed result.")
    aggregates: dict[str, dict[str, float]] = {}
    for metric_name in _AGGREGATE_METRICS:
        values = [
            float(_require_mapping(result["test_metrics"], field_name="test_metrics")[metric_name])
            for result in per_seed_results
        ]
        aggregates[metric_name] = {
            "mean": round(mean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "range": round(max(values) - min(values), 6),
        }
    return aggregates


def _build_stability_comments(aggregates: Mapping[str, Mapping[str, float]]) -> list[str]:
    comments: list[str] = []
    fpr_range = float(aggregates["healthy_to_unhealthy_fpr"]["range"])
    macro_range = float(aggregates["macro_f1"]["range"])
    unhealthy_f1_range = float(aggregates["unhealthy_f1"]["range"])
    if fpr_range == 0.0:
        comments.append("healthy_to_unhealthy_fpr is identical across the initial seed set.")
    elif fpr_range <= 0.01:
        comments.append("healthy_to_unhealthy_fpr remains tightly clustered across seeds.")
    else:
        comments.append("healthy_to_unhealthy_fpr moves enough across seeds to merit a wider repeat check.")
    if macro_range <= 0.02 and unhealthy_f1_range <= 0.02:
        comments.append("macro_f1 and unhealthy_f1 are reasonably stable for a first three-seed sweep.")
    else:
        comments.append("macro_f1 or unhealthy_f1 show noticeable seed sensitivity; keep the aggregate view in future tuning.")
    return comments


def _build_summary_markdown(
    prepared: PreparedLeanLeanSeedSweep,
    summary_rows: Sequence[Mapping[str, object]],
    aggregates: Mapping[str, Mapping[str, float]],
    stability_comments: Sequence[str],
) -> str:
    lines = [
        "# Lean-Lean Seed Sweep",
        "",
        f"Deployment candidate config: `{prepared.deployment_candidate_config_path}`",
        f"Seeds: `{', '.join(str(seed) for seed in prepared.seeds)}`",
        "",
        "## Per-Seed Test Metrics",
        "",
        "| Seed | Test Acc | Test FPR | Unhealthy Precision | Unhealthy Recall | Unhealthy F1 | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {seed} | {accuracy:.6f} | {fpr:.6f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {macro:.6f} |".format(
                seed=int(row["seed"]),
                accuracy=float(row["test_accuracy"]),
                fpr=float(row["healthy_to_unhealthy_fpr"]),
                precision=float(row["unhealthy_precision"]),
                recall=float(row["unhealthy_recall"]),
                f1=float(row["unhealthy_f1"]),
                macro=float(row["macro_f1"]),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Ranges",
            "",
            "| Metric | Mean | Min | Max | Range |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric_name in _AGGREGATE_METRICS:
        aggregate = aggregates[metric_name]
        lines.append(
            "| {metric} | {mean:.6f} | {min:.6f} | {max:.6f} | {range:.6f} |".format(
                metric=metric_name,
                mean=float(aggregate["mean"]),
                min=float(aggregate["min"]),
                max=float(aggregate["max"]),
                range=float(aggregate["range"]),
            )
        )
    lines.extend(["", "## Stability Notes", ""])
    lines.extend(f"- {comment}" for comment in stability_comments)
    return "\n".join(lines) + "\n"


def _normalize_seeds(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractValidationError("`seeds` must be a sequence of integers.")
    seeds: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractValidationError(f"`seeds[{index}]` must be a non-negative integer.")
        seeds.append(int(value))
    if not seeds:
        raise ContractValidationError("`seeds` must include at least one value.")
    if len(set(seeds)) != len(seeds):
        raise ContractValidationError("`seeds` must not contain duplicates.")
    return tuple(seeds)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractValidationError(f"`{path}` must deserialize to a JSON object.")
    return payload


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"`{field_name}` must be a mapping.")
    return value


def _require_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"`{field_name}` must be a sequence of strings.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item.strip() == "":
            raise ContractValidationError(f"`{field_name}[{index}]` must be a non-empty string.")
        normalized.append(item.strip())
    return tuple(normalized)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RUNS_ROOT",
    "PLAN_ARTIFACT_NAME",
    "SUMMARY_ARTIFACT_NAME",
    "PreparedLeanLeanSeedSweep",
    "load_leanlean_seed_sweep_config",
    "prepare_leanlean_seed_sweep",
    "run_leanlean_seed_sweep",
    "run_prepared_leanlean_seed_sweep",
    "write_leanlean_seed_sweep_plan",
]
