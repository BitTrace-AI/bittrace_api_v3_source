#!/usr/bin/env python3
"""Run the bounded release smoke workflow for the BitTrace V3 source lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import py_compile
import re
import shlex
import subprocess
import sys
import sysconfig
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
SUMMARY_ROOT = RUNS_ROOT / "release_smoke"
SOURCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "canonical_source_profile.yaml"
DEPLOYMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "canonical_deployment_candidate.yaml"
QUIET_PERSISTENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "persistence_quiet_scout.yaml"
AGGRESSIVE_PERSISTENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "persistence_aggressive.yaml"
SUMMARY_JSON_NAME = "release_smoke_summary.json"
SUMMARY_MD_NAME = "release_smoke_summary.md"
SUMMARY_SCHEMA_VERSION = "bittrace-bearings-v3-source-release-smoke-summary-1"
_KEY_VALUE_KEY_RE = re.compile(r"^[A-Za-z0-9_.\\/-]+$")


@dataclass(slots=True)
class StepRecord:
    name: str
    status: str
    duration_seconds: float
    command: str | None = None
    note: str | None = None
    parsed_output: dict[str, str] = field(default_factory=dict)
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime("release_smoke_%Y%m%d_%H%M%S")


def _tail_lines(text: str, limit: int = 20) -> list[str]:
    lines = text.strip().splitlines()
    if not lines:
        return []
    return lines[-limit:]


def _parse_key_value_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value or not _KEY_VALUE_KEY_RE.fullmatch(key):
            continue
        parsed[key] = value
    return parsed


def _format_command(argv: list[str]) -> str:
    return shlex.join(argv)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary_payload(
    *,
    smoke_run_id: str,
    summary_dir: Path,
    run_ids: dict[str, str],
    cli_executable: str | None,
    steps: list[StepRecord],
) -> dict[str, object]:
    status = "PASS" if all(step.status == "PASS" for step in steps) else "FAIL"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "smoke_run_id": smoke_run_id,
        "status": status,
        "project_root": str(PROJECT_ROOT),
        "summary_dir": str(summary_dir),
        "summary_json_path": str(summary_dir / SUMMARY_JSON_NAME),
        "summary_md_path": str(summary_dir / SUMMARY_MD_NAME),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "supported_cli_executable": cli_executable,
        "run_ids": run_ids,
        "steps": [asdict(step) for step in steps],
    }


def _summary_markdown(payload: dict[str, object]) -> str:
    run_ids = payload["run_ids"]
    lines = [
        "# Release Smoke Summary",
        "",
        f"- Smoke run id: `{payload['smoke_run_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Python: `{payload['python']['version'].splitlines()[0]}`",
        f"- Supported CLI executable: `{payload['supported_cli_executable']}`",
        f"- Summary JSON: `{payload['summary_json_path']}`",
        "",
        "## Run IDs",
        "",
        f"- campaign: `{run_ids['campaign']}`",
        f"- deployment_candidate: `{run_ids['deployment_candidate']}`",
        f"- quiet_persistence: `{run_ids['quiet_persistence']}`",
        f"- aggressive_persistence: `{run_ids['aggressive_persistence']}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Seconds | Command / Note | Key outputs |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for step in payload["steps"]:
        detail = step["command"] or step["note"] or ""
        outputs = ", ".join(
            f"{key}={value}" for key, value in sorted(step["parsed_output"].items())
        )
        lines.append(
            f"| {step['name']} | {step['status']} | {step['duration_seconds']:.2f} | "
            f"`{detail}` | `{outputs}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _persist_summary(
    *,
    smoke_run_id: str,
    summary_dir: Path,
    run_ids: dict[str, str],
    cli_executable: str | None,
    steps: list[StepRecord],
) -> None:
    payload = _summary_payload(
        smoke_run_id=smoke_run_id,
        summary_dir=summary_dir,
        run_ids=run_ids,
        cli_executable=cli_executable,
        steps=steps,
    )
    _write_text(summary_dir / SUMMARY_JSON_NAME, json.dumps(payload, indent=2, sort_keys=True))
    _write_text(summary_dir / SUMMARY_MD_NAME, _summary_markdown(payload))


def _record_python_compile() -> StepRecord:
    start = time.monotonic()
    compiled_files: list[str] = []
    try:
        for root_name in ("src", "scripts"):
            for path in sorted((PROJECT_ROOT / root_name).rglob("*.py")):
                py_compile.compile(path, doraise=True)
                compiled_files.append(str(path.relative_to(PROJECT_ROOT)))
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        return StepRecord(
            name="py_compile",
            status="FAIL",
            duration_seconds=duration,
            note=f"py_compile failed: {exc}",
        )
    duration = time.monotonic() - start
    return StepRecord(
        name="py_compile",
        status="PASS",
        duration_seconds=duration,
        note=f"Compiled {len(compiled_files)} Python files.",
        parsed_output={"compiled_file_count": str(len(compiled_files))},
    )


def _record_import_smoke() -> StepRecord:
    start = time.monotonic()
    try:
        __import__("bittrace")
        __import__("bittrace.v3")
        __import__("bittrace_bearings_v3_source")
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        return StepRecord(
            name="import_smoke",
            status="FAIL",
            duration_seconds=duration,
            note=f"Import smoke failed: {exc}",
        )
    duration = time.monotonic() - start
    return StepRecord(
        name="import_smoke",
        status="PASS",
        duration_seconds=duration,
        note="Imported bittrace, bittrace.v3, and bittrace_bearings_v3_source.",
    )


def _cli_path() -> Path:
    candidate = Path(sysconfig.get_path("scripts")) / "bittrace-source"
    if os.name == "nt":
        candidate = candidate.with_suffix(".exe")
    return candidate


def _run_command(
    *,
    name: str,
    argv: list[str],
    summary_dir: Path,
) -> StepRecord:
    stdout_log_path = summary_dir / f"{name}.stdout.log"
    stderr_log_path = summary_dir / f"{name}.stderr.log"
    command = _format_command(argv)
    print(f"[release-smoke] {name}: {command}", flush=True)
    start = time.monotonic()
    result = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.monotonic() - start
    _write_text(stdout_log_path, result.stdout)
    _write_text(stderr_log_path, result.stderr)
    status = "PASS" if result.returncode == 0 else "FAIL"
    note = None if status == "PASS" else f"Command exited with status {result.returncode}."
    return StepRecord(
        name=name,
        status=status,
        duration_seconds=duration,
        command=command,
        note=note,
        parsed_output=_parse_key_value_lines(result.stdout),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        stdout_tail=_tail_lines(result.stdout),
        stderr_tail=_tail_lines(result.stderr),
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke_run_id = argv[0] if argv else _default_run_id()
    summary_dir = (SUMMARY_ROOT / smoke_run_id).resolve()
    if summary_dir.exists():
        print(f"Error: summary directory already exists: {summary_dir}", file=sys.stderr)
        return 2
    summary_dir.mkdir(parents=True, exist_ok=False)

    run_ids = {
        "campaign": f"{smoke_run_id}__campaign",
        "deployment_candidate": f"{smoke_run_id}__deployment_candidate",
        "quiet_persistence": f"{smoke_run_id}__quiet_scout",
        "aggressive_persistence": f"{smoke_run_id}__aggressive",
    }
    print(f"[release-smoke] smoke_run_id={smoke_run_id}", flush=True)
    for label, run_id in run_ids.items():
        print(f"[release-smoke] {label}_run_id={run_id}", flush=True)

    steps: list[StepRecord] = []
    cli_executable: str | None = None

    try:
        install_step = _run_command(
            name="editable_install",
            argv=[sys.executable, "-m", "pip", "install", "-e", "."],
            summary_dir=summary_dir,
        )
        steps.append(install_step)
        cli_path = _cli_path()
        cli_executable = str(cli_path)
        if install_step.status != "PASS":
            raise RuntimeError("editable install failed")
        if not cli_path.is_file():
            steps.append(
                StepRecord(
                    name="resolve_cli_executable",
                    status="FAIL",
                    duration_seconds=0.0,
                    note=f"Expected supported CLI executable at {cli_path}.",
                )
            )
            raise RuntimeError("supported CLI executable not found")
        steps.append(
            StepRecord(
                name="resolve_cli_executable",
                status="PASS",
                duration_seconds=0.0,
                note=f"Resolved supported CLI executable at {cli_path}.",
                parsed_output={"cli_executable": str(cli_path)},
            )
        )

        py_compile_step = _record_python_compile()
        steps.append(py_compile_step)
        if py_compile_step.status != "PASS":
            raise RuntimeError("py_compile failed")

        import_step = _record_import_smoke()
        steps.append(import_step)
        if import_step.status != "PASS":
            raise RuntimeError("import smoke failed")

        help_step = _run_command(
            name="cli_help",
            argv=[str(cli_path), "--help"],
            summary_dir=summary_dir,
        )
        steps.append(help_step)
        if help_step.status != "PASS":
            raise RuntimeError("CLI help failed")

        campaign_step = _run_command(
            name="canonical_campaign",
            argv=[
                str(cli_path),
                "campaign",
                "--config",
                str(SOURCE_CONFIG_PATH),
                "--run-id",
                run_ids["campaign"],
                "--runs-root",
                str(RUNS_ROOT),
                "--campaign-seed",
                "31",
            ],
            summary_dir=summary_dir,
        )
        steps.append(campaign_step)
        if campaign_step.status != "PASS":
            raise RuntimeError("canonical campaign failed")
        campaign_run_root = campaign_step.parsed_output["run_root"]

        verify_step = _run_command(
            name="verify_parity",
            argv=[str(cli_path), "verify", campaign_run_root],
            summary_dir=summary_dir,
        )
        steps.append(verify_step)
        if verify_step.status != "PASS":
            raise RuntimeError("verify/parity failed")

        deployment_step = _run_command(
            name="deployment_candidate",
            argv=[
                str(cli_path),
                "deployment-candidate",
                "--config",
                str(DEPLOYMENT_CONFIG_PATH),
                "--run-id",
                run_ids["deployment_candidate"],
                "--runs-root",
                str(RUNS_ROOT),
                "--search-seed",
                "7100",
            ],
            summary_dir=summary_dir,
        )
        steps.append(deployment_step)
        if deployment_step.status != "PASS":
            raise RuntimeError("deployment candidate failed")
        deployment_run_root = deployment_step.parsed_output["run_root"]

        quiet_step = _run_command(
            name="quiet_persistence",
            argv=[
                str(cli_path),
                "persistence",
                "--config",
                str(QUIET_PERSISTENCE_CONFIG_PATH),
                "--source-run-root",
                deployment_run_root,
                "--run-id",
                run_ids["quiet_persistence"],
            ],
            summary_dir=summary_dir,
        )
        steps.append(quiet_step)
        if quiet_step.status != "PASS":
            raise RuntimeError("quiet persistence failed")

        aggressive_step = _run_command(
            name="aggressive_persistence",
            argv=[
                str(cli_path),
                "persistence",
                "--config",
                str(AGGRESSIVE_PERSISTENCE_CONFIG_PATH),
                "--source-run-root",
                deployment_run_root,
                "--run-id",
                run_ids["aggressive_persistence"],
            ],
            summary_dir=summary_dir,
        )
        steps.append(aggressive_step)
        if aggressive_step.status != "PASS":
            raise RuntimeError("aggressive persistence failed")
    except Exception as exc:  # noqa: BLE001
        print(f"[release-smoke] failed: {exc}", file=sys.stderr, flush=True)
        _persist_summary(
            smoke_run_id=smoke_run_id,
            summary_dir=summary_dir,
            run_ids=run_ids,
            cli_executable=cli_executable,
            steps=steps,
        )
        print(f"[release-smoke] summary_json={summary_dir / SUMMARY_JSON_NAME}", file=sys.stderr, flush=True)
        print(f"[release-smoke] summary_md={summary_dir / SUMMARY_MD_NAME}", file=sys.stderr, flush=True)
        return 1

    _persist_summary(
        smoke_run_id=smoke_run_id,
        summary_dir=summary_dir,
        run_ids=run_ids,
        cli_executable=cli_executable,
        steps=steps,
    )
    print(f"[release-smoke] PASS", flush=True)
    print(f"[release-smoke] summary_json={summary_dir / SUMMARY_JSON_NAME}", flush=True)
    print(f"[release-smoke] summary_md={summary_dir / SUMMARY_MD_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
