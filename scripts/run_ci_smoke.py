#!/usr/bin/env python3
"""Run the lightweight BitTrace smoke checks used by CI."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile
import shlex
import subprocess
import sys
import sysconfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_TREES = ("src", "scripts", "tests")
IMPORT_SMOKE = "import bittrace, bittrace.experimental, bittrace.source, bittrace.v3"
HELP_COMMANDS: tuple[tuple[str, list[str]], ...] = (
    ("bittrace_help", ["--help"]),
    ("campaign_help", ["campaign", "--help"]),
    ("verify_help", ["verify", "--help"]),
    ("deployment_candidate_help", ["deployment-candidate", "--help"]),
    ("persistence_help", ["persistence", "--help"]),
    ("experimental_help", ["experimental", "--help"]),
)


def _format_command(argv: list[str]) -> str:
    return shlex.join(argv)


def _cli_path() -> Path:
    executable_name = "bittrace.exe" if os.name == "nt" else "bittrace"
    return Path(sysconfig.get_path("scripts")) / executable_name


def _run_command(name: str, argv: list[str]) -> None:
    print(f"[ci-smoke] {name}: {_format_command(argv)}", flush=True)
    result = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    raise SystemExit(result.returncode)


def _compile_python_trees() -> None:
    compiled_files = 0
    for root_name in PYTHON_TREES:
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            py_compile.compile(path, doraise=True)
            compiled_files += 1
    print(f"[ci-smoke] compiled_file_count={compiled_files}", flush=True)


def main() -> int:
    cli_path = _cli_path()
    if not cli_path.is_file():
        print(
            f"Error: expected installed `bittrace` console script at {cli_path}",
            file=sys.stderr,
        )
        return 2

    _compile_python_trees()
    _run_command(
        "import_smoke",
        [sys.executable, "-c", IMPORT_SMOKE],
    )
    for name, args in HELP_COMMANDS:
        _run_command(name, [str(cli_path), *args])
    _run_command(
        "module_help",
        [sys.executable, "-m", "bittrace", "--help"],
    )
    print("[ci-smoke] ci_smoke=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
