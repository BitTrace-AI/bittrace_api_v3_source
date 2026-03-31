from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("args", "expected_tokens"),
    [
        (
            ["--help"],
            [
                "campaign",
                "verify",
                "deployment-candidate",
                "persistence",
                "experimental",
            ],
        ),
        (["campaign", "--help"], ["--config", "--run-id", "--campaign-seed"]),
        (["verify", "--help"], ["run_root", "--output-dir"]),
        (["deployment-candidate", "--help"], ["--config", "--run-id", "--search-seed"]),
        (["persistence", "--help"], ["--config", "--run-id", "--source-run-root"]),
        (["experimental", "--help"], ["backend-comparison", "seed-sweep", "leandeep-max-search"]),
    ],
)
def test_bittrace_help_surface(
    project_root: Path,
    bittrace_cli: Path,
    args: list[str],
    expected_tokens: list[str],
) -> None:
    result = subprocess.run(
        [str(bittrace_cli), *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    output = result.stdout
    assert "usage:" in output.lower()
    for token in expected_tokens:
        assert token in output


def test_python_module_help_surface(project_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "bittrace", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    output = result.stdout
    assert "usage:" in output.lower()
    assert "campaign" in output
    assert "experimental" in output
