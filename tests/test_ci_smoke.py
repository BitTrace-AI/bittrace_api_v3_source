from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_ci_smoke_script_passes(project_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "run_ci_smoke.py")],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ci_smoke=PASS" in result.stdout
