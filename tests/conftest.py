from __future__ import annotations

import os
from pathlib import Path
import sysconfig

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bittrace_cli_path() -> Path:
    executable_name = "bittrace.exe" if os.name == "nt" else "bittrace"
    return Path(sysconfig.get_path("scripts")) / executable_name


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def bittrace_cli() -> Path:
    path = _bittrace_cli_path()
    if not path.is_file():
        pytest.fail(f"Expected installed `bittrace` console script at {path}")
    return path
