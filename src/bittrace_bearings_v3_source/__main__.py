"""Module entrypoint for `python -m bittrace_bearings_v3_source`."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
