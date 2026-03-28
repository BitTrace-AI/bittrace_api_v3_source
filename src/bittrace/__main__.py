"""Module entrypoint for `python -m bittrace`."""

from __future__ import annotations

from bittrace.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
