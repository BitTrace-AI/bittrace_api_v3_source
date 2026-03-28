"""Main CLI entrypoint for the unified BitTrace package."""

from __future__ import annotations

import argparse
import sys

from bittrace.experimental.cli import register_experimental_commands
from bittrace.source.cli import register_supported_commands
from bittrace.v3 import ContractValidationError


def build_parser() -> argparse.ArgumentParser:
    """Build the unified BitTrace CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bittrace",
        description=(
            "Canonical BitTrace CLI with one supported shipping lane and one "
            "experimental research lane."
        ),
        epilog=(
            "Stable commands stay at the top level. Research-only workflows live under "
            "`bittrace experimental ...` and carry no stability guarantees."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_supported_commands(subparsers)
    register_experimental_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the unified BitTrace CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ContractValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
