"""Agent harness CLI: thesis-stage list + execution status over existing qmrkg tools."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent_harness import (
    STAGES,
    collect_execution_status,
    format_stages_json,
    format_status_human,
    format_status_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmrkg-harness",
        description=(
            "Map graduation-design (task.md) narrative to existing qmrkg stage CLIs; "
            "list commands and inspect data/ artifact counts."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root for status (default: current working directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Print ordered stages and example shell commands")
    p_list.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (for tool-using agents)",
    )

    p_status = sub.add_parser("status", help="Summarize data/ directory signals")
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else None

    if args.command == "list":
        if args.json:
            print(format_stages_json())
            return 0
        print(
            "Stages (thesis anchor -> existing CLI). Run in order unless you resume mid-pipeline.\n"
        )
        for i, s in enumerate(STAGES, 1):
            print(f"{i}. [{s.id}] {s.thesis_anchor}")
            print(f"   CLI: {s.cli_name}")
            print(f"   {s.description}")
            print(f"   Example: {s.example_bash}")
            print()
        print("Discovery: uv run qmrkg --list")
        return 0

    if args.command == "status":
        status = collect_execution_status(root)
        if args.json:
            print(format_status_json(root))
            return 0
        print(format_status_human(status))
        return 0

    raise AssertionError("unhandled harness subcommand")


if __name__ == "__main__":
    raise SystemExit(main())
