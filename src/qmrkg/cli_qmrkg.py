"""Root CLI for listing available qmrkg commands."""

from __future__ import annotations

import argparse


COMMANDS: tuple[tuple[str, str], ...] = (
    ("qmrkg --list", "List all available qmrkg commands"),
    ("pdftopng", "Convert PDF files to PNG images"),
    ("pngtotext", "Convert PNG files to markdown text"),
    ("mdchunk", "Chunk markdown files into JSON"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmrkg",
        description="QmrKG command entrypoint.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show all executable commands",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Available commands:")
        for command, desc in COMMANDS:
            print(f"  - {command:<14} {desc}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
