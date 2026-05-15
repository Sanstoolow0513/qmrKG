"""Root CLI for listing available qmrkg commands."""

from __future__ import annotations

import argparse

COMMANDS: tuple[tuple[str, str], ...] = (
    ("qmrkg --list", "List all available qmrkg commands"),
    ("qmr", "Run the full pipeline (PDF -> Neo4j)"),
    ("pdftopng", "Convert PDF files to PNG images"),
    ("pngtotext", "Convert PNG files to markdown text (VLM OCR)"),
    ("mdchunk", "Chunk markdown files into JSON"),
    ("kgmdcombine", "Merge per-page markdown into one file per book"),
    ("kgextract", "Extract KG triples from markdown chunks"),
    ("kgmerge", "Merge raw triple JSON into a deduplicated graph"),
    ("kgneo4j", "Import merged triples into Neo4j"),
    ("kgeval", "Evaluate merged triples against a gold set"),
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
        width = max(len(cmd) for cmd, _ in COMMANDS)
        for command, desc in COMMANDS:
            print(f"  - {command:<{width}}  {desc}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
