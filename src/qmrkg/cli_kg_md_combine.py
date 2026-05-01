"""CLI: merge per-page OCR markdown files into one file per book."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .markdown_chunker import merge_book_pages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-page markdown (*_page_*.md) under each book subfolder of "
            "run.kg_md_combine.markdown_dir into a single {bookname}.md at the markdown root."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.kg_md_combine",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["kg_md_combine"]

    logging.basicConfig(level=logging.INFO)

    markdown_dir = Path(str(run_cfg["markdown_dir"]))
    if not markdown_dir.exists():
        print(f"Error: Markdown directory not found: {markdown_dir}", file=sys.stderr)
        return 1

    book_dirs = sorted(p for p in markdown_dir.iterdir() if p.is_dir())
    if not book_dirs:
        print(f"No book subdirectories found in {markdown_dir}")
        return 0

    success = 0
    failed = 0

    page_glob = str(run_cfg["page_glob"])
    for book_dir in book_dirs:
        page_files = sorted(book_dir.glob(page_glob))
        if not page_files:
            print(f"Skipping {book_dir.name}: no files matching {page_glob!r}")
            continue

        merged_md_path = markdown_dir / f"{book_dir.name}.md"
        try:
            merge_book_pages(page_files, output_path=merged_md_path)
            print(
                f"Merged: {book_dir.name} — {len(page_files)} page file(s) -> {merged_md_path.name}"
            )
            success += 1
        except OSError as exc:
            failed += 1
            print(f"Failed: {book_dir.name} ({exc})", file=sys.stderr)

    print(f"\nDone: {success} book(s) merged, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
