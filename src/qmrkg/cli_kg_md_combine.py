"""CLI: merge per-page OCR markdown files into one file per book."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .markdown_chunker import merge_book_pages


def _build_parser(run_cfg: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-page markdown (*_page_*.md) under each book subfolder of "
            "--markdown-dir into a single {bookname}.md at the markdown root."
        ),
    )
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=Path(str(run_cfg["markdown_dir"])),
        help="Root directory: each immediate subdirectory is one book (default: data/markdown)",
    )
    parser.add_argument(
        "--page-glob",
        type=str,
        default=str(run_cfg["page_glob"]),
        help="Glob for per-page markdown inside each book subdir (default: run.kg_md_combine.page_glob)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    run_cfg = load_run_config(pre_args.config)["kg_md_combine"]

    parser = _build_parser(run_cfg)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    markdown_dir: Path = args.markdown_dir
    if not markdown_dir.exists():
        print(f"Error: Markdown directory not found: {markdown_dir}", file=sys.stderr)
        return 1

    book_dirs = sorted(p for p in markdown_dir.iterdir() if p.is_dir())
    if not book_dirs:
        print(f"No book subdirectories found in {markdown_dir}")
        return 0

    success = 0
    failed = 0

    page_glob = args.page_glob
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
