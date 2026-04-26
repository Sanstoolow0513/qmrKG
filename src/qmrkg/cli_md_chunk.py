"""CLI for markdown chunking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .markdown_chunker import MarkdownChunker, merge_book_pages


def _build_parser(run_cfg: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk markdown files into JSON documents.")
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument("--markdown", type=Path, help="Process a single markdown file")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=Path(str(run_cfg["markdown_dir"])),
        help="Directory containing markdown files (default: data/markdown)",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path for --markdown mode")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path(str(run_cfg["chunk_dir"])),
        help="Directory for generated JSON chunks (default: data/chunks)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(run_cfg["max_tokens"]),
        help="Maximum tokens per chunk (default: 4000)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=bool(run_cfg["recursive"]),
        help="Search subdirectories when using --markdown-dir",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=bool(run_cfg["merge"]),
        help=(
            "Merge-and-chunk mode: scan --markdown-dir for book subdirectories, "
            "merge per-page MD files into a single clean book MD, save it to "
            "--markdown-dir, then chunk using the heading-based strategy."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def _collect_markdown(markdown_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(markdown_dir.glob(pattern))


def _run_merge_mode(args: argparse.Namespace, chunker: MarkdownChunker) -> int:
    """Merge per-page MD files per book subdirectory, then chunk each merged MD."""
    markdown_dir: Path = args.markdown_dir
    chunk_dir: Path = args.chunk_dir

    if not markdown_dir.exists():
        print(f"Error: Markdown directory not found: {markdown_dir}", file=sys.stderr)
        return 1

    book_dirs = sorted(p for p in markdown_dir.iterdir() if p.is_dir())
    if not book_dirs:
        print(f"No book subdirectories found in {markdown_dir}")
        return 0

    chunk_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    for book_dir in book_dirs:
        page_files = sorted(book_dir.glob("*_page_*.md"))
        if not page_files:
            print(f"Skipping {book_dir.name}: no *_page_*.md files found")
            continue

        merged_md_path = markdown_dir / f"{book_dir.name}.md"
        json_path = chunk_dir / f"{book_dir.name}.json"

        try:
            merged_text = merge_book_pages(page_files, output_path=merged_md_path)
            chunks = chunker.chunk_document(merged_text, source_file=str(merged_md_path))
            chunk_data = [
                {
                    "titles": c.titles,
                    "content": c.content,
                    "token_count": c.token_count,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "source_file": c.source_file,
                }
                for c in chunks
            ]
            json_path.write_text(
                json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"Merged & chunked: {book_dir.name} "
                f"({len(page_files)} pages → {len(chunks)} chunks)"
            )
            print(f"  MD:   {merged_md_path}")
            print(f"  JSON: {json_path}")
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {book_dir.name} ({exc})", file=sys.stderr)

    print(f"\nProcessed {success + failed} books")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    run_cfg = load_run_config(pre_args.config)["md_chunk"]

    parser = _build_parser(run_cfg)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    chunker = MarkdownChunker(max_tokens=args.max_tokens)

    if args.merge:
        return _run_merge_mode(args, chunker)

    if args.markdown:
        if not args.markdown.exists():
            print(f"Error: Markdown file not found: {args.markdown}", file=sys.stderr)
            return 1

        json_path = chunker.process_and_save(
            markdown_path=args.markdown,
            output_path=args.output,
            chunk_dir=args.chunk_dir,
        )
        print(f"Chunked: {args.markdown.name}")
        print(f"JSON saved to: {json_path}")
        return 0

    if args.output:
        print("Error: --output can only be used with --markdown", file=sys.stderr)
        return 1

    if not args.markdown_dir.exists():
        print(f"Error: Markdown directory not found: {args.markdown_dir}", file=sys.stderr)
        return 1

    markdown_files = _collect_markdown(args.markdown_dir, args.recursive)
    if not markdown_files:
        print(f"No markdown files found in {args.markdown_dir}")
        return 0

    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    for markdown_path in markdown_files:
        try:
            chunker.process_and_save(markdown_path=markdown_path, chunk_dir=args.chunk_dir)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {markdown_path} ({exc})", file=sys.stderr)

    print(f"Processed {len(markdown_files)} markdown files")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Chunk dir: {args.chunk_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
