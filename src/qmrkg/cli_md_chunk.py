"""CLI for markdown chunking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .markdown_chunker import MarkdownChunker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk markdown files into JSON documents.")
    parser.add_argument("--markdown", type=Path, help="Process a single markdown file")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=Path("data/markdown"),
        help="Directory containing markdown files (default: data/markdown)",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path for --markdown mode")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Directory for generated JSON chunks (default: data/chunks)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens per chunk (default: 4000)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when using --markdown-dir",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def _collect_markdown(markdown_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(markdown_dir.glob(pattern))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    chunker = MarkdownChunker(max_tokens=args.max_tokens)

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
