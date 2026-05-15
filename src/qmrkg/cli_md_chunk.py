"""CLI for markdown chunking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_run_config, optional_path
from .markdown_chunker import MarkdownChunker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk markdown files into JSON documents.")
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.md_chunk",
    )
    return parser


def _collect_markdown(markdown_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(markdown_dir.glob(pattern))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["md_chunk"]

    markdown = optional_path(run_cfg.get("input_file"))
    markdown_dir = Path(str(run_cfg["markdown_dir"]))
    output = optional_path(run_cfg.get("output"))
    chunk_dir = Path(str(run_cfg["chunk_dir"]))
    max_tokens = int(run_cfg["max_tokens"])
    recursive = bool(run_cfg["recursive"])

    logging.basicConfig(level=logging.INFO)
    chunker = MarkdownChunker(max_tokens=max_tokens)

    if markdown:
        if not markdown.exists():
            print(f"Error: Markdown file not found: {markdown}", file=sys.stderr)
            return 1

        json_path = chunker.process_and_save(
            markdown_path=markdown,
            output_path=output,
            chunk_dir=chunk_dir,
        )
        print(f"Chunked: {markdown.name}")
        print(f"JSON saved to: {json_path}")
        return 0

    if output:
        print("Error: run.md_chunk.output requires run.md_chunk.input_file", file=sys.stderr)
        return 1

    if not markdown_dir.exists():
        print(f"Error: Markdown directory not found: {markdown_dir}", file=sys.stderr)
        return 1

    markdown_files = _collect_markdown(markdown_dir, recursive)
    if not markdown_files:
        print(f"No markdown files found in {markdown_dir}")
        return 0

    chunk_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    for markdown_path in markdown_files:
        try:
            chunker.process_and_save(markdown_path=markdown_path, chunk_dir=chunk_dir)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {markdown_path} ({exc})", file=sys.stderr)

    print(f"Processed {len(markdown_files)} markdown files")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Chunk dir: {chunk_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
