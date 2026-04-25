"""CLI for knowledge graph extraction from chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .kg_extractor import KGExtractor
from .tqdm_logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract KG triples from markdown chunks")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chunks"),
        help="Input chunks directory or single JSON file (default: data/chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/triples/raw"),
        help="Output directory for raw triples (default: data/triples/raw)",
    )
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompt / extraction mode (default: zero-shot)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract all chunks even if output exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging(args.verbose)
    logger.info("kgextract mode: %s", args.mode)

    skip = not args.no_skip
    extractor = KGExtractor(mode=args.mode)

    input_path = args.input
    if input_path.is_file():
        paths = extractor.extract_from_chunks_file(input_path, args.output_dir, skip_existing=skip)
        print(f"Extracted {len(paths)} chunk(s) from {input_path.name}")
    elif input_path.is_dir():
        chunk_files = sorted(input_path.glob("*.json"))
        if not chunk_files:
            print(f"No JSON files found in {input_path}", file=sys.stderr)
            return 1
        total = 0
        multi_file = len(chunk_files) > 1
        file_iter = (
            tqdm(
                chunk_files,
                desc="kgextract",
                unit="file",
                total=len(chunk_files),
                dynamic_ncols=True,
            )
            if multi_file
            else chunk_files
        )
        for cf in file_iter:
            paths = extractor.extract_from_chunks_file(
                cf,
                args.output_dir,
                skip_existing=skip,
                progress_leave=not multi_file,
            )
            total += len(paths)
        print(f"Extracted {total} chunk(s) from {len(chunk_files)} file(s)")
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
