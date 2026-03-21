"""CLI for knowledge graph extraction from chunks."""

import argparse
import logging
import sys
from pathlib import Path

from .kg_extractor import KGExtractor


def main():
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
        "--no-skip",
        action="store_true",
        help="Re-extract all chunks even if output exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    skip = not args.no_skip
    extractor = KGExtractor()

    input_path = args.input
    if input_path.is_file():
        paths = extractor.extract_from_chunks_file(input_path, args.output_dir, skip_existing=skip)
        print(f"Extracted {len(paths)} chunk(s) from {input_path.name}")
    elif input_path.is_dir():
        chunk_files = sorted(input_path.glob("*.json"))
        if not chunk_files:
            print(f"No JSON files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        total = 0
        for cf in chunk_files:
            paths = extractor.extract_from_chunks_file(cf, args.output_dir, skip_existing=skip)
            total += len(paths)
        print(f"Extracted {total} chunk(s) from {len(chunk_files)} file(s)")
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
