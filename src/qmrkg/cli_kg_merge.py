"""CLI for merging raw triples into a deduplicated knowledge graph."""

import argparse
import logging
from pathlib import Path

from .kg_merger import KGMerger


def main():
    parser = argparse.ArgumentParser(description="Merge raw KG triples")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/triples/raw"),
        help="Directory containing raw triple JSON files (default: data/triples/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/triples/merged/merged_triples.json"),
        help="Output path for merged triples (default: data/triples/merged/merged_triples.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    merger = KGMerger()
    output = merger.merge_directory(args.input_dir, args.output)
    print(f"Merged triples saved to: {output}")


if __name__ == "__main__":
    main()
