"""CLI for merging raw triples into a deduplicated knowledge graph."""

import argparse
import logging
from pathlib import Path

from .config import load_run_config
from .kg_merger import KGMerger


def main(argv: list[str] | None = None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    run_cfg = load_run_config(pre_args.config)["kg_merge"]

    parser = argparse.ArgumentParser(description="Merge raw KG triples")
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(str(run_cfg["input_dir"])),
        help="Directory containing raw triple JSON files (default: data/triples/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(str(run_cfg["output"])),
        help="Output path for merged triples (default: data/triples/merged/merged_triples.json)",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding canonicalization even if config enables it",
    )
    parser.add_argument(
        "--embedding-task",
        help="Override embedding task name, e.g. entity_embed",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Override embedding cosine similarity threshold",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    embedding_config = dict(run_cfg.get("embedding", {}))
    if args.no_embedding:
        embedding_config["enabled"] = False
    if args.embedding_task is not None:
        embedding_config["task_name"] = args.embedding_task
    if args.similarity_threshold is not None:
        embedding_config["similarity_threshold"] = args.similarity_threshold

    merger = KGMerger()
    output = merger.merge_directory(
        args.input_dir,
        args.output,
        embedding_config=embedding_config,
        config_path=args.config,
    )
    print(f"Merged triples saved to: {output}")


if __name__ == "__main__":
    main()
