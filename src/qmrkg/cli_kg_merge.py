"""CLI for merging raw triples into a deduplicated knowledge graph."""

import argparse
import logging
from pathlib import Path

from .config import load_run_config
from .kg_merger import KGMerger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge raw KG triples")
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.kg_merge",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["kg_merge"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    embedding_config = dict(run_cfg.get("embedding") or {})
    merger = KGMerger()
    output = merger.merge_directory(
        Path(str(run_cfg["input_dir"])),
        Path(str(run_cfg["output"])),
        embedding_config=embedding_config,
        config_path=args.config,
    )
    print(f"Merged triples saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
