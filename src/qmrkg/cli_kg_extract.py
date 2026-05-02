"""CLI for knowledge graph extraction from chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .kg_extractor import KGExtractor
from .tqdm_logging import setup_logging

logger = logging.getLogger(__name__)

_VALID_MODES = {"fs", "zs", "few-shot", "zero-shot", "few_shot", "zero_shot"}


def build_parser(_run_cfg: dict[str, object] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract KG triples from markdown chunks")
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.kg_extract",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cfg = load_run_config(args.config)["kg_extract"]

    input_path = Path(str(run_cfg["input"]))
    output_dir = Path(str(run_cfg["output_dir"]))
    mode = str(run_cfg["mode"])
    if mode not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES))
        print(f"Error: run.kg_extract.mode must be one of: {valid}", file=sys.stderr)
        return 1

    setup_logging(False)
    logger.info("kgextract mode: %s", mode)

    skip = not bool(run_cfg["no_skip"])
    extractor = KGExtractor(
        config_path=args.config,
        mode=mode,
        enable_review=bool(run_cfg["review"]),
        strict_evidence=bool(run_cfg["strict_evidence"]),
        keep_dropped=bool(run_cfg["keep_dropped"]),
        extractor_version=str(run_cfg["extractor_version"]),
    )

    if input_path.is_file():
        paths = extractor.extract_from_chunks_file(
            input_path,
            output_dir,
            skip_existing=skip,
            progress_desc=f"{input_path.name} 进度",
            progress_position=0,
        )
        print(f"Extracted {len(paths)} chunk(s) from {input_path.name}")
    elif input_path.is_dir():
        chunk_files = sorted(input_path.glob("*.json"))
        if not chunk_files:
            print(f"No JSON files found in {input_path}", file=sys.stderr)
            return 1
        paths = extractor.extract_from_chunks_files(
            chunk_files,
            output_dir,
            skip_existing=skip,
            progress_leave=True,
            progress_desc=f"{len(chunk_files)} file(s) chunk",
            progress_position=0,
        )
        total = len(paths)
        print(f"Extracted {total} chunk(s) from {len(chunk_files)} file(s)")
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
