"""CLI for knowledge graph extraction from chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .config import load_run_config
from .kg_extractor import KGExtractor
from .tqdm_logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser(run_cfg: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract KG triples from markdown chunks")
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(str(run_cfg["input"])),
        help="Input chunks directory or single JSON file (default: data/chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(run_cfg["output_dir"])),
        help="Output directory for raw triples (default: data/triples/raw)",
    )
    parser.add_argument(
        "--mode",
        choices=["fs", "zs", "few-shot", "zero-shot"],
        default=str(run_cfg["mode"]),
        help="Prompt / extraction mode (default: fs)",
    )
    parser.add_argument(
        "--no-skip",
        dest="no_skip",
        action="store_true",
        help="Re-extract all chunks even if output exists",
    )
    parser.add_argument(
        "--skip",
        dest="no_skip",
        action="store_false",
        help="Skip existing extracted files when output already exists",
    )
    parser.set_defaults(no_skip=bool(run_cfg["no_skip"]))
    parser.add_argument(
        "--review",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["review"]),
        help="Enable in-extractor review stage (default: enabled)",
    )
    parser.add_argument(
        "--strict-evidence",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["strict_evidence"]),
        help="Require evidence to be exact chunk substring (default: enabled)",
    )
    parser.add_argument(
        "--keep-dropped",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["keep_dropped"]),
        help="Keep dropped candidates in output for auditing (default: enabled)",
    )
    parser.add_argument(
        "--min-triples",
        type=int,
        default=int(run_cfg["min_triples"]),
        help="Warn when kept triples per chunk is below this value (default: 1)",
    )
    parser.add_argument(
        "--extractor-version",
        type=str,
        default=str(run_cfg["extractor_version"]),
        help="Version label to include in output metadata",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    run_cfg = load_run_config(pre_args.config)["kg_extract"]
    args = build_parser(run_cfg).parse_args(argv)

    setup_logging(args.verbose)
    logger.info("kgextract mode: %s", args.mode)

    skip = not args.no_skip
    extractor = KGExtractor(
        mode=args.mode,
        enable_review=args.review,
        strict_evidence=args.strict_evidence,
        keep_dropped=args.keep_dropped,
        extractor_version=args.extractor_version,
    )

    input_path = args.input
    if input_path.is_file():
        paths = extractor.extract_from_chunks_file(
            input_path,
            args.output_dir,
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
        total = 0
        multi_file = len(chunk_files) > 1
        if multi_file:
            file_pbar = tqdm(
                total=len(chunk_files),
                desc="文件 0/0",
                unit="file",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
            try:
                for idx, cf in enumerate(chunk_files, start=1):
                    file_pbar.set_description_str(f"文件 {idx}/{len(chunk_files)}: {cf.name}")
                    paths = extractor.extract_from_chunks_file(
                        cf,
                        args.output_dir,
                        skip_existing=skip,
                        progress_leave=False,
                        progress_desc=f"{cf.name} chunk",
                        progress_position=1,
                    )
                    total += len(paths)
                    file_pbar.update(1)
            finally:
                file_pbar.close()
        else:
            cf = chunk_files[0]
            paths = extractor.extract_from_chunks_file(
                cf,
                args.output_dir,
                skip_existing=skip,
                progress_leave=True,
                progress_desc=f"{cf.name} 进度",
                progress_position=0,
            )
            total += len(paths)
        print(f"Extracted {total} chunk(s) from {len(chunk_files)} file(s)")
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
