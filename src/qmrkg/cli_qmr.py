"""CLI: run the full QmrKG pipeline (PDF -> Neo4j)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from . import (
    cli_kg_extract,
    cli_kg_md_combine,
    cli_kg_merge,
    cli_kg_neo4j,
    cli_md_chunk,
    cli_pdf_to_png,
    cli_png_to_text,
)

logger = logging.getLogger(__name__)

STAGE_NAMES = (
    "pdftopng",
    "pngtotext",
    "kgmdcombine",
    "mdchunk",
    "kgextract",
    "kgmerge",
    "kgneo4j",
)


def _stages() -> list[tuple[str, Callable[[list[str] | None], int | None]]]:
    return [
        ("pdftopng", cli_pdf_to_png.main),
        ("pngtotext", cli_png_to_text.main),
        ("kgmdcombine", cli_kg_md_combine.main),
        ("mdchunk", cli_md_chunk.main),
        ("kgextract", cli_kg_extract.main),
        ("kgmerge", _wrap_kgmerge),
        ("kgneo4j", _wrap_kgneo4j),
    ]


def _wrap_kgmerge(argv: list[str] | None) -> int | None:
    cli_kg_merge.main(argv)
    return 0


def _wrap_kgneo4j(argv: list[str] | None) -> int | None:
    cli_kg_neo4j.main(argv)
    return 0


def _normalize_token(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _parse_stage_arg(value: str) -> str:
    n = _normalize_token(value)
    aliases = {
        "combine": "kgmdcombine",
        "merge_md": "kgmdcombine",
        "neo4j": "kgneo4j",
    }
    s = aliases.get(n, n)
    if s not in STAGE_NAMES:
        valid = ", ".join(STAGE_NAMES)
        raise argparse.ArgumentTypeError(f"Unknown stage {value!r}; expected one of: {valid}")
    return s


def _effective_to_stage(args: argparse.Namespace) -> str:
    if args.to_stage is not None:
        t = str(args.to_stage)
        if args.no_neo4j and STAGE_NAMES.index(t) > STAGE_NAMES.index("kgmerge"):
            return "kgmerge"
        return t
    return "kgmerge" if args.no_neo4j else "kgneo4j"


def _iter_selected_stages(
    from_stage: str | None, to_stage: str
) -> Iterator[tuple[str, Callable[[list[str] | None], int | None]]]:
    all_stages = _stages()
    names = [n for n, _ in all_stages]
    start = names.index(from_stage) if from_stage is not None else 0
    end = names.index(to_stage)
    if start > end:
        raise ValueError("from_stage must not be after to_stage")
    for i in range(start, end + 1):
        yield all_stages[i]


def _build_sub_argv(config: Path | None, verbose: bool) -> list[str] | None:
    out: list[str] = []
    if config is not None:
        out.extend(["--config", str(config)])
    if verbose:
        out.append("-v")
    return out or None


def _coerce_exit_code(code: int | None) -> int:
    if code is None:
        return 0
    return int(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmr",
        description=(
            "Run the end-to-end pipeline: pdftopng -> pngtotext -> kgmdcombine -> mdchunk -> "
            "kgextract -> kgmerge -> kgneo4j (optional)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path (passed to every sub-command)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Pass -v to each sub-command (verbose logging where supported)",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Stop after kgmerge (skip importing into Neo4j)",
    )
    parser.add_argument(
        "--from-stage",
        type=_parse_stage_arg,
        metavar="STAGE",
        help=f"Start at this stage ({', '.join(STAGE_NAMES)})",
    )
    parser.add_argument(
        "--to-stage",
        type=_parse_stage_arg,
        metavar="STAGE",
        help="End at this stage (inclusive; default: kgneo4j, or kgmerge with --no-neo4j)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    to_stage = _effective_to_stage(args)
    from_stage = str(args.from_stage) if args.from_stage is not None else None

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )

    sub_base = _build_sub_argv(args.config, args.verbose)
    try:
        iterator = _iter_selected_stages(from_stage, to_stage)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for stage_name, run in iterator:
        print(f"\n=== [qmr: {stage_name}] ===\n", flush=True)
        logger.info("Running stage %s", stage_name)
        rc = _coerce_exit_code(run(sub_base))
        if rc != 0:
            print(f"qmr: stage {stage_name!r} exited with {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
