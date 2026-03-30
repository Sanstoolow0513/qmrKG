"""CLI for merging raw triples into a deduplicated knowledge graph."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .kg_merger import KGMerger
from .ner_prompts import ExtractionPromptKind


def resolve_kgmerge_paths(
    input_dir: Path | None,
    output: Path | None,
    prompt_kind: str | None,
) -> tuple[Path, Path]:
    """Align merge I/O with kgextract layout: raw/<prompt-kind>/ and merged file per strategy."""
    if input_dir is not None:
        in_path = Path(input_dir)
    elif prompt_kind is not None:
        in_path = Path("data/triples/raw") / prompt_kind
    else:
        in_path = Path("data/triples/raw")

    if output is not None:
        out_path = Path(output)
    elif prompt_kind is not None:
        out_path = Path("data/triples/merged") / f"merged_triples_{prompt_kind}.json"
    else:
        out_path = Path("data/triples/merged/merged_triples.json")
    return in_path, out_path


def main():
    parser = argparse.ArgumentParser(description="Merge raw KG triples")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory of per-chunk raw JSON (*.json). "
            "If omitted and --prompt-kind is set, uses data/triples/raw/<prompt-kind>. "
            "If both omitted, uses data/triples/raw (flat layout, legacy)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Merged output JSON path. "
            "If omitted and --prompt-kind is set, uses "
            "data/triples/merged/merged_triples_<prompt-kind>.json; "
            "otherwise data/triples/merged/merged_triples.json."
        ),
    )
    parser.add_argument(
        "--prompt-kind",
        choices=[k.value for k in ExtractionPromptKind],
        default=None,
        help=(
            "Match kgextract --prompt-kind: sets default input/output paths so the merge stage "
            "reads the same subtree as extraction (e.g. zero_shot -> raw/zero_shot)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_dir, output_path = resolve_kgmerge_paths(
        args.input_dir,
        args.output,
        args.prompt_kind,
    )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    merger = KGMerger()
    output = merger.merge_directory(input_dir, output_path)
    print(f"Merged triples saved to: {output}")


if __name__ == "__main__":
    main()
