"""CLI for evaluating merged knowledge graph triples against a gold file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .evaluation import EvaluationError, evaluate_files, render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate merged KG triples against pure gold triples."
    )
    parser.add_argument(
        "--pred", type=Path, required=True, help="Predicted merged triples JSON file"
    )
    parser.add_argument("--gold", type=Path, required=True, help="Gold triples JSON file")
    parser.add_argument("--output-json", type=Path, help="Optional path for JSON report output")
    parser.add_argument("--output-md", type=Path, help="Optional path for Markdown report output")
    parser.add_argument(
        "--top-errors",
        type=int,
        default=10,
        help="Maximum false positive/negative samples to include (default: 10)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        report = evaluate_files(args.pred, args.gold, top_errors=args.top_errors)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    json_to_stdout = args.output_json is None
    status_stream = sys.stderr if json_to_stdout else sys.stdout
    json_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_report + "\n", encoding="utf-8")
        print(f"Evaluation JSON written to: {args.output_json}", file=status_stream)
    else:
        print(json_report)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"Evaluation Markdown written to: {args.output_md}", file=status_stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
