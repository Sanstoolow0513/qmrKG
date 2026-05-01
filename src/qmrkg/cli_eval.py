"""CLI for evaluating merged knowledge graph triples against a gold file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_run_config, optional_path
from .evaluation import EvaluationError, evaluate_files, render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate merged KG triples against pure gold triples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.kg_eval",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["kg_eval"]

    logging.basicConfig(level=logging.INFO)

    pred = optional_path(run_cfg.get("pred"))
    gold = optional_path(run_cfg.get("gold"))
    output_json = optional_path(run_cfg.get("output_json"))
    output_md = optional_path(run_cfg.get("output_md"))
    top_errors = int(run_cfg["top_errors"])
    if pred is None:
        print("run.kg_eval.pred must be set", file=sys.stderr)
        return 1
    if gold is None:
        print("run.kg_eval.gold must be set", file=sys.stderr)
        return 1

    try:
        report = evaluate_files(pred, gold, top_errors=top_errors)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    json_to_stdout = output_json is None
    status_stream = sys.stderr if json_to_stdout else sys.stdout
    json_report = json.dumps(report, ensure_ascii=False, indent=2)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json_report + "\n", encoding="utf-8")
        print(f"Evaluation JSON written to: {output_json}", file=status_stream)
    else:
        print(json_report)

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"Evaluation Markdown written to: {output_md}", file=status_stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
