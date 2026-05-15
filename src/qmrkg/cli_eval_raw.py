"""CLI for evaluating raw zs/fs extraction output against gold triples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_run_config, optional_path
from .eval_raw import (
    compare_zs_fs,
    evaluate_raw_directory,
    render_comparison_markdown,
)
from .evaluation import EvaluationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate raw zs/fs extraction triples against gold triples with chunk provenance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; reads run.kg_eval_raw section",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config).get("kg_eval_raw") or {}

    logging.basicConfig(level=logging.INFO)

    zs_dir = optional_path(run_cfg.get("zs_raw_dir"))
    fs_dir = optional_path(run_cfg.get("fs_raw_dir"))
    gold = optional_path(run_cfg.get("gold"))
    output_json = optional_path(run_cfg.get("output_json"))
    output_md = optional_path(run_cfg.get("output_md"))

    if gold is None:
        print("run.kg_eval_raw.gold must be set", file=sys.stderr)
        return 1

    json_to_stdout = output_json is None
    status_stream = sys.stderr if json_to_stdout else sys.stdout

    results: dict = {}

    if zs_dir:
        try:
            zs_report = evaluate_raw_directory(zs_dir, gold)
            results["zs"] = zs_report
            if output_json:
                zs_json_path = output_json.parent / f"{output_json.stem}_zs{output_json.suffix}"
                zs_json_path.parent.mkdir(parents=True, exist_ok=True)
                zs_json_path.write_text(
                    json.dumps(zs_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"ZS evaluation written to: {zs_json_path}", file=status_stream)
        except EvaluationError as exc:
            print(f"ZS evaluation error: {exc}", file=sys.stderr)
            return 1
    else:
        zs_report = None

    if fs_dir:
        try:
            fs_report = evaluate_raw_directory(fs_dir, gold)
            results["fs"] = fs_report
            if output_json:
                fs_json_path = output_json.parent / f"{output_json.stem}_fs{output_json.suffix}"
                fs_json_path.parent.mkdir(parents=True, exist_ok=True)
                fs_json_path.write_text(
                    json.dumps(fs_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"FS evaluation written to: {fs_json_path}", file=status_stream)
        except EvaluationError as exc:
            print(f"FS evaluation error: {exc}", file=sys.stderr)
            return 1
    else:
        fs_report = None

    if zs_report and fs_report:
        comparison = compare_zs_fs(zs_report, fs_report)
        results["comparison"] = comparison
        if output_json:
            comp_json_path = output_json.parent / f"{output_json.stem}_comparison{output_json.suffix}"
            comp_json_path.parent.mkdir(parents=True, exist_ok=True)
            comp_json_path.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Comparison written to: {comp_json_path}", file=status_stream)

        if output_md:
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(render_comparison_markdown(comparison), encoding="utf-8")
            print(f"Comparison Markdown written to: {output_md}", file=status_stream)

    if json_to_stdout:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
