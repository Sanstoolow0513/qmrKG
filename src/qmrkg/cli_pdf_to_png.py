"""CLI for PDF / PowerPoint to PNG conversion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_run_config, optional_path
from .pdf_to_png import PDFConverter, PPTConverter, convert_document_to_pngs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDF or PowerPoint (.ppt, .pptx) files to PNG images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.pdf_to_png",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["pdf_to_png"]

    input_file = optional_path(run_cfg.get("input_file"))
    pdf_dir = Path(str(run_cfg["pdf_dir"]))
    image_dir = Path(str(run_cfg["image_dir"]))
    dpi = int(run_cfg["dpi"])
    recursive = bool(run_cfg["recursive"])
    libreoffice = str(run_cfg["libreoffice"])
    ppt_timeout = int(run_cfg["ppt_timeout"])

    logging.basicConfig(level=logging.INFO)
    pdf_converter = PDFConverter(dpi=dpi, output_dir=image_dir)
    ppt_converter = PPTConverter(
        libreoffice_cmd=libreoffice,
        timeout_sec=ppt_timeout,
    )

    if input_file:
        if not input_file.exists():
            print(f"Error: File not found: {input_file}", file=sys.stderr)
            return 1

        try:
            output_paths = convert_document_to_pngs(input_file, pdf_converter, ppt_converter)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        print(f"Processed: {input_file.name}")
        print(f"Pages: {len(output_paths)}")
        if output_paths:
            print(f"Output folder: {output_paths[0].parent}")
        else:
            print(f"Image root: {image_dir}")
        return 0

    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    results = pdf_converter.convert_all(
        pdf_dir,
        recursive=recursive,
        ppt_converter=ppt_converter,
    )
    if not results:
        print(f"No PDF or presentation files found in {pdf_dir}")
        return 0

    success = sum(1 for pages in results.values() if pages)
    failed = len(results) - success
    print(f"Processed {len(results)} file(s)")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Image root: {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
