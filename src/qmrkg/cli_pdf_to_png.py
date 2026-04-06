"""CLI for PDF / PowerPoint to PNG conversion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pdf_to_png import PDFConverter, PPTConverter, convert_document_to_pngs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDF or PowerPoint (.ppt, .pptx) files to PNG images.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Convert a single file (.pdf, .ppt, .pptx)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdf"),
        help="Directory containing PDF or presentation files (default: data/pdf)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/png"),
        help="Root directory for PNG output; each document is written under a subfolder named "
        "after the file stem (default: data/png)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when using --pdf-dir",
    )
    parser.add_argument(
        "--libreoffice",
        type=str,
        default="libreoffice",
        help="LibreOffice executable for .ppt/.pptx (default: libreoffice)",
    )
    parser.add_argument(
        "--ppt-timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="Timeout in seconds for LibreOffice conversion (default: 300)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    pdf_converter = PDFConverter(dpi=args.dpi, output_dir=args.image_dir)
    ppt_converter = PPTConverter(
        libreoffice_cmd=args.libreoffice,
        timeout_sec=args.ppt_timeout,
    )

    if args.pdf:
        if not args.pdf.exists():
            print(f"Error: File not found: {args.pdf}", file=sys.stderr)
            return 1

        try:
            output_paths = convert_document_to_pngs(args.pdf, pdf_converter, ppt_converter)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        print(f"Processed: {args.pdf.name}")
        print(f"Pages: {len(output_paths)}")
        if output_paths:
            print(f"Output folder: {output_paths[0].parent}")
        else:
            print(f"Image root: {args.image_dir}")
        return 0

    if not args.pdf_dir.exists():
        print(f"Error: PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        return 1

    results = pdf_converter.convert_all(
        args.pdf_dir,
        recursive=args.recursive,
        ppt_converter=ppt_converter,
    )
    if not results:
        print(f"No PDF or presentation files found in {args.pdf_dir}")
        return 0

    success = sum(1 for pages in results.values() if pages)
    failed = len(results) - success
    print(f"Processed {len(results)} file(s)")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Image root: {args.image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
