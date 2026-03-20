"""CLI for PDF to PNG conversion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pdf_to_png import PDFConverter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDF files to PNG images.")
    parser.add_argument("--pdf", type=Path, help="Convert a single PDF file")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdf"),
        help="Directory containing PDF files (default: data/pdf)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/png"),
        help="Directory to save generated PNG files (default: data/png)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when using --pdf-dir",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    converter = PDFConverter(dpi=args.dpi, output_dir=args.image_dir)

    if args.pdf:
        if not args.pdf.exists():
            print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
            return 1

        output_paths = converter.convert(args.pdf)
        print(f"Processed: {args.pdf.name}")
        print(f"Pages: {len(output_paths)}")
        print(f"Image dir: {args.image_dir}")
        return 0

    if not args.pdf_dir.exists():
        print(f"Error: PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        return 1

    results = converter.convert_all(args.pdf_dir, recursive=args.recursive)
    if not results:
        print(f"No PDF files found in {args.pdf_dir}")
        return 0

    success = sum(1 for pages in results.values() if pages)
    failed = len(results) - success
    print(f"Processed {len(results)} PDF files")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Image dir: {args.image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
