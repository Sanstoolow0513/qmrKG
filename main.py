"""CLI entry point for QmrKG PDF pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qmrkg import PDFPipeline


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="QmrKG: PDF to Text Pipeline for Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in data/pdf/
  python main.py

  # Process specific PDF
  python main.py --pdf path/to/file.pdf

  # Don't save intermediate images
  python main.py --no-images

  # Use GPU for OCR (faster but requires CUDA)
  python main.py --gpu

  # English only OCR (faster)
  python main.py --lang en
        """,
    )

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
        help="Directory to save images (default: data/png)",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/markdown"),
        help="Directory to save text files (default: data/markdown)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Process a single PDF file instead of all in directory",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)",
    )
    parser.add_argument(
        "--lang",
        choices=["ch", "en", "korean", "japan", "ch_tra"],
        default="ch",
        help="OCR language (default: ch for Chinese+English)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't save intermediate PNG images",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR (requires CUDA)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for PDFs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics and exit",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Create pipeline
    pipeline = PDFPipeline(
        pdf_dir=args.pdf_dir,
        image_dir=args.image_dir,
        text_dir=args.text_dir,
        dpi=args.dpi,
        ocr_lang=args.lang,
        use_gpu=args.gpu,
    )

    # Show stats only
    if args.stats:
        import json

        print(json.dumps(pipeline.get_stats(), indent=2, ensure_ascii=False))
        return

    # Process single PDF or all
    if args.pdf:
        if not args.pdf.exists():
            print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
            sys.exit(1)

        image_paths, text_path = pipeline.process_pdf(
            args.pdf,
            save_images=not args.no_images,
        )
        print(f"✓ Processed: {args.pdf.name}")
        print(f"  Pages: {len(image_paths)}")
        print(f"  Text saved to: {text_path}")
    else:
        results = pipeline.process_all(
            save_images=not args.no_images,
            recursive=args.recursive,
        )

        if not results:
            print(f"No PDF files found in {args.pdf_dir}")
            return

        # Summary
        success = sum(1 for r in results.values() if r["status"] == "success")
        failed = sum(1 for r in results.values() if r["status"] == "failed")

        print(f"\n{'=' * 50}")
        print(f"Pipeline Complete: {success}/{len(results)} succeeded")
        if failed > 0:
            print(f"Failed: {failed}")
        print(f"{'=' * 50}")

        for name, result in results.items():
            status_icon = "✓" if result["status"] == "success" else "✗"
            print(f"{status_icon} {name}")
            if result["status"] == "success":
                print(f"    Pages: {result['pages']}")
                print(f"    Text: {result['text']}")


if __name__ == "__main__":
    main()
