"""CLI for PNG to markdown text conversion."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

from .png_to_text import OCRProcessor


def _configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PNG files to markdown text via OCR.")
    parser.add_argument("--image", type=Path, help="Process a single PNG image")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/png"),
        help="Directory containing PNG images (default: data/png)",
    )
    parser.add_argument("--output", type=Path, help="Output markdown path for --image mode")
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/markdown"),
        help="Directory for markdown outputs in directory mode (default: data/markdown)",
    )
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when using --image-dir",
    )
    parser.add_argument(
        "--lang",
        choices=["ch", "en", "korean", "japan", "ch_tra"],
        default="ch",
        help="Legacy OCR language flag kept for compatibility",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Legacy GPU flag kept for compatibility",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser


def _collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(image_dir.glob(pattern))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    processor = OCRProcessor(
        use_gpu=args.gpu,
        lang=args.lang,
        show_log=args.verbose,
        config_path=args.config,
    )

    if args.image:
        if not args.image.exists():
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            return 1

        output_path = args.output or (args.text_dir / f"{args.image.stem}.md")
        page_results = processor.extract_from_images([args.image])
        saved = processor.process_and_save(page_results, output_path, pdf_source=args.image.name)
        print(f"Processed image: {args.image.name}")
        print(f"Saved markdown to: {saved}")
        return 0

    if args.output:
        print("Error: --output can only be used with --image", file=sys.stderr)
        return 1

    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}", file=sys.stderr)
        return 1

    image_paths = _collect_images(args.image_dir, args.recursive)
    if not image_paths:
        print(f"No PNG files found in {args.image_dir}")
        return 0

    args.text_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    for image_path in tqdm(
        image_paths,
        desc="OCR",
        unit="page",
        total=len(image_paths),
        dynamic_ncols=True,
    ):
        book_stem = re.sub(r"_page_\d+$", "", image_path.stem)
        output_path = args.text_dir / book_stem / f"{image_path.stem}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            page_results = processor.extract_from_images([image_path])
            processor.process_and_save(page_results, output_path, pdf_source=image_path.name)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {image_path} ({exc})", file=sys.stderr)

    print(f"Processed {len(image_paths)} PNG files")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Text dir: {args.text_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
