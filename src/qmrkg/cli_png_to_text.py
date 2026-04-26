"""CLI for PNG to markdown text conversion."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

from .config import load_run_config
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


def _build_parser(run_cfg: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PNG files to markdown text via OCR.")
    parser.add_argument("--config", type=Path, help="Optional config.yaml path override")
    parser.add_argument("--image", type=Path, help="Process a single PNG image")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(str(run_cfg["image_dir"])),
        help="Root directory of PNG inputs (book subfolders under this path are scanned by default)",
    )
    parser.add_argument("--output", type=Path, help="Output markdown path for --image mode")
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path(str(run_cfg["text_dir"])),
        help="Directory for markdown outputs in directory mode (default: data/markdown)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["recursive"]),
        help="Search subdirectories for PNG files when using --image-dir (default: on; use "
        "--no-recursive for a single flat folder of *.png)",
    )
    parser.add_argument(
        "--lang",
        choices=["ch", "en", "korean", "japan", "ch_tra"],
        default=str(run_cfg["lang"]),
        help="Legacy OCR language flag kept for compatibility",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["gpu"]),
        help="Legacy GPU flag kept for compatibility",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--force-ocr",
        action=argparse.BooleanOptionalAction,
        default=bool(run_cfg["force_ocr"]),
        help="Re-run OCR even when the per-page markdown file already exists and is non-empty",
    )
    return parser


def _collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(image_dir.glob(pattern))


def _group_pngs_by_book(image_dir: Path, paths: list[Path]) -> dict[str, list[Path]]:
    """Group PNG paths by book: subfolder name under image_dir, or stem for files in the root."""
    base = image_dir.resolve()
    groups: dict[str, list[Path]] = {}
    for path in paths:
        abs_path = path.resolve()
        rel = abs_path.relative_to(base)
        if len(rel.parts) == 1:
            book_key = re.sub(r"_page_\d+$", "", path.stem)
        else:
            book_key = rel.parts[0]
        groups.setdefault(book_key, []).append(path)
    for book_key in groups:
        groups[book_key] = sorted(groups[book_key])
    return groups


def _truncate_tqdm_label(text: str, max_chars: int = 48) -> str:
    """Keep book titles readable inside tqdm without overflowing narrow terminals."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"
    return text[: max_chars - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    run_cfg = load_run_config(pre_args.config)["png_to_text"]

    parser = _build_parser(run_cfg)
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

    by_book = _group_pngs_by_book(args.image_dir, image_paths)
    total_pages = len(image_paths)

    args.text_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    book_items = sorted(by_book.items(), key=lambda kv: kv[0])
    book_pbar = tqdm(
        book_items,
        desc="Books",
        unit="book",
        total=len(book_items),
        dynamic_ncols=True,
    )
    for book_key, book_paths in book_pbar:
        book_pbar.set_postfix_str(_truncate_tqdm_label(book_key))
        page_results_list = processor.extract_from_images(
            book_paths,
            text_dir=args.text_dir,
            skip_existing_page_md=not args.force_ocr,
            # During concurrent API calls; Pages bar below covers save/skip phase.
            show_progress=True,
        )
        page_pbar = tqdm(
            zip(book_paths, page_results_list),
            total=len(book_paths),
            desc="Pages",
            unit="page",
            leave=False,
            dynamic_ncols=True,
        )
        for image_path, page_result in page_pbar:
            book_stem = re.sub(r"_page_\d+$", "", image_path.stem)
            output_path = args.text_dir / book_stem / f"{image_path.stem}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.force_ocr and processor.check_page_md_done(image_path, args.text_dir):
                success += 1
                continue
            if page_result.status == "failed":
                failed += 1
                err = page_result.error or "unknown error"
                print(f"Failed: {image_path} ({err})", file=sys.stderr)
                continue
            try:
                processor.process_and_save([page_result], output_path, pdf_source=image_path.name)
                success += 1
            except Exception as exc:
                failed += 1
                print(f"Failed: {image_path} ({exc})", file=sys.stderr)

    print(f"Processed {total_pages} PNG files")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Text dir: {args.text_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
