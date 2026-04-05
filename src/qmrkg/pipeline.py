"""Pipeline: PDF -> PNG -> Text."""

import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .markdown_chunker import MarkdownChunker
from .pdf_to_png import PDFConverter
from .png_to_text import OCRProcessor

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Complete pipeline: PDF -> images -> text."""

    def __init__(
        self,
        pdf_dir: Path,
        image_dir: Optional[Path] = None,
        text_dir: Optional[Path] = None,
        dpi: int = 200,
        ocr_lang: str = "ch",
        use_gpu: bool = False,
        skip_existing_page_md: bool = True,
    ):
        """
        Initialize PDF processing pipeline.

        Args:
            pdf_dir: Directory containing PDF files
            image_dir: Directory to save images (default: data/png/)
            text_dir: Directory to save text files (default: data/markdown/)
            dpi: DPI for PDF to image conversion
            ocr_lang: Legacy OCR language flag kept for compatibility
            use_gpu: Legacy GPU flag kept for compatibility
            skip_existing_page_md: Skip OCR when per-page markdown under text_dir is non-empty
        """
        self.pdf_dir = Path(pdf_dir)
        self.image_dir = Path(image_dir) if image_dir else Path("data/png")
        self.text_dir = Path(text_dir) if text_dir else Path("data/markdown")
        self.skip_existing_page_md = skip_existing_page_md

        # Initialize converters
        self.pdf_converter = PDFConverter(dpi=dpi, output_dir=self.image_dir)
        self.ocr_processor = OCRProcessor(lang=ocr_lang, use_gpu=use_gpu)

    def process_pdf(
        self,
        pdf_path: Path,
        save_images: bool = True,
        save_text: bool = True,
        pdf_progress: str | None = None,
        skip_existing_page_md: bool | None = None,
    ) -> tuple[List[Path], Path | None]:
        """
        Process a single PDF through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            save_images: Whether to keep intermediate images
            save_text: Whether to save extracted text
            skip_existing_page_md: Override pipeline default; when True, skip OCR if page .md exists

        Returns:
            Tuple of (image_paths, text_path). `text_path` is `None` when `save_text=False`.
        """
        pdf_path = Path(pdf_path)
        skip_md = (
            self.skip_existing_page_md if skip_existing_page_md is None else skip_existing_page_md
        )
        progress_prefix = f"[{pdf_progress}] " if pdf_progress else ""
        logger.info("%s%s", progress_prefix, pdf_path.name)

        # Step 1: PDF -> PNG (per-book subfolder under image_dir is handled by PDFConverter)
        image_paths = self.pdf_converter.convert(pdf_path)

        if not image_paths:
            raise RuntimeError(f"No images generated from {pdf_path}")

        # Step 2: PNG -> Text
        text_path: Path | None = None
        if save_text:
            text_output_path = self.text_dir / f"{pdf_path.stem}.md"
            self.text_dir.mkdir(parents=True, exist_ok=True)
            sorted_paths = sorted(image_paths)
            page_results = self.ocr_processor.extract_from_images(
                sorted_paths,
                text_dir=self.text_dir,
                skip_existing_page_md=skip_md,
            )
            text_path = self.ocr_processor.process_and_save(
                page_results,
                text_output_path,
                pdf_source=pdf_path.name,
            )
            page_dir = self.text_dir / pdf_path.stem
            page_dir.mkdir(parents=True, exist_ok=True)
            for result in page_results:
                self.ocr_processor.process_and_save(
                    [result],
                    page_dir / f"{result.image_path.stem}.md",
                    pdf_source=result.image_path.name,
                )

        # Cleanup images if not saving
        if not save_images:
            for img_path in image_paths:
                img_path.unlink()
            # Remove empty image directory
            img_dir = self.image_dir / pdf_path.stem
            if img_dir.exists() and not any(img_dir.iterdir()):
                img_dir.rmdir()
            logger.info("Cleaned %s", pdf_path.name)

        return image_paths, text_path

    def process_all(
        self,
        save_images: bool = True,
        save_text: bool = True,
        recursive: bool = False,
        skip_existing_page_md: bool | None = None,
    ) -> dict[str, dict]:
        """
        Process all PDFs in the pdf_dir.

        Args:
            save_images: Keep intermediate images
            save_text: Save extracted text files
            recursive: Search subdirectories for PDFs
            skip_existing_page_md: Override pipeline default for per-page OCR skip

        Returns:
            Dict mapping PDF filename to processing results
        """
        # Find all PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(self.pdf_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return {}

        logger.info("Found %s PDF(s)", len(pdf_files))

        results = {}
        total_pdfs = len(pdf_files)
        for idx, pdf_path in enumerate(
            tqdm(pdf_files, desc="PDFs", unit="pdf", dynamic_ncols=True),
            1,
        ):
            try:
                pdf_progress = f"{idx}/{total_pdfs}"
                image_paths, text_path = self.process_pdf(
                    pdf_path,
                    save_images=save_images,
                    save_text=save_text,
                    pdf_progress=pdf_progress,
                    skip_existing_page_md=skip_existing_page_md,
                )
                results[pdf_path.name] = {
                    "status": "success",
                    "pages": len(image_paths),
                    "images": [str(p) for p in image_paths] if save_images else [],
                    "text": str(text_path) if text_path else None,
                }
            except Exception as e:
                logger.error("Failed %s: %s", pdf_path.name, e)
                results[pdf_path.name] = {
                    "status": "failed",
                    "error": str(e),
                }

        # Summary
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        logger.info("Complete %s/%s succeeded", success_count, len(pdf_files))

        return results

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        pdf_count = len(list(self.pdf_dir.glob("*.pdf")))
        image_count = (
            sum(1 for _ in self.image_dir.rglob("*.png")) if self.image_dir.exists() else 0
        )
        text_count = len(list(self.text_dir.glob("*.md"))) if self.text_dir.exists() else 0

        return {
            "pdf_dir": str(self.pdf_dir),
            "pdf_files": pdf_count,
            "image_dir": str(self.image_dir),
            "image_files": image_count,
            "text_dir": str(self.text_dir),
            "text_files": text_count,
        }

    def chunk_markdown(
        self,
        markdown_path: Path,
        output_path: Path | None = None,
        chunk_dir: Path | None = None,
        max_tokens: int = 4000,
    ) -> Path:
        """Chunk a markdown file into JSON format.

        Args:
            markdown_path: Path to markdown file
            output_path: Optional explicit output path
            chunk_dir: Optional directory for chunks (default: data/chunks)
            max_tokens: Maximum tokens per chunk (default: 4000)

        Returns:
            Path to JSON output file
        """
        chunker = MarkdownChunker(max_tokens=max_tokens)
        return chunker.process_and_save(
            markdown_path=markdown_path,
            output_path=output_path,
            chunk_dir=chunk_dir,
        )
