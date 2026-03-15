"""Pipeline: PDF -> PNG -> Text."""

import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .pdf_to_png import PDFConverter
from .png_to_text import OCRProcessor

logger = logging.getLogger(__name__)


class PDFPipeline:
    """Complete pipeline: PDF -> Images -> Text."""

    def __init__(
        self,
        pdf_dir: Path,
        image_dir: Optional[Path] = None,
        text_dir: Optional[Path] = None,
        dpi: int = 200,
        ocr_lang: str = "ch",
        use_gpu: bool = False,
    ):
        """
        Initialize PDF processing pipeline.

        Args:
            pdf_dir: Directory containing PDF files
            image_dir: Directory to save images (default: data/png/)
            text_dir: Directory to save text files (default: data/markdown/)
            dpi: DPI for PDF to image conversion
            ocr_lang: OCR language ('ch' for Chinese+English, 'en' for English)
            use_gpu: Use GPU for OCR
        """
        self.pdf_dir = Path(pdf_dir)
        self.image_dir = Path(image_dir) if image_dir else Path("data/png")
        self.text_dir = Path(text_dir) if text_dir else Path("data/markdown")

        # Initialize converters
        self.pdf_converter = PDFConverter(dpi=dpi, output_dir=self.image_dir)
        self.ocr_processor = OCRProcessor(lang=ocr_lang, use_gpu=use_gpu)

    def process_pdf(
        self,
        pdf_path: Path,
        save_images: bool = True,
        save_text: bool = True,
    ) -> tuple[List[Path], Path]:
        """
        Process a single PDF through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            save_images: Whether to keep intermediate images
            save_text: Whether to save extracted text

        Returns:
            Tuple of (image_paths, text_path)
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing: {pdf_path.name}")

        # Step 1: PDF -> PNG
        self.pdf_converter.output_dir = self.image_dir / pdf_path.stem
        image_paths = self.pdf_converter.convert(pdf_path)

        if not image_paths:
            raise RuntimeError(f"No images generated from {pdf_path}")

        # Step 2: PNG -> Text
        text_output_path = self.text_dir / f"{pdf_path.stem}.txt"
        self.text_dir.mkdir(parents=True, exist_ok=True)

        text_path = self.ocr_processor.process_and_save(
            sorted(image_paths),
            text_output_path,
        )

        # Cleanup images if not saving
        if not save_images:
            for img_path in image_paths:
                img_path.unlink()
            # Remove empty image directory
            img_dir = self.image_dir / pdf_path.stem
            if img_dir.exists() and not any(img_dir.iterdir()):
                img_dir.rmdir()
            logger.info(f"Cleaned up intermediate images for {pdf_path.name}")

        return image_paths, text_path

    def process_all(
        self,
        save_images: bool = True,
        save_text: bool = True,
        recursive: bool = False,
    ) -> dict[str, dict]:
        """
        Process all PDFs in the pdf_dir.

        Args:
            save_images: Keep intermediate images
            save_text: Save extracted text files
            recursive: Search subdirectories for PDFs

        Returns:
            Dict mapping PDF filename to processing results
        """
        # Find all PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(self.pdf_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDF(s) to process")

        results = {}
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                image_paths, text_path = self.process_pdf(
                    pdf_path,
                    save_images=save_images,
                    save_text=save_text,
                )
                results[pdf_path.name] = {
                    "status": "success",
                    "pages": len(image_paths),
                    "images": [str(p) for p in image_paths] if save_images else [],
                    "text": str(text_path),
                }
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = {
                    "status": "failed",
                    "error": str(e),
                }

        # Summary
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        logger.info(f"Pipeline complete: {success_count}/{len(pdf_files)} succeeded")

        return results

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        pdf_count = len(list(self.pdf_dir.glob("*.pdf")))
        image_count = (
            sum(1 for _ in self.image_dir.rglob("*.png")) if self.image_dir.exists() else 0
        )
        text_count = len(list(self.text_dir.glob("*.txt"))) if self.text_dir.exists() else 0

        return {
            "pdf_dir": str(self.pdf_dir),
            "pdf_files": pdf_count,
            "image_dir": str(self.image_dir),
            "image_files": image_count,
            "text_dir": str(self.text_dir),
            "text_files": text_count,
        }
