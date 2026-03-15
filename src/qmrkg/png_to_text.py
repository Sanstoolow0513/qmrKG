"""PNG to Text using OCR (PaddleOCR)."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Extract text from images using OCR."""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",  # 'ch' for Chinese+English, 'en' for English only
        show_log: bool = False,
    ):
        """
        Initialize OCR processor.

        Args:
            use_gpu: Whether to use GPU for OCR (default False)
            lang: Language code ('ch', 'en', etc.)
            show_log: Show PaddleOCR internal logs
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.show_log = show_log
        self._ocr = None

    @property
    def ocr(self):
        """Lazy initialization of OCR engine."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                logger.info(f"Initializing PaddleOCR (lang={self.lang}, gpu={self.use_gpu})")
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=self.show_log,
                )
            except ImportError:
                raise ImportError("paddleocr not installed. Run: pip install paddleocr")
        return self._ocr

    def extract_text(
        self,
        image_path: Path,
        return_confidence: bool = False,
    ) -> Union[str, tuple[str, float]]:
        """
        Extract text from a single image.

        Args:
            image_path: Path to image file
            return_confidence: If True, return (text, avg_confidence)

        Returns:
            Extracted text string, or (text, confidence) if return_confidence=True
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug(f"OCR processing: {image_path}")

        # Run OCR
        result = self.ocr.ocr(str(image_path), cls=True)

        # Extract text from result
        texts = []
        confidences = []

        if result and result[0]:
            for line in result[0]:
                if line:
                    text = line[1][0]  # The recognized text
                    confidence = line[1][1]  # Confidence score
                    texts.append(text)
                    confidences.append(confidence)

        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.debug(f"Extracted {len(texts)} lines from {image_path.name}")

        if return_confidence:
            return full_text, avg_confidence
        return full_text

    def extract_from_images(
        self,
        image_paths: List[Path],
        return_confidence: bool = False,
    ) -> Union[List[str], List[tuple[str, float]]]:
        """
        Extract text from multiple images.

        Args:
            image_paths: List of image file paths
            return_confidence: If True, return list of (text, confidence) tuples

        Returns:
            List of extracted texts
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.extract_text(img_path, return_confidence=return_confidence)
                results.append(result)
            except Exception as e:
                logger.error(f"OCR failed for {img_path}: {e}")
                if return_confidence:
                    results.append(("", 0.0))
                else:
                    results.append("")
        return results

    def process_and_save(
        self,
        image_paths: List[Path],
        output_path: Path,
        page_separator: str = "\n\n--- Page {page} ---\n\n",
    ) -> Path:
        """
        Extract text from images and save to file.

        Args:
            image_paths: List of image paths (in order)
            output_path: Path to save the text file
            page_separator: Separator between pages, use {page} for page number

        Returns:
            Path to saved text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_texts = []
        for i, img_path in enumerate(image_paths, 1):
            try:
                text = self.extract_text(img_path)
                if isinstance(text, str) and text.strip():
                    separator = page_separator.format(page=i)
                    all_texts.append(f"{separator}{text}")
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        # Combine and save
        full_content = "".join(all_texts)
        output_path.write_text(full_content, encoding="utf-8")

        logger.info(f"Saved OCR text to: {output_path}")
        return output_path
