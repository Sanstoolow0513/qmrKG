"""PDF to PNG converter using PyMuPDF (fitz)."""

import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFConverter:
    """Convert PDF files to PNG images."""

    def __init__(
        self,
        dpi: int = 200,
        fmt: str = "png",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize PDF converter.

        Args:
            dpi: Resolution for output images (default 200)
            fmt: Output format, 'png' or 'jpeg' (default 'png')
            output_dir: Directory to save images (default: same as PDF)
        """
        self.dpi = dpi
        self.fmt = fmt.lower()
        self.output_dir = output_dir
        self.zoom = dpi / 72  # PDF default is 72 DPI

    def convert(
        self,
        pdf_path: Path,
        page_numbers: Optional[List[int]] = None,
        prefix: Optional[str] = None,
    ) -> List[Path]:
        """
        Convert PDF to images.

        Args:
            pdf_path: Path to PDF file
            page_numbers: Specific pages to convert (None = all pages)
            prefix: Prefix for output filenames

        Returns:
            List of paths to generated images
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Determine output directory
        out_dir = self.output_dir or pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate output prefix
        if prefix is None:
            prefix = pdf_path.stem

        logger.info(f"Converting PDF: {pdf_path} (dpi={self.dpi})")

        # Open PDF
        doc = fitz.open(pdf_path)
        output_paths = []

        try:
            # Determine pages to process
            if page_numbers is None:
                pages_to_convert = range(len(doc))
            else:
                pages_to_convert = [p - 1 for p in page_numbers if 1 <= p <= len(doc)]

            # Convert each page
            for page_num in pages_to_convert:
                page = doc[page_num]

                # Render page to image
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Save image
                output_path = out_dir / f"{prefix}_page_{page_num + 1:04d}.{self.fmt}"
                pix.save(str(output_path))
                output_paths.append(output_path)

                logger.debug(f"Saved: {output_path}")

        finally:
            doc.close()

        logger.info(f"Converted {len(output_paths)} pages from {pdf_path.name}")
        return output_paths

    def convert_all(
        self,
        pdf_dir: Path,
        recursive: bool = False,
    ) -> dict[str, List[Path]]:
        """
        Convert all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files
            recursive: Whether to search subdirectories

        Returns:
            Dict mapping PDF filename to list of image paths
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        # Find all PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(pdf_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {}

        results = {}
        for pdf_path in pdf_files:
            try:
                image_paths = self.convert(pdf_path)
                results[pdf_path.name] = image_paths
            except Exception as e:
                logger.error(f"Failed to convert {pdf_path.name}: {e}")
                results[pdf_path.name] = []

        return results
