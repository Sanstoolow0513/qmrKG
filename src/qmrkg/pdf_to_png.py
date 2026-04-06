"""PDF/PPT to PNG: presentations via LibreOffice to PDF, then PyMuPDF (fitz)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

logger = logging.getLogger(__name__)

_PPT_EXTENSIONS = frozenset({".ppt", ".pptx"})


def is_presentation_path(path: Path) -> bool:
    """Return True if path has a .ppt or .pptx suffix (case-insensitive)."""
    return Path(path).suffix.lower() in _PPT_EXTENSIONS


def iter_input_documents(pdf_dir: Path, recursive: bool) -> list[Path]:
    """List ``.pdf``, ``.ppt``, and ``.pptx`` files under ``pdf_dir``."""
    pdf_dir = Path(pdf_dir)
    exts = {".pdf", ".ppt", ".pptx"}
    if recursive:
        candidates = pdf_dir.rglob("*")
    else:
        candidates = pdf_dir.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in exts)


def convert_document_to_pngs(
    path: Path,
    pdf_converter: "PDFConverter",
    ppt_converter: PPTConverter,
) -> List[Path]:
    """Render a ``.pdf`` or ``.ppt`` / ``.pptx`` to PNG paths (PPT via temp PDF)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in _PPT_EXTENSIONS:
        with tempfile.TemporaryDirectory(prefix="qmrkg-ppt-") as tmp:
            tmp_path = Path(tmp)
            pdf_path = ppt_converter.convert_to_pdf(path, tmp_path)
            return pdf_converter.convert(pdf_path)
    if suf == ".pdf":
        return pdf_converter.convert(path)
    raise ValueError(
        f"Unsupported file type {path.suffix!r}; expected .pdf, .ppt, or .pptx"
    )


class PPTConverter:
    """Convert PowerPoint files to PDF using LibreOffice (headless)."""

    def __init__(
        self,
        libreoffice_cmd: str = "libreoffice",
        timeout_sec: int = 300,
    ):
        """
        Args:
            libreoffice_cmd: Executable name or path for LibreOffice (default: libreoffice).
            timeout_sec: Max seconds for one conversion subprocess.
        """
        self._cmd = libreoffice_cmd
        self.timeout_sec = timeout_sec
        self._resolved_exe: str | None = None

    def _resolve_executable(self) -> str:
        if self._resolved_exe is not None:
            return self._resolved_exe
        which = shutil.which(self._cmd)
        if which:
            self._resolved_exe = which
            return which
        if self._cmd == "libreoffice":
            alt = shutil.which("soffice")
            if alt:
                self._resolved_exe = alt
                return alt
        raise FileNotFoundError(
            f"LibreOffice executable not found in PATH (tried {self._cmd!r} and 'soffice')"
        )

    def convert_to_pdf(self, ppt_path: Path, out_dir: Path) -> Path:
        """
        Run ``libreoffice --headless --convert-to pdf`` and return the output PDF path.

        LibreOffice writes ``{stem}.pdf`` under ``out_dir``.
        """
        ppt_path = Path(ppt_path)
        out_dir = Path(out_dir)
        suf = ppt_path.suffix.lower()
        if suf not in _PPT_EXTENSIONS:
            raise ValueError(f"Expected .ppt or .pptx, got {ppt_path.suffix!r}")
        if not ppt_path.exists():
            raise FileNotFoundError(f"Presentation not found: {ppt_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        exe = self._resolve_executable()
        cmd = [
            exe,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir.resolve()),
            str(ppt_path.resolve()),
        ]
        logger.info("LibreOffice convert %s -> PDF in %s", ppt_path.name, out_dir)

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"LibreOffice timed out after {self.timeout_sec}s for {ppt_path.name}"
            ) from e

        expected = out_dir / f"{ppt_path.stem}.pdf"
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()[:2000]
            raise RuntimeError(
                f"LibreOffice failed (exit {completed.returncode}) for {ppt_path.name}: {err}"
            )
        if not expected.exists():
            raise RuntimeError(
                f"Expected PDF missing at {expected} after converting {ppt_path.name}"
            )
        return expected


def _safe_book_folder_name(stem: str) -> str:
    """Filesystem-safe name for a per-PDF output directory under a shared image root."""
    name = stem.replace("/", "_").replace("\\", "_").strip()
    return name or "document"


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

        # Determine output directory: with a configured root, one subfolder per book (PDF stem)
        if self.output_dir is not None:
            out_dir = Path(self.output_dir) / _safe_book_folder_name(pdf_path.stem)
        else:
            out_dir = pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate output prefix
        if prefix is None:
            prefix = pdf_path.stem

        logger.info("Start %s dpi=%s", pdf_path.name, self.dpi)

        # LibreOffice (and some other tools) emit PDFs whose structure tree triggers MuPDF
        # stderr noise ("No common ancestor in structure tree"); rendering is usually fine.
        _prev_mupdf_stderr = fitz.TOOLS.mupdf_display_errors()
        fitz.TOOLS.mupdf_display_errors(False)
        try:
            doc = fitz.open(pdf_path)
            output_paths = []

            try:
                # Determine pages to process
                if page_numbers is None:
                    pages_to_convert = range(len(doc))
                else:
                    pages_to_convert = [p - 1 for p in page_numbers if 1 <= p <= len(doc)]

                # Convert each page
                for page_num in tqdm(
                    pages_to_convert,
                    desc="PDF pages",
                    unit="page",
                    leave=False,
                    dynamic_ncols=True,
                ):
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

            logger.info("Done %s pages %s", len(output_paths), pdf_path.name)
        finally:
            fitz.TOOLS.mupdf_display_errors(_prev_mupdf_stderr)

        return output_paths

    def convert_all(
        self,
        pdf_dir: Path,
        recursive: bool = False,
        ppt_converter: Optional[PPTConverter] = None,
    ) -> dict[str, List[Path]]:
        """
        Convert all supported documents in a directory.

        When ``ppt_converter`` is set, includes ``.ppt`` / ``.pptx`` (via LibreOffice).
        When ``None``, only ``*.pdf`` files are processed (legacy behavior).

        Args:
            pdf_dir: Directory containing files to convert
            recursive: Whether to search subdirectories
            ppt_converter: Optional converter for PowerPoint inputs

        Returns:
            Dict mapping filename to list of image paths
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        if ppt_converter is not None:
            pdf_files = iter_input_documents(pdf_dir, recursive=recursive)
        else:
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdf_files = list(pdf_dir.glob(pattern))

        if not pdf_files:
            logger.warning("No matching files found in %s", pdf_dir)
            return {}

        results = {}
        for doc_path in pdf_files:
            try:
                if ppt_converter is not None:
                    image_paths = convert_document_to_pngs(doc_path, self, ppt_converter)
                else:
                    image_paths = self.convert(doc_path)
                results[doc_path.name] = image_paths
            except Exception as e:
                logger.error("Failed to convert %s: %s", doc_path.name, e)
                results[doc_path.name] = []

        return results
