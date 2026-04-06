import subprocess
from pathlib import Path

import pytest

import qmrkg.pdf_to_png as pdf_to_png
from qmrkg.pdf_to_png import (
    PDFConverter,
    PPTConverter,
    convert_document_to_pngs,
    is_presentation_path,
)


class FakePixmap:
    def __init__(self):
        self.saved_paths = []

    def save(self, path: str):
        self.saved_paths.append(path)


class FakePage:
    def __init__(self, pixmap: FakePixmap):
        self._pixmap = pixmap

    def get_pixmap(self, matrix, alpha=False):
        return self._pixmap


class FakeDoc:
    def __init__(self, pages):
        self._pages = pages
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        self.closed = True


def test_pdf_converter_uses_tqdm_for_page_progress(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    pixmaps = [FakePixmap(), FakePixmap()]
    fake_doc = FakeDoc([FakePage(pixmaps[0]), FakePage(pixmaps[1])])
    progress_calls = []

    monkeypatch.setattr(pdf_to_png.fitz, "open", lambda path: fake_doc)
    monkeypatch.setattr(
        pdf_to_png,
        "tqdm",
        lambda iterable, **kwargs: progress_calls.append(kwargs) or iterable,
    )

    converter = PDFConverter(output_dir=tmp_path / "png")

    output_paths = converter.convert(pdf_path)

    assert len(output_paths) == 2
    book_dir = tmp_path / "png" / "sample"
    assert all(p.parent == book_dir for p in output_paths)
    assert progress_calls == [
        {
            "desc": "PDF pages",
            "unit": "page",
            "leave": False,
            "dynamic_ncols": True,
        }
    ]
    assert fake_doc.closed is True


def test_is_presentation_path():
    assert is_presentation_path(Path("a.pptx")) is True
    assert is_presentation_path(Path("a.PPTX")) is True
    assert is_presentation_path(Path("a.pdf")) is False


def test_ppt_converter_convert_to_pdf_success(monkeypatch, tmp_path):
    ppt = tmp_path / "slide.pptx"
    ppt.write_bytes(b"x")
    out_dir = tmp_path / "lo_out"

    def fake_run(cmd: list[str], **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "slide.pdf").write_bytes(b"%PDF-1.4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pdf_to_png.subprocess, "run", fake_run)
    monkeypatch.setattr(pdf_to_png.shutil, "which", lambda _name: "/usr/bin/libreoffice")

    converter = PPTConverter()
    result = converter.convert_to_pdf(ppt, out_dir)

    assert result == out_dir / "slide.pdf"
    assert result.exists()


def test_ppt_converter_missing_file(tmp_path):
    converter = PPTConverter()
    with pytest.raises(FileNotFoundError):
        converter.convert_to_pdf(tmp_path / "nope.pptx", tmp_path / "out")


def test_ppt_converter_wrong_suffix(tmp_path):
    converter = PPTConverter()
    p = tmp_path / "x.pdf"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="Expected .ppt or .pptx"):
        converter.convert_to_pdf(p, tmp_path / "out")


def test_convert_document_to_pngs_delegates_to_ppt_then_pdf(tmp_path):
    pptx = tmp_path / "x.pptx"
    pptx.write_bytes(b"x")
    png = tmp_path / "x_page_0001.png"
    lo_calls = []
    pdf_calls = []

    class StubPPT:
        def convert_to_pdf(self, p, out_dir):
            lo_calls.append((p, out_dir))
            (out_dir / "x.pdf").write_bytes(b"%PDF")
            return out_dir / "x.pdf"

    class StubPDF:
        def convert(self, p):
            pdf_calls.append(p)
            return [png]

    out = convert_document_to_pngs(pptx, StubPDF(), StubPPT())
    assert out == [png]
    assert lo_calls[0][0] == pptx
    assert pdf_calls[0].name == "x.pdf"


def test_ppt_converter_libreoffice_failure(monkeypatch, tmp_path):
    ppt = tmp_path / "slide.pptx"
    ppt.write_bytes(b"x")
    out_dir = tmp_path / "lo_out"
    out_dir.mkdir()

    def fake_run(cmd: list[str], **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(pdf_to_png.subprocess, "run", fake_run)
    monkeypatch.setattr(pdf_to_png.shutil, "which", lambda _name: "/usr/bin/libreoffice")

    converter = PPTConverter()
    with pytest.raises(RuntimeError, match="LibreOffice failed"):
        converter.convert_to_pdf(ppt, out_dir)
