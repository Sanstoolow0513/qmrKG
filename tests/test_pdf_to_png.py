from pathlib import Path

import qmrkg.pdf_to_png as pdf_to_png
from qmrkg.pdf_to_png import PDFConverter


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
