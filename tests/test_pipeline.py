from pathlib import Path

import pytest

from qmrkg.pipeline import PDFPipeline
from qmrkg.png_to_text import OCRPageResult


def test_pipeline_still_passes_legacy_ocr_args():
    temp_dir = Path(".pytest_local/pipeline")
    temp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PDFPipeline(pdf_dir=temp_dir, ocr_lang="en", use_gpu=True)

    assert pipeline.pdf_dir == temp_dir
    assert pipeline.ocr_processor.lang == "en"
    assert pipeline.ocr_processor.use_gpu is True


def test_process_pdf_skips_text_output_when_save_text_is_false(monkeypatch):
    pipeline = PDFPipeline(pdf_dir=Path(".pytest_local/pipeline-save-text"))
    image_paths = [Path("page-1.png")]
    process_calls = []

    monkeypatch.setattr(pipeline.pdf_converter, "convert", lambda pdf_path: image_paths)
    monkeypatch.setattr(
        pipeline.ocr_processor,
        "extract_from_images",
        lambda image_paths, **kwargs: [
            OCRPageResult(
                image_path=Path("page-1.png"),
                page_number=1,
                text="test content",
                processed_at="2024-01-01T00:00:00+00:00",
                duration_seconds=1.0,
            )
        ],
    )
    monkeypatch.setattr(
        pipeline.ocr_processor,
        "process_and_save",
        lambda page_results, output_path, pdf_source: process_calls.append(
            (page_results, output_path)
        ),
    )

    returned_images, text_path = pipeline.process_pdf(Path("sample.pdf"), save_text=False)

    assert returned_images == image_paths
    assert text_path is None
    assert process_calls == []


def test_process_pptx_invokes_ppt_converter(monkeypatch, tmp_path):
    pptx = tmp_path / "deck.pptx"
    pptx.write_bytes(b"x")
    fake_pdf = tmp_path / "deck.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    pipeline = PDFPipeline(pdf_dir=tmp_path)
    ppt_calls: list[tuple[Path, Path]] = []

    def fake_convert_to_pdf(ppt: Path, out: Path) -> Path:
        ppt_calls.append((ppt, out))
        return fake_pdf

    monkeypatch.setattr(pipeline._ppt_converter, "convert_to_pdf", fake_convert_to_pdf)

    img = tmp_path / "deck_page_0001.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(pipeline.pdf_converter, "convert", lambda _pdf: [img])

    returned_images, text_path = pipeline.process_pdf(pptx, save_text=False)

    assert len(ppt_calls) == 1
    assert ppt_calls[0][0] == pptx
    assert returned_images == [img]
    assert text_path is None


def test_process_pptx_uses_original_filename_for_pdf_source(monkeypatch, tmp_path):
    pptx = tmp_path / "deck.pptx"
    pptx.write_bytes(b"x")
    fake_pdf = tmp_path / "deck.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    pipeline = PDFPipeline(pdf_dir=tmp_path)
    pdf_sources: list[str | None] = []

    monkeypatch.setattr(
        pipeline._ppt_converter,
        "convert_to_pdf",
        lambda _ppt, _out: fake_pdf,
    )
    img = tmp_path / "deck_page_0001.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(pipeline.pdf_converter, "convert", lambda _pdf: [img])

    monkeypatch.setattr(
        pipeline.ocr_processor,
        "extract_from_images",
        lambda _images, **kwargs: [
            OCRPageResult(
                image_path=img,
                page_number=1,
                text="t",
                processed_at="2024-01-01T00:00:00+00:00",
                duration_seconds=1.0,
            )
        ],
    )

    def capture_save(page_results, output_path, pdf_source=None):
        pdf_sources.append(pdf_source)
        return output_path

    monkeypatch.setattr(pipeline.ocr_processor, "process_and_save", capture_save)

    pipeline.process_pdf(pptx, save_text=True)

    assert pdf_sources[0] == "deck.pptx"


def test_process_pdf_rejects_unsupported_suffix(tmp_path):
    pipeline = PDFPipeline(pdf_dir=tmp_path)
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        pipeline.process_pdf(bad)
