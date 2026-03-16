from pathlib import Path

from qmrkg.pipeline import PDFPipeline


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
        "process_and_save",
        lambda image_paths, output_path: process_calls.append((image_paths, output_path)),
    )

    returned_images, text_path = pipeline.process_pdf(Path("sample.pdf"), save_text=False)

    assert returned_images == image_paths
    assert text_path is None
    assert process_calls == []
