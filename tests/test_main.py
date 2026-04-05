import logging
import sys
from pathlib import Path

import main as main_module
from qmrkg import tqdm_logging as tqdm_logging_mod
from qmrkg.tqdm_logging import CompactFormatter, TqdmLoggingHandler


class StubPipeline:
    init_kwargs = None
    last_process_pdf_call = None

    def __init__(self, **kwargs):
        StubPipeline.init_kwargs = kwargs

    def process_pdf(self, pdf_path, save_images=True, save_text=True):
        StubPipeline.last_process_pdf_call = {
            "pdf_path": pdf_path,
            "save_images": save_images,
            "save_text": save_text,
        }
        return [Path("page-1.png")], Path("output.txt")

    def process_all(self, save_images=True, recursive=False):
        return {}

    def get_stats(self):
        return {"pdf_files": 0}


def test_main_accepts_lang_and_gpu_flags(monkeypatch, capsys):
    temp_dir = Path(".pytest_local/main")
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(main_module, "PDFPipeline", StubPipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--pdf", str(pdf_path), "--lang", "en", "--gpu"],
    )

    exit_code = main_module.main()

    captured = capsys.readouterr()
    assert exit_code is None
    assert StubPipeline.init_kwargs["ocr_lang"] == "en"
    assert StubPipeline.init_kwargs["use_gpu"] is True
    assert "Processed: sample.pdf" in captured.out


def test_compact_formatter_shortens_level_and_module_names():
    formatter = CompactFormatter()
    record = logging.LogRecord(
        name="qmrkg.pdf_to_png",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Start sample.pdf dpi=200",
        args=(),
        exc_info=None,
    )
    record.created = 0

    assert formatter.format(record).endswith("I pdf Start sample.pdf dpi=200")


def test_tqdm_logging_handler_writes_single_line_messages(monkeypatch):
    written = []
    handler = TqdmLoggingHandler()
    handler.setFormatter(CompactFormatter())
    monkeypatch.setattr(tqdm_logging_mod.tqdm, "write", written.append)

    record = logging.LogRecord(
        name="qmrkg.pipeline",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="1/2 sample.pdf",
        args=(),
        exc_info=None,
    )
    record.created = 0

    handler.emit(record)

    assert len(written) == 1
    assert written[0].endswith("W pipe 1/2 sample.pdf")
