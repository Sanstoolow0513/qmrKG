"""Tqdm/compact logging helpers (formerly also covered a removed root `main.py` script)."""

import logging

from qmrkg import tqdm_logging as tqdm_logging_mod
from qmrkg.tqdm_logging import CompactFormatter, TqdmLoggingHandler


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
