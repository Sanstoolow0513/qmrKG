"""Logging setup compatible with tqdm progress bars (use tqdm.write for records)."""

from __future__ import annotations

import logging

from tqdm import tqdm


class CompactFormatter(logging.Formatter):
    """Single-line formatter that plays well with tqdm output."""

    LEVEL_ALIASES = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }
    LOGGER_ALIASES = {
        "qmrkg.pipeline": "pipe",
        "qmrkg.pdf_to_png": "pdf",
        "qmrkg.png_to_text": "ocr",
        "qmrkg.kg_extractor": "kgex",
        "qmrkg.kg_merger": "kgmerge",
        "openai._base_client": "openai",
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = self.LEVEL_ALIASES.get(record.levelname, record.levelname[:1])
        logger_name = self.LOGGER_ALIASES.get(record.name, record.name.rsplit(".", 1)[-1])
        return f"{timestamp} {level} {logger_name} {record.getMessage()}"


class TqdmLoggingHandler(logging.Handler):
    """Write log lines through tqdm so progress bars stay aligned."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for CLI use alongside tqdm."""
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(CompactFormatter())
    root_logger.addHandler(handler)

    logging.getLogger("openai").setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO if verbose else logging.WARNING)
