"""Tests for kg-extract CLI argument parsing and mode wiring."""

import pytest

from qmrkg.config import DEFAULT_RUN_CONFIG
from qmrkg.cli_kg_extract import build_parser

_RUN = DEFAULT_RUN_CONFIG["kg_extract"]


def test_build_parser_default_mode_is_fs() -> None:
    args = build_parser(_RUN).parse_args([])
    assert args.mode == "fs"


def test_build_parser_few_shot_mode() -> None:
    args = build_parser(_RUN).parse_args(["--mode", "few-shot"])
    assert args.mode == "few-shot"


def test_build_parser_rejects_invalid_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser(_RUN).parse_args(["--mode", "invalid"])


def test_main_passes_mode_to_kg_extractor(monkeypatch, tmp_path, capsys) -> None:
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, mode=None, **kwargs):
            calls["mode"] = mode

        def extract_from_chunks_file(self, *args, **kwargs):
            return []

    monkeypatch.setattr(cli, "KGExtractor", StubExtractor)

    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code = cli.main(
        ["--input", str(chunks), "--output-dir", str(out_dir), "--mode", "few-shot"]
    )

    assert exit_code == 0
    assert calls["mode"] == "few-shot"
    assert "kgextract mode: few-shot" in capsys.readouterr().out


def test_main_default_mode_fs(monkeypatch, tmp_path, capsys) -> None:
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, mode=None, **kwargs):
            calls["mode"] = mode

        def extract_from_chunks_file(self, *args, **kwargs):
            return []

    monkeypatch.setattr(cli, "KGExtractor", StubExtractor)

    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code = cli.main(["--input", str(chunks), "--output-dir", str(out_dir)])

    assert exit_code == 0
    assert calls["mode"] == "fs"
    assert "kgextract mode: fs" in capsys.readouterr().out
