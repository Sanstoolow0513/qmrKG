"""Tests for kg-extract CLI argument parsing and mode wiring."""

import pytest


def test_build_parser_default_mode_is_zero_shot():
    from qmrkg.cli_kg_extract import build_parser

    args = build_parser().parse_args([])
    assert args.mode == "zero-shot"


def test_build_parser_few_shot_mode():
    from qmrkg.cli_kg_extract import build_parser

    args = build_parser().parse_args(["--mode", "few-shot"])
    assert args.mode == "few-shot"


def test_build_parser_rejects_invalid_mode():
    from qmrkg.cli_kg_extract import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "invalid"])


def test_main_passes_mode_to_kg_extractor(monkeypatch, tmp_path, capsys):
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, runner=None, config_path=None, mode=None):
            calls["mode"] = mode

        def extract_from_chunks_file(self, *args, **kwargs):
            return []

    monkeypatch.setattr(cli, "KGExtractor", StubExtractor)

    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code = cli.main(["--input", str(chunks), "--output-dir", str(out_dir), "--mode", "few-shot"])

    assert exit_code == 0
    assert calls["mode"] == "few-shot"
    assert "kgextract mode: few-shot" in capsys.readouterr().out


def test_main_default_mode_zero_shot(monkeypatch, tmp_path, capsys):
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, runner=None, config_path=None, mode=None):
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
    assert calls["mode"] == "zero-shot"
    assert "kgextract mode: zero-shot" in capsys.readouterr().out
