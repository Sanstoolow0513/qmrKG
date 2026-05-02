"""Tests for config-driven kg-extract CLI wiring."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from qmrkg.cli_kg_extract import build_parser


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return config_path


def test_build_parser_is_config_only() -> None:
    args = build_parser().parse_args([])
    assert args.config is None


def test_build_parser_rejects_mode_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "few-shot"])


def test_main_passes_mode_from_config_to_kg_extractor(monkeypatch, tmp_path, capsys) -> None:
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
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_extract:
            input: "{chunks}"
            output_dir: "{out_dir}"
            mode: "few-shot"
        """,
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls["mode"] == "few-shot"
    assert "kgextract mode: few-shot" in capsys.readouterr().out


def test_main_passes_config_path_to_kg_extractor(monkeypatch, tmp_path, capsys) -> None:
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, config_path=None, **kwargs):
            calls["config_path"] = config_path

        def extract_from_chunks_file(self, *args, **kwargs):
            return []

    monkeypatch.setattr(cli, "KGExtractor", StubExtractor)

    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_extract:
            input: "{chunks}"
            output_dir: "{out_dir}"
        """,
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls["config_path"] == config_path
    capsys.readouterr()


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
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_extract:
            input: "{chunks}"
            output_dir: "{out_dir}"
        """,
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls["mode"] == "fs"
    assert "kgextract mode: fs" in capsys.readouterr().out


def test_main_batches_directory_chunk_files(monkeypatch, tmp_path, capsys) -> None:
    import qmrkg.cli_kg_extract as cli

    calls: dict = {}

    class StubExtractor:
        def __init__(self, **kwargs):
            pass

        def extract_from_chunks_files(self, chunk_files, output_dir, **kwargs):
            calls["chunk_files"] = [Path(path).name for path in chunk_files]
            calls["output_dir"] = Path(output_dir)
            calls["skip_existing"] = kwargs["skip_existing"]
            return [Path("a"), Path("b")]

    monkeypatch.setattr(cli, "KGExtractor", StubExtractor)

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "b.json").write_text("[]", encoding="utf-8")
    (chunk_dir / "a.json").write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_extract:
            input: "{chunk_dir}"
            output_dir: "{out_dir}"
        """,
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls["chunk_files"] == ["a.json", "b.json"]
    assert calls["output_dir"] == out_dir
    assert calls["skip_existing"] is True
    assert "Extracted 2 chunk(s) from 2 file(s)" in capsys.readouterr().out


def test_main_rejects_invalid_config_mode(tmp_path, capsys) -> None:
    import qmrkg.cli_kg_extract as cli

    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_extract:
            input: "{chunks}"
            mode: "invalid"
        """,
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "run.kg_extract.mode" in capsys.readouterr().err
