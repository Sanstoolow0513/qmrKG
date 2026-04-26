"""Tests for the full-pipeline `qmr` CLI."""

from __future__ import annotations

import argparse
import pytest

from qmrkg import cli_qmr


def test_qmr_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as e:
        cli_qmr.main(["--help"])
    assert e.value.code == 0


def test_iter_selected_stages_slice() -> None:
    got = [n for n, _ in cli_qmr._iter_selected_stages("mdchunk", "kgmerge")]
    assert got == ["mdchunk", "kgextract", "kgmerge"]


def test_iter_full_pipeline_default_ends() -> None:
    full = [n for n, _ in cli_qmr._iter_selected_stages(None, "kgneo4j")]
    assert full == list(cli_qmr.STAGE_NAMES)


def test_iter_from_after_to_raises() -> None:
    with pytest.raises(ValueError, match="from_stage must not be after to_stage"):
        list(cli_qmr._iter_selected_stages("kgmerge", "mdchunk"))


def test_effective_to_no_neo4j() -> None:
    a = argparse.Namespace(to_stage=None, no_neo4j=True)
    assert cli_qmr._effective_to_stage(a) == "kgmerge"


def test_effective_to_default() -> None:
    a = argparse.Namespace(to_stage=None, no_neo4j=False)
    assert cli_qmr._effective_to_stage(a) == "kgneo4j"


def test_effective_to_caps_at_merge_when_no_neo4j() -> None:
    a = argparse.Namespace(to_stage="kgneo4j", no_neo4j=True)
    assert cli_qmr._effective_to_stage(a) == "kgmerge"


def test_qmr_list_includes_qmr_in_qmrkg(capsys) -> None:
    import qmrkg.cli_qmrkg as cli_qmrkg

    exit_code = cli_qmrkg.main(["--list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "qmr" in out
