import json

from qmrkg.agent_harness import STAGES


def test_harness_list_json_parses(capsys):
    import qmrkg.cli_agent_harness as cli_agent_harness

    exit_code = cli_agent_harness.main(["list", "--json"])
    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(STAGES)
    assert rows[0]["cli_name"] == "pdftopng"


def test_harness_status_json_reflects_empty_project(tmp_path):
    import qmrkg.cli_agent_harness as cli_agent_harness

    exit_code = cli_agent_harness.main(["--root", str(tmp_path), "status", "--json"])
    assert exit_code == 0
    # stdout not captured here; validate via library
    from qmrkg.agent_harness import format_status_json

    data = json.loads(format_status_json(tmp_path))
    assert data["counts"]["pdf"] == 0
    assert data["counts"]["png"] == 0
    assert data["artifacts"]["merged_triples_json"]["exists"] is False


def test_harness_status_counts_files(tmp_path):
    from qmrkg.agent_harness import collect_execution_status

    (tmp_path / "data" / "pdf").mkdir(parents=True)
    (tmp_path / "data" / "pdf" / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "data" / "png" / "sub").mkdir(parents=True)
    (tmp_path / "data" / "png" / "sub" / "p.png").write_bytes(b"x")
    (tmp_path / "data" / "markdown").mkdir(parents=True)
    (tmp_path / "data" / "markdown" / "m.md").write_text("# t", encoding="utf-8")
    (tmp_path / "data" / "chunks").mkdir(parents=True)
    (tmp_path / "data" / "chunks" / "c.json").write_text("[]", encoding="utf-8")

    st = collect_execution_status(tmp_path)
    assert st["counts"]["pdf"] == 1
    assert st["counts"]["png"] == 1
    assert st["counts"]["markdown"] == 1
    assert st["counts"]["chunks_json"] == 1


def test_qmrkg_list_includes_harness(capsys):
    import qmrkg.cli_qmrkg as cli_qmrkg

    exit_code = cli_qmrkg.main(["--list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "qmrkg-harness" in out
