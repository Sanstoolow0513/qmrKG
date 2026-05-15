from __future__ import annotations

import json
import tomllib
from pathlib import Path
from textwrap import dedent


def _pred_payload() -> dict:
    return {
        "entities": [
            {"name": "HTTP", "type": "protocol"},
            {"name": "TCP", "type": "protocol"},
        ],
        "triples": [
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "depends_on",
                "tail": "TCP",
                "tail_type": "protocol",
                "evidences": ["HTTP 使用 TCP 作为传输层协议"],
            }
        ],
    }


def _gold_payload() -> dict:
    return {
        "meta": {"schema_version": 1},
        "triples": [
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "depends_on",
                "tail": "TCP",
                "tail_type": "protocol",
            }
        ],
    }


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return config_path


def test_cli_eval_writes_json_and_markdown(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_eval:
            pred: "{pred_path}"
            gold: "{gold_path}"
            output_json: "{output_json}"
            output_md: "{output_md}"
            top_errors: 2
        """,
    )

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Evaluation JSON written to:" in captured.out
    assert "Evaluation Markdown written to:" in captured.out
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["metrics"]["triples"]["f1"] == 1.0
    assert "# QmrKG Evaluation Report" in output_md.read_text(encoding="utf-8")


def test_cli_eval_committed_fixtures_produce_expected_metrics(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    fixture_dir = Path(__file__).parent / "fixtures" / "eval"
    output_json = tmp_path / "fixture-report.json"
    output_md = tmp_path / "fixture-report.md"
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_eval:
            pred: "{fixture_dir / "pred_merged.json"}"
            gold: "{fixture_dir / "gold_triples.json"}"
            output_json: "{output_json}"
            output_md: "{output_md}"
        """,
    )

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Evaluation JSON written to:" in captured.out
    assert "Evaluation Markdown written to:" in captured.out
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["metrics"]["triples"]["fp"] == 1
    assert report["metrics"]["entities"]["tp"] == 2
    assert report["evidence"]["pred_coverage"] == 0.5
    assert "# QmrKG Evaluation Report" in output_md.read_text(encoding="utf-8")


def test_cli_eval_prints_json_to_stdout_when_no_json_output(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_eval:
            pred: "{pred_path}"
            gold: "{gold_path}"
        """,
    )

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["entities"]["f1"] == 1.0


def test_cli_eval_keeps_stdout_json_clean_when_writing_markdown(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    output_md = tmp_path / "nested" / "reports" / "report.md"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_eval:
            pred: "{pred_path}"
            gold: "{gold_path}"
            output_md: "{output_md}"
        """,
    )

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["metrics"]["entities"]["f1"] == 1.0
    assert output_md.exists()
    assert "Evaluation Markdown written to:" not in captured.out
    assert "Evaluation Markdown written to:" in captured.err


def test_cli_eval_returns_nonzero_for_invalid_input(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    gold_path = tmp_path / "gold_triples.json"
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")
    config_path = write_config(
        tmp_path,
        f"""
        run:
          kg_eval:
            pred: "{tmp_path / "missing.json"}"
            gold: "{gold_path}"
        """,
    )

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "pred file not found:" in capsys.readouterr().err


def test_cli_eval_returns_nonzero_without_gold(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    config_path = write_config(tmp_path, "run:\n  kg_eval:\n    pred: data/x.json\n")

    exit_code = cli_eval.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "run.kg_eval.gold must be set" in capsys.readouterr().err


def test_pyproject_registers_kgeval_script() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["kgeval"] == "qmrkg.cli_eval:main"


def test_qmrkg_list_includes_kgeval(capsys) -> None:
    from qmrkg.cli_qmrkg import main

    exit_code = main(["--list"])

    assert exit_code == 0
    assert "kgeval" in capsys.readouterr().out


def test_cli_eval_help_describes_gold_evaluation() -> None:
    from qmrkg.cli_eval import build_parser

    help_text = build_parser().format_help()

    assert "Evaluate merged KG triples against pure gold triples." in help_text
