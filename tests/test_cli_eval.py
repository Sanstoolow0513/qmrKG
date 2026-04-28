import json


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


def test_cli_eval_writes_json_and_markdown(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    exit_code = cli_eval.main(
        [
            "--pred",
            str(pred_path),
            "--gold",
            str(gold_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--top-errors",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Evaluation JSON written to:" in captured.out
    assert "Evaluation Markdown written to:" in captured.out
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["metrics"]["triples"]["f1"] == 1.0
    assert "# QmrKG Evaluation Report" in output_md.read_text(encoding="utf-8")


def test_cli_eval_prints_json_to_stdout_when_no_json_output(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    exit_code = cli_eval.main(["--pred", str(pred_path), "--gold", str(gold_path)])

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

    exit_code = cli_eval.main(
        [
            "--pred",
            str(pred_path),
            "--gold",
            str(gold_path),
            "--output-md",
            str(output_md),
        ]
    )

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

    exit_code = cli_eval.main(["--pred", str(tmp_path / "missing.json"), "--gold", str(gold_path)])

    assert exit_code == 1
    assert "pred file not found:" in capsys.readouterr().err
