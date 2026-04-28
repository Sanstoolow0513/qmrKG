import json
from pathlib import Path

import pytest

from qmrkg.evaluation import (
    EvaluationError,
    evaluate_files,
    evaluate_payloads,
    render_markdown_report,
)


def _pred_payload() -> dict:
    return {
        "entities": [
            {"name": "HTTP", "type": "protocol", "description": "HTTP协议", "frequency": 3},
            {"name": "TCP", "type": "protocol", "description": "传输控制协议", "frequency": 5},
            {"name": "UDP", "type": "protocol", "description": "用户数据报协议", "frequency": 1},
        ],
        "triples": [
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "depends_on",
                "tail": "TCP",
                "tail_type": "protocol",
                "frequency": 2,
                "evidences": ["HTTP 使用 TCP 作为传输层协议"],
            },
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "compared_with",
                "tail": "UDP",
                "tail_type": "protocol",
                "frequency": 1,
                "evidences": [],
            },
        ],
    }


def _gold_payload() -> dict:
    return {
        "meta": {"name": "unit-gold", "schema_version": 1},
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
                "evidence": "HTTP 使用 TCP 作为传输层协议",
            }
        ],
    }


def test_evaluate_payloads_counts_strict_entity_and_triple_metrics() -> None:
    report = evaluate_payloads(
        _pred_payload(),
        _gold_payload(),
        pred_path="pred.json",
        gold_path="gold.json",
        evaluated_at="2026-04-28T00:00:00Z",
        top_errors=5,
    )

    entity = report["metrics"]["entities"]
    assert entity["pred_count"] == 3
    assert entity["gold_count"] == 2
    assert entity["tp"] == 2
    assert entity["fp"] == 1
    assert entity["fn"] == 0
    assert entity["precision"] == pytest.approx(2 / 3)
    assert entity["recall"] == pytest.approx(1.0)
    assert entity["f1"] == pytest.approx(0.8)

    triple = report["metrics"]["triples"]
    assert triple["pred_count"] == 2
    assert triple["gold_count"] == 1
    assert triple["tp"] == 1
    assert triple["fp"] == 1
    assert triple["fn"] == 0
    assert triple["precision"] == pytest.approx(0.5)
    assert triple["recall"] == pytest.approx(1.0)
    assert triple["f1"] == pytest.approx(2 / 3)

    assert report["evidence"]["pred_coverage"] == pytest.approx(0.5)
    assert report["evidence"]["tp_coverage"] == pytest.approx(1.0)
    assert report["errors"]["false_positives"] == [
        {
            "head": "HTTP",
            "head_type": "protocol",
            "relation": "compared_with",
            "tail": "UDP",
            "tail_type": "protocol",
        }
    ]
    assert report["errors"]["false_negatives"] == []


def test_gold_entities_are_derived_from_triples_when_missing() -> None:
    gold = {
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

    report = evaluate_payloads(
        _pred_payload(),
        gold,
        pred_path="pred.json",
        gold_path="gold.json",
        evaluated_at="2026-04-28T00:00:00Z",
    )

    assert report["metrics"]["entities"]["gold_count"] == 2
    assert report["metrics"]["entities"]["tp"] == 2


def test_missing_required_triple_field_raises_clear_error() -> None:
    pred = {
        "entities": [{"name": "HTTP", "type": "protocol"}],
        "triples": [
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "depends_on",
                "tail": "TCP",
            }
        ],
    }

    with pytest.raises(EvaluationError, match="pred.triples\\[0\\] missing required field: tail_type"):
        evaluate_payloads(pred, _gold_payload())


def test_duplicate_predicted_triples_collapse_and_preserve_evidence() -> None:
    pred = {
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
                "evidences": [],
            },
            {
                "head": "HTTP",
                "head_type": "protocol",
                "relation": "depends_on",
                "tail": "TCP",
                "tail_type": "protocol",
                "evidences": ["HTTP 使用 TCP 作为传输层协议"],
            },
        ],
    }

    report = evaluate_payloads(pred, _gold_payload())

    assert report["metrics"]["triples"]["pred_count"] == 1
    assert report["evidence"]["pred_coverage"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "evidences",
            "not-a-list",
            "pred.triples\\[0\\] field evidences must be a list",
        ),
        ("evidences", [1], "pred.triples\\[0\\].evidences\\[0\\] must be a string"),
        ("evidence", 1, "pred.triples\\[0\\] field evidence must be a string"),
    ],
)
def test_malformed_evidence_fields_raise_clear_error(
    field: str, value: object, match: str
) -> None:
    pred = _pred_payload()
    pred["triples"][0][field] = value

    with pytest.raises(EvaluationError, match=match):
        evaluate_payloads(pred, _gold_payload())


def test_negative_top_errors_raises_clear_error() -> None:
    with pytest.raises(EvaluationError, match="top_errors must be greater than or equal to 0"):
        evaluate_payloads(_pred_payload(), _gold_payload(), top_errors=-1)


def test_evaluate_files_loads_json_and_preserves_paths(tmp_path) -> None:
    pred_path = tmp_path / "pred_merged.json"
    gold_path = tmp_path / "gold_triples.json"
    pred_path.write_text(json.dumps(_pred_payload(), ensure_ascii=False), encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    report = evaluate_files(
        pred_path,
        gold_path,
        top_errors=3,
        evaluated_at="2026-04-28T00:00:00Z",
    )

    assert report["meta"]["pred_path"] == str(pred_path)
    assert report["meta"]["gold_path"] == str(gold_path)
    assert report["metrics"]["triples"]["tp"] == 1


def test_render_markdown_report_contains_summary_and_error_samples() -> None:
    report = evaluate_payloads(
        _pred_payload(),
        _gold_payload(),
        pred_path="pred.json",
        gold_path="gold.json",
        evaluated_at="2026-04-28T00:00:00Z",
    )

    markdown = render_markdown_report(report)

    assert "# QmrKG Evaluation Report" in markdown
    assert "| Entity | 3 | 2 | 2 | 1 | 0 | 0.6667 | 1.0000 | 0.8000 |" in markdown
    assert "| Triple | 2 | 1 | 1 | 1 | 0 | 0.5000 | 1.0000 | 0.6667 |" in markdown
    assert "Predicted evidence coverage: 1/2 (0.5000)" in markdown
    assert "`HTTP` | `protocol` | `compared_with` | `UDP` | `protocol`" in markdown


def test_render_markdown_report_escapes_triple_table_cells() -> None:
    report = {
        "meta": {
            "pred_path": "pred`input.json",
            "gold_path": "gold.json",
            "evaluated_at": "2026-04-28T00:00:00Z",
            "gold_schema_version": 1,
        },
        "metrics": {
            "entities": {
                "pred_count": 0,
                "gold_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "triples": {
                "pred_count": 1,
                "gold_count": 0,
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        },
        "evidence": {"pred_coverage": 0.0, "tp_coverage": 0.0},
        "errors": {
            "false_positives": [
                {
                    "head": "HT|TP`name\nnext",
                    "head_type": "protocol",
                    "relation": "depends_on",
                    "tail": "TC``P|name",
                    "tail_type": "protocol",
                }
            ],
            "false_negatives": [],
        },
    }

    markdown = render_markdown_report(report)

    escaped_row = (
        "| `` HT\\|TP`name next `` | `protocol` | `depends_on` | "
        "``` TC``P\\|name ``` | `protocol` |"
    )
    markdown_lines = markdown.splitlines()
    assert escaped_row in markdown_lines
    assert "- Prediction file: `` pred`input.json ``" in markdown_lines
    assert "\\`" not in escaped_row


def test_evaluate_files_missing_pred_file_raises_clear_error(tmp_path) -> None:
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvaluationError, match="pred file not found:"):
        evaluate_files(tmp_path / "missing.json", gold_path)


def test_evaluate_files_invalid_pred_json_raises_clear_error(tmp_path) -> None:
    pred_path = tmp_path / "pred.json"
    gold_path = tmp_path / "gold.json"
    pred_path.write_text("{not-json", encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvaluationError, match="pred file is not valid JSON:"):
        evaluate_files(pred_path, gold_path)


def test_evaluate_files_non_object_pred_json_raises_clear_error(tmp_path) -> None:
    pred_path = tmp_path / "pred.json"
    gold_path = tmp_path / "gold.json"
    pred_path.write_text("[]", encoding="utf-8")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvaluationError, match="pred file must contain a JSON object:"):
        evaluate_files(pred_path, gold_path)


def test_evaluate_files_non_utf8_pred_json_raises_clear_error(tmp_path) -> None:
    pred_path = tmp_path / "pred.json"
    gold_path = tmp_path / "gold.json"
    pred_path.write_bytes(b"\xff")
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvaluationError, match="pred file is not valid UTF-8 text:"):
        evaluate_files(pred_path, gold_path)


def test_committed_eval_fixtures_produce_expected_metrics() -> None:
    report = evaluate_files(
        Path("tests/fixtures/eval/pred_merged.json"),
        Path("tests/fixtures/eval/gold_triples.json"),
        evaluated_at="2026-04-28T00:00:00Z",
        top_errors=10,
    )

    assert report["metrics"]["entities"]["pred_count"] == 3
    assert report["metrics"]["entities"]["gold_count"] == 2
    assert report["metrics"]["entities"]["tp"] == 2
    assert report["metrics"]["triples"]["pred_count"] == 2
    assert report["metrics"]["triples"]["gold_count"] == 1
    assert report["metrics"]["triples"]["fp"] == 1
    assert report["evidence"]["pred_coverage"] == 0.5
