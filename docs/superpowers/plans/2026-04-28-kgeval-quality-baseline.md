# kgeval Quality Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, strict, reproducible `kgeval` quality baseline for evaluating `kgmerge` `merged_triples.json` outputs against pure gold triples.

**Architecture:** Add a focused `qmrkg.evaluation` module for schema validation, strict matching, metric computation, evidence coverage, and report rendering. Add a thin `qmrkg.cli_eval` command wrapper, register `kgeval`, and provide committed fixture data plus documentation so the baseline is testable without local `data/` artifacts.

**Tech Stack:** Python 3.13, pathlib, argparse, dataclasses, pytest, existing `qmrkg.kg_schema` constants, uv.

---

## Scope Check

This plan covers one subsystem: evaluation of final merged graph JSON. It does not implement raw chunk evaluation, audit-table conversion, zero-shot/few-shot experiment orchestration, validator extraction, or normalized matching.

## File Structure

- Create: `src/qmrkg/evaluation.py`
  - Owns strict schema parsing, strict key generation, metric calculation, evidence coverage, file loading, and Markdown rendering.
- Create: `src/qmrkg/cli_eval.py`
  - Owns CLI argument parsing, output writing, stdout behavior, and nonzero error exits.
- Modify: `pyproject.toml`
  - Registers `kgeval = "qmrkg.cli_eval:main"`.
- Modify: `src/qmrkg/cli_qmrkg.py`
  - Lists `kgeval` in `qmrkg --list`.
- Create: `tests/test_evaluation.py`
  - Covers core evaluation behavior.
- Create: `tests/test_cli_eval.py`
  - Covers CLI behavior and command registration.
- Create: `tests/fixtures/eval/pred_merged.json`
  - Small predicted `merged_triples.json` example.
- Create: `tests/fixtures/eval/gold_triples.json`
  - Small pure gold example.
- Create: `docs/evaluation/kgeval.md`
  - Documents gold format, strict matching, command usage, and report interpretation.

## Task 1: Core Strict Evaluation

**Files:**
- Create: `src/qmrkg/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation.py` with this content:

```python
import pytest

from qmrkg.evaluation import EvaluationError, evaluate_payloads


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'qmrkg.evaluation'`.

- [ ] **Step 3: Implement the core evaluator**

Create `src/qmrkg/evaluation.py` with this content:

```python
"""Evaluation utilities for merged QmrKG triples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .kg_schema import ENTITY_TYPES, RELATION_TYPES


class EvaluationError(ValueError):
    """Raised when evaluation input is invalid."""


@dataclass(frozen=True, order=True, slots=True)
class EntityKey:
    name: str
    type: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.type}


@dataclass(frozen=True, order=True, slots=True)
class TripleKey:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "head": self.head,
            "head_type": self.head_type,
            "relation": self.relation,
            "tail": self.tail,
            "tail_type": self.tail_type,
        }


def evaluate_payloads(
    pred: dict[str, Any],
    gold: dict[str, Any],
    *,
    pred_path: str = "",
    gold_path: str = "",
    evaluated_at: str | None = None,
    top_errors: int = 10,
) -> dict[str, Any]:
    """Evaluate predicted merged triples against pure gold triples."""
    if top_errors < 0:
        raise EvaluationError("top_errors must be greater than or equal to 0")

    pred_entities = _parse_entity_list(_require_list(pred, "entities", "pred"), "pred.entities")
    pred_triples = _parse_triple_map(_require_list(pred, "triples", "pred"), "pred.triples")
    gold_triples = _parse_triple_map(_require_list(gold, "triples", "gold"), "gold.triples")
    gold_entities = _parse_gold_entities(gold, set(gold_triples))

    pred_entity_set = set(pred_entities)
    pred_triple_set = set(pred_triples)
    gold_triple_set = set(gold_triples)

    entity_counts = _count_sets(pred_entity_set, gold_entities)
    triple_counts = _count_sets(pred_triple_set, gold_triple_set)
    true_positive_triples = pred_triple_set & gold_triple_set

    false_positives = sorted(pred_triple_set - gold_triple_set)[:top_errors]
    false_negatives = sorted(gold_triple_set - pred_triple_set)[:top_errors]

    pred_with_evidence = sum(1 for has_evidence in pred_triples.values() if has_evidence)
    tp_with_evidence = sum(1 for key in true_positive_triples if pred_triples.get(key, False))

    return {
        "meta": {
            "pred_path": pred_path,
            "gold_path": gold_path,
            "evaluated_at": evaluated_at or datetime.now(UTC).isoformat(),
            "gold_schema_version": _gold_schema_version(gold),
            "matching": "strict",
        },
        "metrics": {
            "entities": _metrics(entity_counts),
            "triples": _metrics(triple_counts),
        },
        "evidence": {
            "pred_with_evidence": pred_with_evidence,
            "pred_total": len(pred_triples),
            "pred_coverage": _safe_div(pred_with_evidence, len(pred_triples)),
            "tp_with_evidence": tp_with_evidence,
            "tp_total": len(true_positive_triples),
            "tp_coverage": _safe_div(tp_with_evidence, len(true_positive_triples)),
        },
        "errors": {
            "false_positives": [key.to_dict() for key in false_positives],
            "false_negatives": [key.to_dict() for key in false_negatives],
        },
    }


def _require_list(payload: dict[str, Any], field: str, source: str) -> list[Any]:
    if field not in payload:
        raise EvaluationError(f"{source} missing required field: {field}")
    value = payload[field]
    if not isinstance(value, list):
        raise EvaluationError(f"{source}.{field} must be a list")
    return value


def _parse_entity_list(items: list[Any], source: str) -> set[EntityKey]:
    entities: set[EntityKey] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise EvaluationError(f"{source}[{idx}] must be an object")
        name = _required_str(item, "name", source, idx)
        type_ = _required_str(item, "type", source, idx)
        _validate_entity_type(type_, source, idx, "type")
        entities.add(EntityKey(name=name, type=type_))
    return entities


def _parse_gold_entities(gold: dict[str, Any], gold_triples: set[TripleKey]) -> set[EntityKey]:
    raw_entities = gold.get("entities")
    if isinstance(raw_entities, list) and raw_entities:
        return _parse_entity_list(raw_entities, "gold.entities")
    if raw_entities not in (None, []):
        raise EvaluationError("gold.entities must be a list when provided")
    entities: set[EntityKey] = set()
    for triple in gold_triples:
        entities.add(EntityKey(name=triple.head, type=triple.head_type))
        entities.add(EntityKey(name=triple.tail, type=triple.tail_type))
    return entities


def _parse_triple_map(items: list[Any], source: str) -> dict[TripleKey, bool]:
    triples: dict[TripleKey, bool] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise EvaluationError(f"{source}[{idx}] must be an object")
        head = _required_str(item, "head", source, idx)
        head_type = _required_str(item, "head_type", source, idx)
        relation = _required_str(item, "relation", source, idx)
        tail = _required_str(item, "tail", source, idx)
        tail_type = _required_str(item, "tail_type", source, idx)
        _validate_entity_type(head_type, source, idx, "head_type")
        _validate_entity_type(tail_type, source, idx, "tail_type")
        if relation not in RELATION_TYPES:
            raise EvaluationError(f"{source}[{idx}] invalid relation: {relation}")
        key = TripleKey(
            head=head,
            head_type=head_type,
            relation=relation,
            tail=tail,
            tail_type=tail_type,
        )
        triples[key] = triples.get(key, False) or _has_evidence(item)
    return triples


def _required_str(item: dict[str, Any], field: str, source: str, idx: int) -> str:
    if field not in item:
        raise EvaluationError(f"{source}[{idx}] missing required field: {field}")
    value = str(item[field]).strip()
    if not value:
        raise EvaluationError(f"{source}[{idx}] field {field} must be non-empty")
    return value


def _validate_entity_type(value: str, source: str, idx: int, field: str) -> None:
    if value not in ENTITY_TYPES:
        raise EvaluationError(f"{source}[{idx}] invalid {field}: {value}")


def _has_evidence(item: dict[str, Any]) -> bool:
    evidences = item.get("evidences")
    if isinstance(evidences, list):
        return any(str(evidence).strip() for evidence in evidences)
    evidence = item.get("evidence")
    if isinstance(evidence, str):
        return bool(evidence.strip())
    return False


def _count_sets(pred: set[Any], gold: set[Any]) -> dict[str, int]:
    return {
        "pred_count": len(pred),
        "gold_count": len(gold),
        "tp": len(pred & gold),
        "fp": len(pred - gold),
        "fn": len(gold - pred),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _gold_schema_version(gold: dict[str, Any]) -> int | None:
    meta = gold.get("meta")
    if not isinstance(meta, dict):
        return None
    version = meta.get("schema_version")
    return int(version) if isinstance(version, int) else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qmrkg/evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): add strict merged graph metrics"
```

## Task 2: File Loading and Markdown Report Rendering

**Files:**
- Modify: `src/qmrkg/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Append these imports and tests to `tests/test_evaluation.py`:

```python
import json

from qmrkg.evaluation import evaluate_files, render_markdown_report


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_evaluate_files_loads_json_and_preserves_paths tests/test_evaluation.py::test_render_markdown_report_contains_summary_and_error_samples -v
```

Expected: FAIL with import errors for `evaluate_files` and `render_markdown_report`.

- [ ] **Step 3: Add file loading and Markdown rendering**

Update imports at the top of `src/qmrkg/evaluation.py`:

```python
import json
from pathlib import Path
```

Add these functions after `evaluate_payloads`:

```python
def evaluate_files(
    pred_path: Path,
    gold_path: Path,
    *,
    top_errors: int = 10,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    """Load prediction and gold files, then evaluate them."""
    pred_path = Path(pred_path)
    gold_path = Path(gold_path)
    pred = _read_json_object(pred_path, "pred")
    gold = _read_json_object(gold_path, "gold")
    return evaluate_payloads(
        pred,
        gold,
        pred_path=str(pred_path),
        gold_path=str(gold_path),
        evaluated_at=evaluated_at,
        top_errors=top_errors,
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render an evaluation report as Markdown."""
    entity = report["metrics"]["entities"]
    triple = report["metrics"]["triples"]
    evidence = report["evidence"]
    meta = report["meta"]
    lines = [
        "# QmrKG Evaluation Report",
        "",
        "## Evaluation Inputs",
        "",
        f"- Prediction: `{meta.get('pred_path', '')}`",
        f"- Gold: `{meta.get('gold_path', '')}`",
        f"- Evaluated at: `{meta.get('evaluated_at', '')}`",
        f"- Gold schema version: `{meta.get('gold_schema_version', '')}`",
        f"- Matching: `{meta.get('matching', 'strict')}`",
        "",
        "## Summary",
        "",
        "| Target | Pred | Gold | TP | FP | FN | Precision | Recall | F1 |",
        "|--------|------|------|----|----|----|-----------|--------|----|",
        _metric_row("Entity", entity),
        _metric_row("Triple", triple),
        "",
        "## Evidence",
        "",
        (
            "Predicted evidence coverage: "
            f"{evidence['pred_with_evidence']}/{evidence['pred_total']} "
            f"({_fmt(evidence['pred_coverage'])})"
        ),
        "",
        (
            "True-positive evidence coverage: "
            f"{evidence['tp_with_evidence']}/{evidence['tp_total']} "
            f"({_fmt(evidence['tp_coverage'])})"
        ),
        "",
        "## Error Samples",
        "",
        "### False Positives",
        "",
        *_triple_table(report["errors"]["false_positives"]),
        "",
        "### False Negatives",
        "",
        *_triple_table(report["errors"]["false_negatives"]),
        "",
        "## Notes",
        "",
        "- Strict entity matching key: `(name, type)`.",
        "- Strict triple matching key: `(head, head_type, relation, tail, tail_type)`.",
        "- Evidence is reported as coverage only and does not affect strict triple matching.",
    ]
    return "\n".join(lines).rstrip() + "\n"
```

Add these helpers near the bottom of `src/qmrkg/evaluation.py`:

```python
def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"{label} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{label} file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise EvaluationError(f"{label} file must contain a JSON object: {path}")
    return raw


def _metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {metrics['pred_count']} | {metrics['gold_count']} | "
        f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
        f"{_fmt(metrics['precision'])} | {_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} |"
    )


def _triple_table(items: list[dict[str, str]]) -> list[str]:
    if not items:
        return ["No samples."]
    rows = [
        "| Head | Head Type | Relation | Tail | Tail Type |",
        "|------|-----------|----------|------|-----------|",
    ]
    for item in items:
        rows.append(
            f"| `{item['head']}` | `{item['head_type']}` | `{item['relation']}` | "
            f"`{item['tail']}` | `{item['tail_type']}` |"
        )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.4f}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qmrkg/evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): render evaluation reports"
```

## Task 3: CLI Entry Point

**Files:**
- Create: `src/qmrkg/cli_eval.py`
- Create: `tests/test_cli_eval.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_cli_eval.py` with this content:

```python
import json
from pathlib import Path


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


def test_cli_eval_returns_nonzero_for_invalid_input(tmp_path, capsys) -> None:
    import qmrkg.cli_eval as cli_eval

    gold_path = tmp_path / "gold_triples.json"
    gold_path.write_text(json.dumps(_gold_payload(), ensure_ascii=False), encoding="utf-8")

    exit_code = cli_eval.main(["--pred", str(tmp_path / "missing.json"), "--gold", str(gold_path)])

    assert exit_code == 1
    assert "pred file not found:" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_cli_eval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'qmrkg.cli_eval'`.

- [ ] **Step 3: Implement the CLI**

Create `src/qmrkg/cli_eval.py` with this content:

```python
"""CLI for evaluating merged QmrKG triples against gold triples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .evaluation import EvaluationError, evaluate_files, render_markdown_report

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kgeval",
        description="Evaluate merged KG triples against pure gold triples.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to predicted merged_triples.json",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to pure gold triples JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the JSON evaluation report",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Optional path for the Markdown evaluation report",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=10,
        help="Number of false-positive and false-negative samples to include",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        report = evaluate_files(args.pred, args.gold, top_errors=args.top_errors)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Evaluation JSON written to: {args.output_json}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"Evaluation Markdown written to: {args.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run:

```bash
uv run pytest tests/test_cli_eval.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related tests**

Run:

```bash
uv run pytest tests/test_evaluation.py tests/test_cli_eval.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qmrkg/cli_eval.py tests/test_cli_eval.py
git commit -m "feat(eval): add kgeval cli"
```

## Task 4: Command Registration

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/qmrkg/cli_qmrkg.py`
- Modify: `tests/test_cli_eval.py`

- [ ] **Step 1: Write the failing registration tests**

Append these tests to `tests/test_cli_eval.py`:

```python
import tomllib


def test_pyproject_registers_kgeval_script() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["kgeval"] == "qmrkg.cli_eval:main"


def test_qmrkg_list_includes_kgeval(capsys) -> None:
    from qmrkg.cli_qmrkg import main

    exit_code = main(["--list"])

    assert exit_code == 0
    assert "kgeval" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_cli_eval.py::test_pyproject_registers_kgeval_script tests/test_cli_eval.py::test_qmrkg_list_includes_kgeval -v
```

Expected: FAIL because `kgeval` is not registered or listed.

- [ ] **Step 3: Register `kgeval` in `pyproject.toml`**

In `[project.scripts]`, add this line after `kgneo4j`:

```toml
kgeval = "qmrkg.cli_eval:main"
```

- [ ] **Step 4: List `kgeval` in `src/qmrkg/cli_qmrkg.py`**

Add this tuple to `COMMANDS` after `kgneo4j`:

```python
    ("kgeval", "Evaluate merged triples against a gold set"),
```

- [ ] **Step 5: Run the registration tests**

Run:

```bash
uv run pytest tests/test_cli_eval.py::test_pyproject_registers_kgeval_script tests/test_cli_eval.py::test_qmrkg_list_includes_kgeval -v
```

Expected: PASS.

- [ ] **Step 6: Verify the installed script through uv**

Run:

```bash
uv run kgeval --help
```

Expected: command exits 0 and prints usage containing `Evaluate merged KG triples against pure gold triples.`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/qmrkg/cli_qmrkg.py tests/test_cli_eval.py
git commit -m "feat(eval): register kgeval command"
```

## Task 5: Fixtures, Documentation, and End-to-End Verification

**Files:**
- Create: `tests/fixtures/eval/pred_merged.json`
- Create: `tests/fixtures/eval/gold_triples.json`
- Create: `docs/evaluation/kgeval.md`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add fixture coverage test**

Append this test to `tests/test_evaluation.py`:

```python
from pathlib import Path


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
```

- [ ] **Step 2: Run the fixture test to verify it fails**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_committed_eval_fixtures_produce_expected_metrics -v
```

Expected: FAIL because `tests/fixtures/eval/pred_merged.json` does not exist.

- [ ] **Step 3: Create fixture directory and predicted fixture**

Create `tests/fixtures/eval/pred_merged.json` with this content:

```json
{
  "entities": [
    {
      "name": "HTTP",
      "type": "protocol",
      "description": "超文本传输协议",
      "frequency": 3
    },
    {
      "name": "TCP",
      "type": "protocol",
      "description": "传输控制协议",
      "frequency": 5
    },
    {
      "name": "UDP",
      "type": "protocol",
      "description": "用户数据报协议",
      "frequency": 1
    }
  ],
  "triples": [
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "depends_on",
      "tail": "TCP",
      "tail_type": "protocol",
      "frequency": 2,
      "evidences": [
        "HTTP 使用 TCP 作为传输层协议"
      ]
    },
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "compared_with",
      "tail": "UDP",
      "tail_type": "protocol",
      "frequency": 1,
      "evidences": []
    }
  ],
  "stats": {
    "total_entities": 3,
    "total_triples": 2
  }
}
```

- [ ] **Step 4: Create gold fixture**

Create `tests/fixtures/eval/gold_triples.json` with this content:

```json
{
  "meta": {
    "name": "qmrkg-minimal-eval-fixture",
    "schema_version": 1
  },
  "entities": [
    {
      "name": "HTTP",
      "type": "protocol"
    },
    {
      "name": "TCP",
      "type": "protocol"
    }
  ],
  "triples": [
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "depends_on",
      "tail": "TCP",
      "tail_type": "protocol",
      "evidence": "HTTP 使用 TCP 作为传输层协议"
    }
  ]
}
```

- [ ] **Step 5: Write documentation**

Create `docs/evaluation/kgeval.md` with this content:

```markdown
# kgeval 质量基线

`kgeval` 用于评估 `kgmerge` 生成的 `merged_triples.json`。第一版只评估最终合并图谱，不评估 `kgextract` 的 raw chunk 输出。

## 命令

```bash
uv run kgeval \
  --pred data/triples/merged/merged_triples.json \
  --gold data/eval/gold_triples.json \
  --output-json docs/reports/eval-baseline.json \
  --output-md docs/reports/eval-baseline.md
```

如果不传 `--output-json`，命令会把 JSON 报告打印到 stdout。

## Gold 格式

Gold 文件使用纯金标格式：

```json
{
  "meta": {
    "name": "qmrkg-minimal-eval",
    "schema_version": 1
  },
  "entities": [
    {"name": "HTTP", "type": "protocol"}
  ],
  "triples": [
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "depends_on",
      "tail": "TCP",
      "tail_type": "protocol",
      "evidence": "HTTP 使用 TCP 作为传输层协议"
    }
  ]
}
```

`entities` 可以省略或留空。省略时，`kgeval` 会从 gold triples 的 `head/head_type` 和 `tail/tail_type` 派生实体集合。

## Strict Matching

实体匹配键：

```text
(name, type)
```

三元组匹配键：

```text
(head, head_type, relation, tail, tail_type)
```

匹配前只去掉字段两端空白，不做别名归一、大小写折叠、后缀剥离或 embedding 语义匹配。

## 指标

报告输出实体和三元组的 micro Precision、Recall、F1，并输出 evidence 覆盖率：

- predicted evidence coverage：预测三元组中 `evidences` 非空的比例。
- true-positive evidence coverage：命中 gold 的预测三元组中 `evidences` 非空的比例。

Evidence 只用于覆盖率统计，不参与 strict triple matching。

## Fixture

仓库提交了轻量示例：

- `tests/fixtures/eval/pred_merged.json`
- `tests/fixtures/eval/gold_triples.json`

真实评估集仍建议放在本地 `data/eval/`，因为 `data/` 是运行数据目录，不进入 git。
```

- [ ] **Step 6: Run fixture and docs-adjacent tests**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_committed_eval_fixtures_produce_expected_metrics -v
```

Expected: PASS.

- [ ] **Step 7: Run CLI against committed fixtures**

Run:

```bash
uv run kgeval --pred tests/fixtures/eval/pred_merged.json --gold tests/fixtures/eval/gold_triples.json --output-json /tmp/qmrkg-eval-report.json --output-md /tmp/qmrkg-eval-report.md
```

Expected: command exits 0 and prints both output paths.

- [ ] **Step 8: Run formatting and tests**

Run:

```bash
uv run pytest tests/ -q
```

Expected: PASS.

Run:

```bash
uv run ruff check .
```

Expected: PASS.

Run:

```bash
uv run black --check .
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tests/fixtures/eval/pred_merged.json tests/fixtures/eval/gold_triples.json docs/evaluation/kgeval.md tests/test_evaluation.py
git commit -m "docs(eval): document kgeval baseline fixtures"
```

## Task 6: Final Smoke Check

**Files:**
- No file changes expected unless verification exposes a defect.

- [ ] **Step 1: Check git status**

Run:

```bash
git status --short
```

Expected: no unstaged implementation changes. If generated `/tmp/qmrkg-eval-report.json` and `/tmp/qmrkg-eval-report.md` exist, they are outside the repository and do not affect status.

- [ ] **Step 2: Verify command listing**

Run:

```bash
uv run qmrkg --list
```

Expected: output includes `kgeval`.

- [ ] **Step 3: Verify help output**

Run:

```bash
uv run kgeval --help
```

Expected: output includes `--pred`, `--gold`, `--output-json`, `--output-md`, and `--top-errors`.

- [ ] **Step 4: Run full test suite**

Run:

```bash
uv run pytest tests/ -q
```

Expected: PASS.

- [ ] **Step 5: Record verification in the final implementation response**

Include these lines in the implementation summary:

```text
Verification:
- uv run pytest tests/ -q
- uv run qmrkg --list
- uv run kgeval --help
```

Do not claim `ruff` or `black` passed unless those commands were run and passed during Task 5.
