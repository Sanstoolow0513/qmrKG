"""Per-chunk raw extraction evaluation against chunk-provenanced gold triples.

Evaluates zero-shot / few-shot extraction output directly (no merge) by:
1. Loading gold triples grouped by (source_file, chunk_index)
2. Finding the corresponding raw per-chunk extraction file
3. Enriching raw triples with head_type/tail_type from the chunk's entity list
4. Computing strict-identity precision, recall, F1 per scope (entity, triple, relation, type)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qmrkg.evaluation import (
    EntityKey,
    TripleKey,
    _set_counts,
    _metrics,
    _safe_divide,
    EvaluationError,
)


@dataclass(frozen=True, slots=True)
class RelationKey:
    """Identity for relation-level evaluation (ignoring entity names)."""

    head_type: str
    relation: str
    tail_type: str


def _triple_to_relation_key(t: TripleKey) -> RelationKey:
    return RelationKey(head_type=t.head_type, relation=t.relation, tail_type=t.tail_type)


def _parse_gold_triples_by_chunk(
    gold_path: Path,
) -> tuple[dict[tuple[str, int], list[TripleKey]], int]:
    """Load gold triples and group by (source_file, chunk_index).

    Returns (grouped_dict, total_gold_count).
    """
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    triples_raw = gold.get("triples", [])
    if not isinstance(triples_raw, list):
        raise EvaluationError("gold.triples must be a list")

    grouped: dict[tuple[str, int], list[TripleKey]] = defaultdict(list)
    for item in triples_raw:
        if not isinstance(item, dict):
            continue
        source_file = str(item.get("source_file", "")).strip()
        chunk_index = item.get("chunk_index", 0)
        if not source_file:
            continue
        key = TripleKey(
            head=str(item.get("head", "")).strip(),
            head_type=str(item.get("head_type", "")).strip(),
            relation=str(item.get("relation", "")).strip(),
            tail=str(item.get("tail", "")).strip(),
            tail_type=str(item.get("tail_type", "")).strip(),
        )
        grouped[(source_file, chunk_index)].append(key)

    return dict(grouped), len(triples_raw)


def _find_raw_file(
    raw_dir: Path,
    source_file: str,
    chunk_index: int,
) -> Path | None:
    """Find the raw extraction file for a given gold (source_file, chunk_index).

    Raw files follow the naming convention:
        {chunk_file_stem}_chunk_{chunk_index:04d}.json

    The gold's source_file is a chunk JSON filename (e.g. "rfc5246.txt.json").
    """
    stem = Path(source_file).stem  # "rfc5246.txt" from "rfc5246.txt.json"
    filename = f"{stem}_chunk_{chunk_index:04d}.json"
    path = raw_dir / filename
    return path if path.exists() else None


def _load_raw_triples(
    raw_path: Path,
) -> tuple[dict[TripleKey, bool], dict[RelationKey, bool], int]:
    """Load and type-enrich raw extraction triples from a per-chunk JSON file.

    Returns:
        (triples_dict, relation_dict, raw_triple_count)
        triples_dict: TripleKey → has_evidence (bool)
        relation_dict: RelationKey → has_evidence (bool) — for relation-level eval
        raw_triple_count: total triples before enrichment (for coverage stats)
    """
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    entities = raw.get("entities", [])
    entity_type_map: dict[str, str] = {}
    for e in entities:
        if isinstance(e, dict):
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "")).strip()
            if name and etype:
                entity_type_map[name] = etype

    triples_raw = raw.get("triples", [])
    if not isinstance(triples_raw, list):
        return {}, {}, 0

    triples_dict: dict[TripleKey, bool] = {}
    relation_dict: dict[RelationKey, bool] = {}

    for item in triples_raw:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail = str(item.get("tail", "")).strip()
        if not head or not tail or not relation:
            continue

        head_type = entity_type_map.get(head, "")
        tail_type = entity_type_map.get(tail, "")

        evidence = str(item.get("evidence", "")).strip()
        has_ev = bool(evidence)

        tk = TripleKey(
            head=head,
            head_type=head_type,
            relation=relation,
            tail=tail,
            tail_type=tail_type,
        )
        rk = RelationKey(
            head_type=head_type,
            relation=relation,
            tail_type=tail_type,
        )

        # Merge evidence flag: OR with existing
        triples_dict[tk] = triples_dict.get(tk, False) or has_ev
        relation_dict[rk] = relation_dict.get(rk, False) or has_ev

    return triples_dict, relation_dict, len(triples_raw)


def evaluate_raw_directory(
    raw_dir: Path,
    gold_path: Path,
    *,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    """Evaluate raw extraction output against gold triples.

    Args:
        raw_dir: Directory containing per-chunk raw extraction JSON files.
        gold_path: Path to gold_triples.json (with source_file + chunk_index provenance).

    Returns:
        Evaluation report dict with entity, triple, relation, endpoint-type metrics.
    """
    raw_dir = Path(raw_dir)
    gold_path = Path(gold_path)

    gold_grouped, total_gold = _parse_gold_triples_by_chunk(gold_path)

    all_pred_triples: set[TripleKey] = set()
    all_pred_relations: set[RelationKey] = set()
    all_pred_entities: set[EntityKey] = set()
    all_gold_triples: set[TripleKey] = set()
    all_gold_relations: set[RelationKey] = set()
    all_gold_entities: set[EntityKey] = set()

    chunks_matched = 0
    chunks_missing_raw = 0
    raw_triple_count = 0

    for (source_file, chunk_index), gold_keys in gold_grouped.items():
        for key in gold_keys:
            all_gold_triples.add(key)
            all_gold_relations.add(_triple_to_relation_key(key))
            all_gold_entities.add(EntityKey(name=key.head, type=key.head_type))
            all_gold_entities.add(EntityKey(name=key.tail, type=key.tail_type))

        raw_path = _find_raw_file(raw_dir, source_file, chunk_index)
        if raw_path is None:
            chunks_missing_raw += 1
            continue

        chunks_matched += 1
        pred_triples, pred_relations, rtc = _load_raw_triples(raw_path)
        raw_triple_count += rtc

        for tk in pred_triples:
            all_pred_triples.add(tk)
            all_pred_relations.add(_triple_to_relation_key(tk))
            all_pred_entities.add(EntityKey(name=tk.head, type=tk.head_type))
            all_pred_entities.add(EntityKey(name=tk.tail, type=tk.tail_type))

    entity_counts = _set_counts(all_pred_entities, all_gold_entities)
    entity_metrics = _metrics(entity_counts)

    triple_counts = _set_counts(all_pred_triples, all_gold_triples)
    triple_metrics = _metrics(triple_counts)

    relation_counts = _set_counts(all_pred_relations, all_gold_relations)
    relation_metrics = _metrics(relation_counts)

    head_type_acc = _attribute_accuracy(all_pred_triples, all_gold_triples, "head_type")
    tail_type_acc = _attribute_accuracy(all_pred_triples, all_gold_triples, "tail_type")
    relation_acc = _attribute_accuracy(all_pred_triples, all_gold_triples, "relation")

    return {
        "meta": {
            "raw_dir": str(raw_dir),
            "gold_path": str(gold_path),
            "evaluated_at": evaluated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "gold_total_triples": total_gold,
            "gold_covered_chunks": len(gold_grouped),
            "chunks_matched_raw": chunks_matched,
            "chunks_missing_raw": chunks_missing_raw,
            "raw_total_triples": raw_triple_count,
        },
        "metrics": {
            "entity": entity_metrics,
            "triple": triple_metrics,
            "relation": relation_metrics,
            "attribute_accuracy": {
                "head_type": head_type_acc,
                "tail_type": tail_type_acc,
                "relation": relation_acc,
            },
        },
    }


def _attribute_accuracy(
    pred_set: set[TripleKey],
    gold_set: set[TripleKey],
    attr: str,
) -> dict[str, float]:
    """Compute accuracy for a single attribute (head_type, tail_type, or relation).

    Accuracy = correct attribute predictions / total predictions (among gold-aligned triples).
    Only triples whose head+tail exist in gold are counted (entity-name alignment).
    """
    # Build lookup: (head, tail) → expected attribute value from gold
    gold_attr_map: dict[tuple[str, str], str] = {}
    for tk in gold_set:
        gold_attr_map[(tk.head, tk.tail)] = getattr(tk, attr)

    total = 0
    correct = 0
    for tk in pred_set:
        key = (tk.head, tk.tail)
        if key in gold_attr_map:
            total += 1
            if getattr(tk, attr) == gold_attr_map[key]:
                correct += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
    }


def compare_zs_fs(
    zs_report: dict[str, Any],
    fs_report: dict[str, Any],
) -> dict[str, Any]:
    """Produce a comparison report between zs and fs raw evaluation results."""
    zs_metrics = zs_report.get("metrics", {})
    fs_metrics = fs_report.get("metrics", {})

    def _diff(zs_val: float, fs_val: float) -> float:
        return fs_val - zs_val

    comparison = {}
    for scope in ("entity", "triple", "relation"):
        zs = zs_metrics.get(scope, {})
        fs = fs_metrics.get(scope, {})
        comparison[scope] = {
            "zs_precision": zs.get("precision", 0),
            "fs_precision": fs.get("precision", 0),
            "delta_precision": _diff(zs.get("precision", 0), fs.get("precision", 0)),
            "zs_recall": zs.get("recall", 0),
            "fs_recall": fs.get("recall", 0),
            "delta_recall": _diff(zs.get("recall", 0), fs.get("recall", 0)),
            "zs_f1": zs.get("f1", 0),
            "fs_f1": fs.get("f1", 0),
            "delta_f1": _diff(zs.get("f1", 0), fs.get("f1", 0)),
        }

    # Attribute accuracy comparison
    zs_attr = zs_metrics.get("attribute_accuracy", {})
    fs_attr = fs_metrics.get("attribute_accuracy", {})
    comparison["attribute_accuracy"] = {}
    for attr in ("head_type", "tail_type", "relation"):
        zs_acc = (zs_attr.get(attr) or {}).get("accuracy", 0)
        fs_acc = (fs_attr.get(attr) or {}).get("accuracy", 0)
        comparison["attribute_accuracy"][attr] = {
            "zs_accuracy": zs_acc,
            "fs_accuracy": fs_acc,
            "delta": _diff(zs_acc, fs_acc),
        }

    return {
        "meta": {
            "zs_raw_dir": zs_report.get("meta", {}).get("raw_dir", ""),
            "fs_raw_dir": fs_report.get("meta", {}).get("raw_dir", ""),
            "gold_path": zs_report.get("meta", {}).get("gold_path", ""),
            "comparison_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        "comparison": comparison,
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a zs vs fs comparison report as Markdown."""
    meta = comparison.get("meta", {})
    comp = comparison.get("comparison", {})

    lines = [
        "# QmrKG Raw Extraction Evaluation: ZS vs FS",
        "",
        "## Inputs",
        "",
        f"- ZS raw dir: `{meta.get('zs_raw_dir', '')}`",
        f"- FS raw dir: `{meta.get('fs_raw_dir', '')}`",
        f"- Gold: `{meta.get('gold_path', '')}`",
        f"- Compared at: `{meta.get('comparison_at', '')}`",
        "",
        "## Scope-level Comparison (P / R / F1)",
        "",
        "| Scope | Metric | ZS | FS | Δ (FS−ZS) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for scope_label in ("entity", "triple", "relation"):
        scope = comp.get(scope_label, {})
        scope_display = scope_label.capitalize()
        for metric in ("precision", "recall", "f1"):
            metric_display = metric.capitalize()
            zs_key = f"zs_{metric}"
            fs_key = f"fs_{metric}"
            delta_key = f"delta_{metric}"
            lines.append(
                f"| {scope_display} | {metric_display} "
                f"| {scope.get(zs_key, 0):.4f} "
                f"| {scope.get(fs_key, 0):.4f} "
                f"| {scope.get(delta_key, 0):+.4f} |"
            )

    lines.extend([
        "",
        "## Attribute Accuracy",
        "",
        "| Attribute | ZS Acc | FS Acc | Δ (FS−ZS) |",
        "| --- | ---: | ---: | ---: |",
    ])

    attr = comp.get("attribute_accuracy", {})
    for attr_name in ("head_type", "tail_type", "relation"):
        a = attr.get(attr_name, {})
        lines.append(
            f"| {attr_name} "
            f"| {a.get('zs_accuracy', 0):.4f} "
            f"| {a.get('fs_accuracy', 0):.4f} "
            f"| {a.get('delta', 0):+.4f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Entity**: `(name, type)` strict match",
        "- **Triple**: `(head, head_type, relation, tail, tail_type)` strict match",
        "- **Relation**: `(head_type, relation, tail_type)` match (ignoring entity names)",
        "- **Attribute accuracy**: proportion of gold-aligned predictions where a single attribute matches gold",
        "- Raw triples are type-enriched via the chunk's entity list before matching",
        "- FPs only counted from chunks with gold coverage; FNs = gold triples not found in pred",
    ])

    return "\n".join(lines) + "\n"
