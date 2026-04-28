"""Strict evaluation utilities for merged knowledge graph payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from qmrkg.kg_schema import ENTITY_TYPES, RELATION_TYPES


class EvaluationError(ValueError):
    """Raised when an evaluation payload is malformed."""


@dataclass(frozen=True, order=True, slots=True)
class EntityKey:
    """Strict entity identity."""

    name: str
    type: str


@dataclass(frozen=True, order=True, slots=True)
class TripleKey:
    """Strict triple identity."""

    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str


def evaluate_payloads(
    pred_payload: dict[str, Any],
    gold_payload: dict[str, Any],
    *,
    pred_path: str | None = None,
    gold_path: str | None = None,
    evaluated_at: str | None = None,
    top_errors: int = 20,
) -> dict[str, Any]:
    """Evaluate predicted merged KG payload against a strict gold payload."""

    if top_errors < 0:
        raise EvaluationError("top_errors must be greater than or equal to 0")

    pred_entities = _parse_entities(_list_field(pred_payload, "entities", "pred"), "pred.entities")
    pred_triples = _parse_triples(_list_field(pred_payload, "triples", "pred"), "pred.triples")
    gold_triples = _parse_triples(_list_field(gold_payload, "triples", "gold"), "gold.triples")

    gold_entity_items = _list_field(gold_payload, "entities", "gold", required=False)
    if gold_entity_items:
        gold_entities = _parse_entities(gold_entity_items, "gold.entities")
    else:
        gold_entities = _entities_from_triples(gold_triples)

    pred_triple_keys = set(pred_triples)
    gold_triple_keys = set(gold_triples)
    triple_tp_keys = pred_triple_keys & gold_triple_keys

    entity_counts = _set_counts(set(pred_entities), set(gold_entities))
    triple_counts = _set_counts(pred_triple_keys, gold_triple_keys)

    false_positives = sorted(pred_triple_keys - gold_triple_keys)
    false_negatives = sorted(gold_triple_keys - pred_triple_keys)

    return {
        "meta": {
            "pred_path": pred_path,
            "gold_path": gold_path,
            "evaluated_at": evaluated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "gold_schema_version": _gold_schema_version(gold_payload),
        },
        "metrics": {
            "entities": _metrics(entity_counts),
            "triples": _metrics(triple_counts),
        },
        "evidence": {
            "pred_coverage": _safe_divide(
                sum(1 for has_evidence in pred_triples.values() if has_evidence),
                len(pred_triple_keys),
            ),
            "tp_coverage": _safe_divide(
                sum(1 for key in triple_tp_keys if pred_triples[key]),
                len(triple_tp_keys),
            ),
        },
        "errors": {
            "false_positives": [_triple_to_dict(key) for key in false_positives[:top_errors]],
            "false_negatives": [_triple_to_dict(key) for key in false_negatives[:top_errors]],
        },
    }


def _list_field(
    payload: dict[str, Any],
    field: str,
    payload_name: str,
    *,
    required: bool = True,
) -> list[Any]:
    if not isinstance(payload, dict):
        raise EvaluationError(f"{payload_name} payload must be an object")

    if field not in payload:
        if required:
            raise EvaluationError(f"{payload_name} missing required field: {field}")
        return []

    value = payload[field]
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise EvaluationError(f"{payload_name}.{field} must be a list")
    return value


def _parse_entities(items: list[Any], path: str) -> set[EntityKey]:
    entities: set[EntityKey] = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise EvaluationError(f"{item_path} must be an object")

        name = _required_str(item, "name", item_path)
        entity_type = _required_str(item, "type", item_path)
        _validate_entity_type(entity_type, f"{item_path}.type")
        entities.add(EntityKey(name=name, type=entity_type))
    return entities


def _parse_triples(items: list[Any], path: str) -> dict[TripleKey, bool]:
    triples: dict[TripleKey, bool] = {}
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise EvaluationError(f"{item_path} must be an object")

        key = TripleKey(
            head=_required_str(item, "head", item_path),
            head_type=_required_str(item, "head_type", item_path),
            relation=_required_str(item, "relation", item_path),
            tail=_required_str(item, "tail", item_path),
            tail_type=_required_str(item, "tail_type", item_path),
        )
        _validate_entity_type(key.head_type, f"{item_path}.head_type")
        _validate_entity_type(key.tail_type, f"{item_path}.tail_type")
        _validate_relation(key.relation, f"{item_path}.relation")
        triples[key] = triples.get(key, False) or _has_evidence(item, item_path)
    return triples


def _required_str(item: dict[str, Any], field: str, path: str) -> str:
    if field not in item:
        raise EvaluationError(f"{path} missing required field: {field}")
    value = item[field]
    if not isinstance(value, str):
        raise EvaluationError(f"{path}.{field} must be a string")
    value = value.strip()
    if not value:
        raise EvaluationError(f"{path}.{field} must not be empty")
    return value


def _validate_entity_type(value: str, path: str) -> None:
    if value not in ENTITY_TYPES:
        allowed = ", ".join(sorted(ENTITY_TYPES))
        raise EvaluationError(f"{path} has invalid entity type: {value!r}; expected one of: {allowed}")


def _validate_relation(value: str, path: str) -> None:
    if value not in RELATION_TYPES:
        allowed = ", ".join(sorted(RELATION_TYPES))
        raise EvaluationError(
            f"{path} has invalid relation: {value!r}; expected one of: {allowed}"
        )


def _has_evidence(item: dict[str, Any], path: str) -> bool:
    evidence = item.get("evidence")
    if "evidence" in item and not isinstance(evidence, str):
        raise EvaluationError(f"{path} field evidence must be a string")

    if "evidences" in item:
        evidences = item["evidences"]
        if not isinstance(evidences, list):
            raise EvaluationError(f"{path} field evidences must be a list")
        for index, value in enumerate(evidences):
            if not isinstance(value, str):
                raise EvaluationError(f"{path}.evidences[{index}] must be a string")
        if any(value.strip() for value in evidences):
            return True

    if "evidence" not in item:
        return False

    return bool(evidence.strip())


def _entities_from_triples(triples: dict[TripleKey, bool]) -> set[EntityKey]:
    entities: set[EntityKey] = set()
    for triple in triples:
        entities.add(EntityKey(name=triple.head, type=triple.head_type))
        entities.add(EntityKey(name=triple.tail, type=triple.tail_type))
    return entities


def _set_counts(pred: set[Any], gold: set[Any]) -> dict[str, int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return {
        "pred_count": len(pred),
        "gold_count": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _gold_schema_version(gold_payload: dict[str, Any]) -> int | None:
    meta = gold_payload.get("meta")
    if not isinstance(meta, dict):
        return None
    value = meta.get("schema_version")
    if value is None:
        return None
    if not isinstance(value, int):
        raise EvaluationError("gold.meta.schema_version must be an integer")
    return value


def _triple_to_dict(key: TripleKey) -> dict[str, str]:
    return {
        "head": key.head,
        "head_type": key.head_type,
        "relation": key.relation,
        "tail": key.tail,
        "tail_type": key.tail_type,
    }
