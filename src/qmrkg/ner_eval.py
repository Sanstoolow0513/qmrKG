"""课程 NER 评估：实体 (name, type) 集合上的 P/R/F1，按 split 汇总以对比准确率与泛化。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .kg_schema import ENTITY_TYPES, Entity


def normalize_entity_key(name: str, entity_type: str) -> tuple[str, str]:
    """用于匹配的规范化键：去空白、类型小写。"""
    n = " ".join(name.strip().split())
    t = entity_type.strip().lower()
    return (n, t)


def entities_to_pred_set(entities: list[Entity]) -> set[tuple[str, str]]:
    return {normalize_entity_key(e.name, e.type) for e in entities if e.is_valid()}


def entities_from_gold_dict(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        et = str(it.get("type", "")).strip().lower()
        if et in ENTITY_TYPES and len(name) >= 2:
            out.add(normalize_entity_key(name, et))
    return out


@dataclass(frozen=True, slots=True)
class EntityPRF:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_entity_sets(
    predicted: set[tuple[str, str]],
    gold: set[tuple[str, str]],
) -> EntityPRF:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return EntityPRF(true_positive=tp, false_positive=fp, false_negative=fn)


SplitName = Literal["in_domain_cn", "cross_course"]


@dataclass(slots=True)
class NerEvalSample:
    sample_id: str
    split: SplitName
    text: str
    gold_entities: set[tuple[str, str]]


def load_ner_eval_gold(path: Path) -> list[NerEvalSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    out: list[NerEvalSample] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", "")).strip()
        split = str(s.get("split", "")).strip()
        text = str(s.get("text", "")).strip()
        gold_raw = s.get("gold_entities", [])
        if split not in ("in_domain_cn", "cross_course"):
            continue
        if not isinstance(gold_raw, list):
            continue
        gold_set = entities_from_gold_dict(gold_raw)
        out.append(
            NerEvalSample(
                sample_id=sid or "unknown",
                split=cast(SplitName, split),
                text=text,
                gold_entities=gold_set,
            )
        )
    return out


def aggregate_prf_by_split(
    per_sample: list[tuple[NerEvalSample, EntityPRF]],
) -> dict[SplitName, EntityPRF]:
    """按 split 微平均式汇总（对 TP/FP/FN 求和再算 P/R/F1）。"""
    sums: dict[SplitName, list[int]] = {
        "in_domain_cn": [0, 0, 0],
        "cross_course": [0, 0, 0],
    }
    for sample, prf in per_sample:
        bucket = sums[sample.split]
        bucket[0] += prf.true_positive
        bucket[1] += prf.false_positive
        bucket[2] += prf.false_negative
    result: dict[SplitName, EntityPRF] = {}
    for split, (tp, fp, fn) in sums.items():
        result[split] = EntityPRF(true_positive=tp, false_positive=fp, false_negative=fn)
    return result
