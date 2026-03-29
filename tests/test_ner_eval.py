from pathlib import Path

from qmrkg.kg_schema import Entity
from qmrkg.ner_eval import (
    aggregate_prf_by_split,
    entities_to_pred_set,
    load_ner_eval_gold,
    score_entity_sets,
)


def test_score_entity_sets_perfect():
    gold = {("TCP", "protocol"), ("UDP", "protocol")}
    pred = {("TCP", "protocol"), ("UDP", "protocol")}
    m = score_entity_sets(pred, gold)
    assert m.true_positive == 2 and m.false_positive == 0 and m.false_negative == 0
    assert m.precision == m.recall == m.f1 == 1.0


def test_score_entity_sets_partial():
    gold = {("TCP", "protocol"), ("UDP", "protocol")}
    pred = {("TCP", "protocol"), ("HTTP", "protocol")}
    m = score_entity_sets(pred, gold)
    assert m.true_positive == 1 and m.false_positive == 1 and m.false_negative == 1
    assert abs(m.f1 - 0.5) < 1e-9


def test_entities_to_pred_set_filters_invalid_type():
    entities = [
        Entity(name="TCP", type="protocol", description=""),
        Entity(name="Bad", type="nope", description=""),
    ]
    s = entities_to_pred_set(entities)
    assert s == {("TCP", "protocol")}


def test_load_ner_eval_gold_and_aggregate():
    path = Path(__file__).resolve().parent / "fixtures" / "ner_eval_gold.json"
    samples = load_ner_eval_gold(path)
    assert len(samples) >= 5
    splits = {s.split for s in samples}
    assert splits == {"in_domain_cn", "cross_course"}

    per = []
    for s in samples:
        prf = score_entity_sets(s.gold_entities, s.gold_entities)
        per.append((s, prf))
    agg = aggregate_prf_by_split(per)
    for split in ("in_domain_cn", "cross_course"):
        m = agg[split]
        assert m.false_positive == 0 and m.false_negative == 0
        assert m.f1 == 1.0
