from qmrkg.kg_merger import normalize_entity_name, KGMerger
from qmrkg.kg_schema import Entity, Triple


def test_normalize_alias():
    assert normalize_entity_name("传输控制协议") == "TCP"
    assert normalize_entity_name("往返时延") == "RTT"
    assert normalize_entity_name("域名系统") == "DNS"


def test_normalize_suffix_removal():
    assert normalize_entity_name("慢启动算法") == "慢启动"
    assert normalize_entity_name("滑动窗口机制") == "滑动窗口"


def test_normalize_passthrough():
    assert normalize_entity_name("TCP") == "TCP"
    assert normalize_entity_name("拥塞控制") == "拥塞控制"


def test_normalize_strips_whitespace():
    assert normalize_entity_name("  TCP  ") == "TCP"


def test_normalize_short_after_suffix_removal():
    assert normalize_entity_name("X协议") == "X协议"


def test_merge_entities_dedup():
    merger = KGMerger()
    entities = [
        Entity(name="TCP", type="protocol"),
        Entity(name="传输控制协议", type="protocol"),
        Entity(name="TCP", type="protocol"),
    ]
    merged = merger._merge_entities(entities)
    assert len(merged) == 1
    assert merged[0].name == "TCP"
    assert merged[0].frequency == 3


def test_merge_entities_keeps_description():
    merger = KGMerger()
    entities = [
        Entity(name="TCP", type="protocol", description=""),
        Entity(name="TCP", type="protocol", description="传输控制协议"),
    ]
    merged = merger._merge_entities(entities)
    assert len(merged) == 1
    assert merged[0].description == "传输控制协议"


def test_merge_entities_filters_invalid():
    merger = KGMerger()
    entities = [
        Entity(name="TCP", type="protocol"),
        Entity(name="X", type="protocol"),
        Entity(name="TCP", type="unknown_type"),
    ]
    merged = merger._merge_entities(entities)
    assert len(merged) == 1


def test_merge_triples_dedup():
    merger = KGMerger()
    valid_names = {"TCP", "UDP"}
    triples = [
        Triple(head="TCP", relation="compared_with", tail="UDP", evidence="ev1"),
        Triple(head="TCP", relation="compared_with", tail="UDP", evidence="ev2"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 1
    assert merged[0].frequency == 2
    assert len(merged[0].evidences) == 2
    assert "ev1" in merged[0].evidences
    assert "ev2" in merged[0].evidences


def test_merge_triples_dedup_same_evidence():
    merger = KGMerger()
    valid_names = {"TCP", "UDP"}
    triples = [
        Triple(head="TCP", relation="compared_with", tail="UDP", evidence="same"),
        Triple(head="TCP", relation="compared_with", tail="UDP", evidence="same"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 1
    assert merged[0].frequency == 2
    assert len(merged[0].evidences) == 1


def test_merge_triples_filters_unknown_entity():
    merger = KGMerger()
    valid_names = {"TCP"}
    triples = [
        Triple(head="TCP", relation="contains", tail="UNKNOWN", evidence="x"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 0


def test_merge_triples_normalizes_names():
    merger = KGMerger()
    valid_names = {"TCP", "UDP"}
    triples = [
        Triple(head="传输控制协议", relation="compared_with", tail="用户数据报协议", evidence="x"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 1
    assert merged[0].head == "TCP"
    assert merged[0].tail == "UDP"


def test_merge_triples_filters_self_loop_after_normalize():
    merger = KGMerger()
    valid_names = {"TCP"}
    triples = [
        Triple(head="TCP", relation="contains", tail="传输控制协议", evidence="x"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 0


def test_compute_stats():
    entities = [
        Entity(name="TCP", type="protocol"),
        Entity(name="UDP", type="protocol"),
        Entity(name="拥塞控制", type="concept"),
    ]
    triples = [
        Triple(head="TCP", relation="compared_with", tail="UDP"),
        Triple(head="TCP", relation="contains", tail="拥塞控制"),
    ]
    stats = KGMerger._compute_stats(entities, triples)
    assert stats["total_entities"] == 3
    assert stats["total_triples"] == 2
    assert stats["entities_by_type"]["protocol"] == 2
    assert stats["entities_by_type"]["concept"] == 1
    assert stats["triples_by_relation"]["compared_with"] == 1
    assert stats["triples_by_relation"]["contains"] == 1
