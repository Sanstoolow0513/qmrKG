from qmrkg.kg_schema import Entity, Triple, ENTITY_TYPES, RELATION_TYPES


def test_entity_types_count():
    assert len(ENTITY_TYPES) == 4


def test_relation_types_count():
    assert len(RELATION_TYPES) == 4


def test_entity_valid():
    e = Entity(name="TCP", type="protocol")
    assert e.is_valid()


def test_entity_invalid_type():
    e = Entity(name="TCP", type="unknown")
    assert not e.is_valid()


def test_entity_too_short():
    e = Entity(name="X", type="protocol")
    assert not e.is_valid()


def test_entity_too_long():
    e = Entity(name="A" * 31, type="protocol")
    assert not e.is_valid()


def test_entity_empty_name():
    e = Entity(name="  ", type="protocol")
    assert not e.is_valid()


def test_triple_valid():
    t = Triple(head="TCP", relation="compared_with", tail="UDP")
    assert t.is_valid()


def test_triple_self_loop():
    t = Triple(head="TCP", relation="contains", tail="TCP")
    assert not t.is_valid()


def test_triple_invalid_relation():
    t = Triple(head="TCP", relation="unknown", tail="UDP")
    assert not t.is_valid()


def test_triple_empty_head():
    t = Triple(head="", relation="contains", tail="UDP")
    assert not t.is_valid()


def test_triple_empty_tail():
    t = Triple(head="TCP", relation="contains", tail="  ")
    assert not t.is_valid()
