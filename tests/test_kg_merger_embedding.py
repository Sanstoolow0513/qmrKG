from __future__ import annotations

import json
import logging

from qmrkg.kg_merger import KGMerger
from qmrkg.kg_schema import Entity


class FakeEmbeddingProcessor:
    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text
        self.calls: list[list[str]] = []

    def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
        self.calls.append(inputs)
        return [self.vectors_by_text[text] for text in inputs]


def make_canonicalizer(
    fake_processor: FakeEmbeddingProcessor,
    *,
    threshold: float = 0.85,
    bucket_by_type: bool = True,
    cache_path=None,
):
    from qmrkg.kg_merger import EmbeddingCanonicalizer

    return EmbeddingCanonicalizer(
        task_name="entity_embed",
        encode_fields=["type", "name", "description"],
        similarity_threshold=threshold,
        bucket_by_type=bucket_by_type,
        batch_size=2,
        cache_path=cache_path,
        config_path=None,
        processor=fake_processor,
    )


def test_canonicalizer_merges_synonyms():
    fake = FakeEmbeddingProcessor(
        {
            "protocol | TCP | 传输控制协议": [1.0, 0.0],
            "protocol | 传输控制协议 | TCP 中文名": [1.0, 0.0],
            "protocol | UDP | 用户数据报协议": [0.0, 1.0],
        }
    )
    canonicalizer = make_canonicalizer(fake)
    entities = [
        Entity(name="TCP", type="protocol", description="传输控制协议", frequency=3),
        Entity(name="传输控制协议", type="protocol", description="TCP 中文名", frequency=1),
        Entity(name="UDP", type="protocol", description="用户数据报协议", frequency=2),
    ]

    mapping = canonicalizer.build_canonical_map(entities)

    assert mapping["TCP"] == "TCP"
    assert mapping["传输控制协议"] == "TCP"
    assert mapping["UDP"] == "UDP"


def test_canonicalizer_respects_type_bucket():
    fake = FakeEmbeddingProcessor(
        {
            "protocol | TCP": [1.0, 0.0],
            "concept | TCP概念": [1.0, 0.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="TCP概念", type="concept", frequency=1),
    ]

    bucketed = make_canonicalizer(fake, bucket_by_type=True).build_canonical_map(entities)
    unbucketed = make_canonicalizer(fake, bucket_by_type=False).build_canonical_map(entities)

    assert bucketed["TCP"] == "TCP"
    assert bucketed["TCP概念"] == "TCP概念"
    assert unbucketed["TCP"] == "TCP"
    assert unbucketed["TCP概念"] == "TCP"


def test_canonicalizer_threshold():
    fake = FakeEmbeddingProcessor(
        {
            "concept | 三次握手": [1.0, 0.0],
            "concept | 三路握手": [0.8, 0.6],
        }
    )
    entities = [
        Entity(name="三次握手", type="concept", frequency=1),
        Entity(name="三路握手", type="concept", frequency=1),
    ]

    high_threshold = make_canonicalizer(fake, threshold=0.95).build_canonical_map(entities)
    low_threshold = make_canonicalizer(fake, threshold=0.7).build_canonical_map(entities)

    assert high_threshold["三路握手"] == "三路握手"
    assert low_threshold["三路握手"] == "三次握手"


def test_canonical_pick_rule():
    fake = FakeEmbeddingProcessor(
        {
            "mechanism | 慢启动": [1.0, 0.0],
            "mechanism | 慢启动算法": [1.0, 0.0],
            "mechanism | TCP慢启动": [1.0, 0.0],
        }
    )
    canonicalizer = make_canonicalizer(fake)
    entities = [
        Entity(name="慢启动", type="mechanism", description="", frequency=2),
        Entity(name="慢启动算法", type="mechanism", description="", frequency=2),
        Entity(name="TCP慢启动", type="mechanism", description="", frequency=5),
    ]

    mapping = canonicalizer.build_canonical_map(entities)

    assert mapping["慢启动"] == "TCP慢启动"
    assert mapping["慢启动算法"] == "TCP慢启动"
    assert mapping["TCP慢启动"] == "TCP慢启动"


def test_canonical_pick_rule_tiebreaks_by_short_name():
    fake = FakeEmbeddingProcessor(
        {
            "mechanism | 慢启动": [1.0, 0.0],
            "mechanism | 慢启动算法": [1.0, 0.0],
        }
    )
    canonicalizer = make_canonicalizer(fake)
    entities = [
        Entity(name="慢启动", type="mechanism", description="", frequency=2),
        Entity(name="慢启动算法", type="mechanism", description="", frequency=2),
    ]

    mapping = canonicalizer.build_canonical_map(entities)

    assert mapping["慢启动"] == "慢启动"
    assert mapping["慢启动算法"] == "慢启动"


def test_merger_with_embedding_disabled_matches_legacy_output(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_payload = {
        "entities": [
            {"name": "TCP", "type": "protocol", "description": "传输控制协议"},
            {"name": "传输控制协议", "type": "protocol", "description": "TCP 中文名"},
            {"name": "UDP", "type": "protocol", "description": "用户数据报协议"},
        ],
        "triples": [
            {
                "head": "传输控制协议",
                "relation": "compared_with",
                "tail": "用户数据报协议",
                "evidence": "TCP 与 UDP 对比。",
            }
        ],
    }
    (raw_dir / "chunk.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8"
    )

    legacy_output = tmp_path / "legacy.json"
    disabled_output = tmp_path / "disabled.json"
    KGMerger().merge_directory(raw_dir, legacy_output)
    KGMerger().merge_directory(raw_dir, disabled_output, embedding_config={"enabled": False})

    assert disabled_output.read_text(encoding="utf-8") == legacy_output.read_text(encoding="utf-8")


def test_merger_writes_cache_and_reuses(tmp_path):
    cache_path = tmp_path / ".embed_cache.json"
    fake = FakeEmbeddingProcessor(
        {
            "protocol | TCP": [1.0, 0.0],
            "protocol | 传输控制协议": [1.0, 0.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="传输控制协议", type="protocol", frequency=1),
    ]

    first = make_canonicalizer(fake, cache_path=cache_path).build_canonical_map(entities)
    second_fake = FakeEmbeddingProcessor(
        {
            "protocol | TCP": [0.0, 1.0],
            "protocol | 传输控制协议": [0.0, 1.0],
        }
    )
    second = make_canonicalizer(second_fake, cache_path=cache_path).build_canonical_map(entities)

    assert first == second
    assert cache_path.exists()
    assert second_fake.calls == []


def test_canonicalizer_invalid_cache_json_uses_empty_and_reembeds(caplog, tmp_path) -> None:
    caplog.set_level(logging.WARNING)
    cache_path = tmp_path / ".embed_cache.json"
    cache_path.write_text("{bad json", encoding="utf-8")
    fake = FakeEmbeddingProcessor(
        {
            "protocol | TCP": [1.0, 0.0],
            "protocol | 传输控制协议": [1.0, 0.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="传输控制协议", type="protocol", frequency=1),
    ]
    canonicalizer = make_canonicalizer(fake, cache_path=cache_path)
    mapping = canonicalizer.build_canonical_map(entities)
    assert mapping["传输控制协议"] == "TCP"
    assert len(fake.calls) >= 1
    assert "Invalid embedding cache" in caplog.text
    new_raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert any(isinstance(v, list) for v in new_raw.values())
