from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from qmrkg.kg_merger import KGMerger
from qmrkg.kg_schema import Entity


class FakeEmbeddingProcessor:
    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text
        self.calls: list[list[str]] = []
        self.settings = SimpleNamespace(
            model="qwen/qwen3-embedding-8b",
            encoding_format="float",
            embedding_dimensions=1024,
        )

    def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
        self.calls.append(inputs)
        return [self.vectors_by_text[text] for text in inputs]


class FakeFaissIndexFlatIP:
    def __init__(self, dim: int):
        self.dim = dim
        self._matrix = None

    def add(self, mat):
        self._matrix = mat

    def search(self, query, k: int):
        import numpy as np

        scores = query @ self._matrix.T
        idx = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, idx, axis=1)
        return sorted_scores.astype(np.float32), idx.astype(np.int64)


def make_canonicalizer(
    fake_processor: FakeEmbeddingProcessor,
    *,
    threshold: float = 0.85,
    bucket_by_type: bool = True,
    cache_path=None,
    cache_format: str = "binary",
    max_desc_chars: int = 160,
    faiss_top_k: int = 50,
):
    from qmrkg.kg_merger import EmbeddingCanonicalizer

    return EmbeddingCanonicalizer(
        task_name="entity_embed",
        encode_fields=["type", "name", "description"],
        similarity_threshold=threshold,
        bucket_by_type=bucket_by_type,
        batch_size=2,
        cache_path=cache_path,
        cache_format=cache_format,
        encoding_template="structured_zh",
        max_desc_chars=max_desc_chars,
        faiss_top_k=faiss_top_k,
        config_path=None,
        processor=fake_processor,
    )


def test_canonicalizer_merges_synonyms(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP；定义：传输控制协议": [1.0, 0.0],
            "实体类型：protocol；实体名：传输控制协议；定义：TCP 中文名": [1.0, 0.0],
            "实体类型：protocol；实体名：UDP；定义：用户数据报协议": [0.0, 1.0],
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


def test_canonicalizer_respects_type_bucket(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [1.0, 0.0],
            "实体类型：concept；实体名：TCP概念": [1.0, 0.0],
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


def test_canonicalizer_threshold(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：concept；实体名：三次握手": [1.0, 0.0],
            "实体类型：concept；实体名：三路握手": [0.8, 0.6],
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


def test_canonical_pick_rule(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：mechanism；实体名：慢启动": [1.0, 0.0],
            "实体类型：mechanism；实体名：慢启动算法": [1.0, 0.0],
            "实体类型：mechanism；实体名：TCP慢启动": [1.0, 0.0],
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


def test_canonical_pick_rule_tiebreaks_by_short_name(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：mechanism；实体名：慢启动": [1.0, 0.0],
            "实体类型：mechanism；实体名：慢启动算法": [1.0, 0.0],
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


def test_merger_writes_cache_and_reuses(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    cache_path = tmp_path / ".embed_cache.json"
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [1.0, 0.0],
            "实体类型：protocol；实体名：传输控制协议": [1.0, 0.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="传输控制协议", type="protocol", frequency=1),
    ]

    first = make_canonicalizer(fake, cache_path=cache_path).build_canonical_map(entities)
    second_fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [0.0, 1.0],
            "实体类型：protocol；实体名：传输控制协议": [0.0, 1.0],
        }
    )
    second = make_canonicalizer(second_fake, cache_path=cache_path).build_canonical_map(entities)

    assert first == second
    assert (tmp_path / ".embed_cache.meta.json").exists()
    assert (tmp_path / ".embed_cache.npy").exists()
    assert second_fake.calls == []


def test_canonicalizer_invalid_cache_json_uses_empty_and_reembeds(
    monkeypatch, caplog, tmp_path
) -> None:
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    caplog.set_level(logging.WARNING)
    cache_path = tmp_path / ".embed_cache.json"
    cache_path.write_text("{bad json", encoding="utf-8")
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [1.0, 0.0],
            "实体类型：protocol；实体名：传输控制协议": [1.0, 0.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="传输控制协议", type="protocol", frequency=1),
    ]
    canonicalizer = make_canonicalizer(fake, cache_path=cache_path, cache_format="json")
    mapping = canonicalizer.build_canonical_map(entities)
    assert mapping["传输控制协议"] == "TCP"
    assert len(fake.calls) >= 1
    assert "Invalid embedding cache" in caplog.text
    new_raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert any(isinstance(v, list) for v in new_raw.values())


def test_encode_row_structured_and_clipped(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake = FakeEmbeddingProcessor({})
    canonicalizer = make_canonicalizer(fake, max_desc_chars=4)
    text = canonicalizer._encode_row("TCP", "protocol", "传输控制协议")
    assert text == "实体类型：protocol；实体名：TCP；定义：传输控制"


def test_faiss_missing_raises(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", None)
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [1.0, 0.0],
            "实体类型：protocol；实体名：UDP": [0.0, 1.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="UDP", type="protocol", frequency=1),
    ]
    canonicalizer = make_canonicalizer(fake)
    try:
        canonicalizer.build_canonical_map(entities)
        assert False, "expected ImportError"
    except ImportError as exc:
        assert "faiss is required" in str(exc)


def test_binary_cache_signature_mismatch_reembeds(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    cache_path = tmp_path / ".embed_cache.json"
    base = tmp_path / ".embed_cache"
    (base.with_name(".embed_cache.meta.json")).write_text(
        json.dumps(
            {
                "version": 1,
                "signature": "mismatch-signature",
                "dim": 2,
                "key_to_idx": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(base.with_name(".embed_cache.npy"), np.zeros((0, 2), dtype=np.float32))
    fake = FakeEmbeddingProcessor(
        {
            "实体类型：protocol；实体名：TCP": [1.0, 0.0],
            "实体类型：protocol；实体名：UDP": [0.0, 1.0],
        }
    )
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="UDP", type="protocol", frequency=1),
    ]
    canonicalizer = make_canonicalizer(fake, cache_path=cache_path, faiss_top_k=2)
    canonicalizer.build_canonical_map(entities)
    assert fake.calls != []
