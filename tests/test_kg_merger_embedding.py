from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

from qmrkg.kg_merger import KGMerger
from qmrkg.kg_schema import Entity


class FakeEmbeddingProcessor:
    def __init__(self, vectors_by_text: dict[str, list[float]], max_concurrency: int = 1):
        self.vectors_by_text = vectors_by_text
        self.calls: list[list[str]] = []
        self.settings = SimpleNamespace(
            model="qwen/qwen3-embedding-8b",
            encoding_format="float",
            embedding_dimensions=1024,
            max_concurrency=max_concurrency,
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


def test_canonicalizer_embeds_batches_concurrently(monkeypatch):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))

    class BlockingEmbeddingProcessor(FakeEmbeddingProcessor):
        def __init__(self):
            super().__init__(
                {
                    "实体类型：concept；实体名：A1；定义：a": [1.0, 0.0],
                    "实体类型：concept；实体名：B1；定义：b": [0.9, 0.1],
                    "实体类型：concept；实体名：C1；定义：c": [0.0, 1.0],
                    "实体类型：concept；实体名：D1；定义：d": [0.1, 0.9],
                },
                max_concurrency=2,
            )
            self._lock = threading.Lock()
            self._two_started = threading.Event()
            self._started = 0
            self._active = 0
            self.max_active = 0

        def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
            with self._lock:
                self._started += 1
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                if self._started >= 2:
                    self._two_started.set()
            try:
                assert self._two_started.wait(timeout=2), "embedding batches did not overlap"
                return super().embed(inputs, batch_size=batch_size)
            finally:
                with self._lock:
                    self._active -= 1

    fake = BlockingEmbeddingProcessor()
    canonicalizer = make_canonicalizer(fake, threshold=0.8)
    entities = [
        Entity(name="A1", type="concept", description="a", frequency=1),
        Entity(name="B1", type="concept", description="b", frequency=1),
        Entity(name="C1", type="concept", description="c", frequency=1),
        Entity(name="D1", type="concept", description="d", frequency=1),
    ]

    candidates = canonicalizer.build_candidates(entities)

    assert fake.max_active == 2
    assert len(fake.calls) == 2
    assert candidates


class FakeMergeJudgeProcessor:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.prompts: list[str] = []

    def run_text(self, prompt: str, *, system_prompt: str | None = None):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.response_text)


class FakeQueuedMergeJudgeProcessor:
    def __init__(self, response_texts: list[str]):
        self.response_texts = list(response_texts)
        self.prompts: list[str] = []

    def run_text(self, prompt: str, *, system_prompt: str | None = None):
        self.prompts.append(prompt)
        if not self.response_texts:
            raise AssertionError("no queued LLM response")
        return SimpleNamespace(text=self.response_texts.pop(0))


def test_merger_progress_labels_cover_load_embed_and_recheck(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    progress_descs: list[str | None] = []

    def fake_tqdm(iterable=None, **kwargs):
        progress_descs.append(kwargs.get("desc"))
        if iterable is not None:
            return iterable

        class _FakePbar:
            def update(self, _n=1): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass

        return _FakePbar()

    monkeypatch.setattr(kg_merger, "tqdm", fake_tqdm)
    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：concept；实体名：拥塞控制；定义：避免网络过载": [1.0, 0.0],
            "实体类型：concept；实体名：拥塞管理；定义：避免网络拥塞": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    judge = FakeMergeJudgeProcessor(
        json.dumps(
            {
                "decision": "unsure",
                "canonical_name": None,
                "reason_code": "INSUFFICIENT_EVIDENCE",
                "reason": "证据不足。",
                "supporting_evidence_ids": [],
                "conflict_evidence_ids": [],
            },
            ensure_ascii=False,
        )
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "拥塞控制", "type": "concept", "description": "避免网络过载"},
                    {"name": "拥塞管理", "type": "concept", "description": "避免网络拥塞"},
                ],
                "triples": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        tmp_path / "merged.json",
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {"enabled": True},
        },
    )

    assert "kgmerge load" in progress_descs
    assert "kgmerge embed" in progress_descs
    assert "kgmerge recheck" in progress_descs


def test_merger_llm_recheck_merges_validated_candidate(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：mechanism；实体名：三次握手；定义：TCP 建立连接的三次握手": [1.0, 0.0],
            "实体类型：mechanism；实体名：三路握手；定义：三次握手的别称": [1.0, 0.0],
            "实体类型：concept；实体名：可靠传输；定义：保证数据可靠到达": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    judge = FakeMergeJudgeProcessor(
        json.dumps(
            {
                "decision": "merge",
                "canonical_name": "三次握手",
                "reason_code": "SAME_MECHANISM",
                "reason": "证据显示二者描述同一连接建立机制。",
                "supporting_evidence_ids": ["L_DESC_1", "R_DESC_1"],
                "conflict_evidence_ids": [],
            },
            ensure_ascii=False,
        )
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "chunk_index": 1,
                "source_file": "book.md",
                "entities": [
                    {
                        "name": "三次握手",
                        "type": "mechanism",
                        "description": "TCP 建立连接的三次握手",
                    },
                    {"name": "三路握手", "type": "mechanism", "description": "三次握手的别称"},
                    {"name": "可靠传输", "type": "concept", "description": "保证数据可靠到达"},
                ],
                "triples": [
                    {
                        "head": "三次握手",
                        "relation": "applied_to",
                        "tail": "可靠传输",
                        "evidence": "三次握手用于建立可靠传输连接。",
                    },
                    {
                        "head": "三路握手",
                        "relation": "applied_to",
                        "tail": "可靠传输",
                        "evidence": "三路握手用于建立可靠传输连接。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {"enabled": True},
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert "三次握手" in names
    assert "三路握手" not in names
    assert data["merge_audit"]["decisions"][0]["method"] == "llm_recheck"
    assert "embedding_score" not in judge.prompts[0]


def test_merger_llm_recheck_rejects_unverifiable_llm_support(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：concept；实体名：拥塞控制；定义：避免网络过载": [1.0, 0.0],
            "实体类型：concept；实体名：拥塞管理；定义：避免网络拥塞": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    judge = FakeMergeJudgeProcessor(
        json.dumps(
            {
                "decision": "merge",
                "canonical_name": "拥塞控制",
                "reason_code": "SAME_CONCEPT",
                "reason": "看起来类似。",
                "supporting_evidence_ids": ["FAKE_EVIDENCE"],
                "conflict_evidence_ids": [],
            },
            ensure_ascii=False,
        )
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "拥塞控制", "type": "concept", "description": "避免网络过载"},
                    {"name": "拥塞管理", "type": "concept", "description": "避免网络拥塞"},
                ],
                "triples": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {"enabled": True},
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert {"拥塞控制", "拥塞管理"}.issubset(names)
    assert data["merge_audit"]["decisions"][0]["decision"] == "unsure"


def test_merger_rules_reject_direct_relation_before_llm(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：concept；实体名：拥塞控制；定义：避免网络过载": [1.0, 0.0],
            "实体类型：concept；实体名：流量控制；定义：避免接收方过载": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    judge = FakeMergeJudgeProcessor("{}")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "拥塞控制", "type": "concept", "description": "避免网络过载"},
                    {"name": "流量控制", "type": "concept", "description": "避免接收方过载"},
                ],
                "triples": [
                    {
                        "head": "拥塞控制",
                        "relation": "compared_with",
                        "tail": "流量控制",
                        "evidence": "拥塞控制与流量控制不同。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {"enabled": True},
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert {"拥塞控制", "流量控制"}.issubset(names)
    assert judge.prompts == []
    assert data["merge_audit"]["decisions"][0]["reason_code"] == "DIRECT_RELATION_SELF_LOOP_RISK"


def test_merger_embedding_without_recheck_does_not_merge_ambiguous_candidate(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：concept；实体名：TCP流量控制；定义：TCP 中控制发送速率": [1.0, 0.0],
            "实体类型：concept；实体名：UDP流量控制；定义：UDP 场景中的流量控制": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "TCP流量控制", "type": "concept", "description": "TCP 中控制发送速率"},
                    {
                        "name": "UDP流量控制",
                        "type": "concept",
                        "description": "UDP 场景中的流量控制",
                    },
                ],
                "triples": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger().merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {"enabled": False},
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert {"TCP流量控制", "UDP流量控制"}.issubset(names)
    assert data["merge_audit"]["decisions"][0]["reason_code"] == "LLM_RECHECK_DISABLED"


def test_merger_rejects_transitive_cluster_without_complete_pairwise_review(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：mechanism；实体名：A_TCP重传；定义：TCP 特定重传机制": [1.0, 0.0],
            "实体类型：mechanism；实体名：B_重传；定义：通用重传机制": [0.8, 0.6],
            "实体类型：mechanism；实体名：C_ARQ重传；定义：ARQ 特定重传机制": [0.28, 0.96],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    merge_response = json.dumps(
        {
            "decision": "merge",
            "canonical_name": "B_重传",
            "reason_code": "LOCAL_EQUIVALENCE",
            "reason": "局部证据支持合并。",
            "supporting_evidence_ids": ["L_DESC_1", "R_DESC_1"],
            "conflict_evidence_ids": [],
        },
        ensure_ascii=False,
    )
    judge = FakeQueuedMergeJudgeProcessor([merge_response, merge_response])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "A_TCP重传", "type": "mechanism", "description": "TCP 特定重传机制"},
                    {"name": "B_重传机制", "type": "mechanism", "description": "通用重传机制"},
                    {"name": "C_ARQ重传", "type": "mechanism", "description": "ARQ 特定重传机制"},
                ],
                "triples": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.75,
            "cache_path": None,
            "llm_recheck": {
                "enabled": True,
                "require_complete_pairwise_cluster": True,
            },
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert {"A_TCP重传", "B_重传", "C_ARQ重传"}.issubset(names)
    assert data["merge_audit"]["cluster_conflicts"][0]["reason_code"] == (
        "INCOMPLETE_PAIRWISE_CLUSTER"
    )


def test_merger_rejects_llm_merge_when_context_is_truncated(monkeypatch, tmp_path):
    from qmrkg import kg_merger

    monkeypatch.setattr(kg_merger, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndexFlatIP))
    fake_embed = FakeEmbeddingProcessor(
        {
            "实体类型：mechanism；实体名：三次握手；定义：TCP 建立连接机制": [1.0, 0.0],
            "实体类型：mechanism；实体名：三路握手；定义：TCP 建立连接机制": [1.0, 0.0],
            "实体类型：concept；实体名：可靠传输；定义：可靠数据传输": [0.0, 1.0],
            "实体类型：concept；实体名：连接建立；定义：建立连接": [0.0, -1.0],
        }
    )
    monkeypatch.setattr(
        kg_merger,
        "EmbeddingTaskProcessor",
        lambda task_name, config_path=None: fake_embed,
    )
    judge = FakeMergeJudgeProcessor(
        json.dumps(
            {
                "decision": "merge",
                "canonical_name": "三次握手",
                "reason_code": "SAME_MECHANISM",
                "reason": "证据显示二者相同。",
                "supporting_evidence_ids": ["L_DESC_1", "R_DESC_1"],
                "conflict_evidence_ids": [],
            },
            ensure_ascii=False,
        )
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "chunk.json").write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "三次握手", "type": "mechanism", "description": "TCP 建立连接机制"},
                    {"name": "三路握手", "type": "mechanism", "description": "TCP 建立连接机制"},
                    {"name": "可靠传输", "type": "concept", "description": "可靠数据传输"},
                    {"name": "连接建立", "type": "concept", "description": "建立连接"},
                ],
                "triples": [
                    {
                        "head": "三次握手",
                        "relation": "applied_to",
                        "tail": "可靠传输",
                        "evidence": "三次握手应用于可靠传输。",
                    },
                    {
                        "head": "三次握手",
                        "relation": "depends_on",
                        "tail": "连接建立",
                        "evidence": "三次握手依赖连接建立过程。",
                    },
                    {
                        "head": "三路握手",
                        "relation": "applied_to",
                        "tail": "可靠传输",
                        "evidence": "三路握手应用于可靠传输。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.json"
    KGMerger(merge_judge_processor=judge).merge_directory(
        raw_dir,
        output_path,
        embedding_config={
            "enabled": True,
            "candidate_threshold": 0.8,
            "cache_path": None,
            "llm_recheck": {
                "enabled": True,
                "context_triples_per_entity": 1,
                "allow_merge_with_truncated_context": False,
            },
        },
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    names = {entity["name"] for entity in data["entities"]}
    assert {"三次握手", "三路握手"}.issubset(names)
    assert data["merge_audit"]["decisions"][0]["reason_code"] == "TRUNCATED_CONTEXT"
