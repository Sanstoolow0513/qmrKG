# kgmerge Embedding Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional embedding-driven entity synonym canonicalization step to `kgmerge` while preserving the existing `merged_triples.json` schema and disabled-by-default behavior.

**Architecture:** Keep the current rule canonicalization (`ALIAS_MAP` + suffix stripping) as canonical_v1, then optionally apply an `EmbeddingCanonicalizer` that buckets canonical_v1 entities by type, embeds encoded entity text, unions pairs above a cosine threshold, and rewrites entity/triple names to canonical_v2 before final exact deduplication. All model calls go through `llm_factory`, with config parsed through the existing task/profile structure.

**Tech Stack:** Python 3.13, `uv`, pytest, PPIO OpenAI-compatible embeddings, numpy, existing `qmrkg` CLI/config/test patterns.

---

## Scope Check

This plan covers one subsystem: `kgmerge` entity canonicalization and the LLM embedding plumbing it requires. It does not change relation normalization, Neo4j import, frontend visualization, or the output JSON schema.

## File Structure

- Modify: `pyproject.toml`
  - Add `numpy` as a runtime dependency because cosine similarity is part of normal `kgmerge` execution when embedding is enabled.
- Modify: `config.yaml`
  - Add `embedding_qwen3_8b` profile, `entity_embed` task, and disabled-by-default `run.kg_merge.embedding` settings.
- Modify: `src/qmrkg/llm_types.py`
  - Extend `LLMModality` and add `LLMEmbeddingResponse`.
- Modify: `src/qmrkg/llm_config.py`
  - Parse embedding modality, optional prompt, `encoding_format`, and `dimensions`.
- Modify: `src/qmrkg/llm_factory.py`
  - Add embedding request execution and batching via `EmbeddingTaskProcessor`.
- Modify: `src/qmrkg/kg_merger.py`
  - Add `EmbeddingCanonicalizer`, cache handling, cosine unioning, canonical selection, and optional integration in `KGMerger.merge_directory`.
- Modify: `src/qmrkg/cli_kg_merge.py`
  - Pass config embedding settings to `KGMerger`, and add override flags.
- Modify: `src/qmrkg/config.py`
  - Add default `run.kg_merge.embedding` values so CLI config merging remains predictable.
- Modify: `tests/test_llm_factory.py`
  - Cover embedding config, runner validation, batching, and retry behavior with fake clients.
- Create: `tests/test_kg_merger_embedding.py`
  - Cover embedding canonicalization behavior, threshold/type-bucket rules, cache reuse, and disabled-path regression.

## Task 1: Add Failing Tests for Embedding LLM Config

**Files:**
- Modify: `tests/test_llm_factory.py`
- Later implementation: `src/qmrkg/llm_types.py`
- Later implementation: `src/qmrkg/llm_config.py`

- [ ] **Step 1: Add config tests to `tests/test_llm_factory.py`**

Append these tests after `test_factory_rejects_unknown_llm_profile`:

```python

def test_embedding_modality_no_prompt_required(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        base_url: https://api.ppinfra.com/openai
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        timeout_seconds: 30
        max_retries: 2
        encoding_format: float
        dimensions: 1024
      rate_limit:
        rpm: 100
        max_concurrency: 4
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    runner = LLMFactory(config_path).create("entity_embed", client=FakeClient(lambda **_: None))

    assert runner.settings.modality == "embedding"
    assert runner.settings.prompt == ""
    assert runner.settings.encoding_format == "float"
    assert runner.settings.embedding_dimensions == 1024


def test_embedding_modality_rejects_thinking(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        thinking:
          enabled: true
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    with pytest.raises(ValueError, match="request.thinking.enabled"):
        LLMFactory(config_path).create("entity_embed", client=FakeClient(lambda **_: None))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run pytest tests/test_llm_factory.py::test_embedding_modality_no_prompt_required tests/test_llm_factory.py::test_embedding_modality_rejects_thinking -v
```

Expected: the first test fails because `provider.modality` only accepts `text` and `multimodal`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_llm_factory.py
git commit -m "test: cover embedding llm configuration"
```

## Task 2: Implement Embedding Modality Config

**Files:**
- Modify: `src/qmrkg/llm_types.py`
- Modify: `src/qmrkg/llm_config.py`
- Test: `tests/test_llm_factory.py`

- [ ] **Step 1: Extend `src/qmrkg/llm_types.py`**

Change the modality literal and add the embedding response dataclass after `LLMResponse`:

```python
LLMModality = Literal["text", "multimodal", "embedding"]
```

```python
@dataclass(slots=True)
class LLMEmbeddingResponse:
    vectors: list[list[float]]
    model: str | None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float = 0.0
    processed_at: str = ""
```

- [ ] **Step 2: Extend `TaskLLMSettings` in `src/qmrkg/llm_config.py`**

Add constants near the existing defaults:

```python
DEFAULT_EMBEDDING_ENCODING_FORMAT = "float"
```

Update `_default_modality`:

```python
def _default_modality(task_name: str) -> str:
    if task_name.endswith("_embed") or task_name == "entity_embed":
        return "embedding"
    return "multimodal" if task_name == "ocr" else "text"
```

Add fields to `TaskLLMSettings`:

```python
    encoding_format: str = DEFAULT_EMBEDDING_ENCODING_FORMAT
    embedding_dimensions: int | None = None
```

Update modality validation:

```python
        if modality not in {"text", "multimodal", "embedding"}:
            raise ValueError("provider.modality must be one of: text, multimodal, embedding")
```

After `max_retries` parsing, add request parsing for embedding options:

```python
        encoding_format = _read_str(
            tuple(),
            _get_nested_value(request_config, "encoding_format"),
            DEFAULT_EMBEDDING_ENCODING_FORMAT,
        )
        dimensions_config = _get_nested_value(request_config, "dimensions")
        embedding_dimensions = None
        if dimensions_config not in (None, ""):
            embedding_dimensions = int(dimensions_config)
            if embedding_dimensions <= 0:
                raise ValueError("request.dimensions must be greater than 0")
```

Pass the new fields in the returned settings:

```python
            encoding_format=encoding_format,
            embedding_dimensions=embedding_dimensions,
```

- [ ] **Step 3: Run config tests**

Run:

```bash
uv run pytest tests/test_llm_factory.py::test_embedding_modality_no_prompt_required tests/test_llm_factory.py::test_embedding_modality_rejects_thinking -v
```

Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/llm_types.py src/qmrkg/llm_config.py tests/test_llm_factory.py
git commit -m "feat: support embedding llm configuration"
```

## Task 3: Add Failing Tests for Embedding Runner and Batching

**Files:**
- Modify: `tests/test_llm_factory.py`
- Later implementation: `src/qmrkg/llm_factory.py`

- [ ] **Step 1: Add fake embedding client classes**

Add these classes after `FakeClient`:

```python
class FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]], model: str = "fake-embedding-model"):
        self.model = model
        self.data = [
            type("EmbeddingItem", (), {"embedding": vector})()
            for vector in vectors
        ]
        self.usage = type("Usage", (), {"prompt_tokens": 3, "total_tokens": 3})()


class FakeEmbeddings:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class FakeEmbeddingClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.embeddings = FakeEmbeddings(self._record_and_handle)

    def _record_and_handle(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(**kwargs)
```

- [ ] **Step 2: Import `EmbeddingTaskProcessor`**

Change the import:

```python
from qmrkg.llm_factory import EmbeddingTaskProcessor, LLMFactory
```

- [ ] **Step 3: Add embedding runner tests**

Append these tests:

```python

def test_run_embeddings_validates_modality(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    text_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
ner:
  llm_profile: text_profile
  prompts: {}
""".strip(),
    )
    runner = LLMFactory(config_path).create(
        "ner",
        client=FakeEmbeddingClient(lambda **_: FakeEmbeddingResponse([[1.0, 0.0]])),
    )

    with pytest.raises(ValueError, match="modality=text"):
        runner.run_embeddings(["TCP"])


def test_run_embeddings_sends_request_options(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        timeout_seconds: 12
        max_retries: 1
        encoding_format: float
        dimensions: 1024
      rate_limit:
        rpm: 100
        max_concurrency: 4
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )
    fake_client = FakeEmbeddingClient(lambda **_: FakeEmbeddingResponse([[1.0, 0.0]]))
    runner = LLMFactory(config_path).create("entity_embed", client=fake_client)

    response = runner.run_embeddings(["protocol | TCP | 传输控制协议"])

    assert response.vectors == [[1.0, 0.0]]
    assert fake_client.calls[0]["model"] == "qwen/qwen3-embedding-8b"
    assert fake_client.calls[0]["input"] == ["protocol | TCP | 传输控制协议"]
    assert fake_client.calls[0]["encoding_format"] == "float"
    assert fake_client.calls[0]["dimensions"] == 1024


def test_embedding_task_processor_batches(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    def handler(**kwargs):
        vectors = [[float(index), 0.0] for index, _ in enumerate(kwargs["input"])]
        return FakeEmbeddingResponse(vectors)

    fake_client = FakeEmbeddingClient(handler)
    processor = EmbeddingTaskProcessor("entity_embed", config_path=config_path, client=fake_client)

    vectors = processor.embed(["a", "b", "c"], batch_size=2)

    assert vectors == [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    assert [call["input"] for call in fake_client.calls] == [["a", "b"], ["c"]]


def test_run_embeddings_retries_on_transient(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    monkeypatch.setattr("qmrkg.llm_factory.time.sleep", lambda _: None)
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        max_retries: 1
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )
    attempts = {"count": 0}

    def handler(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            error = RuntimeError("rate limited")
            error.status_code = 429
            raise error
        return FakeEmbeddingResponse([[1.0, 0.0]])

    runner = LLMFactory(config_path).create("entity_embed", client=FakeEmbeddingClient(handler))

    response = runner.run_embeddings(["TCP"])

    assert response.vectors == [[1.0, 0.0]]
    assert attempts["count"] == 2
```

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_llm_factory.py::test_run_embeddings_validates_modality tests/test_llm_factory.py::test_run_embeddings_sends_request_options tests/test_llm_factory.py::test_embedding_task_processor_batches tests/test_llm_factory.py::test_run_embeddings_retries_on_transient -v
```

Expected: import or attribute failure because `EmbeddingTaskProcessor` and `run_embeddings` do not exist.

- [ ] **Step 5: Commit**

```bash
git add tests/test_llm_factory.py
git commit -m "test: cover embedding runner behavior"
```

## Task 4: Implement Embedding Runner

**Files:**
- Modify: `src/qmrkg/llm_factory.py`
- Test: `tests/test_llm_factory.py`

- [ ] **Step 1: Import `LLMEmbeddingResponse`**

Change the import in `src/qmrkg/llm_factory.py`:

```python
from .llm_types import LLMContentPart, LLMEmbeddingResponse, LLMMessage, LLMResponse
```

- [ ] **Step 2: Add embedding public method and retry path to `TaskLLMRunner`**

Insert after `run_messages`:

```python
    def run_embeddings(self, inputs: list[str]) -> LLMEmbeddingResponse:
        if self.settings.modality != "embedding":
            raise ValueError(
                f"task '{self.settings.task_name}' is configured with modality={self.settings.modality}"
            )
        if not inputs:
            return LLMEmbeddingResponse(
                vectors=[],
                model=self.settings.model,
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._invoke_embeddings(inputs)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Transient embedding failure for task=%s on attempt %s/%s: %s",
                    self.settings.task_name,
                    attempt,
                    attempts,
                    self._format_exception_summary(exc),
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("unreachable")
```

- [ ] **Step 3: Add embedding invocation helpers**

Insert before `_request_kwargs`:

```python
    def _invoke_embeddings(self, inputs: list[str]) -> LLMEmbeddingResponse:
        start_time = time.perf_counter()
        self.rate_limiter.acquire()

        response = self.client.embeddings.create(
            model=self.settings.model,
            input=inputs,
            timeout=self.settings.timeout_seconds,
            **self._embedding_request_kwargs(),
        )

        duration = time.perf_counter() - start_time
        prompt_tokens, _, total_tokens = self._extract_usage(response)
        vectors = self._extract_embedding_vectors(response)
        return LLMEmbeddingResponse(
            vectors=vectors,
            model=getattr(response, "model", None) or self.settings.model,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _embedding_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"encoding_format": self.settings.encoding_format}
        if self.settings.embedding_dimensions is not None:
            kwargs["dimensions"] = self.settings.embedding_dimensions
        return kwargs

    @staticmethod
    def _extract_embedding_vectors(response) -> list[list[float]]:
        data = getattr(response, "data", None) or []
        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding")
            vectors.append([float(value) for value in embedding])
        return vectors
```

- [ ] **Step 4: Add `EmbeddingTaskProcessor`**

Append after `MultimodalTaskProcessor`:

```python

class EmbeddingTaskProcessor:
    """Convenience wrapper for embedding tasks with deterministic batching."""

    def __init__(self, task_name: str, config_path: Path | None = None, client=None):
        self.task_name = task_name
        self._factory = LLMFactory(config_path)
        self._runner = self._factory.create(task_name, client=client)

    @property
    def settings(self) -> TaskLLMSettings:
        return self._runner.settings

    def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            response = self._runner.run_embeddings(batch)
            if len(response.vectors) != len(batch):
                raise ValueError(
                    f"embedding response length mismatch: expected {len(batch)}, got {len(response.vectors)}"
                )
            vectors.extend(response.vectors)
        return vectors
```

- [ ] **Step 5: Run embedding runner tests**

Run:

```bash
uv run pytest tests/test_llm_factory.py::test_run_embeddings_validates_modality tests/test_llm_factory.py::test_run_embeddings_sends_request_options tests/test_llm_factory.py::test_embedding_task_processor_batches tests/test_llm_factory.py::test_run_embeddings_retries_on_transient -v
```

Expected: all four tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qmrkg/llm_factory.py tests/test_llm_factory.py
git commit -m "feat: add embedding task processor"
```

## Task 5: Add Failing Tests for Embedding Canonicalizer

**Files:**
- Create: `tests/test_kg_merger_embedding.py`
- Later implementation: `src/qmrkg/kg_merger.py`

- [ ] **Step 1: Create `tests/test_kg_merger_embedding.py`**

```python
from __future__ import annotations

import json

from qmrkg.kg_merger import EmbeddingCanonicalizer, KGMerger
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
) -> EmbeddingCanonicalizer:
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
            "concept | TCP": [1.0, 0.0],
        }
    )
    canonicalizer = make_canonicalizer(fake, bucket_by_type=True)
    entities = [
        Entity(name="TCP", type="protocol", frequency=1),
        Entity(name="TCP", type="concept", frequency=1),
    ]

    mapping = canonicalizer.build_canonical_map(entities)

    assert mapping["TCP"] == "TCP"


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
            "mechanism | 慢启动 |": [1.0, 0.0],
            "mechanism | 慢启动算法 |": [1.0, 0.0],
            "mechanism | TCP慢启动 |": [1.0, 0.0],
        }
    )
    canonicalizer = EmbeddingCanonicalizer(
        task_name="entity_embed",
        encode_fields=["type", "name", "description"],
        similarity_threshold=0.85,
        bucket_by_type=True,
        batch_size=2,
        cache_path=None,
        config_path=None,
        processor=fake,
    )
    entities = [
        Entity(name="慢启动", type="mechanism", description="", frequency=2),
        Entity(name="慢启动算法", type="mechanism", description="", frequency=2),
        Entity(name="TCP慢启动", type="mechanism", description="", frequency=5),
    ]

    mapping = canonicalizer.build_canonical_map(entities)

    assert mapping["慢启动"] == "TCP慢启动"
    assert mapping["慢启动算法"] == "TCP慢启动"
    assert mapping["TCP慢启动"] == "TCP慢启动"


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
    (raw_dir / "chunk.json").write_text(json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8")

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
```

- [ ] **Step 2: Run canonicalizer tests to verify they fail**

Run:

```bash
uv run pytest tests/test_kg_merger_embedding.py -v
```

Expected: import failure because `EmbeddingCanonicalizer` does not exist and `merge_directory` does not accept `embedding_config`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kg_merger_embedding.py
git commit -m "test: cover embedding canonicalization"
```

## Task 6: Implement Embedding Canonicalizer

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/qmrkg/kg_merger.py`
- Test: `tests/test_kg_merger_embedding.py`

- [ ] **Step 1: Add numpy dependency**

Run:

```bash
uv add numpy
```

Expected: `pyproject.toml` and lockfile update with `numpy`.

- [ ] **Step 2: Add imports to `src/qmrkg/kg_merger.py`**

Add these imports:

```python
import hashlib
from typing import Any

import numpy as np

from .llm_factory import EmbeddingTaskProcessor
```

- [ ] **Step 3: Add union-find helpers above `KGMerger`**

```python
class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root
```

- [ ] **Step 4: Add `EmbeddingCanonicalizer` above `KGMerger`**

```python
class EmbeddingCanonicalizer:
    """Build an entity canonicalization map from embedding similarity."""

    def __init__(
        self,
        *,
        task_name: str,
        encode_fields: list[str],
        similarity_threshold: float,
        bucket_by_type: bool,
        batch_size: int,
        cache_path: Path | None,
        config_path: Path | None,
        processor=None,
    ):
        self.task_name = task_name
        self.encode_fields = encode_fields
        self.similarity_threshold = similarity_threshold
        self.bucket_by_type = bucket_by_type
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.config_path = config_path
        self.processor = processor or EmbeddingTaskProcessor(task_name, config_path=config_path)
        self._cache: dict[str, list[float]] = self._load_cache()

    def build_canonical_map(self, entities: list[Entity]) -> dict[str, str]:
        unique_entities = self._unique_entities(entities)
        mapping = {entity.name: entity.name for entity in unique_entities}
        buckets = self._bucket_entities(unique_entities)
        for bucket_entities in buckets.values():
            if len(bucket_entities) < 2:
                continue
            texts = [self._encode_entity(entity) for entity in bucket_entities]
            vectors = self._embed_texts(texts)
            bucket_mapping = self._build_bucket_mapping(bucket_entities, vectors)
            mapping.update(bucket_mapping)
        self._write_cache()
        return mapping

    def _unique_entities(self, entities: list[Entity]) -> list[Entity]:
        grouped: dict[tuple[str, str], Entity] = {}
        for entity in entities:
            if not entity.name.strip():
                continue
            key = (entity.name, entity.type)
            if key in grouped:
                grouped[key].frequency += entity.frequency
                if not grouped[key].description and entity.description:
                    grouped[key].description = entity.description
            else:
                grouped[key] = Entity(
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    frequency=entity.frequency,
                )
        return list(grouped.values())

    def _bucket_entities(self, entities: list[Entity]) -> dict[str, list[Entity]]:
        buckets: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            key = entity.type if self.bucket_by_type else "__all__"
            buckets[key].append(entity)
        return buckets

    def _encode_entity(self, entity: Entity) -> str:
        values: list[str] = []
        for field_name in self.encode_fields:
            value = getattr(entity, field_name, "")
            if value:
                values.append(str(value))
        return " | ".join(values)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if self._cache_key(text) not in self._cache]
        if missing:
            vectors = self.processor.embed(missing, batch_size=self.batch_size)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[self._cache_key(text)] = vector
        return [self._cache[self._cache_key(text)] for text in texts]

    def _build_bucket_mapping(
        self,
        entities: list[Entity],
        vectors: list[list[float]],
    ) -> dict[str, str]:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        similarity = normalized @ normalized.T

        union_find = _UnionFind(len(entities))
        for left in range(len(entities)):
            for right in range(left + 1, len(entities)):
                if float(similarity[left, right]) >= self.similarity_threshold:
                    union_find.union(left, right)

        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index in range(len(entities)):
            grouped_indices[union_find.find(index)].append(index)

        mapping: dict[str, str] = {}
        for indices in grouped_indices.values():
            canonical = self._pick_canonical([entities[index] for index in indices])
            for index in indices:
                mapping[entities[index].name] = canonical.name
        return mapping

    @staticmethod
    def _pick_canonical(entities: list[Entity]) -> Entity:
        return sorted(entities, key=lambda entity: (-entity.frequency, len(entity.name), entity.name))[0]

    def _cache_key(self, text: str) -> str:
        model = getattr(self.processor, "settings", None)
        model_id = getattr(model, "model", self.task_name)
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{model_id}::{digest}"

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {
            str(key): [float(value) for value in vector]
            for key, vector in data.items()
            if isinstance(vector, list)
        }

    def _write_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 5: Add mapping helpers inside `KGMerger`**

Insert before `_merge_entities`:

```python
    def _apply_entity_mapping(
        self,
        entities: list[Entity],
        mapping: dict[str, str],
    ) -> list[Entity]:
        return [
            Entity(
                name=mapping.get(entity.name, entity.name),
                type=entity.type,
                description=entity.description,
                frequency=entity.frequency,
            )
            for entity in entities
        ]

    def _apply_triple_mapping(
        self,
        triples: list[Triple],
        mapping: dict[str, str],
    ) -> list[Triple]:
        return [
            Triple(
                head=mapping.get(normalize_entity_name(triple.head), normalize_entity_name(triple.head)),
                relation=triple.relation,
                tail=mapping.get(normalize_entity_name(triple.tail), normalize_entity_name(triple.tail)),
                evidence=triple.evidence,
                evidence_span=triple.evidence_span,
                frequency=triple.frequency,
                evidences=list(triple.evidences),
                review_decision=triple.review_decision,
                review_reason_code=triple.review_reason_code,
                review_reason=triple.review_reason,
            )
            for triple in triples
        ]
```

- [ ] **Step 6: Change `merge_directory` signature and integration**

Change the signature:

```python
    def merge_directory(
        self,
        raw_dir: Path,
        output_path: Path,
        embedding_config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> Path:
```

Replace the current merge block:

```python
        merged_entities = self._merge_entities(all_entities)
        entity_names = {e.name for e in merged_entities}
        merged_triples = self._merge_triples(all_triples, entity_names)
```

with:

```python
        merged_entities = self._merge_entities(all_entities)
        if embedding_config and embedding_config.get("enabled"):
            canonicalizer = EmbeddingCanonicalizer(
                task_name=str(embedding_config.get("task_name", "entity_embed")),
                encode_fields=list(
                    embedding_config.get("encode_fields", ["type", "name", "description"])
                ),
                similarity_threshold=float(embedding_config.get("similarity_threshold", 0.85)),
                bucket_by_type=bool(embedding_config.get("bucket_by_type", True)),
                batch_size=int(embedding_config.get("batch_size", 1024)),
                cache_path=(
                    Path(str(embedding_config["cache_path"]))
                    if embedding_config.get("cache_path")
                    else None
                ),
                config_path=config_path,
            )
            canonical_map = canonicalizer.build_canonical_map(merged_entities)
            mapped_entities = self._apply_entity_mapping(merged_entities, canonical_map)
            mapped_triples = self._apply_triple_mapping(all_triples, canonical_map)
            merged_entities = self._merge_entities(mapped_entities)
            entity_names = {e.name for e in merged_entities}
            merged_triples = self._merge_triples(mapped_triples, entity_names)
        else:
            entity_names = {e.name for e in merged_entities}
            merged_triples = self._merge_triples(all_triples, entity_names)
```

- [ ] **Step 7: Run canonicalizer tests**

Run:

```bash
uv run pytest tests/test_kg_merger_embedding.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/qmrkg/kg_merger.py tests/test_kg_merger_embedding.py
git commit -m "feat: canonicalize kg entities with embeddings"
```

## Task 7: Wire Runtime Config and CLI Overrides

**Files:**
- Modify: `config.yaml`
- Modify: `src/qmrkg/config.py`
- Modify: `src/qmrkg/cli_kg_merge.py`
- Test: manual CLI smoke command

- [ ] **Step 1: Add default embedding config to `src/qmrkg/config.py`**

Change the `kg_merge` section in `DEFAULT_RUN_CONFIG`:

```python
    "kg_merge": {
        "input_dir": "data/triples/raw",
        "output": "data/triples/merged/merged_triples.json",
        "embedding": {
            "enabled": False,
            "task_name": "entity_embed",
            "encode_fields": ["type", "name", "description"],
            "similarity_threshold": 0.85,
            "bucket_by_type": True,
            "batch_size": 1024,
            "cache_path": "data/triples/merged/.embed_cache.json",
        },
    },
```

- [ ] **Step 2: Add `config.yaml` embedding profile and task**

Add this profile under `llm.profiles`:

```yaml
    embedding_qwen3_8b:
      provider:
        name: ppio
        base_url: "https://api.ppinfra.com/openai"
        model: "qwen/qwen3-embedding-8b"
        modality: "embedding"
        supports_thinking: false
      request:
        timeout_seconds: 60.0
        max_retries: 3
        encoding_format: "float"
        dimensions: 1024
      rate_limit:
        rpm: 100
        max_concurrency: 4
```

Add this top-level task near the existing `ocr`, `extract`, `ner`, and `re` tasks:

```yaml
entity_embed:
  llm_profile: embedding_qwen3_8b
```

Add this under `run.kg_merge`:

```yaml
    embedding:
      enabled: false
      task_name: "entity_embed"
      encode_fields: ["type", "name", "description"]
      similarity_threshold: 0.85
      bucket_by_type: true
      batch_size: 1024
      cache_path: "data/triples/merged/.embed_cache.json"
```

- [ ] **Step 3: Add CLI flags in `src/qmrkg/cli_kg_merge.py`**

Add parser arguments after `--output`:

```python
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding canonicalization even if config enables it",
    )
    parser.add_argument(
        "--embedding-task",
        help="Override embedding task name, e.g. entity_embed",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Override embedding cosine similarity threshold",
    )
```

Replace the merge invocation:

```python
    embedding_config = dict(run_cfg.get("embedding", {}))
    if args.no_embedding:
        embedding_config["enabled"] = False
    if args.embedding_task:
        embedding_config["task_name"] = args.embedding_task
    if args.similarity_threshold is not None:
        embedding_config["similarity_threshold"] = args.similarity_threshold

    merger = KGMerger()
    output = merger.merge_directory(
        args.input_dir,
        args.output,
        embedding_config=embedding_config,
        config_path=args.config,
    )
```

- [ ] **Step 4: Run disabled-path CLI smoke test**

Run:

```bash
uv run kgmerge --no-embedding --input-dir data/triples/raw --output /tmp/qmrkg-kgmerge-disabled.json
```

Expected: command completes without embedding API calls and prints `Merged triples saved to: /tmp/qmrkg-kgmerge-disabled.json`.

- [ ] **Step 5: Commit**

```bash
git add config.yaml src/qmrkg/config.py src/qmrkg/cli_kg_merge.py
git commit -m "feat: wire kgmerge embedding configuration"
```

## Task 8: Full Regression and Quality Checks

**Files:**
- Verify all modified Python files and tests.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_kg_merger.py tests/test_kg_merger_embedding.py tests/test_llm_factory.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Run format and lint checks**

Run:

```bash
uv run black --check src/qmrkg tests
uv run ruff check src/qmrkg tests
```

Expected: both commands pass. If black reports formatting changes are needed, run `uv run black src/qmrkg tests`, then repeat the checks.

- [ ] **Step 4: Run import smoke test**

Run:

```bash
uv run python -c "from qmrkg.kg_merger import EmbeddingCanonicalizer, KGMerger; from qmrkg.llm_factory import EmbeddingTaskProcessor; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 5: Commit verification fixes**

If Step 3 formatting changed files, commit them:

```bash
git add src/qmrkg tests
git commit -m "style: format embedding canonicalization changes"
```

If no files changed, do not create an empty commit.

## Self-Review

**Spec coverage:** The plan covers embedding modality config, task-scoped `entity_embed`, embedding runner through `llm_factory`, type-bucketed pairwise cosine, threshold unioning, canonical selection by frequency/length/name, cache read/write, disabled-by-default runtime config, CLI overrides, unchanged output schema, and regression tests for the legacy path.

**Placeholder scan:** The plan contains concrete file paths, concrete test code, concrete implementation snippets, commands, and expected outcomes. It avoids deferred implementation language and does not require unlisted files.

**Type consistency:** The plan consistently uses `EmbeddingTaskProcessor.embed(inputs, batch_size)`, `TaskLLMRunner.run_embeddings(inputs)`, `LLMEmbeddingResponse.vectors`, `embedding_dimensions`, and `EmbeddingCanonicalizer.build_canonical_map(entities)`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-kgmerge-embedding-canonicalization.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
