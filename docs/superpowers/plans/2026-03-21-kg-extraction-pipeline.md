# 知识图谱抽取管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 chunks 管线基础上，实现 LLM 联合知识抽取、三元组融合、Neo4j 导入的完整知识图谱构建管线。

**Architecture:** 三个新模块串联：`kg_extractor` 读取 chunks 调用 LLM 联合抽取三元组；`kg_merger` 对原始三元组做实体归一化、去重和过滤；`kg_neo4j` 将融合结果导入 Neo4j 图数据库。每个模块有独立 CLI 入口。

**Tech Stack:** Python 3.13, OpenAI SDK (PPIO), DeepSeek V3.2, Neo4j 5.x, neo4j Python driver

**Spec:** `docs/superpowers/specs/2026-03-21-kg-extraction-pipeline-design.md`

---

## File Structure

| 操作 | 文件 | 职责 |
|---|---|---|
| Create | `src/qmrkg/kg_schema.py` | 实体/关系类型常量与数据类 |
| Create | `src/qmrkg/kg_extractor.py` | 读取 chunks, 调用 LLM, 输出原始三元组 |
| Create | `src/qmrkg/kg_merger.py` | 三元组归一化、去重、过滤、融合 |
| Create | `src/qmrkg/kg_neo4j.py` | Neo4j 导入与查询 |
| Create | `src/qmrkg/cli_kg_extract.py` | 抽取 CLI |
| Create | `src/qmrkg/cli_kg_merge.py` | 融合 CLI |
| Create | `src/qmrkg/cli_kg_neo4j.py` | Neo4j 导入 CLI |
| Modify | `src/qmrkg/__init__.py` | 导出新模块 |
| Modify | `pyproject.toml` | 新增 neo4j 依赖和 CLI 入口 |
| Modify | `config.yaml` | 新增 extract 任务段 |
| Create | `tests/test_kg_schema.py` | schema 测试 |
| Create | `tests/test_kg_extractor.py` | extractor 测试 |
| Create | `tests/test_kg_merger.py` | merger 测试 |
| Create | `tests/test_kg_neo4j.py` | neo4j 测试 |

---

### Task 1: 类型定义 — `kg_schema.py`

**Files:**
- Create: `src/qmrkg/kg_schema.py`
- Test: `tests/test_kg_schema.py`

- [ ] **Step 1: 创建 `kg_schema.py`**

```python
"""Entity and relation type definitions for knowledge graph extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EntityType = Literal["protocol", "concept", "mechanism", "metric"]
RelationType = Literal["contains", "depends_on", "compared_with", "applied_to"]

ENTITY_TYPES: set[str] = {"protocol", "concept", "mechanism", "metric"}
RELATION_TYPES: set[str] = {"contains", "depends_on", "compared_with", "applied_to"}

ENTITY_TYPE_LABELS: dict[str, str] = {
    "protocol": "Protocol",
    "concept": "Concept",
    "mechanism": "Mechanism",
    "metric": "Metric",
}

RELATION_TYPE_LABELS: dict[str, str] = {
    "contains": "CONTAINS",
    "depends_on": "DEPENDS_ON",
    "compared_with": "COMPARED_WITH",
    "applied_to": "APPLIED_TO",
}


@dataclass(slots=True)
class Entity:
    name: str
    type: str
    description: str = ""
    frequency: int = 1

    def is_valid(self) -> bool:
        return (
            self.type in ENTITY_TYPES
            and 2 <= len(self.name) <= 30
            and bool(self.name.strip())
        )


@dataclass(slots=True)
class Triple:
    head: str
    relation: str
    tail: str
    evidence: str = ""
    frequency: int = 1
    evidences: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return (
            self.relation in RELATION_TYPES
            and self.head != self.tail
            and bool(self.head.strip())
            and bool(self.tail.strip())
        )


@dataclass(slots=True)
class ChunkExtractionResult:
    chunk_index: int
    source_file: str
    titles: list[str]
    entities: list[Entity]
    triples: list[Triple]
```

- [ ] **Step 2: 写测试 `tests/test_kg_schema.py`**

```python
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


def test_triple_valid():
    t = Triple(head="TCP", relation="compared_with", tail="UDP")
    assert t.is_valid()


def test_triple_self_loop():
    t = Triple(head="TCP", relation="contains", tail="TCP")
    assert not t.is_valid()


def test_triple_invalid_relation():
    t = Triple(head="TCP", relation="unknown", tail="UDP")
    assert not t.is_valid()
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_kg_schema.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/kg_schema.py tests/test_kg_schema.py
git commit -m "feat: add kg_schema with entity/relation type definitions"
```

---

### Task 2: LLM 联合抽取 — `kg_extractor.py`

**Files:**
- Create: `src/qmrkg/kg_extractor.py`
- Test: `tests/test_kg_extractor.py`

- [ ] **Step 1: 创建 `kg_extractor.py`**

核心类 `KGExtractor`：

```python
"""Knowledge graph extraction from markdown chunks using LLM."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .kg_schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    ChunkExtractionResult,
    Entity,
    Triple,
)
from .llm_factory import LLMFactory, TaskLLMRunner

logger = logging.getLogger(__name__)

EXTRACT_TASK_NAME = "extract"

EXTRACT_PROMPT = """\
你是一个计算机网络课程知识图谱构建专家。

任务：从给定的教材文本中，识别命名实体并抽取实体间的关系，生成知识三元组。

## 实体类型（4类）
- protocol: 协议名称（如 TCP、HTTP、DNS、ARP）
- concept: 概念术语（如 拥塞控制、子网掩码、路由表）
- mechanism: 机制算法（如 三次握手、慢启动算法、CSMA/CD）
- metric: 性能指标（如 吞吐量、RTT、丢包率、带宽）

## 关系类型（4类）
- contains: 包含关系（A 包含 B）
- depends_on: 依赖关系（A 依赖 B）
- compared_with: 对比关系（A 与 B 对比）
- applied_to: 应用关系（A 应用于 B）

## 输出格式
严格输出 JSON，不要输出任何其他内容：
{
  "entities": [
    {"name": "TCP", "type": "protocol", "description": "传输控制协议"}
  ],
  "triples": [
    {"head": "TCP", "relation": "compared_with", "tail": "UDP", "evidence": "原文依据"}
  ]
}

## 规则
1. 只从给定文本中抽取，不要编造
2. entity.name 使用文本中出现的原始名称
3. 每个 triple 必须附带 evidence（原文中支持该关系的关键句）
4. 如果文本中没有可抽取的实体或关系，返回空列表\
"""


class KGExtractor:
    """Extract entities and relations from markdown chunks via LLM."""

    def __init__(self, runner: TaskLLMRunner | None = None, config_path: Path | None = None):
        if runner is not None:
            self._runner = runner
        else:
            factory = LLMFactory(config_path)
            self._runner = factory.create(EXTRACT_TASK_NAME)

    def extract_from_chunk(self, chunk: dict) -> ChunkExtractionResult:
        """Extract entities and relations from a single chunk dict."""
        content = chunk.get("content", "")
        if not content.strip():
            return ChunkExtractionResult(
                chunk_index=chunk.get("chunk_index", 0),
                source_file=chunk.get("source_file", ""),
                titles=chunk.get("titles", []),
                entities=[],
                triples=[],
            )

        response = self._runner.run_text(content, system_prompt=EXTRACT_PROMPT)
        raw = self._parse_json_response(response.text)
        entities = self._parse_entities(raw.get("entities", []))
        triples = self._parse_triples(raw.get("triples", []))

        return ChunkExtractionResult(
            chunk_index=chunk.get("chunk_index", 0),
            source_file=chunk.get("source_file", ""),
            titles=chunk.get("titles", []),
            entities=entities,
            triples=triples,
        )

    def extract_from_chunks_file(
        self,
        chunks_path: Path,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Extract from all chunks in a JSON file, saving per-chunk results."""
        chunks_path = Path(chunks_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        result_paths: list[Path] = []

        for chunk in chunks:
            idx = chunk.get("chunk_index", 0)
            out_path = output_dir / f"{chunks_path.stem}_chunk_{idx:04d}.json"

            if skip_existing and out_path.exists():
                logger.info("Skipping existing %s", out_path.name)
                result_paths.append(out_path)
                continue

            try:
                result = self.extract_from_chunk(chunk)
                self._save_result(result, out_path)
                result_paths.append(out_path)
                logger.info(
                    "Extracted chunk %d: %d entities, %d triples",
                    idx, len(result.entities), len(result.triples),
                )
            except Exception as e:
                logger.error("Failed chunk %d: %s", idx, e)

        return result_paths

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code fences."""
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")
            return {"entities": [], "triples": []}

    @staticmethod
    def _parse_entities(raw_entities: list) -> list[Entity]:
        entities = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            entity = Entity(
                name=str(item.get("name", "")).strip(),
                type=str(item.get("type", "")).strip().lower(),
                description=str(item.get("description", "")).strip(),
            )
            if entity.is_valid():
                entities.append(entity)
        return entities

    @staticmethod
    def _parse_triples(raw_triples: list) -> list[Triple]:
        triples = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue
            triple = Triple(
                head=str(item.get("head", "")).strip(),
                relation=str(item.get("relation", "")).strip().lower(),
                tail=str(item.get("tail", "")).strip(),
                evidence=str(item.get("evidence", "")).strip(),
            )
            if triple.is_valid():
                triples.append(triple)
        return triples

    @staticmethod
    def _save_result(result: ChunkExtractionResult, path: Path) -> None:
        data = {
            "chunk_index": result.chunk_index,
            "source_file": result.source_file,
            "titles": result.titles,
            "entities": [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in result.entities
            ],
            "triples": [
                {"head": t.head, "relation": t.relation, "tail": t.tail, "evidence": t.evidence}
                for t in result.triples
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 2: 写测试 `tests/test_kg_extractor.py`**

测试 JSON 解析和实体/三元组校验逻辑（mock LLM 调用）：

```python
from qmrkg.kg_extractor import KGExtractor


def test_parse_json_response_plain():
    text = '{"entities": [{"name": "TCP", "type": "protocol", "description": "x"}], "triples": []}'
    result = KGExtractor._parse_json_response(text)
    assert len(result["entities"]) == 1


def test_parse_json_response_fenced():
    text = '```json\n{"entities": [], "triples": []}\n```'
    result = KGExtractor._parse_json_response(text)
    assert result == {"entities": [], "triples": []}


def test_parse_json_response_invalid():
    result = KGExtractor._parse_json_response("not json at all")
    assert result == {"entities": [], "triples": []}


def test_parse_entities_filters_invalid():
    raw = [
        {"name": "TCP", "type": "protocol", "description": "x"},
        {"name": "X", "type": "protocol", "description": "too short"},
        {"name": "TCP", "type": "unknown", "description": "bad type"},
    ]
    entities = KGExtractor._parse_entities(raw)
    assert len(entities) == 1
    assert entities[0].name == "TCP"


def test_parse_triples_filters_invalid():
    raw = [
        {"head": "TCP", "relation": "compared_with", "tail": "UDP", "evidence": "x"},
        {"head": "TCP", "relation": "contains", "tail": "TCP", "evidence": "self loop"},
        {"head": "TCP", "relation": "bad_rel", "tail": "UDP", "evidence": "bad"},
    ]
    triples = KGExtractor._parse_triples(raw)
    assert len(triples) == 1
    assert triples[0].head == "TCP"
    assert triples[0].tail == "UDP"
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_kg_extractor.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/kg_extractor.py tests/test_kg_extractor.py
git commit -m "feat: add kg_extractor for LLM-based joint entity and relation extraction"
```

---

### Task 3: 三元组融合 — `kg_merger.py`

**Files:**
- Create: `src/qmrkg/kg_merger.py`
- Test: `tests/test_kg_merger.py`

- [ ] **Step 1: 创建 `kg_merger.py`**

```python
"""Merge, normalize and deduplicate extracted triples."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from .kg_schema import ENTITY_TYPES, RELATION_TYPES, Entity, Triple

logger = logging.getLogger(__name__)

SUFFIX_PATTERN = re.compile(r"(协议|算法|机制|方法|技术|方式)$")

ALIAS_MAP: dict[str, str] = {
    "传输控制协议": "TCP",
    "用户数据报协议": "UDP",
    "超文本传输协议": "HTTP",
    "域名系统": "DNS",
    "文件传输协议": "FTP",
    "简单邮件传输协议": "SMTP",
    "地址解析协议": "ARP",
    "网际控制报文协议": "ICMP",
    "网际协议": "IP",
    "往返时延": "RTT",
    "往返时间": "RTT",
    "最大传输单元": "MTU",
    "服务质量": "QoS",
}


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to a canonical form."""
    name = name.strip()
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    without_suffix = SUFFIX_PATTERN.sub("", name)
    if len(without_suffix) >= 2 and without_suffix in ALIAS_MAP:
        return ALIAS_MAP[without_suffix]
    if without_suffix != name and len(without_suffix) >= 2:
        return without_suffix
    return name


class KGMerger:
    """Merge raw extraction results into a deduplicated knowledge graph."""

    def __init__(self, alias_map: dict[str, str] | None = None):
        if alias_map:
            ALIAS_MAP.update(alias_map)

    def merge_directory(self, raw_dir: Path, output_path: Path) -> Path:
        """Load all raw triple files from a directory and merge them."""
        raw_dir = Path(raw_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(raw_dir.glob("*.json"))
        if not raw_files:
            logger.warning("No raw triple files found in %s", raw_dir)

        all_entities: list[Entity] = []
        all_triples: list[Triple] = []

        for f in raw_files:
            data = json.loads(f.read_text(encoding="utf-8"))
            for e in data.get("entities", []):
                all_entities.append(
                    Entity(name=e["name"], type=e["type"], description=e.get("description", ""))
                )
            for t in data.get("triples", []):
                all_triples.append(
                    Triple(
                        head=t["head"],
                        relation=t["relation"],
                        tail=t["tail"],
                        evidence=t.get("evidence", ""),
                    )
                )

        merged_entities = self._merge_entities(all_entities)
        entity_names = {e.name for e in merged_entities}
        merged_triples = self._merge_triples(all_triples, entity_names)

        entity_type_map = {e.name: e.type for e in merged_entities}

        result = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "frequency": e.frequency,
                }
                for e in sorted(merged_entities, key=lambda x: x.frequency, reverse=True)
            ],
            "triples": [
                {
                    "head": t.head,
                    "head_type": entity_type_map.get(t.head, ""),
                    "relation": t.relation,
                    "tail": t.tail,
                    "tail_type": entity_type_map.get(t.tail, ""),
                    "frequency": t.frequency,
                    "evidences": t.evidences,
                }
                for t in sorted(merged_triples, key=lambda x: x.frequency, reverse=True)
            ],
            "stats": self._compute_stats(merged_entities, merged_triples),
        }

        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "Merged: %d entities, %d triples -> %s",
            len(merged_entities), len(merged_triples), output_path,
        )
        return output_path

    def _merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """Normalize names, deduplicate, accumulate frequency."""
        grouped: dict[str, Entity] = {}
        for e in entities:
            if not e.is_valid():
                continue
            canonical = normalize_entity_name(e.name)
            if canonical in grouped:
                grouped[canonical].frequency += 1
                if not grouped[canonical].description and e.description:
                    grouped[canonical].description = e.description
            else:
                grouped[canonical] = Entity(
                    name=canonical,
                    type=e.type,
                    description=e.description,
                    frequency=1,
                )
        return list(grouped.values())

    def _merge_triples(
        self, triples: list[Triple], valid_entity_names: set[str]
    ) -> list[Triple]:
        """Normalize head/tail, deduplicate, accumulate frequency and evidences."""
        grouped: dict[tuple[str, str, str], Triple] = {}
        for t in triples:
            if not t.is_valid():
                continue
            head = normalize_entity_name(t.head)
            tail = normalize_entity_name(t.tail)
            if head == tail:
                continue
            if head not in valid_entity_names or tail not in valid_entity_names:
                continue
            key = (head, t.relation, tail)
            if key in grouped:
                grouped[key].frequency += 1
                if t.evidence and t.evidence not in grouped[key].evidences:
                    grouped[key].evidences.append(t.evidence)
            else:
                grouped[key] = Triple(
                    head=head,
                    relation=t.relation,
                    tail=tail,
                    frequency=1,
                    evidences=[t.evidence] if t.evidence else [],
                )
        return list(grouped.values())

    @staticmethod
    def _compute_stats(
        entities: list[Entity], triples: list[Triple]
    ) -> dict:
        entities_by_type: dict[str, int] = defaultdict(int)
        for e in entities:
            entities_by_type[e.type] += 1
        triples_by_relation: dict[str, int] = defaultdict(int)
        for t in triples:
            triples_by_relation[t.relation] += 1
        return {
            "total_entities": len(entities),
            "total_triples": len(triples),
            "entities_by_type": dict(entities_by_type),
            "triples_by_relation": dict(triples_by_relation),
        }
```

- [ ] **Step 2: 写测试 `tests/test_kg_merger.py`**

```python
from qmrkg.kg_merger import normalize_entity_name, KGMerger
from qmrkg.kg_schema import Entity, Triple


def test_normalize_alias():
    assert normalize_entity_name("传输控制协议") == "TCP"
    assert normalize_entity_name("往返时延") == "RTT"


def test_normalize_suffix_removal():
    assert normalize_entity_name("慢启动算法") == "慢启动"


def test_normalize_passthrough():
    assert normalize_entity_name("TCP") == "TCP"


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


def test_merge_triples_filters_unknown_entity():
    merger = KGMerger()
    valid_names = {"TCP"}
    triples = [
        Triple(head="TCP", relation="contains", tail="UNKNOWN", evidence="x"),
    ]
    merged = merger._merge_triples(triples, valid_names)
    assert len(merged) == 0
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_kg_merger.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/kg_merger.py tests/test_kg_merger.py
git commit -m "feat: add kg_merger for triple normalization, dedup and fusion"
```

---

### Task 4: Neo4j 导入 — `kg_neo4j.py`

**Files:**
- Create: `src/qmrkg/kg_neo4j.py`
- Test: `tests/test_kg_neo4j.py`

- [ ] **Step 1: 创建 `kg_neo4j.py`**

```python
"""Load merged triples into Neo4j graph database."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .kg_schema import ENTITY_TYPE_LABELS, RELATION_TYPE_LABELS

logger = logging.getLogger(__name__)

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"


def _read_neo4j_env() -> tuple[str, str, str]:
    uri = os.getenv("NEO4J_URI", DEFAULT_URI)
    user = os.getenv("NEO4J_USER", DEFAULT_USER)
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is required")
    return uri, user, password


class KGNeo4jLoader:
    """Import merged triples into Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("neo4j not installed. Run: pip install neo4j") from exc

        env_uri, env_user, env_password = _read_neo4j_env()
        self._uri = uri or env_uri
        self._user = user or env_user
        self._password = password or env_password
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def clear_database(self) -> None:
        """Delete all nodes and relationships."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all data from Neo4j")

    def import_from_file(self, merged_path: Path, clear: bool = False) -> dict:
        """Import merged triples JSON file into Neo4j."""
        data = json.loads(Path(merged_path).read_text(encoding="utf-8"))

        if clear:
            self.clear_database()

        entities_created = self._create_entities(data.get("entities", []))
        relations_created = self._create_relations(data.get("triples", []))

        logger.info("Imported %d entities, %d relations", entities_created, relations_created)
        return {"entities_created": entities_created, "relations_created": relations_created}

    def _create_entities(self, entities: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for entity in entities:
                label = ENTITY_TYPE_LABELS.get(entity["type"])
                if not label:
                    continue
                session.run(
                    f"MERGE (n:{label} {{name: $name}}) "
                    "SET n.description = $description, n.frequency = $frequency",
                    name=entity["name"],
                    description=entity.get("description", ""),
                    frequency=entity.get("frequency", 1),
                )
                count += 1
        return count

    def _create_relations(self, triples: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for triple in triples:
                head_label = ENTITY_TYPE_LABELS.get(triple.get("head_type", ""))
                tail_label = ENTITY_TYPE_LABELS.get(triple.get("tail_type", ""))
                rel_type = RELATION_TYPE_LABELS.get(triple["relation"])
                if not head_label or not tail_label or not rel_type:
                    continue
                evidences_json = json.dumps(
                    triple.get("evidences", []), ensure_ascii=False
                )
                session.run(
                    f"MATCH (a:{head_label} {{name: $head}}) "
                    f"MATCH (b:{tail_label} {{name: $tail}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.frequency = $frequency, r.evidences = $evidences",
                    head=triple["head"],
                    tail=triple["tail"],
                    frequency=triple.get("frequency", 1),
                    evidences=evidences_json,
                )
                count += 1
        return count

    def get_stats(self) -> dict:
        """Return node and relationship counts from Neo4j."""
        with self._driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}
```

- [ ] **Step 2: 写测试 `tests/test_kg_neo4j.py`**

Neo4j 依赖外部服务，测试仅覆盖可离线验证的部分：

```python
import json
import os
import pytest
from unittest.mock import patch

from qmrkg.kg_neo4j import _read_neo4j_env, DEFAULT_URI, DEFAULT_USER


def test_read_neo4j_env_defaults():
    with patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"}, clear=False):
        uri, user, password = _read_neo4j_env()
        assert uri == DEFAULT_URI
        assert user == DEFAULT_USER
        assert password == "test123"


def test_read_neo4j_env_missing_password():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            _read_neo4j_env()


def test_read_neo4j_env_custom():
    env = {
        "NEO4J_URI": "bolt://custom:7687",
        "NEO4J_USER": "admin",
        "NEO4J_PASSWORD": "secret",
    }
    with patch.dict(os.environ, env, clear=True):
        uri, user, password = _read_neo4j_env()
        assert uri == "bolt://custom:7687"
        assert user == "admin"
        assert password == "secret"
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_kg_neo4j.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/kg_neo4j.py tests/test_kg_neo4j.py
git commit -m "feat: add kg_neo4j for importing triples into Neo4j"
```

---

### Task 5: CLI 入口

**Files:**
- Create: `src/qmrkg/cli_kg_extract.py`
- Create: `src/qmrkg/cli_kg_merge.py`
- Create: `src/qmrkg/cli_kg_neo4j.py`

- [ ] **Step 1: 创建 `cli_kg_extract.py`**

```python
"""CLI for knowledge graph extraction from chunks."""

import argparse
import logging
import sys
from pathlib import Path

from .kg_extractor import KGExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract KG triples from markdown chunks")
    parser.add_argument(
        "--input", type=Path, default=Path("data/chunks"),
        help="Input chunks directory or single JSON file (default: data/chunks)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/triples/raw"),
        help="Output directory for raw triples (default: data/triples/raw)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip chunks that already have output files (default: True)",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-extract all chunks even if output exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    skip = not args.no_skip
    extractor = KGExtractor()

    input_path = args.input
    if input_path.is_file():
        paths = extractor.extract_from_chunks_file(input_path, args.output_dir, skip_existing=skip)
        print(f"Extracted {len(paths)} chunk(s) from {input_path.name}")
    elif input_path.is_dir():
        chunk_files = sorted(input_path.glob("*.json"))
        if not chunk_files:
            print(f"No JSON files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        total = 0
        for cf in chunk_files:
            paths = extractor.extract_from_chunks_file(cf, args.output_dir, skip_existing=skip)
            total += len(paths)
        print(f"Extracted {total} chunk(s) from {len(chunk_files)} file(s)")
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建 `cli_kg_merge.py`**

```python
"""CLI for merging raw triples into a deduplicated knowledge graph."""

import argparse
import logging
from pathlib import Path

from .kg_merger import KGMerger


def main():
    parser = argparse.ArgumentParser(description="Merge raw KG triples")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/triples/raw"),
        help="Directory containing raw triple JSON files (default: data/triples/raw)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/triples/merged/merged_triples.json"),
        help="Output path for merged triples (default: data/triples/merged/merged_triples.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    merger = KGMerger()
    output = merger.merge_directory(args.input_dir, args.output)
    print(f"Merged triples saved to: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 创建 `cli_kg_neo4j.py`**

```python
"""CLI for importing merged triples into Neo4j."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .kg_neo4j import KGNeo4jLoader


def main():
    parser = argparse.ArgumentParser(description="Import KG triples into Neo4j")
    parser.add_argument(
        "--import", dest="import_file", type=Path,
        help="Path to merged triples JSON file to import",
    )
    parser.add_argument("--uri", type=str, help="Neo4j URI (default: env NEO4J_URI)")
    parser.add_argument("--user", type=str, help="Neo4j user (default: env NEO4J_USER)")
    parser.add_argument("--password", type=str, help="Neo4j password (default: env NEO4J_PASSWORD)")
    parser.add_argument("--clear", action="store_true", help="Clear database before import")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.import_file and not args.stats:
        parser.print_help()
        sys.exit(1)

    with KGNeo4jLoader(uri=args.uri, user=args.user, password=args.password) as loader:
        if args.import_file:
            if not args.import_file.exists():
                print(f"File not found: {args.import_file}", file=sys.stderr)
                sys.exit(1)
            result = loader.import_from_file(args.import_file, clear=args.clear)
            print(f"Imported: {result['entities_created']} entities, {result['relations_created']} relations")

        if args.stats:
            stats = loader.get_stats()
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add src/qmrkg/cli_kg_extract.py src/qmrkg/cli_kg_merge.py src/qmrkg/cli_kg_neo4j.py
git commit -m "feat: add CLI entry points for kg extract, merge, and neo4j import"
```

---

### Task 6: 项目配置更新

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.yaml`
- Modify: `src/qmrkg/__init__.py`

- [ ] **Step 1: 更新 `pyproject.toml`**

在 `dependencies` 中添加 `"neo4j>=5.0.0"`。

在 `[project.scripts]` 中添加：
```toml
kgextract = "qmrkg.cli_kg_extract:main"
kgmerge = "qmrkg.cli_kg_merge:main"
kgneo4j = "qmrkg.cli_kg_neo4j:main"
```

- [ ] **Step 2: 更新 `config.yaml`**

将现有的 `ner` 和 `re` 段替换为统一的 `extract` 段：

```yaml
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3-0324"
    modality: "text"
    supports_thinking: false
  prompts:
    default: |
      你是一个计算机网络课程知识图谱构建专家。
      （完整 prompt 同 kg_extractor.py 中的 EXTRACT_PROMPT）
  request:
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false
  rate_limit:
    rpm: 30
    max_concurrency: 4
```

- [ ] **Step 3: 更新 `__init__.py`**

添加新模块的导出。

- [ ] **Step 4: 安装依赖并验证**

Run: `uv pip install -e .`
Expected: 安装成功，`kgextract --help`、`kgmerge --help`、`kgneo4j --help` 正常输出

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.yaml src/qmrkg/__init__.py
git commit -m "feat: update project config with extract task, neo4j dep, and new CLI entries"
```

---

### Task 7: 端到端集成验证

- [ ] **Step 1: 确保 chunks 数据存在**

如果 `data/chunks/` 为空，先运行 chunking：
```bash
python main.py --chunk-all
```

- [ ] **Step 2: 运行抽取（先对单个 chunk 文件测试）**

```bash
kgextract --input data/chunks/<first_file>.json --output-dir data/triples/raw -v
```

检查 `data/triples/raw/` 下生成的 JSON 文件内容是否合理。

- [ ] **Step 3: 运行融合**

```bash
kgmerge --input-dir data/triples/raw --output data/triples/merged/merged_triples.json -v
```

检查输出的 stats 统计是否合理。

- [ ] **Step 4: 导入 Neo4j（需要先启动 Neo4j 服务）**

```bash
kgneo4j --import data/triples/merged/merged_triples.json --clear -v
kgneo4j --stats
```

- [ ] **Step 5: 在 Neo4j Browser 中验证**

打开 `http://localhost:7474`，运行：
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: complete KG extraction pipeline integration"
```
