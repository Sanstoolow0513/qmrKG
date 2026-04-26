"""Merge, normalize and deduplicate extracted triples."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .kg_schema import Entity, Triple
from .llm_factory import EmbeddingTaskProcessor

logger = logging.getLogger(__name__)

SUFFIX_PATTERN = re.compile(r"(协议|算法|机制|方法|技术|方式)$")

ALIAS_MAP: dict[str, str] = {
    "传输控制协议": "TCP",
    "用户数据报协议": "UDP",
    "超文本传输协议": "HTTP",
    "超文本传送协议": "HTTP",
    "域名系统": "DNS",
    "文件传输协议": "FTP",
    "简单邮件传输协议": "SMTP",
    "简单邮件传送协议": "SMTP",
    "地址解析协议": "ARP",
    "逆地址解析协议": "RARP",
    "网际控制报文协议": "ICMP",
    "网际协议": "IP",
    "往返时延": "RTT",
    "往返时间": "RTT",
    "最大传输单元": "MTU",
    "最大报文段长度": "MSS",
    "服务质量": "QoS",
    "开放系统互连": "OSI",
    "网络地址转换": "NAT",
    "动态主机配置协议": "DHCP",
    "简单网络管理协议": "SNMP",
    "资源预留协议": "RSVP",
    "多协议标签交换": "MPLS",
    "边界网关协议": "BGP",
    "内部网关协议": "IGP",
    "开放最短路径优先": "OSPF",
    "路由信息协议": "RIP",
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


class EmbeddingCanonicalizer:
    """Group similar entity names in buckets using embedding cosine similarity."""

    def __init__(
        self,
        task_name: str,
        encode_fields: list[str],
        similarity_threshold: float,
        bucket_by_type: bool,
        batch_size: int,
        cache_path: Path | None,
        config_path: Path | None,
        processor: Any = None,
    ):
        self.task_name = task_name
        self.encode_fields = encode_fields
        self.similarity_threshold = similarity_threshold
        self.bucket_by_type = bucket_by_type
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.processor = (
            processor
            if processor is not None
            else EmbeddingTaskProcessor(task_name, config_path=config_path)
        )

    def _model_id(self) -> str:
        settings = getattr(self.processor, "settings", None)
        model = getattr(settings, "model", "") if settings is not None else ""
        return model or self.task_name

    def _encode_row(self, name: str, type_: str, description: str) -> str | None:
        if not (name and name.strip()):
            return None
        parts: list[str] = []
        for field in self.encode_fields:
            if field == "type":
                v = (type_ or "").strip()
            elif field == "name":
                v = (name or "").strip()
            elif field == "description":
                v = (description or "").strip()
            else:
                continue
            if not v:
                continue
            parts.append(v)
        return " | ".join(parts)

    def _embed_texts_resolved(
        self, texts: list[str], cache: dict[str, list[float]]
    ) -> list[list[float]]:
        """Return embedding vectors, same order as `texts`."""
        model_id = self._model_id()
        out: list[list[float]] = []
        pending_idx: list[int] = []
        pending_text: list[str] = []
        for i, text in enumerate(texts):
            key = f"{model_id}::{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
            if key in cache:
                out.append([float(x) for x in cache[key]])
            else:
                out.append([])  # placeholder
                pending_idx.append(i)
                pending_text.append(text)
        if pending_text:
            new_vecs = self.processor.embed(pending_text, batch_size=self.batch_size)
            for j, i in enumerate(pending_idx):
                text = pending_text[j]
                vec = new_vecs[j]
                key = f"{model_id}::{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
                cache[key] = [float(x) for x in vec]
                out[i] = [float(x) for x in vec]
        return out

    def build_canonical_map(self, entities: list[Entity]) -> dict[str, str]:
        # Dedupe (name, type), sum frequency, first non-empty description
        agg_freq: dict[tuple[str, str], int] = defaultdict(int)
        agg_desc: dict[tuple[str, str], str] = {}
        for e in entities:
            if not e.is_valid():
                continue
            name = e.name.strip()
            if not name:
                continue
            key = (name, e.type)
            agg_freq[key] += e.frequency
            if key not in agg_desc and (e.description and e.description.strip()):
                agg_desc[key] = e.description

        rows: list[tuple[str, str, int, str]] = []
        for (name, type_), freq in sorted(agg_freq.items()):
            rows.append((name, type_, freq, agg_desc.get((name, type_), "")))

        if not rows:
            return {}

        n = len(rows)
        mapping: dict[str, str] = {r[0]: r[0] for r in rows}
        cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cache = {k: v for k, v in raw.items() if isinstance(v, list)}

        uf = _UnionFind(n)

        buckets: dict[str, list[int]] = defaultdict(list)
        for i, (name, type_, _freq, _desc) in enumerate(rows):
            bkey = type_ if self.bucket_by_type else "_all"
            buckets[bkey].append(i)

        for _bkey, indices in buckets.items():
            if len(indices) < 2:
                continue
            texts: list[str] = []
            valid: list[int] = []
            for i in indices:
                name, type_, _f, desc = rows[i]
                text = self._encode_row(name, type_, desc)
                if text is None:
                    continue
                texts.append(text)
                valid.append(i)
            if len(texts) < 2:
                continue
            vecs = self._embed_texts_resolved(texts, cache)
            m = len(vecs)
            mat = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, np.float32(1e-12))
            sim = mat @ mat.T
            for a in range(m):
                for b in range(a + 1, m):
                    if float(sim[a, b]) >= self.similarity_threshold:
                        uf.union(valid[a], valid[b])

        members_by_root: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            members_by_root[uf.find(i)].append(i)

        for _root, mems in members_by_root.items():
            candidates = [rows[i] for i in mems]
            candidates.sort(key=lambda r: (-r[2], len(r[0]), r[0]))
            canonical = candidates[0][0]
            for i in mems:
                mapping[rows[i][0]] = canonical

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return mapping


class KGMerger:
    """Merge raw extraction results into a deduplicated knowledge graph."""

    def __init__(self, alias_map: dict[str, str] | None = None):
        if alias_map:
            ALIAS_MAP.update(alias_map)

    def merge_directory(
        self,
        raw_dir: Path,
        output_path: Path,
        embedding_config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> Path:
        """Load all raw triple files from a directory and merge them."""
        raw_dir = Path(raw_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(raw_dir.glob("*.json"))
        if not raw_files:
            logger.warning("No raw triple files found in %s", raw_dir)

        all_entities: list[Entity] = []
        all_triples: list[Triple] = []

        for f in tqdm(
            raw_files,
            desc="kgmerge",
            unit="file",
            total=len(raw_files),
            dynamic_ncols=True,
        ):
            data = json.loads(f.read_text(encoding="utf-8"))
            for e in data.get("entities", []):
                all_entities.append(
                    Entity(
                        name=e["name"],
                        type=e["type"],
                        description=e.get("description", ""),
                    )
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
        if embedding_config and embedding_config.get("enabled"):
            task_name = str(embedding_config.get("task_name", "entity_embed"))
            encode_fields = list(
                embedding_config.get("encode_fields", ["type", "name", "description"])
            )
            threshold = float(embedding_config.get("similarity_threshold", 0.85))
            bucket_by_type = bool(embedding_config.get("bucket_by_type", True))
            batch_size = int(embedding_config.get("batch_size", 1024))
            emb_cache = embedding_config.get("cache_path")
            cache_path = Path(emb_cache) if emb_cache is not None else None
            canonicalizer = EmbeddingCanonicalizer(
                task_name=task_name,
                encode_fields=encode_fields,
                similarity_threshold=threshold,
                bucket_by_type=bucket_by_type,
                batch_size=batch_size,
                cache_path=cache_path,
                config_path=config_path,
            )
            canonical_map = canonicalizer.build_canonical_map(merged_entities)
            merged_entities = self._merge_entities(
                self._apply_entity_mapping(merged_entities, canonical_map)
            )
            all_triples = self._apply_triple_mapping(all_triples, canonical_map)
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
            len(merged_entities),
            len(merged_triples),
            output_path,
        )
        return output_path

    @staticmethod
    def _apply_entity_mapping(entities: list[Entity], mapping: dict[str, str]) -> list[Entity]:
        out: list[Entity] = []
        for e in entities:
            new_name = mapping.get(e.name, e.name)
            out.append(
                Entity(
                    name=new_name,
                    type=e.type,
                    description=e.description,
                    frequency=e.frequency,
                )
            )
        return out

    @staticmethod
    def _apply_triple_mapping(triples: list[Triple], mapping: dict[str, str]) -> list[Triple]:
        out: list[Triple] = []
        for t in triples:
            head = normalize_entity_name(t.head)
            tail = normalize_entity_name(t.tail)
            head = mapping.get(head, head)
            tail = mapping.get(tail, tail)
            out.append(
                Triple(
                    head=head,
                    relation=t.relation,
                    tail=tail,
                    evidence=t.evidence,
                    evidence_span=t.evidence_span,
                    frequency=t.frequency,
                    evidences=list(t.evidences),
                    review_decision=t.review_decision,
                    review_reason_code=t.review_reason_code,
                    review_reason=t.review_reason,
                )
            )
        return out

    def _merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """Normalize names, deduplicate, accumulate frequency."""
        grouped: dict[str, Entity] = {}
        for e in entities:
            if not e.is_valid():
                continue
            canonical = normalize_entity_name(e.name)
            if len(canonical) < 2:
                continue
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

    def _merge_triples(self, triples: list[Triple], valid_entity_names: set[str]) -> list[Triple]:
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
    def _compute_stats(entities: list[Entity], triples: list[Triple]) -> dict:
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
