"""Merge, normalize and deduplicate extracted triples."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .kg_schema import Entity, Triple
from .llm_factory import EmbeddingTaskProcessor, TextTaskProcessor

logger = logging.getLogger(__name__)
try:  # pragma: no cover - import behavior covered indirectly
    import faiss
except ImportError:  # pragma: no cover - validated at runtime
    faiss = None

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

_PUNCT_OR_SPACE_PATTERN = re.compile(r"[\s\-_/\\.,，。:：;；()（）\[\]【】{}<>《》\"'`]+")


@dataclass(slots=True)
class EntityMergeCandidate:
    """Embedding-retrieved candidate pair for entity canonicalization."""

    left: str
    right: str
    left_type: str
    right_type: str
    left_frequency: int
    right_frequency: int
    score: float


@dataclass(slots=True)
class EntityMergeDecision:
    """Auditable decision for one candidate pair."""

    left: str
    right: str
    decision: str
    method: str
    reason_code: str
    canonical_name: str | None = None
    reason: str = ""
    embedding_score: float | None = None
    supporting_evidence_ids: list[str] = field(default_factory=list)
    conflict_evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _EntityContext:
    name: str
    type: str = ""
    frequency: int = 0
    descriptions: list[str] = field(default_factory=list)
    triples: list[dict[str, str]] = field(default_factory=list)


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


def _compact_entity_name(name: str) -> str:
    """Normalize superficial punctuation/casing for strong exact-name checks."""
    return _PUNCT_OR_SPACE_PATTERN.sub("", normalize_entity_name(name)).lower()


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


class _EmbeddingBinaryCache:
    """Binary embedding cache backed by metadata JSON and NumPy matrix."""

    VERSION = 1

    def __init__(self, base_path: Path, signature: str):
        self.base_path = base_path
        self.signature = signature
        self.meta_path = base_path.with_name(f"{base_path.name}.meta.json")
        self.vec_path = base_path.with_name(f"{base_path.name}.npy")
        self.key_to_idx: dict[str, int] = {}
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.dim: int | None = None

    @staticmethod
    def _coerce_base_path(cache_path: Path) -> Path:
        return cache_path.with_suffix("") if cache_path.suffix else cache_path

    @classmethod
    def from_cache_path(cls, cache_path: Path, signature: str) -> "_EmbeddingBinaryCache":
        return cls(cls._coerce_base_path(cache_path), signature)

    def load(self) -> None:
        if not self.meta_path.exists() or not self.vec_path.exists():
            return
        try:
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                logger.warning("Invalid binary cache meta at %s, ignoring", self.meta_path)
                return
            if raw.get("version") != self.VERSION or raw.get("signature") != self.signature:
                logger.warning("Embedding cache signature/version mismatch, ignoring old cache")
                return
            key_to_idx = raw.get("key_to_idx")
            if not isinstance(key_to_idx, dict):
                logger.warning("Invalid key_to_idx in cache meta at %s", self.meta_path)
                return
            vectors = np.load(self.vec_path, allow_pickle=False)
            if vectors.ndim != 2:
                logger.warning("Invalid vector shape in cache at %s", self.vec_path)
                return
            self.vectors = np.asarray(vectors, dtype=np.float32)
            self.dim = int(self.vectors.shape[1]) if self.vectors.shape[0] > 0 else None
            cleaned: dict[str, int] = {}
            for key, idx in key_to_idx.items():
                if not isinstance(key, str) or not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= self.vectors.shape[0]:
                    continue
                cleaned[key] = idx
            self.key_to_idx = cleaned
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load embedding cache, starting empty: %s", exc)
            self.key_to_idx = {}
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self.dim = None

    def get(self, key: str) -> list[float] | None:
        idx = self.key_to_idx.get(key)
        if idx is None:
            return None
        return self.vectors[idx].astype(np.float32).tolist()

    def put(self, key: str, vec: list[float]) -> None:
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] == 0:
            return
        if self.dim is None:
            self.dim = int(arr.shape[1])
            self.vectors = arr.copy()
            self.key_to_idx[key] = 0
            return
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dim}, got {arr.shape[1]}"
            )
        next_idx = int(self.vectors.shape[0])
        self.vectors = np.vstack([self.vectors, arr])
        self.key_to_idx[key] = next_idx

    def save(self) -> None:
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.vectors.astype(np.float32))
        payload = {
            "version": self.VERSION,
            "signature": self.signature,
            "dim": self.dim,
            "key_to_idx": self.key_to_idx,
        }
        self.meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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
        cache_format: str,
        encoding_template: str,
        max_desc_chars: int,
        faiss_top_k: int,
        config_path: Path | None,
        processor: Any = None,
    ):
        self.task_name = task_name
        self.encode_fields = encode_fields
        self.similarity_threshold = similarity_threshold
        self.bucket_by_type = bucket_by_type
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache_format = cache_format
        self.encoding_template = encoding_template
        self.max_desc_chars = max_desc_chars
        self.faiss_top_k = faiss_top_k
        self.processor = (
            processor
            if processor is not None
            else EmbeddingTaskProcessor(task_name, config_path=config_path)
        )

    def _model_id(self) -> str:
        settings = getattr(self.processor, "settings", None)
        model = getattr(settings, "model", "") if settings is not None else ""
        return model or self.task_name

    def _embedding_signature(self) -> str:
        settings = getattr(self.processor, "settings", None)
        model = self._model_id()
        encoding_format = (
            getattr(settings, "encoding_format", None) if settings is not None else None
        ) or "float"
        dimensions = (
            getattr(settings, "embedding_dimensions", None) if settings is not None else None
        )
        return f"{model}::{encoding_format}::{dimensions or 'native'}"

    def _cache_key(self, text: str) -> str:
        signature = self._embedding_signature()
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{signature}::{digest}"

    def _encode_row(self, name: str, type_: str, description: str) -> str | None:
        if not (name and name.strip()):
            return None
        values: dict[str, str] = {
            "type": (type_ or "").strip(),
            "name": (name or "").strip(),
            "description": (description or "").strip(),
        }
        desc = values["description"]
        if desc and self.max_desc_chars > 0:
            values["description"] = desc[: self.max_desc_chars]
        if self.encoding_template != "structured_zh":
            parts = [values[f] for f in self.encode_fields if values.get(f)]
            return " | ".join(parts)
        labels = {"type": "实体类型", "name": "实体名", "description": "定义"}
        chunks: list[str] = []
        for field_name in self.encode_fields:
            value = values.get(field_name, "")
            if not value:
                continue
            label = labels.get(field_name, field_name)
            chunks.append(f"{label}：{value}")
        return "；".join(chunks) if chunks else None

    def _embed_texts_resolved(
        self,
        texts: list[str],
        cache_json: dict[str, list[float]],
        cache_bin: _EmbeddingBinaryCache | None,
    ) -> list[list[float]]:
        """Return embedding vectors, same order as `texts`."""
        out: list[list[float]] = []
        pending_idx: list[int] = []
        pending_text: list[str] = []
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if cache_bin is not None:
                vec = cache_bin.get(key)
                if vec is not None:
                    out.append(vec)
                    continue
            if key in cache_json:
                out.append([float(x) for x in cache_json[key]])
            else:
                out.append([])  # placeholder
                pending_idx.append(i)
                pending_text.append(text)
        if pending_text:
            new_vecs: list[list[float]] = []
            batches = [
                pending_text[start : start + self.batch_size]
                for start in range(0, len(pending_text), self.batch_size)
            ]
            batch_results: list[list[list[float]] | None] = [None] * len(batches)
            max_workers = self._resolve_max_workers(len(batches))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.processor.embed, batch, batch_size=self.batch_size): i
                    for i, batch in enumerate(batches)
                }
                completed = as_completed(futures)
                for future in tqdm(
                    completed,
                    desc="kgmerge embed",
                    unit="batch",
                    total=len(futures),
                    dynamic_ncols=True,
                ):
                    batch_index = futures[future]
                    vectors = future.result()
                    if len(vectors) != len(batches[batch_index]):
                        raise ValueError(
                            "embedding response length mismatch: expected "
                            f"{len(batches[batch_index])}, got {len(vectors)}"
                        )
                    batch_results[batch_index] = vectors
            for vectors in batch_results:
                if vectors is None:
                    raise RuntimeError("embedding batch did not complete")
                new_vecs.extend(vectors)
            for j, i in enumerate(pending_idx):
                text = pending_text[j]
                vec = new_vecs[j]
                key = self._cache_key(text)
                if cache_bin is not None:
                    cache_bin.put(key, vec)
                else:
                    cache_json[key] = [float(x) for x in vec]
                out[i] = [float(x) for x in vec]
        return out

    def _resolve_max_workers(self, pending_batch_count: int) -> int:
        configured = getattr(getattr(self.processor, "settings", None), "max_concurrency", 1)
        if not isinstance(configured, int) or configured <= 0:
            configured = 1
        return max(1, min(pending_batch_count, configured))

    @staticmethod
    def _build_rows(entities: list[Entity]) -> list[tuple[str, str, int, str]]:
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
        return rows

    def build_candidates(self, entities: list[Entity]) -> list[EntityMergeCandidate]:
        """Return embedding-similar entity pairs without deciding whether to merge."""
        rows = self._build_rows(entities)
        return self._find_candidates(rows)

    def build_canonical_map(self, entities: list[Entity]) -> dict[str, str]:
        """Legacy embedding-only canonicalization used when LLM recheck is disabled."""
        rows = self._build_rows(entities)
        if not rows:
            return {}

        n = len(rows)
        mapping: dict[str, str] = {r[0]: r[0] for r in rows}
        index_by_name: dict[str, int] = {
            name: i for i, (name, _type, _freq, _desc) in enumerate(rows)
        }
        uf = _UnionFind(n)

        for candidate in self._find_candidates(rows):
            left_idx = index_by_name.get(candidate.left)
            right_idx = index_by_name.get(candidate.right)
            if left_idx is not None and right_idx is not None:
                uf.union(left_idx, right_idx)

        members_by_root: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            members_by_root[uf.find(i)].append(i)

        for _root, mems in members_by_root.items():
            candidates = [rows[i] for i in mems]
            candidates.sort(key=lambda r: (-r[2], len(r[0]), r[0]))
            canonical = candidates[0][0]
            for i in mems:
                mapping[rows[i][0]] = canonical

        return mapping

    def _find_candidates(self, rows: list[tuple[str, str, int, str]]) -> list[EntityMergeCandidate]:
        if not rows:
            return []

        cache_json: dict[str, list[float]] = {}
        cache_bin: _EmbeddingBinaryCache | None = None
        if self.cache_path:
            if self.cache_format == "binary":
                cache_bin = _EmbeddingBinaryCache.from_cache_path(
                    self.cache_path, self._embedding_signature()
                )
                cache_bin.load()
            elif self.cache_path.exists():
                try:
                    raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Invalid embedding cache at %s, starting empty: %s",
                        self.cache_path,
                        exc,
                    )
                    raw = None
                if isinstance(raw, dict):
                    cache_json = {k: v for k, v in raw.items() if isinstance(v, list)}
                elif raw is not None:
                    logger.warning(
                        "Invalid embedding cache at %s (not a JSON object), starting empty",
                        self.cache_path,
                    )

        buckets: dict[str, list[int]] = defaultdict(list)
        for i, (name, type_, _freq, _desc) in enumerate(rows):
            bkey = type_ if self.bucket_by_type else "_all"
            buckets[bkey].append(i)

        candidates: list[EntityMergeCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()
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
            vecs = self._embed_texts_resolved(texts, cache_json, cache_bin)
            m = len(vecs)
            mat = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, np.float32(1e-12))
            if faiss is None:
                raise ImportError(
                    "faiss is required for embedding canonicalization. Install faiss-cpu."
                )
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            top_k = max(2, min(self.faiss_top_k, m))
            scores, neighbors = index.search(mat, top_k)
            for a in range(m):
                for rank in range(1, top_k):
                    b = int(neighbors[a, rank])
                    if b < 0 or b <= a:
                        continue
                    if float(scores[a, rank]) >= self.similarity_threshold:
                        left = rows[valid[a]]
                        right = rows[valid[b]]
                        pair_key = tuple(sorted((left[0], right[0])))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        candidates.append(
                            EntityMergeCandidate(
                                left=left[0],
                                right=right[0],
                                left_type=left[1],
                                right_type=right[1],
                                left_frequency=left[2],
                                right_frequency=right[2],
                                score=float(scores[a, rank]),
                            )
                        )

        if self.cache_path is not None:
            if self.cache_format == "binary" and cache_bin is not None:
                cache_bin.save()
            else:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(cache_json, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        candidates.sort(key=lambda c: (-c.score, c.left, c.right))
        return candidates


class EntityMergeJudge:
    """LLM rechecker for ambiguous embedding candidates.

    The judge output is treated as an untrusted classification and is validated by
    KGMerger before any merge decision is applied.
    """

    VALID_DECISIONS = {"merge", "keep_separate", "unsure"}

    def __init__(
        self,
        task_name: str,
        config_path: Path | None,
        processor: Any = None,
        require_supporting_evidence: bool = True,
    ):
        self.task_name = task_name
        self.processor = (
            processor
            if processor is not None
            else TextTaskProcessor(task_name, config_path=config_path)
        )
        self.require_supporting_evidence = require_supporting_evidence

    def judge(
        self,
        candidate: EntityMergeCandidate,
        left_context: dict[str, Any],
        right_context: dict[str, Any],
    ) -> EntityMergeDecision:
        prompt, allowed_evidence_ids = self._build_prompt(candidate, left_context, right_context)
        try:
            response = self.processor.run_text(prompt)
        except Exception as exc:
            logger.warning(
                "Entity merge LLM recheck failed for %s/%s: %s",
                candidate.left,
                candidate.right,
                exc,
            )
            return EntityMergeDecision(
                left=candidate.left,
                right=candidate.right,
                decision="unsure",
                method="llm_recheck",
                reason_code="LLM_REQUEST_FAILED",
                reason=str(exc),
                embedding_score=candidate.score,
            )

        raw = self._parse_json_response(response.text)
        if not isinstance(raw, dict):
            return EntityMergeDecision(
                left=candidate.left,
                right=candidate.right,
                decision="unsure",
                method="llm_recheck",
                reason_code="LLM_INVALID_JSON",
                embedding_score=candidate.score,
            )

        decision = str(raw.get("decision", "")).strip().lower()
        if decision not in self.VALID_DECISIONS:
            decision = "unsure"

        allowed_names = {candidate.left, candidate.right}
        canonical_raw = raw.get("canonical_name")
        canonical_name = str(canonical_raw).strip() if canonical_raw is not None else None
        if decision == "merge" and canonical_name not in allowed_names:
            decision = "unsure"
            canonical_name = None

        supporting_ids = self._clean_evidence_ids(raw.get("supporting_evidence_ids"))
        conflict_ids = self._clean_evidence_ids(raw.get("conflict_evidence_ids"))
        if not set(supporting_ids).issubset(allowed_evidence_ids):
            decision = "unsure"
            supporting_ids = [eid for eid in supporting_ids if eid in allowed_evidence_ids]
        if not set(conflict_ids).issubset(allowed_evidence_ids):
            conflict_ids = [eid for eid in conflict_ids if eid in allowed_evidence_ids]
        if decision == "merge" and conflict_ids:
            decision = "unsure"
            canonical_name = None
        if decision == "merge" and self.require_supporting_evidence and not supporting_ids:
            decision = "unsure"
            canonical_name = None

        return EntityMergeDecision(
            left=candidate.left,
            right=candidate.right,
            decision=decision,
            method="llm_recheck",
            reason_code=str(raw.get("reason_code", "LLM_RECHECK")).strip() or "LLM_RECHECK",
            canonical_name=canonical_name if decision == "merge" else None,
            reason=str(raw.get("reason", "")).strip(),
            embedding_score=candidate.score,
            supporting_evidence_ids=supporting_ids,
            conflict_evidence_ids=conflict_ids,
        )

    def _build_prompt(
        self,
        candidate: EntityMergeCandidate,
        left_context: dict[str, Any],
        right_context: dict[str, Any],
    ) -> tuple[str, set[str]]:
        payload = {
            "task": "判断两个计算机网络知识图谱实体 term 是否指向同一个实体。",
            "decision_rules": [
                "只能根据给定 evidence 判断，不要使用自报置信度。",
                "merge 表示两个 term 在所有三元组中可互换且语义不变。",
                "keep_separate 表示二者相关但不是同一实体，或存在对比/包含/依赖等反证。",
                "unsure 表示证据不足，不能安全合并。",
                "canonical_name 只能从 allowed_canonical_names 中选择。",
            ],
            "allowed_decisions": ["merge", "keep_separate", "unsure"],
            "allowed_canonical_names": [candidate.left, candidate.right],
            "left": left_context,
            "right": right_context,
            "output_schema": {
                "decision": "merge|keep_separate|unsure",
                "canonical_name": "string or null",
                "reason_code": "short_code",
                "reason": "简短中文原因",
                "supporting_evidence_ids": ["evidence id list"],
                "conflict_evidence_ids": ["evidence id list"],
            },
        }
        allowed_ids = set()
        for side in (left_context, right_context):
            for desc in side.get("descriptions", []):
                if isinstance(desc, dict) and isinstance(desc.get("id"), str):
                    allowed_ids.add(desc["id"])
            for triple in side.get("triples", []):
                if isinstance(triple, dict) and isinstance(triple.get("id"), str):
                    allowed_ids.add(triple["id"])
        return json.dumps(payload, ensure_ascii=False, indent=2), allowed_ids

    @staticmethod
    def _clean_evidence_ids(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any] | None:
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity merge LLM JSON response")
            return None
        return raw if isinstance(raw, dict) else None


class KGMerger:
    """Merge raw extraction results into a deduplicated knowledge graph."""

    def __init__(self, alias_map: dict[str, str] | None = None, merge_judge_processor: Any = None):
        if alias_map:
            ALIAS_MAP.update(alias_map)
        self._merge_judge_processor = merge_judge_processor

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
        entity_contexts: dict[str, _EntityContext] = {}

        for f in tqdm(
            raw_files,
            desc="kgmerge load",
            unit="file",
            total=len(raw_files),
            dynamic_ncols=True,
        ):
            data = json.loads(f.read_text(encoding="utf-8"))
            for e in data.get("entities", []):
                entity = Entity(
                    name=e["name"],
                    type=e["type"],
                    description=e.get("description", ""),
                )
                all_entities.append(entity)
                self._record_entity_context(entity_contexts, entity)
            for idx, t in enumerate(data.get("triples", []), start=1):
                triple = Triple(
                    head=t["head"],
                    relation=t["relation"],
                    tail=t["tail"],
                    evidence=t.get("evidence", ""),
                )
                all_triples.append(triple)
                self._record_triple_context(
                    entity_contexts,
                    triple,
                    evidence_id=f"{f.stem}:T{idx}",
                    source_file=str(data.get("source_file") or f.name),
                    chunk_index=str(data.get("chunk_index", "")),
                )

        entity_type_conflicts = self._entity_type_conflicts(all_entities)
        merged_entities = self._merge_entities(all_entities)
        merge_audit: dict[str, Any] | None = None
        if embedding_config and embedding_config.get("enabled"):
            task_name = str(embedding_config.get("task_name", "entity_embed"))
            encode_fields = list(
                embedding_config.get("encode_fields", ["type", "name", "description"])
            )
            recheck_config = dict(embedding_config.get("llm_recheck") or {})
            recheck_enabled = bool(recheck_config.get("enabled", False))
            threshold = float(
                embedding_config.get(
                    "candidate_threshold" if recheck_enabled else "similarity_threshold",
                    embedding_config.get("similarity_threshold", 0.85),
                )
            )
            bucket_by_type = bool(embedding_config.get("bucket_by_type", True))
            batch_size = int(embedding_config.get("batch_size", 1024))
            emb_cache = embedding_config.get("cache_path")
            cache_path = Path(emb_cache) if emb_cache is not None else None
            cache_format = str(embedding_config.get("cache_format", "binary"))
            encoding_template = str(embedding_config.get("encoding_template", "structured_zh"))
            max_desc_chars = int(embedding_config.get("max_desc_chars", 160))
            faiss_top_k = int(embedding_config.get("faiss_top_k", 50))
            canonicalizer = EmbeddingCanonicalizer(
                task_name=task_name,
                encode_fields=encode_fields,
                similarity_threshold=threshold,
                bucket_by_type=bucket_by_type,
                batch_size=batch_size,
                cache_path=cache_path,
                cache_format=cache_format,
                encoding_template=encoding_template,
                max_desc_chars=max_desc_chars,
                faiss_top_k=faiss_top_k,
                config_path=config_path,
            )
            if recheck_enabled:
                canonical_map, merge_audit = self._build_rechecked_canonical_map(
                    merged_entities,
                    all_triples,
                    entity_contexts,
                    canonicalizer,
                    recheck_config,
                    config_path=config_path,
                )
            elif bool(embedding_config.get("direct_merge_without_recheck", False)):
                logger.warning(
                    "kgmerge direct embedding merge is enabled without LLM recheck; "
                    "this can merge related but non-equivalent entities."
                )
                canonical_map = canonicalizer.build_canonical_map(merged_entities)
                merge_audit = {
                    "strategy": "direct_embedding_merge_without_recheck",
                    "warning": (
                        "Unsafe compatibility mode: embedding similarity directly produced "
                        "the canonical mapping."
                    ),
                }
            else:
                canonical_map, merge_audit = self._build_rule_only_canonical_map(
                    merged_entities,
                    all_triples,
                    canonicalizer,
                )
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
        if entity_type_conflicts:
            if merge_audit is None:
                merge_audit = {"strategy": "legacy_rule_merge"}
            merge_audit["entity_type_conflicts"] = entity_type_conflicts
        if merge_audit is not None:
            result["merge_audit"] = merge_audit

        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "Merged: %d entities, %d triples -> %s",
            len(merged_entities),
            len(merged_triples),
            output_path,
        )
        return output_path

    @staticmethod
    def _entity_type_conflicts(entities: list[Entity]) -> list[dict[str, Any]]:
        types_by_name: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entity in entities:
            if not entity.is_valid():
                continue
            canonical = normalize_entity_name(entity.name)
            if not canonical:
                continue
            types_by_name[canonical][entity.type] += entity.frequency
        conflicts = []
        for name, counts in sorted(types_by_name.items()):
            if len(counts) > 1:
                conflicts.append({"name": name, "types": dict(sorted(counts.items()))})
        return conflicts

    @staticmethod
    def _record_entity_context(
        contexts: dict[str, _EntityContext],
        entity: Entity,
    ) -> None:
        canonical = normalize_entity_name(entity.name)
        if not canonical:
            return
        ctx = contexts.setdefault(canonical, _EntityContext(name=canonical))
        ctx.frequency += 1
        if not ctx.type and entity.type:
            ctx.type = entity.type
        description = (entity.description or "").strip()
        if description and description not in ctx.descriptions:
            ctx.descriptions.append(description)

    @staticmethod
    def _record_triple_context(
        contexts: dict[str, _EntityContext],
        triple: Triple,
        *,
        evidence_id: str,
        source_file: str,
        chunk_index: str,
    ) -> None:
        head = normalize_entity_name(triple.head)
        tail = normalize_entity_name(triple.tail)
        triple_payload = {
            "id": evidence_id,
            "head": head,
            "relation": triple.relation,
            "tail": tail,
            "evidence": triple.evidence,
            "source_file": source_file,
            "chunk_index": chunk_index,
        }
        for name in (head, tail):
            if not name:
                continue
            ctx = contexts.setdefault(name, _EntityContext(name=name))
            if triple_payload not in ctx.triples:
                ctx.triples.append(triple_payload)

    def _build_rechecked_canonical_map(
        self,
        entities: list[Entity],
        triples: list[Triple],
        contexts: dict[str, _EntityContext],
        canonicalizer: EmbeddingCanonicalizer,
        recheck_config: dict[str, Any],
        *,
        config_path: Path | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        all_candidates = canonicalizer.build_candidates(entities)
        max_pairs = int(recheck_config.get("max_pairs", 200))
        llm_budget = max_pairs if max_pairs > 0 else len(all_candidates)
        context_triples_limit = int(recheck_config.get("context_triples_per_entity", 8))
        require_supporting = bool(recheck_config.get("require_supporting_evidence", True))
        allow_truncated_context = bool(
            recheck_config.get("allow_merge_with_truncated_context", False)
        )
        max_cluster_size = int(recheck_config.get("max_cluster_size", 4))
        require_complete_pairwise = bool(
            recheck_config.get("require_complete_pairwise_cluster", True)
        )
        decisions: list[EntityMergeDecision] = []
        llm_jobs: list[tuple[EntityMergeCandidate, dict, dict]] = []
        llm_reviewed_count = 0
        for candidate in tqdm(
            all_candidates,
            desc="kgmerge rules",
            unit="pair",
            total=len(all_candidates),
            dynamic_ncols=True,
        ):
            rule_decision = self._rule_decision(candidate, triples, entities)
            if rule_decision.decision != "needs_llm":
                decisions.append(rule_decision)
                continue
            if llm_reviewed_count >= llm_budget:
                decisions.append(
                    EntityMergeDecision(
                        left=candidate.left,
                        right=candidate.right,
                        decision="unsure",
                        method="llm_recheck_skipped",
                        reason_code="MAX_PAIRS_EXCEEDED",
                        reason="LLM 复核预算已用完，该候选未进入合并。",
                        embedding_score=candidate.score,
                    )
                )
                continue

            left_context = self._context_for_llm(
                candidate.left,
                contexts,
                side="L",
                triples_limit=context_triples_limit,
            )
            right_context = self._context_for_llm(
                candidate.right,
                contexts,
                side="R",
                triples_limit=context_triples_limit,
            )
            llm_jobs.append((candidate, left_context, right_context))
            llm_reviewed_count += 1

        if llm_jobs:
            judge = EntityMergeJudge(
                str(recheck_config.get("task_name", "entity_merge_review")),
                config_path=config_path,
                processor=self._merge_judge_processor,
                require_supporting_evidence=require_supporting,
            )
            max_workers = self._resolve_recheck_max_workers(recheck_config, judge, len(llm_jobs))
            llm_slots: list[EntityMergeDecision | None] = [None] * len(llm_jobs)

            job_iter = iter(enumerate(llm_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                _futures: dict[Any, int] = {}

                def _submit_next_job() -> None:  # pragma: no cover - closure
                    try:
                        idx, (cand, lctx, rctx) = next(job_iter)
                    except StopIteration:
                        return
                    _futures[
                        executor.submit(
                            self._recheck_single_pair,
                            judge,
                            cand,
                            lctx,
                            rctx,
                            triples,
                            entities,
                            allow_truncated_context=allow_truncated_context,
                        )
                    ] = idx

                for _ in range(max_workers):
                    _submit_next_job()

                with tqdm(
                    total=len(llm_jobs),
                    desc="kgmerge recheck",
                    unit="pair",
                    dynamic_ncols=True,
                ) as pbar:
                    while _futures:
                        done, _ = wait(_futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            idx = _futures.pop(future)
                            try:
                                llm_slots[idx] = future.result()
                            except Exception as exc:
                                llm_slots[idx] = EntityMergeDecision(
                                    left=llm_jobs[idx][0].left,
                                    right=llm_jobs[idx][0].right,
                                    decision="unsure",
                                    method="llm_recheck",
                                    reason_code="LLM_REQUEST_FAILED",
                                    reason=str(exc),
                                    embedding_score=llm_jobs[idx][0].score,
                                )
                            finally:
                                pbar.update(1)
                            _submit_next_job()

            for decision in llm_slots:
                if decision is not None:
                    decisions.append(decision)

        canonical_map, cluster_conflicts = self._mapping_from_merge_decisions(
            entities,
            triples,
            decisions,
            max_cluster_size=max_cluster_size,
            require_complete_pairwise=require_complete_pairwise,
        )
        audit = {
            "strategy": "embedding_rules_llm_recheck",
            "candidate_count": len(all_candidates),
            "llm_reviewed_candidate_count": llm_reviewed_count,
            "llm_skipped_candidate_count": len(
                [d for d in decisions if d.reason_code == "MAX_PAIRS_EXCEEDED"]
            ),
            "decisions": [self._decision_to_dict(d) for d in decisions],
            "cluster_conflicts": cluster_conflicts,
        }
        return canonical_map, audit

    @staticmethod
    def _resolve_recheck_max_workers(
        recheck_config: dict[str, Any],
        judge: EntityMergeJudge,
        pending_count: int,
    ) -> int:
        configured = recheck_config.get("max_concurrency")
        if not (isinstance(configured, int) and configured > 0):
            settings = getattr(getattr(judge, "processor", None), "settings", None)
            configured = getattr(settings, "max_concurrency", 1) if settings is not None else 1
        if not isinstance(configured, int) or configured <= 0:
            configured = 1
        return max(1, min(pending_count, configured))

    def _recheck_single_pair(
        self,
        judge: EntityMergeJudge,
        candidate: EntityMergeCandidate,
        left_context: dict[str, Any],
        right_context: dict[str, Any],
        triples: list[Triple],
        entities: list[Entity],
        *,
        allow_truncated_context: bool,
    ) -> EntityMergeDecision:
        llm_decision = judge.judge(candidate, left_context, right_context)
        return self._validate_llm_decision(
            llm_decision,
            candidate,
            triples,
            entities,
            contexts=[left_context, right_context],
            allow_truncated_context=allow_truncated_context,
        )

    def _build_rule_only_canonical_map(
        self,
        entities: list[Entity],
        triples: list[Triple],
        canonicalizer: EmbeddingCanonicalizer,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        candidates = canonicalizer.build_candidates(entities)
        decisions: list[EntityMergeDecision] = []
        for candidate in tqdm(
            candidates,
            desc="kgmerge rules",
            unit="pair",
            total=len(candidates),
            dynamic_ncols=True,
        ):
            rule_decision = self._rule_decision(candidate, triples, entities)
            if rule_decision.decision == "needs_llm":
                rule_decision.decision = "unsure"
                rule_decision.method = "rules_only"
                rule_decision.reason_code = "LLM_RECHECK_DISABLED"
                rule_decision.reason = "embedding 只召回候选；LLM 复核关闭时不做语义合并。"
            decisions.append(rule_decision)
        canonical_map, cluster_conflicts = self._mapping_from_merge_decisions(
            entities,
            triples,
            decisions,
            max_cluster_size=4,
            require_complete_pairwise=True,
        )
        return canonical_map, {
            "strategy": "embedding_rules_only",
            "candidate_count": len(candidates),
            "decisions": [self._decision_to_dict(d) for d in decisions],
            "cluster_conflicts": cluster_conflicts,
        }

    def _rule_decision(
        self,
        candidate: EntityMergeCandidate,
        triples: list[Triple],
        entities: list[Entity],
    ) -> EntityMergeDecision:
        if candidate.left_type != candidate.right_type:
            return EntityMergeDecision(
                left=candidate.left,
                right=candidate.right,
                decision="keep_separate",
                method="rules",
                reason_code="TYPE_MISMATCH",
                reason="实体类型不一致，禁止自动归并。",
                embedding_score=candidate.score,
            )
        if self._strong_name_equivalence(candidate.left, candidate.right):
            return EntityMergeDecision(
                left=candidate.left,
                right=candidate.right,
                decision="merge",
                method="rules",
                reason_code="STRONG_NAME_EQUIVALENCE",
                canonical_name=self._select_canonical_name(
                    [candidate.left, candidate.right],
                    entities,
                ),
                reason="规则判断为强名称等价。",
                embedding_score=candidate.score,
            )
        if self._has_direct_relation_between(candidate.left, candidate.right, triples):
            return EntityMergeDecision(
                left=candidate.left,
                right=candidate.right,
                decision="keep_separate",
                method="rules",
                reason_code="DIRECT_RELATION_SELF_LOOP_RISK",
                reason="两个实体之间已有关系，合并会产生自环或丢失语义。",
                embedding_score=candidate.score,
            )
        return EntityMergeDecision(
            left=candidate.left,
            right=candidate.right,
            decision="needs_llm",
            method="rules",
            reason_code="AMBIGUOUS_REQUIRES_LLM",
            embedding_score=candidate.score,
        )

    def _validate_llm_decision(
        self,
        decision: EntityMergeDecision,
        candidate: EntityMergeCandidate,
        triples: list[Triple],
        entities: list[Entity],
        *,
        contexts: list[dict[str, Any]],
        allow_truncated_context: bool,
    ) -> EntityMergeDecision:
        hard_rule = self._rule_decision(candidate, triples, entities)
        if hard_rule.decision == "keep_separate":
            hard_rule.method = "rules_after_llm"
            return hard_rule
        if decision.decision != "merge":
            return decision
        if decision.canonical_name not in {candidate.left, candidate.right}:
            decision.decision = "unsure"
            decision.canonical_name = None
            decision.reason_code = "INVALID_CANONICAL_NAME"
            return decision
        if not allow_truncated_context and any(ctx.get("truncated") for ctx in contexts):
            decision.decision = "unsure"
            decision.canonical_name = None
            decision.reason_code = "TRUNCATED_CONTEXT"
            decision.reason = "LLM 未看到该实体的完整三元组上下文，合并不落地。"
            return decision
        return decision

    def _mapping_from_merge_decisions(
        self,
        entities: list[Entity],
        triples: list[Triple],
        decisions: list[EntityMergeDecision],
        *,
        max_cluster_size: int,
        require_complete_pairwise: bool,
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        names = sorted({e.name for e in entities})
        index_by_name = {name: idx for idx, name in enumerate(names)}
        uf = _UnionFind(len(names))
        accepted_pairs: set[frozenset[str]] = set()
        for decision in decisions:
            if decision.decision != "merge":
                continue
            left_idx = index_by_name.get(decision.left)
            right_idx = index_by_name.get(decision.right)
            if left_idx is not None and right_idx is not None:
                uf.union(left_idx, right_idx)
                accepted_pairs.add(frozenset((decision.left, decision.right)))

        members_by_root: dict[int, list[str]] = defaultdict(list)
        for name, idx in index_by_name.items():
            members_by_root[uf.find(idx)].append(name)

        mapping = {name: name for name in names}
        conflicts: list[dict[str, Any]] = []

        for members in members_by_root.values():
            if len(members) < 2:
                continue
            conflict = self._cluster_conflict(
                members,
                triples,
                accepted_pairs=accepted_pairs,
                max_cluster_size=max_cluster_size,
                require_complete_pairwise=require_complete_pairwise,
            )
            if conflict is not None:
                conflicts.append(conflict)
                continue
            preferred: list[str] = []
            for decision in decisions:
                if (
                    decision.decision == "merge"
                    and decision.canonical_name
                    and decision.left in members
                    and decision.right in members
                ):
                    preferred.append(decision.canonical_name)
            canonical = self._select_canonical_name(members, entities, preferred)
            for name in members:
                mapping[name] = canonical
        return mapping, conflicts

    def _cluster_conflict(
        self,
        members: list[str],
        triples: list[Triple],
        *,
        accepted_pairs: set[frozenset[str]],
        max_cluster_size: int,
        require_complete_pairwise: bool,
    ) -> dict[str, Any] | None:
        if max_cluster_size > 0 and len(members) > max_cluster_size:
            return {
                "members": members,
                "reason_code": "CLUSTER_TOO_LARGE",
                "reason": f"聚类大小 {len(members)} 超过上限 {max_cluster_size}，整组保持分离。",
            }
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if self._strong_name_equivalence(left, right):
                    continue
                if require_complete_pairwise and frozenset((left, right)) not in accepted_pairs:
                    return {
                        "members": members,
                        "reason_code": "INCOMPLETE_PAIRWISE_CLUSTER",
                        "reason": f"{left} 与 {right} 未被直接复核为等价，禁止传递合并。",
                    }
                if self._has_direct_relation_between(left, right, triples):
                    return {
                        "members": members,
                        "reason_code": "DIRECT_RELATION_IN_CLUSTER",
                        "reason": f"{left} 与 {right} 之间存在原始关系，整组保持分离。",
                    }
        return None

    @staticmethod
    def _context_for_llm(
        name: str,
        contexts: dict[str, _EntityContext],
        *,
        side: str,
        triples_limit: int,
    ) -> dict[str, Any]:
        ctx = contexts.get(name, _EntityContext(name=name))
        descriptions = [
            {"id": f"{side}_DESC_{idx}", "text": text}
            for idx, text in enumerate(ctx.descriptions[:3], start=1)
        ]
        selected_triples = KGMerger._select_context_triples(ctx.triples, triples_limit)
        triples = []
        for idx, triple in enumerate(selected_triples, start=1):
            triples.append(
                {
                    "id": f"{side}_TRIPLE_{idx}",
                    "triple": f"{triple['head']} {triple['relation']} {triple['tail']}",
                    "evidence": triple.get("evidence", ""),
                    "source_file": triple.get("source_file", ""),
                    "chunk_index": triple.get("chunk_index", ""),
                }
            )
        return {
            "name": name,
            "type": ctx.type,
            "frequency": ctx.frequency,
            "total_descriptions": len(ctx.descriptions),
            "omitted_descriptions": max(0, len(ctx.descriptions) - len(descriptions)),
            "total_triples": len(ctx.triples),
            "omitted_triples": max(0, len(ctx.triples) - len(triples)),
            "truncated": len(ctx.descriptions) > len(descriptions)
            or len(ctx.triples) > len(triples),
            "relation_summary": KGMerger._relation_summary(ctx.triples),
            "descriptions": descriptions,
            "triples": triples,
        }

    @staticmethod
    def _select_context_triples(
        triples: list[dict[str, str]],
        limit: int,
    ) -> list[dict[str, str]]:
        if limit <= 0 or len(triples) <= limit:
            return list(triples)
        by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
        for triple in triples:
            by_relation[triple.get("relation", "")].append(triple)
        selected: list[dict[str, str]] = []
        relation_names = sorted(by_relation)
        while len(selected) < limit and relation_names:
            next_relation_names: list[str] = []
            for relation in relation_names:
                bucket = by_relation[relation]
                if bucket and len(selected) < limit:
                    selected.append(bucket.pop(0))
                if bucket:
                    next_relation_names.append(relation)
            relation_names = next_relation_names
        return selected

    @staticmethod
    def _relation_summary(triples: list[dict[str, str]]) -> dict[str, int]:
        summary: dict[str, int] = defaultdict(int)
        for triple in triples:
            summary[triple.get("relation", "")] += 1
        return dict(sorted(summary.items()))

    @staticmethod
    def _decision_to_dict(decision: EntityMergeDecision) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "left": decision.left,
            "right": decision.right,
            "decision": decision.decision,
            "method": decision.method,
            "reason_code": decision.reason_code,
            "canonical_name": decision.canonical_name,
            "reason": decision.reason,
            "supporting_evidence_ids": decision.supporting_evidence_ids,
            "conflict_evidence_ids": decision.conflict_evidence_ids,
        }
        if decision.embedding_score is not None:
            payload["embedding_score"] = decision.embedding_score
        return payload

    @staticmethod
    def _has_direct_relation_between(left: str, right: str, triples: list[Triple]) -> bool:
        pair = {left, right}
        for triple in triples:
            head = normalize_entity_name(triple.head)
            tail = normalize_entity_name(triple.tail)
            if {head, tail} == pair:
                return True
        return False

    @staticmethod
    def _strong_name_equivalence(left: str, right: str) -> bool:
        if normalize_entity_name(left) == normalize_entity_name(right):
            return True
        return _compact_entity_name(left) == _compact_entity_name(right)

    @staticmethod
    def _select_canonical_name(
        names: list[str],
        entities: list[Entity],
        preferred: list[str] | None = None,
    ) -> str:
        preferred_set = set(preferred or [])
        freq_by_name: dict[str, int] = defaultdict(int)
        for entity in entities:
            freq_by_name[entity.name] += entity.frequency
        candidates = sorted(
            set(names),
            key=lambda name: (
                0 if name in preferred_set else 1,
                -freq_by_name.get(name, 0),
                len(name),
                name,
            ),
        )
        return candidates[0]

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
                grouped[canonical].frequency += e.frequency
                if not grouped[canonical].description and e.description:
                    grouped[canonical].description = e.description
            else:
                grouped[canonical] = Entity(
                    name=canonical,
                    type=e.type,
                    description=e.description,
                    frequency=e.frequency,
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
