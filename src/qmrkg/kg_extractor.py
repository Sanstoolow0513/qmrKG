"""Knowledge graph extraction from markdown chunks using LLM."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .kg_schema import ChunkExtractionResult, Entity, Triple
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

REVIEW_PROMPT = """\
你是一个知识图谱三元组审核器。

任务：审核候选三元组并输出修正后的结果。

审核规则（必须严格执行）：
1. evidence 必须是输入 chunk 原文中的逐字片段，不能改写或拼接
2. head 与 tail 必须都出现在 evidence 中
3. relation 必须属于 contains / depends_on / compared_with / applied_to
4. 不满足条件时必须 decision=drop，并给出 reason_code

输出必须为 JSON：
{
  "triples": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "evidence_span": {"start": 0, "end": 10},
      "review": {
        "decision": "keep|revise|drop",
        "reason_code": "SUPPORTED|EVIDENCE_NOT_IN_CHUNK|SPAN_MISMATCH|HEAD_NOT_IN_EVIDENCE|TAIL_NOT_IN_EVIDENCE|INVALID_RELATION|SELF_LOOP|LOW_CONFIDENCE_DROP",
        "reason": "..."
      }
    }
  ]
}
"""

_MODE_TO_PROMPT_KEY: dict[str, str] = {
    "default": "fs",
    "fs": "fs",
    "zs": "zs",
    "zero_shot": "zs",
    "zero-shot": "zs",
    "few_shot": "fs",
    "few-shot": "fs",
}


def _find_qmrkg_repo_root() -> Path | None:
    """Return the project root (directory whose pyproject.toml names this package), if found."""
    start = Path(__file__).resolve().parent
    for base in (start, *start.parents):
        pyproject = base / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception:
            continue
        project = data.get("project")
        if isinstance(project, dict) and project.get("name") == "qmrkg":
            return base
    return None


def _discover_extract_config_paths(config_path: Path | None) -> list[Path]:
    """Resolve candidate config files for extract prompt loading.

    When ``config_path`` is None, only these locations are considered (in order):
    1. ``<cwd>/config.yaml`` or ``config.yml`` (optional local override)
    2. ``<qmrkg repo root>/config.yaml`` or ``config.yml`` (project defaults)

    This avoids walking unbounded ``cwd`` ancestor chains, which can pick up unrelated
    config files outside the project.
    """
    if config_path is not None:
        return [Path(config_path).resolve()]

    candidates: list[Path] = []
    seen: set[Path] = set()
    cwd = Path.cwd()
    for name in ("config.yaml", "config.yml"):
        p = cwd / name
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(p)
    root = _find_qmrkg_repo_root()
    if root is not None:
        for name in ("config.yaml", "config.yml"):
            p = root / name
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(p)
    return candidates


def _load_extract_prompts(config_path: Path | None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    for path in _discover_extract_config_paths(config_path):
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read extract prompts from %s: %s", path, exc)
            continue
        extract_cfg = data.get("extract")
        if not isinstance(extract_cfg, dict):
            continue
        prompts = extract_cfg.get("prompts")
        if isinstance(prompts, dict):
            return prompts
    return {}


def _mode_to_prompt_key(mode: str | None) -> str:
    if not mode or not str(mode).strip():
        return "fs"
    key = str(mode).strip().lower()
    return _MODE_TO_PROMPT_KEY.get(key, "fs")


class KGExtractor:
    """Extract entities and relations from markdown chunks via LLM."""

    def __init__(
        self,
        runner: TaskLLMRunner | None = None,
        config_path: Path | None = None,
        mode: str | None = None,
        *,
        enable_review: bool = True,
        strict_evidence: bool = True,
        keep_dropped: bool = True,
        extractor_version: str = "kgextract_v2",
    ):
        self._config_path = Path(config_path) if config_path is not None else None
        self._mode = mode
        self._enable_review = enable_review
        self._strict_evidence = strict_evidence
        self._keep_dropped = keep_dropped
        self._extractor_version = extractor_version
        if runner is not None:
            self._runner = runner
        else:
            factory = LLMFactory(config_path)
            self._runner = factory.create(EXTRACT_TASK_NAME)
        self._system_prompt = self._resolve_system_prompt()

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
                dropped=[],
            )

        propose_response = self._runner.run_text(content, system_prompt=self._system_prompt)
        proposed_raw = self._parse_json_response(propose_response.text)
        entities = self._parse_entities(proposed_raw.get("entities", []))
        proposed_triples = self._parse_triples(proposed_raw.get("triples", []))
        reviewed_triples = (
            self._review_triples(content, proposed_triples) if self._enable_review else proposed_triples
        )
        triples, dropped = self._apply_gate(
            reviewed_triples, content, strict_evidence=self._strict_evidence
        )

        return ChunkExtractionResult(
            chunk_index=chunk.get("chunk_index", 0),
            source_file=chunk.get("source_file", ""),
            titles=chunk.get("titles", []),
            entities=entities,
            triples=triples,
            dropped=dropped if self._keep_dropped else [],
        )

    def extract_from_chunks_file(
        self,
        chunks_path: Path,
        output_dir: Path,
        skip_existing: bool = True,
        *,
        progress_leave: bool = True,
    ) -> list[Path]:
        """Extract from all chunks in a JSON file, saving per-chunk results."""
        chunks_path = Path(chunks_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        result_paths: list[Path] = []

        chunk_iter = tqdm(
            chunks,
            desc="kgextract",
            unit="chunk",
            total=len(chunks),
            dynamic_ncols=True,
            leave=progress_leave,
        )
        for chunk in chunk_iter:
            idx = chunk.get("chunk_index", 0)
            out_path = output_dir / f"{chunks_path.stem}_chunk_{idx:04d}.json"

            if skip_existing and out_path.exists():
                tqdm.write(f"{out_path.name} skip for existing output")
                result_paths.append(out_path)
                continue

            try:
                result = self.extract_from_chunk(chunk)
                self._save_result(result, out_path, extractor_version=self._extractor_version)
                result_paths.append(out_path)
                logger.info(
                    "Extracted chunk %d: %d entities, %d triples",
                    idx,
                    len(result.entities),
                    len(result.triples),
                )
            except Exception as e:
                logger.error("Failed chunk %d: %s", idx, e)

        return result_paths

    def resolve_prompt(self) -> str:
        """System prompt used for extraction (config + mode, or built-in default)."""
        return self._system_prompt

    def _resolve_system_prompt(self) -> str:
        prompts = _load_extract_prompts(self._config_path)
        key = _mode_to_prompt_key(self._mode)
        text = prompts.get(key)
        if not text:
            # 默认优先 fs，并兼容历史键名。
            text = prompts.get("fs") or prompts.get("few_shot") or prompts.get("default")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return EXTRACT_PROMPT

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
            review = item.get("review", {}) if isinstance(item.get("review"), dict) else {}
            evidence_span = item.get("evidence_span")
            if not isinstance(evidence_span, dict):
                evidence_span = None
            triple = Triple(
                head=str(item.get("head", "")).strip(),
                relation=str(item.get("relation", "")).strip().lower(),
                tail=str(item.get("tail", "")).strip(),
                evidence=str(item.get("evidence", "")).strip(),
                evidence_span=evidence_span,
                review_decision=str(review.get("decision", "keep")).strip().lower() or "keep",
                review_reason_code=(
                    str(review.get("reason_code", "SUPPORTED")).strip().upper() or "SUPPORTED"
                ),
                review_reason=str(review.get("reason", "")).strip(),
            )
            if triple.is_valid():
                triples.append(triple)
        return triples

    def _review_triples(self, content: str, triples: list[Triple]) -> list[Triple]:
        if not triples:
            return []
        payload = {
            "chunk": content,
            "triples": [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence,
                }
                for t in triples
            ],
        }
        review_response = self._runner.run_text(
            json.dumps(payload, ensure_ascii=False),
            system_prompt=REVIEW_PROMPT,
        )
        reviewed_raw = self._parse_json_response(review_response.text)
        reviewed = self._parse_triples(reviewed_raw.get("triples", []))
        return reviewed or triples

    @staticmethod
    def _apply_gate(
        triples: list[Triple],
        chunk_content: str,
        *,
        strict_evidence: bool = True,
    ) -> tuple[list[Triple], list[dict[str, Any]]]:
        kept: list[Triple] = []
        dropped: list[dict[str, Any]] = []

        for triple in triples:
            if triple.review_decision == "drop":
                dropped.append(
                    KGExtractor._dropped_item(triple, "LOW_CONFIDENCE_DROP", triple.review_reason)
                )
                continue
            if strict_evidence and triple.evidence not in chunk_content:
                dropped.append(KGExtractor._dropped_item(triple, "EVIDENCE_NOT_IN_CHUNK"))
                continue
            actual_start = chunk_content.find(triple.evidence)
            actual_end = actual_start + len(triple.evidence)
            if strict_evidence and actual_start < 0:
                dropped.append(KGExtractor._dropped_item(triple, "EVIDENCE_NOT_IN_CHUNK"))
                continue
            if (
                strict_evidence
                and triple.evidence_span
                and (
                    triple.evidence_span.get("start") != actual_start
                    or triple.evidence_span.get("end") != actual_end
                )
            ):
                dropped.append(KGExtractor._dropped_item(triple, "SPAN_MISMATCH"))
                continue
            if strict_evidence:
                triple.evidence_span = {"start": actual_start, "end": actual_end}
            if strict_evidence and triple.head not in triple.evidence:
                dropped.append(KGExtractor._dropped_item(triple, "HEAD_NOT_IN_EVIDENCE"))
                continue
            if strict_evidence and triple.tail not in triple.evidence:
                dropped.append(KGExtractor._dropped_item(triple, "TAIL_NOT_IN_EVIDENCE"))
                continue
            kept.append(triple)

        return kept, dropped

    @staticmethod
    def _dropped_item(triple: Triple, reason_code: str, reason: str = "") -> dict[str, Any]:
        return {
            "candidate": {
                "head": triple.head,
                "relation": triple.relation,
                "tail": triple.tail,
                "evidence": triple.evidence,
                "evidence_span": triple.evidence_span,
            },
            "review": {
                "decision": "drop",
                "reason_code": reason_code,
                "reason": reason,
            },
        }

    @staticmethod
    def _save_result(
        result: ChunkExtractionResult,
        path: Path,
        *,
        extractor_version: str,
    ) -> None:
        data = {
            "chunk_index": result.chunk_index,
            "source_file": result.source_file,
            "titles": result.titles,
            "extractor_version": extractor_version,
            "stats": {
                "proposed_entities": len(result.entities),
                "proposed_triples": len(result.triples) + len(result.dropped),
                "kept_triples": len(result.triples),
                "dropped_triples": len(result.dropped),
            },
            "entities": [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in result.entities
            ],
            "triples": [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "evidence_span": t.evidence_span,
                    "review": {
                        "decision": t.review_decision,
                        "reason_code": t.review_reason_code,
                        "reason": t.review_reason,
                    },
                }
                for t in result.triples
            ],
            "dropped": result.dropped,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
