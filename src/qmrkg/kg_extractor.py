"""Knowledge graph extraction from markdown chunks using LLM."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from tqdm import tqdm

from .config import load_config
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
5. 不要计算字符级起止位置，不要猜测 span，evidence_span 固定输出为 null

输出必须为 JSON：
{
  "triples": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "evidence_span": null,
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


@dataclass(slots=True)
class _ChunkJob:
    file_index: int
    chunks_path: Path
    chunk_index: int
    chunk: dict[str, Any]
    out_path: Path


def _load_extract_prompts(config_path: Path | None) -> dict[str, Any]:
    data = load_config(config_path)
    extract_cfg = data.get("extract")
    if not isinstance(extract_cfg, dict):
        return {}
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
        self._review_prompt = self._resolve_review_prompt()

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
            self._review_triples(content, proposed_triples)
            if self._enable_review
            else proposed_triples
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
        progress_desc: str = "kgextract",
        progress_position: int = 0,
    ) -> list[Path]:
        """Extract from all chunks in a JSON file, saving per-chunk results."""
        result_paths = self.extract_from_chunks_files(
            [chunks_path],
            output_dir,
            skip_existing=skip_existing,
            progress_leave=progress_leave,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )
        return result_paths

    def extract_from_chunks_files(
        self,
        chunks_paths: Sequence[Path],
        output_dir: Path,
        skip_existing: bool = True,
        *,
        progress_leave: bool = True,
        progress_desc: str = "kgextract",
        progress_position: int = 0,
    ) -> list[Path]:
        """Extract from multiple chunk JSON files through one shared worker pool."""
        normalized_paths = [Path(path) for path in chunks_paths]
        if not normalized_paths:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_maps: list[dict[int, Path]] = []
        pending_indices_by_file: list[set[int]] = []
        pending_count = 0
        skipped_count = 0

        for file_index, chunks_path in enumerate(normalized_paths):
            result_path_map, pending_indices, file_skipped_count = self._scan_chunks_file(
                chunks_path,
                output_dir,
                skip_existing=skip_existing,
            )
            result_maps.append(result_path_map)
            pending_indices_by_file.append(pending_indices)
            pending_count += len(pending_indices)
            skipped_count += file_skipped_count

        self._run_chunk_jobs(
            self._iter_chunk_jobs(normalized_paths, output_dir, pending_indices_by_file),
            pending_count,
            result_maps,
            progress_leave=progress_leave,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )

        if skipped_count:
            tqdm.write(f"skipped existing outputs: {skipped_count}")

        result_paths: list[Path] = []
        for result_path_map in result_maps:
            result_paths.extend(result_path_map[idx] for idx in sorted(result_path_map))
        return result_paths

    def _scan_chunks_file(
        self,
        chunks_path: Path,
        output_dir: Path,
        *,
        skip_existing: bool,
    ) -> tuple[dict[int, Path], set[int], int]:
        """Scan one chunks JSON and record output bookkeeping without retaining chunks."""
        chunks_path = Path(chunks_path)
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        result_path_map: dict[int, Path] = {}
        pending_indices: set[int] = set()
        skipped_count = 0

        for chunk in chunks:
            idx = chunk.get("chunk_index", 0)
            out_path = output_dir / f"{chunks_path.stem}_chunk_{idx:04d}.json"

            if skip_existing and out_path.exists():
                skipped_count += 1
                result_path_map[idx] = out_path
                continue
            pending_indices.add(idx)

        return result_path_map, pending_indices, skipped_count

    def _iter_chunk_jobs(
        self,
        chunks_paths: Sequence[Path],
        output_dir: Path,
        pending_indices_by_file: Sequence[set[int]],
    ) -> Iterator[_ChunkJob]:
        """Yield chunk jobs one file at a time so pending chunks are not all retained."""
        for file_index, chunks_path in enumerate(chunks_paths):
            pending_indices = pending_indices_by_file[file_index]
            if not pending_indices:
                continue
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            for chunk in chunks:
                idx = chunk.get("chunk_index", 0)
                if idx not in pending_indices:
                    continue
                yield _ChunkJob(
                    file_index=file_index,
                    chunks_path=chunks_path,
                    chunk_index=idx,
                    chunk=chunk,
                    out_path=output_dir / f"{chunks_path.stem}_chunk_{idx:04d}.json",
                )

    def _run_chunk_jobs(
        self,
        jobs: Iterator[_ChunkJob],
        total_jobs: int,
        result_maps: Sequence[dict[int, Path]],
        *,
        progress_leave: bool,
        progress_desc: str,
        progress_position: int,
    ) -> None:
        if total_jobs <= 0:
            return

        max_workers = self._resolve_max_workers(total_jobs)
        job_iter = iter(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, _ChunkJob] = {}

            def submit_next() -> bool:
                try:
                    job = next(job_iter)
                except StopIteration:
                    return False
                futures[executor.submit(self._extract_and_save_chunk, job.chunk, job.out_path)] = (
                    job
                )
                return True

            for _ in range(max_workers):
                if not submit_next():
                    break

            with tqdm(
                total=total_jobs,
                desc=progress_desc,
                unit="chunk",
                dynamic_ncols=True,
                leave=progress_leave,
                position=progress_position,
            ) as pbar:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = futures.pop(future)
                        try:
                            out_path = future.result()
                            result_maps[job.file_index][job.chunk_index] = out_path
                        except Exception as e:
                            logger.error(
                                "Failed %s chunk %d: %s",
                                job.chunks_path.name,
                                job.chunk_index,
                                e,
                            )
                        finally:
                            pbar.update(1)
                        submit_next()

    def _resolve_max_workers(self, pending_count: int) -> int:
        configured = getattr(getattr(self._runner, "settings", None), "max_concurrency", 1)
        if not isinstance(configured, int) or configured <= 0:
            configured = 1
        return max(1, min(pending_count, configured))

    def _extract_and_save_chunk(self, chunk: dict[str, Any], out_path: Path) -> Path:
        idx = chunk.get("chunk_index", 0)
        result = self.extract_from_chunk(chunk)
        self._save_result(result, out_path, extractor_version=self._extractor_version)
        logger.info(
            "Extracted chunk %d: %d entities, %d triples",
            idx,
            len(result.entities),
            len(result.triples),
        )
        return out_path

    def resolve_prompt(self) -> str:
        """System prompt used for extraction (config + mode, or built-in default)."""
        return self._system_prompt

    def resolve_review_prompt(self) -> str:
        """System prompt used for review (config + mode, or built-in default)."""
        return self._review_prompt

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

    def _resolve_review_prompt(self) -> str:
        prompts = _load_extract_prompts(self._config_path)
        key = _mode_to_prompt_key(self._mode)
        review_mode_key = f"review_{key}"
        text = prompts.get(review_mode_key)
        if not text:
            text = prompts.get("review")
        if not text:
            text = prompts.get("review_default")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return REVIEW_PROMPT

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
            system_prompt=self._review_prompt,
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
