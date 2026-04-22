"""Knowledge graph extraction from markdown chunks using LLM."""

from __future__ import annotations

import json
import logging
import re
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

_MODE_TO_PROMPT_KEY: dict[str, str] = {
    "default": "default",
    "zero_shot": "zero_shot",
    "zero-shot": "zero_shot",
    "few_shot": "few_shot",
    "few-shot": "few_shot",
}


def _discover_extract_config_paths(config_path: Path | None) -> list[Path]:
    if config_path is not None:
        return [Path(config_path)]
    return [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).parent.parent.parent.parent / "config.yaml",
    ]


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
        return "default"
    key = str(mode).strip().lower()
    return _MODE_TO_PROMPT_KEY.get(key, "default")


class KGExtractor:
    """Extract entities and relations from markdown chunks via LLM."""

    def __init__(
        self,
        runner: TaskLLMRunner | None = None,
        config_path: Path | None = None,
        mode: str | None = None,
    ):
        self._config_path = Path(config_path) if config_path is not None else None
        self._mode = mode
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
            )

        response = self._runner.run_text(content, system_prompt=self._system_prompt)
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
                self._save_result(result, out_path)
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
        text = prompts.get(key) or prompts.get("default")
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
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence,
                }
                for t in result.triples
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
