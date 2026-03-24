"""Thesis-oriented pipeline stages: maps graduation-design wording to existing CLIs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStage:
    """One executable stage in the QmrKG stack."""

    id: str
    thesis_anchor: str
    description: str
    cli_name: str
    example_bash: str


# Order matches typical execution. Wording mirrors task.md bullets without adding scope.
STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        id="corpus_pdf",
        thesis_anchor="课程语料构建与预处理（PDF 渲染）",
        description="将课程 PDF 转为页面图像，作为多源语料的标准化输入。",
        cli_name="pdftopng",
        example_bash="uv run pdftopng --pdf-dir data/pdf",
    ),
    PipelineStage(
        id="corpus_ocr",
        thesis_anchor="课程语料构建与预处理（结构化 Markdown）",
        description="对页面图做 VLM OCR，输出保留标题层级的 Markdown 语料。",
        cli_name="pngtotext",
        example_bash="uv run pngtotext --image-dir data/png --text-dir data/markdown --recursive",
    ),
    PipelineStage(
        id="chunking",
        thesis_anchor="结构化输入语料库（分块 JSON）",
        description="按 token 预算将 Markdown 切成可供抽取的 JSON chunk。",
        cli_name="mdchunk",
        example_bash="uv run mdchunk --markdown-dir data/markdown --chunk-dir data/chunks --recursive",
    ),
    PipelineStage(
        id="kg_extract",
        thesis_anchor="命名实体识别与关系抽取（LLM 抽取）",
        description="从 chunk JSON 中抽取实体与三元组（NER+RE，配置见 config.yaml extract）。",
        cli_name="kgextract",
        example_bash="uv run kgextract --input data/chunks --output-dir data/triples/raw",
    ),
    PipelineStage(
        id="kg_merge",
        thesis_anchor="知识验证与融合（去重合并）",
        description="合并原始三元组 JSON，去重并输出可导入的图数据文件。",
        cli_name="kgmerge",
        example_bash="uv run kgmerge --input-dir data/triples/raw --output data/triples/merged/merged_triples.json",
    ),
    PipelineStage(
        id="kg_neo4j",
        thesis_anchor="知识图谱构建（Neo4j 导入，可支撑后续 Agent/应用）",
        description="将合并后的三元组导入 Neo4j（需环境变量 NEO4J_*）。",
        cli_name="kgneo4j",
        example_bash="uv run kgneo4j --import data/triples/merged/merged_triples.json",
    ),
)


def stages_as_dicts() -> list[dict[str, str]]:
    return [asdict(s) for s in STAGES]


def _count_pdf(root: Path) -> int:
    d = root / "data" / "pdf"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.pdf") if p.is_file())


def _count_png(root: Path) -> int:
    d = root / "data" / "png"
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob("*.png"))


def _count_md(root: Path) -> int:
    d = root / "data" / "markdown"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.md") if p.is_file())


def _count_chunk_json(root: Path) -> int:
    d = root / "data" / "chunks"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.json") if p.is_file())


def _count_raw_triple_files(root: Path) -> int:
    d = root / "data" / "triples" / "raw"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.rglob("*.json") if p.is_file())


def _merged_path(root: Path) -> Path:
    return root / "data" / "triples" / "merged" / "merged_triples.json"


def collect_execution_status(root: Path | None = None) -> dict[str, object]:
    """Lightweight filesystem signals for agents (no DB calls)."""

    base = Path(root or Path.cwd()).resolve()
    merged = _merged_path(base)
    return {
        "project_root": str(base),
        "counts": {
            "pdf": _count_pdf(base),
            "png": _count_png(base),
            "markdown": _count_md(base),
            "chunks_json": _count_chunk_json(base),
            "triples_raw_json_files": _count_raw_triple_files(base),
        },
        "artifacts": {
            "merged_triples_json": {
                "path": str(merged),
                "exists": merged.is_file(),
                "bytes": merged.stat().st_size if merged.is_file() else 0,
            }
        },
    }


def format_status_human(status: dict[str, object]) -> str:
    lines = [
        f"Project root: {status['project_root']}",
        "Counts:",
    ]
    counts = status["counts"]
    assert isinstance(counts, dict)
    for k, v in counts.items():
        lines.append(f"  {k}: {v}")
    art = status["artifacts"]
    assert isinstance(art, dict)
    merged = art["merged_triples_json"]
    assert isinstance(merged, dict)
    lines.append(
        f"Merged triples: {'yes' if merged['exists'] else 'no'} "
        f"({merged['bytes']} bytes) -> {merged['path']}"
    )
    return "\n".join(lines)


def format_stages_json() -> str:
    return json.dumps(stages_as_dicts(), ensure_ascii=False, indent=2)


def format_status_json(root: Path | None = None) -> str:
    return json.dumps(collect_execution_status(root), ensure_ascii=False, indent=2)
