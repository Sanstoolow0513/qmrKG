#!/usr/bin/env python3
"""Generate a Markdown inventory report for files and KG data under data/."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ENCODING = "cl100k_base"
DEFAULT_CHUNK_FIELD = "content"

PDF_EXTENSIONS = {".pdf"}
PPT_EXTENSIONS = {".ppt", ".pptx"}
PNG_EXTENSIONS = {".png"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
JSON_EXTENSIONS = {".json"}


@dataclass(frozen=True)
class SourceStats:
    source_dir: Path
    pdf_files: int
    ppt_files: int


@dataclass(frozen=True)
class PngStats:
    png_dir: Path
    png_files: int


@dataclass(frozen=True)
class MarkdownStats:
    markdown_dir: Path
    files: int
    chars: int
    non_whitespace_chars: int


@dataclass(frozen=True)
class ChunkStats:
    chunks_dir: Path
    tokenizer: str
    files: int
    chunks: int
    tokens: int


@dataclass(frozen=True)
class TripleGroupStats:
    group: str
    json_files: int = 0
    entities: int = 0
    triples: int = 0


@dataclass(frozen=True)
class TripleStats:
    triples_dir: Path
    json_files: int
    entities: int
    triples: int
    by_group: list[TripleGroupStats] = field(default_factory=list)


@dataclass(frozen=True)
class DataAnalysis:
    data_dir: Path
    generated_at: str
    source: SourceStats
    png: PngStats
    markdown: MarkdownStats
    chunks: ChunkStats
    triples: TripleStats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Analyze qmrKG data assets and write a Markdown report.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help="Root data directory. Default: the directory containing this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "data-analysis.md",
        help="Markdown report path. Default: data/data-analysis.md.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"tiktoken encoding for chunks token counting. Default: {DEFAULT_ENCODING}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional GPT model name for tiktoken.encoding_for_model, e.g. gpt-4o.",
    )
    parser.add_argument(
        "--chunk-field",
        default=DEFAULT_CHUNK_FIELD,
        help=f"Chunk text field to count. Default: {DEFAULT_CHUNK_FIELD}.",
    )
    return parser.parse_args(argv)


def iter_files(root: Path, extensions: set[str] | None = None) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if extensions is None or path.suffix.lower() in extensions:
            yield path


def count_files(root: Path, extensions: set[str]) -> int:
    return sum(1 for _path in iter_files(root, extensions))


def load_encoding(encoding_name: str, model_name: str | None) -> tuple[Any, str]:
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: tiktoken. Run with `uv run python data/data-analisys.py`."
        ) from exc

    if model_name:
        return tiktoken.encoding_for_model(model_name), f"model:{model_name}"
    return tiktoken.get_encoding(encoding_name), f"encoding:{encoding_name}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunk_entries(path: Path) -> list[Any]:
    payload = load_json(path)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("chunks", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"{path} is not a chunk list or a dict with chunks/items/data list")


def extract_chunk_text(entry: Any, field_name: str, path: Path, index: int) -> str:
    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        value = entry.get(field_name)
        if isinstance(value, str):
            return value
        raise ValueError(f"{path} chunk {index} has no string field {field_name!r}")

    raise ValueError(f"{path} chunk {index} is not an object or string")


def count_chunk_tokens(
    chunks_dir: Path, encoding: Any, tokenizer_name: str, field_name: str
) -> ChunkStats:
    json_files = list(iter_files(chunks_dir, JSON_EXTENSIONS))
    chunk_count = 0
    token_count = 0

    for path in json_files:
        entries = load_chunk_entries(path)
        chunk_count += len(entries)
        for index, entry in enumerate(entries):
            text = extract_chunk_text(entry, field_name, path, index)
            token_count += len(encoding.encode(text, disallowed_special=()))

    return ChunkStats(
        chunks_dir=chunks_dir,
        tokenizer=tokenizer_name,
        files=len(json_files),
        chunks=chunk_count,
        tokens=token_count,
    )


def count_markdown(markdown_dir: Path) -> MarkdownStats:
    files = list(iter_files(markdown_dir, MARKDOWN_EXTENSIONS))
    chars = 0
    non_whitespace_chars = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        chars += len(text)
        non_whitespace_chars += len(re.sub(r"\s+", "", text))

    return MarkdownStats(
        markdown_dir=markdown_dir,
        files=len(files),
        chars=chars,
        non_whitespace_chars=non_whitespace_chars,
    )


def count_list_field(payload: Any, field_name: str) -> int:
    if isinstance(payload, dict) and isinstance(payload.get(field_name), list):
        return len(payload[field_name])
    return 0


def triple_group_name(triples_dir: Path, path: Path) -> str:
    relative = path.relative_to(triples_dir)
    if len(relative.parts) == 1:
        return "."
    return relative.parts[0]


def count_triples(triples_dir: Path) -> TripleStats:
    json_files = list(iter_files(triples_dir, JSON_EXTENSIONS))
    groups: dict[str, TripleGroupStats] = {}

    for path in json_files:
        payload = load_json(path)
        group_name = triple_group_name(triples_dir, path)
        previous = groups.get(group_name, TripleGroupStats(group=group_name))
        groups[group_name] = TripleGroupStats(
            group=group_name,
            json_files=previous.json_files + 1,
            entities=previous.entities + count_list_field(payload, "entities"),
            triples=previous.triples + count_list_field(payload, "triples"),
        )

    by_group = sorted(groups.values(), key=lambda item: item.group)
    return TripleStats(
        triples_dir=triples_dir,
        json_files=sum(item.json_files for item in by_group),
        entities=sum(item.entities for item in by_group),
        triples=sum(item.triples for item in by_group),
        by_group=by_group,
    )


def collect_analysis(
    data_dir: Path, encoding_name: str, model_name: str | None, chunk_field: str
) -> DataAnalysis:
    data_dir = data_dir.resolve()
    encoding, tokenizer_name = load_encoding(encoding_name, model_name)

    return DataAnalysis(
        data_dir=data_dir,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source=SourceStats(
            source_dir=data_dir / "pdf",
            pdf_files=count_files(data_dir / "pdf", PDF_EXTENSIONS),
            ppt_files=count_files(data_dir / "pdf", PPT_EXTENSIONS),
        ),
        png=PngStats(
            png_dir=data_dir / "png",
            png_files=count_files(data_dir / "png", PNG_EXTENSIONS),
        ),
        markdown=count_markdown(data_dir / "markdown"),
        chunks=count_chunk_tokens(data_dir / "chunks", encoding, tokenizer_name, chunk_field),
        triples=count_triples(data_dir / "triples"),
    )


def fmt(number: int) -> str:
    return f"{number:,}"


def render_report(analysis: DataAnalysis) -> str:
    lines = [
        "# Data Analysis Report",
        "",
        f"- Generated at: `{analysis.generated_at}`",
        f"- Data directory: `{analysis.data_dir}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| PDF files | {fmt(analysis.source.pdf_files)} |",
        f"| PPT/PPTX files | {fmt(analysis.source.ppt_files)} |",
        f"| PNG files | {fmt(analysis.png.png_files)} |",
        f"| Markdown files | {fmt(analysis.markdown.files)} |",
        f"| Markdown characters | {fmt(analysis.markdown.chars)} |",
        f"| Markdown non-whitespace characters | {fmt(analysis.markdown.non_whitespace_chars)} |",
        f"| Chunk JSON files | {fmt(analysis.chunks.files)} |",
        f"| Chunks | {fmt(analysis.chunks.chunks)} |",
        f"| Chunk tokens ({analysis.chunks.tokenizer}) | {fmt(analysis.chunks.tokens)} |",
        f"| Triple JSON files | {fmt(analysis.triples.json_files)} |",
        f"| Entities | {fmt(analysis.triples.entities)} |",
        f"| Triples | {fmt(analysis.triples.triples)} |",
        "",
        "## Source Files",
        "",
        f"- Source directory: `{analysis.source.source_dir}`",
        f"- PDF files: {fmt(analysis.source.pdf_files)}",
        f"- PPT/PPTX files: {fmt(analysis.source.ppt_files)}",
        "",
        "## PNG",
        "",
        f"- PNG directory: `{analysis.png.png_dir}`",
        f"- PNG files: {fmt(analysis.png.png_files)}",
        "",
        "## Markdown",
        "",
        f"- Markdown directory: `{analysis.markdown.markdown_dir}`",
        f"- Markdown files: {fmt(analysis.markdown.files)}",
        f"- Characters: {fmt(analysis.markdown.chars)}",
        f"- Non-whitespace characters: {fmt(analysis.markdown.non_whitespace_chars)}",
        "",
        "## Chunks",
        "",
        f"- Chunks directory: `{analysis.chunks.chunks_dir}`",
        f"- Tokenizer: `{analysis.chunks.tokenizer}`",
        f"- Chunk JSON files: {fmt(analysis.chunks.files)}",
        f"- Chunks: {fmt(analysis.chunks.chunks)}",
        f"- Tokens: {fmt(analysis.chunks.tokens)}",
        "",
        "## Triples",
        "",
        f"- Triples directory: `{analysis.triples.triples_dir}`",
        f"- Triple JSON files: {fmt(analysis.triples.json_files)}",
        f"- Entities: {fmt(analysis.triples.entities)}",
        f"- Triples: {fmt(analysis.triples.triples)}",
        "",
        "### Triples By Directory",
        "",
        "| Directory | JSON files | Entities | Triples |",
        "|---|---:|---:|---:|",
    ]

    for group in analysis.triples.by_group:
        lines.append(
            f"| `{group.group}` | {fmt(group.json_files)} | {fmt(group.entities)} | {fmt(group.triples)} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_report(report: str, output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = collect_analysis(args.data_dir, args.encoding, args.model, args.chunk_field)
    output_path = write_report(render_report(analysis), args.output)
    print(f"Wrote data analysis report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
