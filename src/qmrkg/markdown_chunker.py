"""Markdown chunking module for splitting documents into token-bounded chunks.

This module provides data structures for representing markdown chunks and header
hierarchies, supporting downstream knowledge graph construction workflows.

Example:
    from qmrkg.markdown_chunker import MarkdownChunker, MarkdownChunk

    chunker = MarkdownChunker(max_tokens=4000)
    chunks = chunker.chunk_file("path/to/file.md")
    for chunk in chunks:
        print(chunk.titles, chunk.token_count)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_TOKENS = 1500
ENCODING = "cl100k_base"

# Metadata prefixes that should be stripped during cleaning
METADATA_PREFIXES = [
    "**Image:**",
    "**Processed:**",
    "**Duration:**",
    "**Model:**",
    "**Prompt Tokens:**",
    "**Completion Tokens:**",
    "**Total Tokens:**",
    "**Status:**",
    "**Error:**",
    "**Confidence:**",
]

# Regex pattern for matching markdown headers
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Icon block pattern: ```icon: ... ```
ICON_BLOCK_PATTERN = re.compile(r"```icon:[^`]*```", re.DOTALL)

# Page number symbols like "• 115 •" or "· 115 ·"
PAGE_NUMBER_PATTERN = re.compile(r"^[\s]*[•·]\s*\d+\s*[•·][\s]*$")

# Page header/footer patterns (e.g., "运输层 119", "122 第 3 章")
PAGE_HEADER_PATTERN = re.compile(
    r"^[\s]*(?:第\s*\d+\s*章|运输层|网络层|应用层|[^a-zA-Z]{2,})\s+\d+[\s]*$"
)

# Table of contents ellipsis pattern
TOC_ELLIPSIS_PATTERN = re.compile(r"\s*\.{3,}\s*\d+")

# HTML comment pattern for page markers
PAGE_COMMENT_PATTERN = re.compile(r"<!--\s*Page\s*\d+\s*-->")

# Re-export for backward compatibility
__all__ = [
    "MarkdownChunk",
    "MarkdownChunker",
    "HeaderNode",
    "MAX_TOKENS",
    "ENCODING",
    "HEADER_PATTERN",
    "clean_markdown",
    "clean_markdown_file",
    "batch_clean_markdown_files",
]


def clean_markdown(
    text: str,
    remove_icon_blocks: bool = True,
    remove_page_comments: bool = False,
    remove_page_numbers: bool = True,
    remove_page_headers: bool = True,
    normalize_whitespace: bool = True,
) -> tuple[str, dict[int, dict]]:
    """Clean markdown by removing OCR artifacts and formatting issues.

    Returns:
        Tuple of (cleaned_text, page_metadata_dict)
    """
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL, count=1)

    if remove_icon_blocks:
        text = ICON_BLOCK_PATTERN.sub("", text)

    lines = text.split("\n")
    result_lines = []
    skip_metadata = False
    page_metadata: dict[int, dict] = {}
    current_page: int | None = None
    current_metadata: dict = {}
    prev_line_empty = False

    for line in lines:
        stripped = line.strip()

        if normalize_whitespace and not stripped:
            if prev_line_empty:
                continue
            prev_line_empty = True
        else:
            prev_line_empty = False

        page_match = re.match(r"^## Page (\d+)", stripped)
        if page_match:
            if current_page is not None and current_metadata:
                page_metadata[current_page] = current_metadata

            page_num = int(page_match.group(1))
            if not remove_page_comments:
                result_lines.append(f"<!-- Page {page_num} -->")
            skip_metadata = True
            current_page = page_num
            current_metadata = {}
            continue

        if PAGE_COMMENT_PATTERN.match(stripped):
            if not remove_page_comments:
                result_lines.append(line)
            continue

        if remove_page_numbers and PAGE_NUMBER_PATTERN.match(stripped):
            continue

        if remove_page_headers and PAGE_HEADER_PATTERN.match(stripped):
            continue

        if skip_metadata and stripped:
            is_metadata = any(stripped.startswith(prefix) for prefix in METADATA_PREFIXES)
            if is_metadata:
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    key = key.strip().strip("*")
                    value = value.strip().strip("`")
                    current_metadata[key] = value
                continue
            skip_metadata = False

        if TOC_ELLIPSIS_PATTERN.search(line):
            line = TOC_ELLIPSIS_PATTERN.sub("", line)
            stripped = line.strip()
            if not stripped:
                continue

        line = line.lstrip("　 ")

        result_lines.append(line)

    if current_page is not None and current_metadata:
        page_metadata[current_page] = current_metadata

    return "\n".join(result_lines), page_metadata


def clean_markdown_file(
    file_path: Path,
    output_path: Path | None = None,
    remove_icon_blocks: bool = True,
    remove_page_comments: bool = False,
    remove_page_numbers: bool = True,
    remove_page_headers: bool = True,
    normalize_whitespace: bool = True,
) -> tuple[str, dict[int, dict]]:
    """Clean a markdown file and optionally write the output."""
    text = Path(file_path).read_text(encoding="utf-8")
    cleaned, metadata = clean_markdown(
        text,
        remove_icon_blocks=remove_icon_blocks,
        remove_page_comments=remove_page_comments,
        remove_page_numbers=remove_page_numbers,
        remove_page_headers=remove_page_headers,
        normalize_whitespace=normalize_whitespace,
    )

    if output_path:
        output_path.write_text(cleaned, encoding="utf-8")
        logger.info("Cleaned markdown written to %s", output_path)

    return cleaned, metadata


def batch_clean_markdown_files(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.md",
    remove_icon_blocks: bool = True,
    remove_page_comments: bool = False,
    remove_page_numbers: bool = True,
    remove_page_headers: bool = True,
    normalize_whitespace: bool = True,
) -> dict[Path, dict[int, dict]]:
    """Batch clean multiple markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: dict[Path, dict[int, dict]] = {}

    for input_file in input_dir.glob(pattern):
        output_file = output_dir / input_file.name
        logger.info("Cleaning %s -> %s", input_file, output_file)

        _cleaned, metadata = clean_markdown_file(
            input_file,
            output_path=output_file,
            remove_icon_blocks=remove_icon_blocks,
            remove_page_comments=remove_page_comments,
            remove_page_numbers=remove_page_numbers,
            remove_page_headers=remove_page_headers,
            normalize_whitespace=normalize_whitespace,
        )
        all_metadata[input_file] = metadata

    return all_metadata


@dataclass(slots=True)
class MarkdownChunk:
    """Represents a chunk of markdown content with metadata.

    Attributes:
        titles: Hierarchical path of headers (e.g., ["Chapter 1", "Section 1.1"])
        content: The actual markdown content with headers
        token_count: Number of tokens in the content
        page_number: Page number where this chunk originates (from <!-- Page N --> markers)
        chunk_index: Sequential index of this chunk in the document
        source_file: Path to the source markdown file
    """

    titles: List[str]
    content: str
    token_count: int
    page_number: int = 0
    chunk_index: int = 0
    source_file: Optional[str] = None


@dataclass(slots=True)
class HeaderNode:
    """Represents a header node in the markdown document tree."""

    level: int
    title: str
    content: str
    children: List["HeaderNode"] = field(default_factory=list)


class MarkdownChunker:
    """Chunk markdown files by headers with token limit enforcement."""

    def __init__(self, max_tokens: int = MAX_TOKENS, encoding: str = ENCODING):
        self.max_tokens = max_tokens
        self.encoding = encoding
        self._encoding_cache = None
        self._tokenizer_fallback_warned = False

    def _get_encoding(self):
        """Lazy-load tiktoken encoding."""
        if self._encoding_cache is None:
            import tiktoken

            self._encoding_cache = tiktoken.get_encoding(self.encoding)
        return self._encoding_cache

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken with an offline-safe fallback."""
        if not text:
            return 0
        try:
            encoding = self._get_encoding()
            return len(encoding.encode(text))
        except Exception as exc:  # pragma: no cover - exercised in offline/runtime environments
            if not self._tokenizer_fallback_warned:
                logger.warning(
                    "Falling back to approximate token counting for encoding %s: %s",
                    self.encoding,
                    exc,
                )
                self._tokenizer_fallback_warned = True
            return self._approximate_token_count(text)

    @staticmethod
    def _approximate_token_count(text: str) -> int:
        """Approximate tokens without network-dependent tokenizer downloads."""
        token_like_units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
        return len(token_like_units)

    def clean_markdown(self, text: str) -> str:
        """Clean markdown by removing YAML frontmatter and per-page metadata.

        This is a wrapper around the standalone clean_markdown function
        for backward compatibility. Returns only the cleaned text (not metadata).

        Args:
            text: Raw markdown text to clean

        Returns:
            Cleaned markdown text
        """
        cleaned, _metadata = clean_markdown(text)
        return cleaned

    def parse_headers(self, text: str) -> List[HeaderNode]:
        """Parse markdown headers into hierarchical tree structure."""
        headers = []
        for match in HEADER_PATTERN.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            headers.append((level, title, start_pos))

        if not headers:
            return [HeaderNode(level=0, title="", content=text.strip(), children=[])]

        root_nodes = []
        stack = []

        for i, (level, title, start_pos) in enumerate(headers):
            if i + 1 < len(headers):
                end_pos = headers[i + 1][2]
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()
            header_line = f"{'#' * level} {title}"
            content = content[len(header_line) :].strip()

            node = HeaderNode(level=level, title=title, content=content, children=[])

            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                root_nodes.append(node)

            stack.append(node)

        return root_nodes

    def _extract_page_from_content(self, content: str) -> tuple[int, str]:
        page_match = re.search(r"<!-- Page (\d+) -->", content)
        if page_match:
            return int(page_match.group(1)), re.sub(r"<!-- Page \d+ -->\n?", "", content)
        return 0, content

    def _chunk_recursive(
        self,
        node: HeaderNode,
        parent_titles: List[str],
        page_number: int = 0,
        source_file: Optional[str] = None,
    ) -> List[MarkdownChunk]:
        """Recursively chunk a header node and its children."""
        current_titles = parent_titles + [node.title] if node.title else parent_titles
        full_content = (
            f"{'#' * node.level} {node.title}\n\n{node.content}" if node.title else node.content
        )

        if not full_content.strip():
            return []

        extracted_page, cleaned_content = self._extract_page_from_content(full_content)
        if extracted_page > 0:
            page_number = extracted_page

        token_count = self.count_tokens(cleaned_content)

        if token_count <= self.max_tokens:
            return [
                MarkdownChunk(
                    titles=current_titles,
                    content=cleaned_content,
                    token_count=token_count,
                    page_number=page_number,
                    source_file=source_file,
                )
            ]

        if node.children:
            chunks = []
            for child in node.children:
                chunks.extend(
                    self._chunk_recursive(child, current_titles, page_number, source_file)
                )
            return chunks

        chunks: List[MarkdownChunk] = []
        paragraphs = node.content.split("\n\n")
        current_chunk_content = f"{'#' * node.level} {node.title}\n\n" if node.title else ""
        current_chunk_tokens = self.count_tokens(current_chunk_content) if node.title else 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            page_match = re.match(r"<!-- Page (\d+) -->", para)
            if page_match:
                page_number = int(page_match.group(1))
                continue

            para_tokens = self.count_tokens(para)

            if (
                current_chunk_tokens + para_tokens > self.max_tokens
                and current_chunk_content.strip()
            ):
                chunks.append(
                    MarkdownChunk(
                        titles=current_titles,
                        content=current_chunk_content.strip(),
                        token_count=current_chunk_tokens,
                        page_number=page_number,
                        source_file=source_file,
                    )
                )
                current_chunk_content = (
                    f"{'#' * node.level} {node.title}\n\n{para}\n\n"
                    if node.title
                    else f"{para}\n\n"
                )
                current_chunk_tokens = self.count_tokens(current_chunk_content)
            else:
                current_chunk_content += para + "\n\n"
                current_chunk_tokens += para_tokens

        if current_chunk_content.strip():
            chunks.append(
                MarkdownChunk(
                    titles=current_titles,
                    content=current_chunk_content.strip(),
                    token_count=self.count_tokens(current_chunk_content.strip()),
                    page_number=page_number,
                    source_file=source_file,
                )
            )

        return chunks

    def chunk_text(self, text: str, source_file: Optional[str] = None) -> List[MarkdownChunk]:
        """Chunk markdown text into structured pieces."""
        cleaned, _page_metadata = clean_markdown(text)
        headers = self.parse_headers(cleaned)

        all_chunks: List[MarkdownChunk] = []
        for header in headers:
            chunks = self._chunk_recursive(header, [], 0, source_file)
            all_chunks.extend(chunks)

        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx

        return all_chunks

    def chunk_file(self, file_path: Path) -> List[MarkdownChunk]:
        """Chunk a markdown file."""
        text = Path(file_path).read_text(encoding="utf-8")
        return self.chunk_text(text, source_file=str(file_path))

    def chunks_to_json(self, chunks: List[MarkdownChunk]) -> str:
        """Convert chunks to JSON string."""
        data = [
            {
                "titles": chunk.titles,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
            }
            for chunk in chunks
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def process_and_save(
        self,
        markdown_path: Path,
        output_path: Optional[Path] = None,
        chunk_dir: Optional[Path] = None,
    ) -> Path:
        """Process a markdown file and save chunks to JSON.

        Args:
            markdown_path: Path to markdown file
            output_path: Optional explicit output path
            chunk_dir: Optional directory for chunks (default: data/chunks)

        Returns:
            Path to JSON output file
        """
        markdown_path = Path(markdown_path)

        if output_path is None:
            if chunk_dir is None:
                chunk_dir = Path("data/chunks")
            chunk_dir = Path(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            output_path = chunk_dir / f"{markdown_path.stem}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = self.chunk_file(markdown_path)
        json_content = self.chunks_to_json(chunks)
        output_path.write_text(json_content, encoding="utf-8")

        logger.info(
            "Chunked %s -> %s (%s chunks)", markdown_path.name, output_path.name, len(chunks)
        )
        return output_path
