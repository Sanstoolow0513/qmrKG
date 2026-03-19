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

from .markdown_cleaner import clean_markdown, HEADER_PATTERN

logger = logging.getLogger(__name__)

MAX_TOKENS = 1500
ENCODING = "cl100k_base"

# Re-export for backward compatibility
__all__ = [
    "MarkdownChunk",
    "MarkdownChunker",
    "HeaderNode",
    "MAX_TOKENS",
    "ENCODING",
    "HEADER_PATTERN",
    "clean_markdown",
]


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

    def _get_encoding(self):
        """Lazy-load tiktoken encoding."""
        if self._encoding_cache is None:
            import tiktoken

            self._encoding_cache = tiktoken.get_encoding(self.encoding)
        return self._encoding_cache

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        encoding = self._get_encoding()
        return len(encoding.encode(text))

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
