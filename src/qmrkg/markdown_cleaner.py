"""Markdown cleaning module for preprocessing markdown files.

This module provides functions to clean markdown content by removing
YAML frontmatter, per-page metadata, icon blocks, page markers, and other
OCR artifacts.

Example:
    from qmrkg.markdown_cleaner import clean_markdown

    cleaned_text, metadata = clean_markdown(raw_markdown)
    print(f"Cleaned {len(metadata)} pages of metadata")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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


def clean_markdown(
    text: str,
    remove_icon_blocks: bool = True,
    remove_page_comments: bool = False,
    remove_page_numbers: bool = True,
    remove_page_headers: bool = True,
    normalize_whitespace: bool = True,
) -> tuple[str, dict[int, dict]]:
    """Clean markdown by removing OCR artifacts and formatting issues.

    Removes:
    - YAML frontmatter (--- to ---)
    - Per-page metadata lines (**Key:** Value)
    - Icon blocks (```icon: ... ```)
    - Page number symbols (• 115 •)
    - Page header/footer lines (e.g., "运输层 119")
    - Table of contents ellipsis
    - Optionally: HTML page comments (<!-- Page N -->)

    Args:
        text: Raw markdown text to clean
        remove_icon_blocks: Remove ```icon:...``` code blocks
        remove_page_comments: Remove <!-- Page N --> comments
        remove_page_numbers: Remove page number symbols like "• 115 •"
        remove_page_headers: Remove page header/footer lines
        normalize_whitespace: Collapse multiple empty lines

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
    """Clean a markdown file.

    Args:
        file_path: Path to the markdown file
        output_path: Optional path to write cleaned output
        remove_icon_blocks: Remove ```icon:...``` code blocks
        remove_page_comments: Remove <!-- Page N --> comments
        remove_page_numbers: Remove page number symbols
        remove_page_headers: Remove page header/footer lines
        normalize_whitespace: Collapse multiple empty lines

    Returns:
        Tuple of (cleaned_text, page_metadata_dict)
    """
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
        logger.info(f"Cleaned markdown written to {output_path}")

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
    """Batch clean multiple markdown files.

    Args:
        input_dir: Directory containing markdown files
        output_dir: Directory to write cleaned files
        pattern: File pattern to match (default: *.md)
        remove_icon_blocks: Remove ```icon:...``` code blocks
        remove_page_comments: Remove <!-- Page N --> comments
        remove_page_numbers: Remove page number symbols
        remove_page_headers: Remove page header/footer lines
        normalize_whitespace: Collapse multiple empty lines

    Returns:
        Dictionary mapping input paths to their metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: dict[Path, dict[int, dict]] = {}

    for input_file in input_dir.glob(pattern):
        output_file = output_dir / input_file.name
        logger.info(f"Cleaning {input_file} -> {output_file}")

        cleaned, metadata = clean_markdown_file(
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
