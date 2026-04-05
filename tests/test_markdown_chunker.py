"""Tests for markdown chunking functionality."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qmrkg.markdown_chunker import (
    HEADER_PATTERN,
    ENCODING,
    MAX_TOKENS,
    HeaderNode,
    MarkdownChunk,
    MarkdownChunker,
)


class TestMarkdownChunk:
    """Tests for MarkdownChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a MarkdownChunk."""
        chunk = MarkdownChunk(
            titles=["Chapter 1", "Section 1.1"],
            content="Test content",
            token_count=100,
        )
        assert chunk.titles == ["Chapter 1", "Section 1.1"]
        assert chunk.content == "Test content"
        assert chunk.token_count == 100

    def test_chunk_with_empty_titles(self):
        """Test chunk with empty titles list."""
        chunk = MarkdownChunk(titles=[], content="Content", token_count=50)
        assert chunk.titles == []


class TestHeaderNode:
    """Tests for HeaderNode dataclass."""

    def test_node_creation(self):
        """Test creating a HeaderNode."""
        node = HeaderNode(level=1, title="Title", content="Content", children=[])
        assert node.level == 1
        assert node.title == "Title"
        assert node.content == "Content"
        assert node.children == []

    def test_node_with_children(self):
        """Test HeaderNode with children."""
        child = HeaderNode(level=2, title="Child", content="Child content", children=[])
        parent = HeaderNode(level=1, title="Parent", content="Parent content", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Child"


class TestMarkdownChunkerInit:
    """Tests for MarkdownChunker initialization."""

    def test_default_init(self):
        """Test default initialization."""
        chunker = MarkdownChunker()
        assert chunker.max_tokens == MAX_TOKENS
        assert chunker.encoding == ENCODING

    def test_custom_init(self):
        """Test initialization with custom values."""
        chunker = MarkdownChunker(max_tokens=2000, encoding="gpt2")
        assert chunker.max_tokens == 2000
        assert chunker.encoding == "gpt2"


class TestCountTokens:
    """Tests for token counting."""

    def test_count_english(self):
        """Test counting English text tokens."""
        chunker = MarkdownChunker()
        tokens = chunker.count_tokens("Hello world")
        assert tokens > 0

    def test_count_cjk(self):
        """Test counting CJK text tokens."""
        chunker = MarkdownChunker()
        tokens = chunker.count_tokens("你好世界")
        assert tokens > 0

    def test_count_empty(self):
        """Test counting empty string."""
        chunker = MarkdownChunker()
        tokens = chunker.count_tokens("")
        assert tokens == 0


class TestCleanMarkdown:
    """Tests for markdown cleaning."""

    def test_remove_yaml_frontmatter(self):
        """Test removing YAML frontmatter."""
        chunker = MarkdownChunker()
        text = "---\nsource: test.pdf\npages: 10\n---\n\n# Content"
        result = chunker.clean_markdown(text)
        assert "---" not in result
        assert "source:" not in result
        assert "# Content" in result

    def test_remove_metadata_lines(self):
        """Test removing metadata lines."""
        chunker = MarkdownChunker()
        text = "## Page 1\n\n**Image:** path.png\n**Processed:** time\n\nContent"
        result = chunker.clean_markdown(text)
        assert "**Image:**" not in result
        assert "**Processed:**" not in result
        assert "Content" in result

    def test_convert_page_markers(self):
        """Test converting page markers to HTML comments."""
        chunker = MarkdownChunker()
        text = "## Page 5\n\nContent"
        result = chunker.clean_markdown(text)
        assert "## Page 5" not in result
        assert "<!-- Page 5 -->" in result

    def test_preserve_actual_content(self):
        """Test that actual content is preserved."""
        chunker = MarkdownChunker()
        text = "# Title\n\nParagraph 1\n\nParagraph 2"
        result = chunker.clean_markdown(text)
        assert "# Title" in result
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result


class TestParseHeaders:
    """Tests for header parsing."""

    def test_parse_single_header(self):
        """Test parsing single header."""
        chunker = MarkdownChunker()
        text = "# Title\n\nContent"
        headers = chunker.parse_headers(text)
        assert len(headers) == 1
        assert headers[0].level == 1
        assert headers[0].title == "Title"

    def test_parse_multiple_headers(self):
        """Test parsing multiple headers."""
        chunker = MarkdownChunker()
        text = "# H1\n\nContent 1\n\n## H2\n\nContent 2"
        headers = chunker.parse_headers(text)
        assert len(headers) == 1
        assert headers[0].level == 1
        assert headers[0].title == "H1"
        assert len(headers[0].children) == 1
        assert headers[0].children[0].level == 2
        assert headers[0].children[0].title == "H2"

    def test_parse_no_headers(self):
        """Test parsing text with no headers."""
        chunker = MarkdownChunker()
        text = "Just some text without headers"
        headers = chunker.parse_headers(text)
        assert len(headers) == 1
        assert headers[0].level == 0
        assert headers[0].title == ""
        assert "Just some text" in headers[0].content

    def test_parse_nested_headers(self):
        """Test parsing nested header structure."""
        chunker = MarkdownChunker()
        text = """# Chapter 1
Content 1
## Section 1.1
Content 2
### Subsection 1.1.1
Content 3
## Section 1.2
Content 4"""
        headers = chunker.parse_headers(text)
        assert len(headers) == 1
        assert headers[0].title == "Chapter 1"
        assert len(headers[0].children) == 2
        assert headers[0].children[0].title == "Section 1.1"
        assert len(headers[0].children[0].children) == 1
        assert headers[0].children[0].children[0].title == "Subsection 1.1.1"


class TestChunkText:
    """Tests for the main chunk_text method."""

    def test_chunk_simple_document(self):
        """Test chunking a simple document."""
        chunker = MarkdownChunker()
        text = "# Title\n\nThis is content."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert chunks[0].titles == ["Title"]
        assert "This is content" in chunks[0].content

    def test_chunk_returns_valid_token_counts(self):
        """Test that all chunks have valid token counts."""
        chunker = MarkdownChunker(max_tokens=4000)
        text = "# Title\n\nParagraph 1\n\nParagraph 2"
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count <= 4000

    def test_chunk_preserves_header_hierarchy(self):
        """Test that chunk preserves header hierarchy in titles."""
        chunker = MarkdownChunker()
        text = "# Chapter\n\nContent\n\n## Section\n\nMore content"
        chunks = chunker.chunk_text(text)
        # Should have chunks with proper title hierarchies
        for chunk in chunks:
            if "Section" in chunk.titles:
                assert "Chapter" in chunk.titles


class TestChunksToJson:
    """Tests for JSON output conversion."""

    def test_chunks_to_json_format(self):
        """Test JSON output format."""
        chunker = MarkdownChunker()
        chunks = [
            MarkdownChunk(titles=["Title"], content="Content", token_count=10),
        ]
        json_str = chunker.chunks_to_json(chunks)
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["titles"] == ["Title"]
        assert data[0]["content"] == "Content"
        assert data[0]["token_count"] == 10

    def test_chunks_to_json_unicode(self):
        """Test JSON output with Unicode characters."""
        chunker = MarkdownChunker()
        chunks = [
            MarkdownChunk(titles=["中文标题"], content="中文内容", token_count=10),
        ]
        json_str = chunker.chunks_to_json(chunks)
        assert "中文标题" in json_str
        assert "中文内容" in json_str


class TestChunkFile:
    """Tests for file-based chunking."""

    def test_chunk_file(self, tmp_path):
        """Test chunking a file."""
        chunker = MarkdownChunker()
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nContent", encoding="utf-8")

        chunks = chunker.chunk_file(md_file)
        assert len(chunks) >= 1
        assert chunks[0].titles == ["Title"]

    def test_chunk_file_with_yaml(self, tmp_path):
        """Test chunking a file with YAML frontmatter."""
        chunker = MarkdownChunker()
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "---\nsource: test.pdf\n---\n\n# Title\n\nContent",
            encoding="utf-8",
        )

        chunks = chunker.chunk_file(md_file)
        assert len(chunks) >= 1
        # YAML should be removed
        for chunk in chunks:
            assert "source:" not in chunk.content


class TestProcessAndSave:
    """Tests for process_and_save method."""

    def test_process_and_save_creates_json(self, tmp_path):
        """Test that process_and_save creates JSON file."""
        chunker = MarkdownChunker()
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nContent", encoding="utf-8")

        output_path = chunker.process_and_save(md_file, chunk_dir=tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_process_and_save_content(self, tmp_path):
        """Test that saved JSON contains correct content."""
        chunker = MarkdownChunker()
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nContent", encoding="utf-8")

        output_path = chunker.process_and_save(md_file, chunk_dir=tmp_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 1


class TestHeaderPattern:
    """Tests for the header regex pattern."""

    def test_header_pattern_matches_h1(self):
        """Test pattern matches H1 headers."""
        match = HEADER_PATTERN.match("# Title")
        assert match is not None
        assert match.group(1) == "#"
        assert match.group(2) == "Title"

    def test_header_pattern_matches_h6(self):
        """Test pattern matches H6 headers."""
        match = HEADER_PATTERN.match("###### Deep nested")
        assert match is not None
        assert match.group(1) == "######"
        assert match.group(2) == "Deep nested"

    def test_header_pattern_no_match_in_text(self):
        """Test pattern doesn't match text with # in middle."""
        match = HEADER_PATTERN.match("This is # not a header")
        assert match is None


class TestIntegration:
    """Integration tests for the complete chunking flow."""

    def test_full_pipeline_with_complex_document(self, tmp_path):
        """Test complete pipeline with a complex document."""
        chunker = MarkdownChunker(max_tokens=100)  # Small limit to force splitting

        text = """---
source: test.pdf
pages: 5
---

## Page 1

**Image:** path.png
**Processed:** 2024-01-01

# Chapter 1

This is the first chapter with some content that might be long enough to trigger chunking if we set a low token limit.

## Section 1.1

This is a subsection with even more content to ensure we have enough tokens.

### Subsection 1.1.1

Deep nested content here.

## Page 2

**Image:** path2.png

# Chapter 2

Second chapter content.
"""

        md_file = tmp_path / "complex.md"
        md_file.write_text(text, encoding="utf-8")

        chunks = chunker.chunk_file(md_file)

        # Verify chunks were created
        assert len(chunks) > 0

        # Verify all chunks are under limit
        for chunk in chunks:
            assert chunk.token_count <= 100

        # Verify titles are preserved
        chapter_chunks = [c for c in chunks if "Chapter 1" in c.titles or "Chapter 2" in c.titles]
        assert len(chapter_chunks) > 0
