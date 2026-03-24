from pathlib import Path

import pytest


def test_pdftopng_single_pdf(monkeypatch, capsys):
    import qmrkg.cli_pdf_to_png as cli_pdf_to_png

    temp_dir = Path(".pytest_local/cli_pdftopng_single")
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    calls = {}

    class StubConverter:
        def __init__(self, dpi=200, fmt="png", output_dir=None):
            calls["init"] = {"dpi": dpi, "fmt": fmt, "output_dir": output_dir}

        def convert(self, path):
            calls["convert"] = Path(path)
            return [Path("sample_page_0001.png")]

    monkeypatch.setattr(cli_pdf_to_png, "PDFConverter", StubConverter)

    exit_code = cli_pdf_to_png.main(["--pdf", str(pdf_path), "--dpi", "300"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert calls["init"]["dpi"] == 300
    assert calls["convert"] == pdf_path
    assert "Processed: sample.pdf" in out


def test_pdftopng_directory_mode_passes_recursive(monkeypatch):
    import qmrkg.cli_pdf_to_png as cli_pdf_to_png

    temp_dir = Path(".pytest_local/cli_pdftopng_dir")
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "a.pdf").write_bytes(b"%PDF-1.4")

    calls = {}

    class StubConverter:
        def __init__(self, dpi=200, fmt="png", output_dir=None):
            calls["init"] = {"dpi": dpi, "fmt": fmt, "output_dir": output_dir}

        def convert_all(self, pdf_dir, recursive=False):
            calls["convert_all"] = {"pdf_dir": Path(pdf_dir), "recursive": recursive}
            return {"a.pdf": [Path("a_page_0001.png")]}

    monkeypatch.setattr(cli_pdf_to_png, "PDFConverter", StubConverter)

    exit_code = cli_pdf_to_png.main(["--pdf-dir", str(temp_dir), "--recursive"])

    assert exit_code == 0
    assert calls["convert_all"]["pdf_dir"] == temp_dir
    assert calls["convert_all"]["recursive"] is True


def test_pngtotext_single_image(monkeypatch, capsys):
    import qmrkg.cli_png_to_text as cli_png_to_text
    from qmrkg.png_to_text import OCRPageResult

    temp_dir = Path(".pytest_local/cli_pngtotext_single")
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / "page_1.png"
    image_path.write_bytes(b"fakepng")
    output_path = temp_dir / "result.md"

    calls = {}

    class StubProcessor:
        def __init__(self, use_gpu=False, lang="ch", show_log=False, config_path=None):
            calls["init"] = {
                "use_gpu": use_gpu,
                "lang": lang,
                "show_log": show_log,
                "config_path": config_path,
            }

        def extract_from_images(self, image_paths):
            calls["extract_from_images"] = [Path(p) for p in image_paths]
            return [
                OCRPageResult(
                    image_path=image_paths[0],
                    page_number=1,
                    text="ok",
                    processed_at="2026-01-01T00:00:00+00:00",
                    duration_seconds=0.1,
                )
            ]

        def process_and_save(self, page_results, output_path, pdf_source=None):
            calls["process_and_save"] = {"output_path": Path(output_path), "pdf_source": pdf_source}
            return Path(output_path)

    monkeypatch.setattr(cli_png_to_text, "OCRProcessor", StubProcessor)

    exit_code = cli_png_to_text.main(["--image", str(image_path), "--output", str(output_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert calls["extract_from_images"] == [image_path]
    assert calls["process_and_save"]["output_path"] == output_path
    assert "Saved markdown to:" in out


def test_pngtotext_missing_input_returns_error(capsys):
    import qmrkg.cli_png_to_text as cli_png_to_text

    exit_code = cli_png_to_text.main(["--image", "not-exist.png"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Image file not found" in captured.err


def test_mdchunk_single_markdown(monkeypatch, capsys):
    import qmrkg.cli_md_chunk as cli_md_chunk

    temp_dir = Path(".pytest_local/cli_mdchunk_single")
    temp_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = temp_dir / "doc.md"
    markdown_path.write_text("# title\n\ncontent", encoding="utf-8")
    chunk_dir = temp_dir / "chunks"

    calls = {}

    class StubChunker:
        def __init__(self, max_tokens=1500, encoding="cl100k_base"):
            calls["init"] = {"max_tokens": max_tokens, "encoding": encoding}

        def process_and_save(self, markdown_path, output_path=None, chunk_dir=None):
            calls["process_and_save"] = {
                "markdown_path": Path(markdown_path),
                "output_path": output_path,
                "chunk_dir": Path(chunk_dir),
            }
            return Path(chunk_dir) / "doc.json"

    monkeypatch.setattr(cli_md_chunk, "MarkdownChunker", StubChunker)

    exit_code = cli_md_chunk.main(
        ["--markdown", str(markdown_path), "--chunk-dir", str(chunk_dir), "--max-tokens", "2048"]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert calls["init"]["max_tokens"] == 2048
    assert calls["process_and_save"]["markdown_path"] == markdown_path
    assert calls["process_and_save"]["chunk_dir"] == chunk_dir
    assert "Chunked: doc.md" in out


def test_mdchunk_missing_markdown_returns_error(capsys):
    import qmrkg.cli_md_chunk as cli_md_chunk

    exit_code = cli_md_chunk.main(["--markdown", "missing.md"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Markdown file not found" in captured.err


def test_pngtotext_dir_outputs_to_book_subdir(monkeypatch, capsys, tmp_path):
    """Directory mode must write each page MD into a book-named subdirectory."""
    import qmrkg.cli_png_to_text as cli_png_to_text
    from qmrkg.png_to_text import OCRPageResult

    image_dir = tmp_path / "png"
    image_dir.mkdir()
    # Image filename with _page_ suffix so the book stem can be stripped
    image_path = image_dir / "mybook_page_0001.png"
    image_path.write_bytes(b"fakepng")

    text_dir = tmp_path / "markdown"
    captured_output_paths: list[Path] = []

    class StubProcessor:
        def __init__(self, **_kwargs):
            pass

        def extract_from_images(self, image_paths):
            return [
                OCRPageResult(
                    image_path=image_paths[0],
                    page_number=1,
                    text="content",
                    processed_at="2026-01-01T00:00:00+00:00",
                    duration_seconds=0.1,
                )
            ]

        def process_and_save(self, page_results, output_path, pdf_source=None):
            captured_output_paths.append(Path(output_path))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("# content", encoding="utf-8")
            return Path(output_path)

    monkeypatch.setattr(cli_png_to_text, "OCRProcessor", StubProcessor)

    exit_code = cli_png_to_text.main(
        ["--image-dir", str(image_dir), "--text-dir", str(text_dir)]
    )

    assert exit_code == 0
    assert len(captured_output_paths) == 1
    output = captured_output_paths[0]
    # Parent directory should be the book stem (no _page_NNNN suffix)
    assert output.parent.name == "mybook"
    assert output.name == "mybook_page_0001.md"


def test_mdchunk_merge_mode(monkeypatch, capsys, tmp_path):
    """--merge mode must merge per-page MDs and chunk each book subdirectory."""
    import qmrkg.cli_md_chunk as cli_md_chunk
    from qmrkg.markdown_chunker import MarkdownChunk

    markdown_dir = tmp_path / "markdown"
    book_dir = markdown_dir / "mybook"
    book_dir.mkdir(parents=True)
    chunk_dir = tmp_path / "chunks"

    # Create two fake per-page MD files with ```markdown fences
    (book_dir / "mybook_page_0001.md").write_text(
        "---\nsource: x\n---\n\n## Page 1\n\n```markdown\n# Chapter 1\n\nIntro text.\n```",
        encoding="utf-8",
    )
    (book_dir / "mybook_page_0002.md").write_text(
        "---\nsource: x\n---\n\n## Page 2\n\n```markdown\n## Section 1.1\n\nMore content.\n```",
        encoding="utf-8",
    )

    merge_calls: dict = {}
    chunk_calls: dict = {}

    original_merge = cli_md_chunk.merge_book_pages

    def stub_merge(page_files, output_path=None):
        merge_calls["files"] = [Path(f).name for f in page_files]
        merge_calls["output_path"] = output_path
        if output_path:
            Path(output_path).write_text("# Chapter 1\n\nMerged.", encoding="utf-8")
        return "# Chapter 1\n\nMerged."

    class StubChunker:
        def __init__(self, max_tokens=4000, encoding="cl100k_base"):
            pass

        def chunk_document(self, text, source_file=None):
            chunk_calls["text"] = text
            chunk_calls["source_file"] = source_file
            return [
                MarkdownChunk(
                    titles=["Chapter 1"],
                    content="# Chapter 1\n\nMerged.",
                    token_count=10,
                    chunk_index=0,
                    source_file=source_file,
                )
            ]

    monkeypatch.setattr(cli_md_chunk, "merge_book_pages", stub_merge)
    monkeypatch.setattr(cli_md_chunk, "MarkdownChunker", StubChunker)

    exit_code = cli_md_chunk.main(
        [
            "--merge",
            "--markdown-dir", str(markdown_dir),
            "--chunk-dir", str(chunk_dir),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    # merge_book_pages should have been called with both page files
    assert set(merge_calls["files"]) == {"mybook_page_0001.md", "mybook_page_0002.md"}
    # Merged MD saved to markdown_dir root
    assert merge_calls["output_path"] == markdown_dir / "mybook.md"
    # chunk_document should have been called
    assert "Chapter 1" in chunk_calls["text"] or chunk_calls["text"] == "# Chapter 1\n\nMerged."
    # JSON chunk file created
    assert (chunk_dir / "mybook.json").exists()
    assert "mybook" in out


def test_qmrkg_list_shows_available_commands(capsys):
    import qmrkg.cli_qmrkg as cli_qmrkg

    exit_code = cli_qmrkg.main(["--list"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Available commands:" in out
    assert "pdftopng" in out
    assert "pngtotext" in out
    assert "mdchunk" in out
    assert "kgextract" in out
    assert "kgmerge" in out
    assert "kgneo4j" in out
