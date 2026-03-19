# OCR Page Results Markdown Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor OCR result handling so page-level metadata survives concurrent OCR and is rendered into the final Markdown output.

**Architecture:** Keep `OCRProcessor` as the single OCR module, add one `OCRPageResult` dataclass there, and split OCR execution from Markdown rendering. `extract_from_images()` becomes a page-object API with no `return_confidence` flag, and `PDFPipeline` only adapts to that new return type while keeping its external workflow unchanged.

**Tech Stack:** Python 3.13, pytest, dataclasses, pathlib, openai-compatible SiliconFlow client

---

## Chunk 1: Page Result Model And OCR Return Shape

### Task 1: Add failing tests for page result objects

**Files:**
- Modify: `tests/test_png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_from_images_returns_page_results_in_input_order(...):
    results = processor.extract_from_images([page1, page2])

    assert [result.page_number for result in results] == [1, 2]
    assert results[0].image_path == page1
    assert results[0].text == "text for page1.png"
    assert results[0].status == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_returns_page_results_in_input_order -v`
Expected: FAIL because `extract_from_images()` still returns `list[str]` and still exposes the old compatibility shape.

- [ ] **Step 3: Write minimal implementation**

```python
def extract_from_images(self, image_paths: list[Path]) -> list[OCRPageResult]:
    ...

@dataclass(slots=True)
class OCRPageResult:
    image_path: Path
    page_number: int
    text: str
    processed_at: str
    duration_seconds: float
    confidence: float | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: str = "success"
    error: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_returns_page_results_in_input_order -v`
Expected: PASS

### Task 2: Collect page metadata during OCR

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Modify: `tests/test_png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write failing tests for metadata capture**

```python
def test_extract_from_images_captures_model_timestamp_duration_and_usage(...):
    result = processor.extract_from_images([page1])[0]

    assert result.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert result.processed_at.endswith("+00:00")
    assert result.duration_seconds >= 0
    assert result.total_tokens == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_captures_model_timestamp_duration_and_usage -v`
Expected: FAIL because the metadata is not collected yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _extract_page_result_with_retries(self, image_path: Path, page_number: int) -> OCRPageResult:
    ...

def _extract_page_result(self, image_path: Path, page_number: int) -> OCRPageResult:
    started = time.perf_counter()
    response = self.client.chat.completions.create(...)
    text = self._extract_message_text(response).strip()
    return OCRPageResult(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_captures_model_timestamp_duration_and_usage -v`
Expected: PASS

### Task 3: Represent page failures explicitly while keeping single-page API behavior

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Modify: `tests/test_png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_from_images_returns_failed_page_result_on_final_failure(...):
    result = processor.extract_from_images([page1])[0]

    assert result.status == "failed"
    assert result.text == ""
    assert "status_code=400" in result.error


def test_extract_text_still_raises_on_final_failure(...):
    with pytest.raises(FakeAPIStatusError):
        processor.extract_text(page1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_returns_failed_page_result_on_final_failure tests/test_png_to_text.py::test_extract_text_still_raises_on_final_failure -v`
Expected: FAIL because failures still return empty strings in the batch path and behavior is not pinned for `extract_text()`.

- [ ] **Step 3: Write minimal implementation**

```python
except Exception as exc:
    results[index] = OCRPageResult(
        image_path=image_path,
        page_number=index + 1,
        text="",
        processed_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=0.0,
        status="failed",
        model=self.settings.model,
        error=self._format_exception_summary(exc),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_returns_failed_page_result_on_final_failure tests/test_png_to_text.py::test_extract_text_still_raises_on_final_failure -v`
Expected: PASS

## Chunk 2: Markdown Rendering Boundary

### Task 4: Refactor `process_and_save()` to accept page results

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Modify: `tests/test_png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing test**

```python
def test_process_and_save_renders_frontmatter_and_page_sections(scratch_dir):
    page_results = [
        OCRPageResult(... page_number=1, text="first page"),
        OCRPageResult(... page_number=2, text="second page", confidence=0.95),
    ]

    processor.process_and_save(page_results, output_path, pdf_source="sample.pdf")

    content = output_path.read_text(encoding="utf-8")
    assert "source: sample.pdf" in content
    assert "## Page 1" in content
    assert "**Image:**" in content
    assert "**Confidence:** 0.95" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_process_and_save_renders_frontmatter_and_page_sections -v`
Expected: FAIL because `process_and_save()` still expects image paths and runs OCR internally.

- [ ] **Step 3: Write minimal implementation**

```python
def process_and_save(self, page_results: list[OCRPageResult], output_path: Path, pdf_source: str | None = None) -> Path:
    content = self._render_markdown(page_results, pdf_source)
    # processed_at = latest page processed_at
    # ocr_model = first non-empty page model
    output_path.write_text(content, encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_png_to_text.py::test_process_and_save_renders_frontmatter_and_page_sections -v`
Expected: PASS

### Task 5: Cover failed-page rendering and remove old assumptions

**Files:**
- Modify: `tests/test_png_to_text.py`
- Modify: `src/qmrkg/png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Replace outdated tests**

Remove or rewrite tests that assume:
- `process_and_save()` skips blank pages with separator-based output
- `process_and_save()` raises OCR configuration errors

- [ ] **Step 2: Write the failing test for failed-page rendering**

```python
def test_process_and_save_renders_failed_page_metadata(scratch_dir):
    failed_page = OCRPageResult(... status="failed", error="bad request")
    processor.process_and_save([failed_page], output_path, pdf_source="sample.pdf")
    content = output_path.read_text(encoding="utf-8")
    assert "**Status:** failed" in content
    assert "**Error:** bad request" in content
    assert "_No text extracted._" in content
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_process_and_save_renders_failed_page_metadata -v`
Expected: FAIL because no failure-section rendering exists.

- [ ] **Step 4: Write minimal implementation**

```python
if page_result.status != "success":
    section_lines.append(f"**Status:** {page_result.status}  ")
    section_lines.append(f"**Error:** {page_result.error or 'unknown'}")
    section_lines.append("")
    section_lines.append("_No text extracted._")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_png_to_text.py::test_process_and_save_renders_failed_page_metadata -v`
Expected: PASS

## Chunk 3: Pipeline Adaptation And Optional API Cleanup

### Task 6: Update pipeline to use the new OCR flow

**Files:**
- Modify: `src/qmrkg/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_process_pdf_extracts_page_results_before_saving(monkeypatch):
    page_results = [OCRPageResult(...)]
    monkeypatch.setattr(pipeline.ocr_processor, "extract_from_images", lambda image_paths: page_results)
    monkeypatch.setattr(pipeline.ocr_processor, "process_and_save", lambda page_results, output_path, pdf_source=None: output_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_process_pdf_extracts_page_results_before_saving -v`
Expected: FAIL because the pipeline still passes image paths directly into `process_and_save()`.

- [ ] **Step 3: Write minimal implementation**

```python
sorted_paths = sorted(image_paths)
page_results = self.ocr_processor.extract_from_images(sorted_paths)
text_path = self.ocr_processor.process_and_save(page_results, text_output_path, pdf_source=pdf_path.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py::test_process_pdf_extracts_page_results_before_saving -v`
Expected: PASS

### Task 7: Optional package export and docs follow-up

**Files:**
- Modify: `src/qmrkg/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Export `OCRPageResult` from package root if direct callers need it**

```python
from .png_to_text import OCRProcessor, OCRPageResult
__all__ = ["PDFPipeline", "PDFConverter", "OCRProcessor", "OCRPageResult"]
```

- [ ] **Step 2: Update README examples and output format if public docs should match the new API**

```python
page_results = ocr.extract_from_images(image_paths)
ocr.process_and_save(page_results, Path("data/markdown/document.md"), pdf_source="document.pdf")
```

- [ ] **Step 3: Run targeted verification**

Run: `pytest tests/test_png_to_text.py tests/test_pipeline.py -v`
Expected: PASS
