# OCR Page Results Markdown Design

**Goal:** Refactor OCR result composition so each PNG page keeps its own metadata and the final Markdown output preserves both document-level and page-level OCR metadata.

## Scope

- Introduce a page result object that carries OCR text plus per-page metadata.
- Change `OCRProcessor.extract_from_images()` to return page result objects instead of raw strings.
- Refactor `OCRProcessor.process_and_save()` so it only renders and saves Markdown from page results.
- Update `PDFPipeline.process_pdf()` to call the new two-step OCR flow.
- Keep CLI behavior and output file paths unchanged.

## Recommended Shape

Use one new dataclass in `src/qmrkg/png_to_text.py`:

```python
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

This is enough for the current requirement without adding a second document-level dataclass. Top-level
Markdown metadata can be derived when saving.

## OCR Flow

`extract_text()` stays as a compatibility helper that still returns text (or `(text, confidence)`).
It should continue to retry and raise on final OCR failure.

`extract_from_images()` should:

- normalize paths
- preserve caller-provided order
- assign `page_number` as the 1-based index of that order
- run per-page OCR concurrently
- return `list[OCRPageResult]`
- remove the old `return_confidence` parameter from this batch API because the page object now carries optional confidence directly
- convert final per-page failures into `OCRPageResult(status="failed", error=...)`

The worker path should move from `_extract_page_text_with_retries()` to something like
`_extract_page_result_with_retries(image_path, page_number)` so page metadata is collected in one place.
The lower-level request helper should return enough information to extract both text and token usage.

## Metadata Collection

For each page capture:

- `image_path`: exact PNG path used for OCR
- `page_number`: 1-based page index from the caller-provided order
- `text`: OCR output text
- `processed_at`: UTC ISO-8601 timestamp when the page finished processing
- `duration_seconds`: elapsed OCR time for the page
- `confidence`: optional, default `None`
- `model`: copy from `self.settings.model`
- `prompt_tokens`, `completion_tokens`, `total_tokens`: parse from `response.usage` when present
- `status`: `success` or `failed`
- `error`: formatted error summary when OCR fails

Do not add speculative metadata that is not already available from the request/response path.

## Markdown Output

Top-level output keeps YAML frontmatter. Derive `processed_at` as the latest non-empty page `processed_at` value, and derive `ocr_model` from the first non-empty page `model` value under the assumption that one batch uses one OCR model. If no page result has those values, omit the field.


```markdown
---
source: example.pdf
pages: 10
successful_pages: 9
failed_pages: 1
ocr_model: Qwen/Qwen3-VL-8B-Instruct
processed_at: 2026-03-18T12:00:00+00:00
---
```

Then render one section per page:

```markdown
## Page 1

**Image:** `data/png/example/example_page_0001.png`  
**Processed:** 2026-03-18T12:00:00+00:00  
**Duration:** 1.23s  
**Model:** `Qwen/Qwen3-VL-8B-Instruct`  
**Confidence:** 0.95  
**Prompt Tokens:** 120  
**Completion Tokens:** 450  
**Total Tokens:** 570

<raw OCR text>

---
```

On failure, render `Status` and `Error` lines and use `_No text extracted._` as the content placeholder.
If `confidence` or token usage is missing, omit those lines instead of printing placeholder values.

## `process_and_save()` Boundary

`process_and_save()` should stop triggering OCR itself. It should accept `list[OCRPageResult]`, derive
document-level metadata, render Markdown through a helper like `_render_markdown(page_results, pdf_source)`,
and write the final file.

This separates:

- OCR execution: `extract_from_images()`
- document rendering: `process_and_save()`

`process_and_save()` should no longer depend on `self.settings`, should not validate OCR configuration, and
should remove the old `page_separator` argument because the output is now structured sections.

## Pipeline Update

`src/qmrkg/pipeline.py` should change from:

```python
text_path = self.ocr_processor.process_and_save(sorted(image_paths), text_output_path, pdf_source=pdf_path.name)
```

to:

```python
sorted_paths = sorted(image_paths)
page_results = self.ocr_processor.extract_from_images(sorted_paths)
text_path = self.ocr_processor.process_and_save(page_results, text_output_path, pdf_source=pdf_path.name)
```

`PDFPipeline.process_pdf()` can keep its current return type `(image_paths, text_path)`.

## Optional Follow-Up

These are useful but not required for the minimal refactor:

- export `OCRPageResult` from `src/qmrkg/__init__.py`
- update `README.md` usage examples for direct OCR page-result workflows
- document the richer Markdown output format

## Testing

Update `tests/test_png_to_text.py` to cover:

- `extract_from_images()` returns ordered `OCRPageResult` objects
- page result includes `image_path`, `page_number`, `processed_at`, `duration_seconds`, and token usage
- failed OCR pages return a failed result object with `status` and `error`
- `extract_text()` still raises on final failure
- `process_and_save()` renders the new frontmatter and page metadata block
- `process_and_save()` renders failed pages deterministically with `_No text extracted._`

Replace tests that assume the old behavior:

- blank pages being skipped by separator-based output
- `process_and_save()` raising OCR configuration errors

Update `tests/test_pipeline.py` to cover:

- `process_pdf()` calls `extract_from_images()` before `process_and_save()`
- `process_and_save()` receives `list[OCRPageResult]`
