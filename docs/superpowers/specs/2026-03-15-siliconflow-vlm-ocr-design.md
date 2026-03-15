# SiliconFlow VLM OCR Design

## Context

The current PNG-to-text stage in `src/qmrkg/png_to_text.py` uses local PaddleOCR and processes
images serially. The user wants to replace the local OCR dependency with a direct SiliconFlow
VLM API integration using the OpenAI-compatible interface, with configuration stored in `.env`
and request throughput constrained by RPM.

## Goals

- Replace PaddleOCR with a direct SiliconFlow VLM API implementation.
- Keep the existing synchronous `OCRProcessor` public shape so the rest of the pipeline changes
  minimally.
- Load API settings and tuning parameters from `.env`.
- Support concurrent page processing while enforcing a configurable RPM cap.
- Preserve deterministic output ordering and file-writing behavior.
- Make the new implementation testable without live network access.

## Non-Goals

- Supporting both PaddleOCR and VLM providers at runtime.
- Building a generic provider abstraction for multiple VLM vendors.
- Preserving GPU or local OCR flags as meaningful runtime behavior.

## Proposed Design

### Module Boundaries

`src/qmrkg/png_to_text.py` remains the orchestration point for the PNG-to-text stage and is
rewritten around three focused responsibilities:

1. `VLMSettings` dataclass
   - Loads `.env` and environment variables.
   - Validates required settings and numeric limits.
   - Holds API key, base URL, model, prompt, timeout, retry count, max concurrency, and RPM.

2. OpenAI-compatible SiliconFlow client wrapper
   - Encodes local PNG files as base64 data URLs.
   - Builds the `chat.completions.create(...)` request in the format documented by SiliconFlow.
   - Extracts response text from the first completion choice.

3. Batch executor
   - Applies both a concurrency cap and an RPM limiter.
   - Retries transient failures such as timeouts, HTTP 429, and HTTP 5xx.
   - Returns page results in input order regardless of completion order.

The public `OCRProcessor` class remains the entry point for `extract_text`,
`extract_from_images`, and `process_and_save`. Internally it delegates to the VLM settings,
client, and batch executor.

### Compatibility Boundary

The compatibility target is the current synchronous interface, not byte-for-byte behavioral parity
with PaddleOCR internals.

- `OCRProcessor.__init__(use_gpu=False, lang="ch", show_log=False)` remains accepted so existing
  callers do not break. These arguments become compatibility placeholders and are ignored by the
  SiliconFlow implementation.
- `extract_text(image_path, return_confidence=False)` remains synchronous.
- `extract_from_images(image_paths, return_confidence=False)` remains synchronous.
- `return_confidence=True` remains supported for compatibility and returns `(text, 1.0)` when text
  extraction succeeds, or `("", 0.0)` for failed batch items. The VLM provider does not expose a
  meaningful OCR confidence score, so this value is documented as a compatibility sentinel rather
  than a model confidence metric.

### Configuration

The implementation reads configuration from `.env` via `python-dotenv`. Supported variables:

- `SILICONFLOW_API_KEY`
- `SILICONFLOW_BASE_URL` default `https://api.siliconflow.com/v1`
- `SILICONFLOW_VLM_MODEL` default `Qwen/Qwen2-VL-72B-Instruct`
- `SILICONFLOW_RPM`
- `SILICONFLOW_MAX_CONCURRENCY`
- `SILICONFLOW_TIMEOUT_SECONDS`
- `SILICONFLOW_MAX_RETRIES`
- `SILICONFLOW_VLM_PROMPT`

`SILICONFLOW_API_KEY` is required. The prompt defaults to an OCR-oriented instruction that asks
the model to transcribe all visible text faithfully and avoid commentary.

### Runtime Behavior

For a single image, `extract_text()`:

1. Validates that the PNG file exists.
2. Loads settings and client lazily on first use.
3. Converts the image to a base64 data URL.
4. Sends a SiliconFlow OpenAI-compatible chat completion request.
5. Returns the extracted text string.

For multiple images, `extract_from_images()`:

1. Builds one work item per page.
2. Uses a `ThreadPoolExecutor` to cap active requests to `SILICONFLOW_MAX_CONCURRENCY`.
3. Uses a per-process rate limiter to ensure no more than `SILICONFLOW_RPM` requests are started
   during any rolling 60-second window.
4. Retries transient failures with exponential backoff.
5. Returns results aligned to the original image order.

`process_and_save()` preserves the current page separator behavior and writes the merged text file
to disk in UTF-8.

### Sync/Concurrency Boundary

The public API stays synchronous. Internally, batch work uses the synchronous OpenAI-compatible
client from the `openai` package and parallelizes requests with `ThreadPoolExecutor`. This avoids
introducing `asyncio.run()` into synchronous methods and keeps behavior safe for callers that may
already run inside an event loop.

### CLI and Pipeline Changes

`src/qmrkg/pipeline.py` continues to instantiate `OCRProcessor`, but no longer relies on local OCR
behavior. `main.py` is updated to remove or reword options that imply PaddleOCR or GPU-backed OCR.
`--lang` and `--gpu` remain accepted for compatibility in this change and are documented as ignored
by the SiliconFlow-backed implementation. They can be removed in a later cleanup once callers have
been migrated.

### Dependencies

`pyproject.toml` is updated to:

- remove `paddlepaddle`
- remove `paddleocr`
- add `openai`
- add `python-dotenv`

### Error Handling

The implementation distinguishes:

- Configuration errors: raised immediately with clear messages.
- Transient API failures: retried up to `SILICONFLOW_MAX_RETRIES`.
- Per-page permanent failures: logged and converted to empty page text in batch mode so a single
  page does not abort the whole document unless the caller is processing a single image directly.

Blank pages and failed pages continue to follow the current `process_and_save()` behavior: a page
separator is only written when the page text is non-empty after stripping. This preserves existing
output semantics even though it means page numbering in the output file reflects only retained
pages.

### Testing Strategy

Tests are added under `tests/` and avoid live API calls by mocking the OpenAI client:

- settings loading and validation
- request payload construction for a single image
- ordered results across concurrent page processing
- retry behavior for transient failures
- rate limiter behavior
- `process_and_save()` output formatting and persistence
- constructor compatibility for ignored `use_gpu`, `lang`, and `show_log` arguments
- CLI compatibility for accepted-but-ignored `--lang` and `--gpu` flags

## Tradeoffs

Keeping `OCRProcessor` preserves pipeline compatibility but leaves a legacy class name that no
longer reflects a traditional OCR engine. That is acceptable here because minimizing surface-area
change is more valuable than renaming the public entry point in the same refactor.

Using a thread pool adds some complexity, but it satisfies both concurrency and RPM constraints
while preserving the existing synchronous API surface.

## Implementation Notes

- Prefer the synchronous `OpenAI` client inside a thread pool so external callers remain fully
  synchronous.
- Keep the rate limiter local to `png_to_text.py`; no global shared process manager is needed.
- Default prompt should bias toward verbatim transcription and preserve line breaks where possible.
- Tests should use fake clocks or patchable time sources where needed so RPM assertions stay fast.
