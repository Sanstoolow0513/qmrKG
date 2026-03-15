# SiliconFlow VLM OCR Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local PaddleOCR-based PNG-to-text stage with a SiliconFlow OpenAI-compatible VLM implementation that loads settings from `.env` and enforces configurable RPM-aware concurrency.

**Architecture:** Keep the existing synchronous `OCRProcessor` entry point in `src/qmrkg/png_to_text.py`, but reimplement its internals around `.env`-loaded settings, a synchronous OpenAI-compatible SiliconFlow client, a rolling-window rate limiter, and a thread-pooled batch executor. Update the pipeline and CLI only where needed to preserve compatibility while removing PaddleOCR-specific behavior.

**Tech Stack:** Python 3.13, `openai`, `python-dotenv`, `pytest`, `ThreadPoolExecutor`, `pathlib`

---

## Chunk 1: Test Harness And Dependency Surface

### Task 1: Add Tests For The New OCR Surface

**Files:**
- Create: `tests/test_png_to_text.py`
- Modify: `pyproject.toml`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing test for settings validation**

```python
def test_settings_require_api_key(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        VLMSettings.from_env()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_settings_require_api_key -v`
Expected: FAIL because `VLMSettings` does not exist yet

- [ ] **Step 3: Write the failing test for constructor compatibility**

```python
def test_ocr_processor_accepts_legacy_constructor_args(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    processor = OCRProcessor(use_gpu=True, lang="en", show_log=True)

    assert processor is not None
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_ocr_processor_accepts_legacy_constructor_args -v`
Expected: FAIL because the implementation still depends on PaddleOCR internals

- [ ] **Step 5: Write the failing test for `.env`-driven defaults**

```python
def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    settings = VLMSettings.from_env()

    assert settings.base_url == "https://api.siliconflow.com/v1"
    assert settings.model == "Qwen/Qwen2-VL-72B-Instruct"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_png_to_text.py -k "settings or constructor" -v`
Expected: FAIL with missing settings/model support

- [ ] **Step 7: Write minimal implementation for settings and constructor compatibility**

```python
@dataclass(slots=True)
class VLMSettings:
    api_key: str
    base_url: str = "https://api.siliconflow.com/v1"
    model: str = "Qwen/Qwen2-VL-72B-Instruct"
    ...

    @classmethod
    def from_env(cls) -> "VLMSettings":
        load_dotenv()
        ...


class OCRProcessor:
    def __init__(self, use_gpu: bool = False, lang: str = "ch", show_log: bool = False):
        self.use_gpu = use_gpu
        self.lang = lang
        self.show_log = show_log
        self._settings = None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_png_to_text.py -k "settings or constructor" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml tests/test_png_to_text.py src/qmrkg/png_to_text.py
git commit -m "test: add SiliconFlow OCR settings coverage"
```

## Chunk 2: Single-Image SiliconFlow Request Path

### Task 2: Replace PaddleOCR Single-Image Extraction

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing test for request payload construction**

```python
def test_extract_text_sends_openai_compatible_vision_request(tmp_path, monkeypatch):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-image")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    fake_client = FakeOpenAIClient("recognized text")
    processor = OCRProcessor()
    processor._client = fake_client

    text = processor.extract_text(image_path)

    assert text == "recognized text"
    assert fake_client.last_model == "Qwen/Qwen2-VL-72B-Instruct"
    assert fake_client.last_messages[0]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_extract_text_sends_openai_compatible_vision_request -v`
Expected: FAIL because `extract_text()` still calls PaddleOCR

- [ ] **Step 3: Write the failing test for `return_confidence=True` compatibility**

```python
def test_extract_text_returns_compatibility_confidence(tmp_path, monkeypatch):
    ...
    result = processor.extract_text(image_path, return_confidence=True)
    assert result == ("recognized text", 1.0)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_extract_text_returns_compatibility_confidence -v`
Expected: FAIL because the new compatibility behavior is not implemented yet

- [ ] **Step 5: Write minimal implementation for base64 encoding and OpenAI request handling**

```python
def _build_image_data_url(image_path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")


def extract_text(self, image_path: Path, return_confidence: bool = False):
    text = self._extract_text_with_client(Path(image_path))
    if return_confidence:
        return text, 1.0
    return text
```

- [ ] **Step 6: Run targeted tests to verify they pass**

Run: `pytest tests/test_png_to_text.py -k "vision_request or compatibility_confidence" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/qmrkg/png_to_text.py tests/test_png_to_text.py
git commit -m "feat: add SiliconFlow single-image OCR requests"
```

## Chunk 3: RPM-Constrained Concurrent Batch Execution

### Task 3: Add Rate Limiting, Retry Logic, And Ordered Batch Processing

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing test for ordered batch results**

```python
def test_extract_from_images_preserves_input_order(tmp_path, monkeypatch):
    ...
    results = processor.extract_from_images([page2, page1])
    assert results == ["text for page2", "text for page1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_preserves_input_order -v`
Expected: FAIL because batch handling is still serial or unordered

- [ ] **Step 3: Write the failing test for retry on transient failure**

```python
def test_extract_from_images_retries_transient_failures(tmp_path, monkeypatch):
    ...
    assert fake_client.call_count == 2
    assert results == ["recovered text"]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_extract_from_images_retries_transient_failures -v`
Expected: FAIL because retry logic does not exist yet

- [ ] **Step 5: Write the failing test for RPM limiter**

```python
def test_rate_limiter_blocks_requests_over_rpm(monkeypatch):
    limiter = RollingRateLimiter(rpm=2, time_fn=fake_time, sleep_fn=fake_sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert fake_sleep.calls == [30.0]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_rate_limiter_blocks_requests_over_rpm -v`
Expected: FAIL because the limiter does not exist yet

- [ ] **Step 7: Write minimal implementation for rate limiting, retries, and thread-pooled batch execution**

```python
class RollingRateLimiter:
    def acquire(self) -> None:
        ...


def extract_from_images(self, image_paths: list[Path], return_confidence: bool = False):
    with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as executor:
        ...
```

- [ ] **Step 8: Run targeted tests to verify they pass**

Run: `pytest tests/test_png_to_text.py -k "preserves_input_order or retries_transient_failures or rate_limiter" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/qmrkg/png_to_text.py tests/test_png_to_text.py
git commit -m "feat: add rate-limited batch SiliconFlow OCR"
```

## Chunk 4: Pipeline, CLI, And Output Integration

### Task 4: Update Pipeline And CLI Compatibility

**Files:**
- Modify: `src/qmrkg/pipeline.py`
- Modify: `main.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test for pipeline compatibility**

```python
def test_pipeline_still_passes_legacy_ocr_args(tmp_path):
    pipeline = PDFPipeline(pdf_dir=tmp_path, ocr_lang="en", use_gpu=True)
    assert pipeline.ocr_processor.lang == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_still_passes_legacy_ocr_args -v`
Expected: FAIL because there are no compatibility tests yet

- [ ] **Step 3: Write the failing test for CLI accepted-but-ignored flags**

```python
def test_main_accepts_lang_and_gpu_flags(monkeypatch, tmp_path, capsys):
    ...
    exit_code = main()
    assert exit_code in (None, 0)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_main_accepts_lang_and_gpu_flags -v`
Expected: FAIL until CLI compatibility is pinned

- [ ] **Step 5: Write minimal implementation to keep pipeline and CLI compatible**

```python
parser.add_argument("--lang", ...)
parser.add_argument("--gpu", ...)
# help text notes these are retained for compatibility
```

- [ ] **Step 6: Run targeted tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/test_main.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add main.py src/qmrkg/pipeline.py tests/test_pipeline.py tests/test_main.py
git commit -m "feat: preserve pipeline and CLI compatibility for SiliconFlow OCR"
```

## Chunk 5: File Output Semantics And Final Verification

### Task 5: Preserve Page-Separator Output Semantics And Verify The End-To-End Surface

**Files:**
- Modify: `src/qmrkg/png_to_text.py`
- Modify: `.gitignore`
- Create: `.env.example`
- Test: `tests/test_png_to_text.py`
- Test: `README.md`

- [ ] **Step 1: Write the failing test for `process_and_save()` separator behavior**

```python
def test_process_and_save_skips_blank_pages_but_keeps_numbering_behavior(tmp_path, monkeypatch):
    ...
    content = output_path.read_text(encoding="utf-8")
    assert "--- Page 2 ---" not in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_process_and_save_skips_blank_pages_but_keeps_numbering_behavior -v`
Expected: FAIL until the SiliconFlow path preserves current semantics

- [ ] **Step 3: Write the failing test for `.env.example` completeness**

```python
def test_env_example_lists_siliconflow_variables():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "SILICONFLOW_API_KEY=" in content
    assert "SILICONFLOW_RPM=" in content
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_env_example_lists_siliconflow_variables -v`
Expected: FAIL because the example file does not exist yet

- [ ] **Step 5: Write minimal implementation for output semantics, env example, and docs**

```python
if isinstance(text, str) and text.strip():
    all_texts.append(f"{page_separator.format(page=i)}{text}")
```

- [ ] **Step 6: Run the focused tests to verify they pass**

Run: `pytest tests/test_png_to_text.py -k "process_and_save or env_example" -v`
Expected: PASS

- [ ] **Step 7: Run full verification**

Run: `pytest -v`
Expected: PASS

Run: `ruff check .`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add .env.example .gitignore README.md src/qmrkg/png_to_text.py tests
git commit -m "docs: document SiliconFlow OCR configuration"
```
