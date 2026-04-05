# SiliconFlow Official VLM Defaults Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the OCR defaults to SiliconFlow's current official endpoint and Qwen3-VL model while adding configurable image detail support for OCR requests.

**Architecture:** Keep the existing synchronous OCR processor and request path intact. Change only configuration defaults, add validation for image detail, and thread the new setting into the existing multimodal request payload.

**Tech Stack:** Python 3.13, pytest, openai-compatible SiliconFlow client

---

## Chunk 1: Settings And Request Payload

### Task 1: Update tests for official defaults and image detail

**Files:**
- Modify: `tests/test_png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    settings = VLMSettings.from_env()

    assert settings.base_url == "https://api.siliconflow.cn/v1"
    assert settings.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings.image_detail == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_png_to_text.py::test_settings_load_defaults tests/test_png_to_text.py::test_extract_text_sends_openai_compatible_vision_request -v`
Expected: FAIL because defaults and payload do not match the new official values.

- [ ] **Step 3: Write minimal implementation**

```python
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

image_detail = _read_image_detail("SILICONFLOW_IMAGE_DETAIL", "high")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_png_to_text.py::test_settings_load_defaults tests/test_png_to_text.py::test_extract_text_sends_openai_compatible_vision_request -v`
Expected: PASS

## Chunk 2: Validation And Docs

### Task 2: Validate image detail configuration

**Files:**
- Modify: `tests/test_png_to_text.py`
- Modify: `src/qmrkg/png_to_text.py`
- Test: `tests/test_png_to_text.py`

- [ ] **Step 1: Write the failing test**

```python
def test_settings_reject_invalid_image_detail(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_IMAGE_DETAIL", "invalid")

    with pytest.raises(ValueError, match="SILICONFLOW_IMAGE_DETAIL"):
        VLMSettings.from_env()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_png_to_text.py::test_settings_reject_invalid_image_detail -v`
Expected: FAIL because no validation exists yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _read_image_detail(name: str, default: str) -> str:
    value = (os.getenv(name) or default).strip().lower() or default
    if value not in {"auto", "low", "high"}:
        raise ValueError(f"{name} must be one of auto, low, high")
    return value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_png_to_text.py::test_settings_reject_invalid_image_detail -v`
Expected: PASS

### Task 3: Update environment example and README

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Update documented defaults**

```dotenv
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
SILICONFLOW_IMAGE_DETAIL=high
```

- [ ] **Step 2: Run targeted tests and smoke-check docs references**

Run: `pytest tests/test_png_to_text.py -v`
Expected: PASS
