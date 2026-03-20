from __future__ import annotations

import base64
import logging
import shutil
import threading
import uuid
from pathlib import Path

import pytest

import qmrkg.png_to_text as png_to_text
from qmrkg.png_to_text import OCRPageResult, OCRProcessor, RollingRateLimiter, VLMSettings


class FakeResponse:
    def __init__(self, text: str):
        self.model = "fake-model"
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": text})()})()]


class FakeCompletions:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.chat = type("Chat", (), {"completions": FakeCompletions(self._record_and_handle)})()
        self.calls = []

    def _record_and_handle(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(**kwargs)


class FakeSleep:
    def __init__(self, timeline):
        self.timeline = timeline
        self.calls = []

    def __call__(self, seconds: float):
        self.calls.append(seconds)
        self.timeline[0] += seconds


class FakeHTTPResponse:
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self._body = body
        self.text = body

    def json(self):
        return {"error": {"message": self._body}}


class FakeAPIStatusError(Exception):
    def __init__(self, message: str, status_code: int, body: str):
        super().__init__(message)
        self.status_code = status_code
        self.response = FakeHTTPResponse(status_code, body)


def build_processor(monkeypatch, handler, **settings_overrides):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    for key, value in settings_overrides.items():
        monkeypatch.setenv(key, str(value))
    processor = OCRProcessor(use_gpu=True, lang="en", show_log=True)
    processor._client = FakeClient(handler)
    return processor


@pytest.fixture
def scratch_dir() -> Path:
    scratch_dir = Path(__file__).resolve().parent / "_scratch" / uuid.uuid4().hex
    scratch_dir.mkdir(parents=True, exist_ok=False)
    yield scratch_dir
    shutil.rmtree(scratch_dir, ignore_errors=True)


def write_image(tmp_path: Path, name: str, content: bytes = b"fake-image") -> Path:
    image_path = tmp_path / name
    image_path.write_bytes(content)
    return image_path


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_settings_require_api_key(monkeypatch):
    monkeypatch.delenv("PPIO_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr(png_to_text, "VLMSettings", VLMSettings)

    with pytest.raises(ValueError, match="PPIO_API_KEY"):
        VLMSettings.from_env()


def test_ocr_processor_accepts_legacy_constructor_args(monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")

    processor = OCRProcessor(use_gpu=True, lang="en", show_log=True)

    assert processor.use_gpu is True
    assert processor.lang == "en"
    assert processor.show_log is True


def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")

    settings = VLMSettings.from_env()

    assert settings.base_url == "https://api.ppio.com/openai"
    assert settings.model == "qwen/qwen3-vl-8b-instruct"
    assert settings.image_detail == "high"
    assert settings.modality == "multimodal"
    assert settings.thinking_enabled is False


def test_settings_support_legacy_siliconflow_aliases(monkeypatch):
    monkeypatch.delenv("PPIO_API_KEY", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "legacy-test-key")
    monkeypatch.setenv("SILICONFLOW_VLM_MODEL", "legacy-model")

    settings = VLMSettings.from_env()

    assert settings.api_key == "legacy-test-key"
    assert settings.model == "legacy-model"


def test_settings_load_task_scoped_ocr_config(scratch_dir, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    monkeypatch.setenv("PPIO_PROMPT_KEY", "structured")
    config_path = write_config(
        scratch_dir,
        """
ocr:
  provider:
    name: ppio
    base_url: https://example.invalid/openai
    model: test-vlm
    modality: multimodal
    supports_thinking: true
  prompts:
    default: default prompt
    structured: structured prompt
  request:
    image_detail: low
    timeout_seconds: 12.5
    max_retries: 7
    thinking:
      enabled: true
  rate_limit:
    rpm: 123
    max_concurrency: 9
""".strip(),
    )

    settings = VLMSettings.from_env(config_path)

    assert settings.base_url == "https://example.invalid/openai"
    assert settings.model == "test-vlm"
    assert settings.prompt == "structured prompt"
    assert settings.image_detail == "low"
    assert settings.timeout_seconds == 12.5
    assert settings.max_retries == 7
    assert settings.rpm == 123
    assert settings.max_concurrency == 9
    assert settings.thinking_enabled is True
    assert settings.supports_thinking is True


def test_settings_reject_legacy_openai_yaml_shape(scratch_dir, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        scratch_dir,
        """
openai:
  base_url: https://legacy.invalid/v1
  model: legacy-model
""".strip(),
    )

    with pytest.raises(ValueError, match="legacy 'openai'"):
        VLMSettings.from_env(config_path)


def test_settings_reject_invalid_image_detail(monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    monkeypatch.setenv("PPIO_IMAGE_DETAIL", "ultra")

    with pytest.raises(ValueError, match="PPIO_IMAGE_DETAIL"):
        VLMSettings.from_env()


def test_settings_reject_thinking_without_provider_support(scratch_dir, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        scratch_dir,
        """
ocr:
  provider:
    name: ppio
    model: test-vlm
    modality: multimodal
    supports_thinking: false
  request:
    thinking:
      enabled: true
""".strip(),
    )

    with pytest.raises(ValueError, match="supports_thinking is false"):
        VLMSettings.from_env(config_path)


def test_extract_text_sends_openai_compatible_vision_request(scratch_dir, monkeypatch):
    image_path = write_image(scratch_dir, "page.png")

    processor = build_processor(monkeypatch, lambda **_: FakeResponse("recognized text"))

    text = processor.extract_text(image_path)

    assert text == "recognized text"
    assert processor._client.calls[0]["model"] == "qwen/qwen3-vl-8b-instruct"
    assert processor._client.calls[0]["messages"][0]["role"] == "user"
    image_part = processor._client.calls[0]["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["image_url"]["detail"] == "high"
    assert "reasoning_enabled" not in processor._client.calls[0]


def test_extract_text_returns_compatibility_confidence(scratch_dir, monkeypatch):
    image_path = write_image(scratch_dir, "page.png")
    processor = build_processor(monkeypatch, lambda **_: FakeResponse("recognized text"))

    result = processor.extract_text(image_path, return_confidence=True)

    assert result == ("recognized text", 1.0)


def test_extract_text_retries_transient_failures(scratch_dir, monkeypatch):
    image_path = write_image(scratch_dir, "page.png")
    attempts = {"count": 0}

    def handler(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary timeout")
        return FakeResponse("recovered text")

    processor = build_processor(
        monkeypatch,
        handler,
        PPIO_MAX_RETRIES=2,
        PPIO_MAX_CONCURRENCY=1,
    )

    result = processor.extract_text(image_path)

    assert result == "recovered text"
    assert attempts["count"] == 2


@pytest.mark.parametrize("exception_type", ["APITimeoutError", "APIConnectionError"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_extract_text_retries_openai_sdk_timeout_and_connection_failures(
    scratch_dir, monkeypatch, exception_type, wrapped
):
    image_path = write_image(scratch_dir, "page.png")
    attempts = {"count": 0}
    sdk_exception = type(exception_type, (Exception,), {})

    def handler(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            if wrapped:
                try:
                    raise sdk_exception("sdk timeout")
                except sdk_exception as exc:
                    raise RuntimeError("wrapped sdk timeout") from exc
            raise sdk_exception("sdk timeout")
        return FakeResponse("recovered text")

    processor = build_processor(
        monkeypatch,
        handler,
        PPIO_MAX_RETRIES=2,
        PPIO_MAX_CONCURRENCY=1,
    )

    result = processor.extract_text(image_path)

    assert result == "recovered text"
    assert attempts["count"] == 2


def test_extract_text_logs_api_error_details_on_retry(scratch_dir, monkeypatch, caplog):
    image_path = write_image(scratch_dir, "page.png")
    attempts = {"count": 0}

    def handler(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FakeAPIStatusError(
                "rate limited", 429, '{"error":{"message":"too many requests"}}'
            )
        return FakeResponse("recovered text")

    processor = build_processor(
        monkeypatch,
        handler,
        PPIO_MAX_RETRIES=2,
        PPIO_MAX_CONCURRENCY=1,
    )

    with caplog.at_level(logging.WARNING):
        result = processor.extract_text(image_path)

    assert result == "recovered text"
    assert "status_code=429" in caplog.text
    assert 'response_body={"error":{"message":"too many requests"}}' in caplog.text
    assert "exception_type=FakeAPIStatusError" in caplog.text


def test_extract_from_images_logs_api_error_details_on_final_failure(
    scratch_dir, monkeypatch, caplog
):
    image_path = write_image(scratch_dir, "page.png")

    processor = build_processor(
        monkeypatch,
        lambda **_kwargs: (_ for _ in ()).throw(
            FakeAPIStatusError("bad request", 400, '{"error":{"message":"invalid image"}}')
        ),
        PPIO_MAX_RETRIES=1,
        PPIO_MAX_CONCURRENCY=1,
    )

    with caplog.at_level(logging.ERROR):
        results = processor.extract_from_images([image_path])

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].text == ""
    assert "status_code=400" in caplog.text
    assert 'response_body={"error":{"message":"invalid image"}}' in caplog.text
    assert "exception_type=FakeAPIStatusError" in caplog.text


def test_extract_from_images_preserves_input_order(scratch_dir, monkeypatch):
    page1 = write_image(scratch_dir, "page1.png", b"page-1")
    page2 = write_image(scratch_dir, "page2.png", b"page-2")
    page3 = write_image(scratch_dir, "page3.png", b"page-3")
    release_page1 = threading.Event()

    def handler(**kwargs):
        image_url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
        payload = image_url.split(",", 1)[1]
        image_bytes = base64.b64decode(payload)
        if image_bytes == b"page-2":
            release_page1.set()
            return FakeResponse("text for page2.png")
        if image_bytes == b"page-3":
            return FakeResponse("text for page3.png")
        release_page1.wait(timeout=1)
        return FakeResponse("text for page1.png")

    processor = build_processor(
        monkeypatch,
        handler,
        PPIO_MAX_CONCURRENCY=3,
        PPIO_RPM=60,
    )

    results = processor.extract_from_images([page1, page2, page3])

    assert len(results) == 3
    assert results[0].text == "text for page1.png"
    assert results[1].text == "text for page2.png"
    assert results[2].text == "text for page3.png"
    assert results[0].page_number == 1
    assert results[1].page_number == 2
    assert results[2].page_number == 3


def test_extract_from_images_retries_transient_failures(scratch_dir, monkeypatch):
    image_path = write_image(scratch_dir, "page.png")
    attempts = {"count": 0}

    def handler(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary timeout")
        return FakeResponse("recovered text")

    processor = build_processor(
        monkeypatch,
        handler,
        PPIO_MAX_RETRIES=2,
        PPIO_MAX_CONCURRENCY=1,
    )

    results = processor.extract_from_images([image_path])

    assert len(results) == 1
    assert results[0].text == "recovered text"
    assert results[0].status == "success"
    assert attempts["count"] == 2


def test_rate_limiter_blocks_requests_over_rpm():
    timeline = [0.0]
    fake_sleep = FakeSleep(timeline)
    limiter = RollingRateLimiter(rpm=2, time_fn=lambda: timeline[0], sleep_fn=fake_sleep)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert fake_sleep.calls == [60.0]


def test_process_and_save_renders_page_results_with_metadata(scratch_dir, monkeypatch):
    from datetime import datetime, timezone

    page1 = write_image(scratch_dir, "page1.png")
    page2 = write_image(scratch_dir, "page2.png")
    output_path = scratch_dir / "out.md"
    processor = build_processor(monkeypatch, lambda **_: FakeResponse("unused"))

    page_results = [
        OCRPageResult(
            image_path=page1,
            page_number=1,
            text="First page content",
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=1.5,
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        OCRPageResult(
            image_path=page2,
            page_number=2,
            text="   ",
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=1.2,
            model="test-model",
        ),
    ]

    saved_path = processor.process_and_save(page_results, output_path, pdf_source="test.pdf")

    assert saved_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "## Page 1" in content
    assert "First page content" in content
    assert "test-model" in content
    assert "## Page 2" in content
    assert "source: test.pdf" in content


def test_process_and_save_with_failed_pages(scratch_dir, monkeypatch):
    from datetime import datetime, timezone

    page1 = write_image(scratch_dir, "page1.png")
    output_path = scratch_dir / "out.md"
    processor = build_processor(monkeypatch, lambda **_: FakeResponse("unused"))

    page_results = [
        OCRPageResult(
            image_path=page1,
            page_number=1,
            text="",
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0.0,
            status="failed",
            error="API timeout",
        ),
    ]

    saved_path = processor.process_and_save(page_results, output_path)

    assert saved_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "## Page 1" in content
    assert "**Status:** failed" in content
    assert "API timeout" in content
    assert "failed_pages: 1" in content


def test_env_example_lists_ppio_variables():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text(encoding="utf-8")

    assert "PPIO_API_KEY=" in content
    assert "PPIO_BASE_URL=" in content
    assert "PPIO_VLM_MODEL=" in content
    assert "PPIO_IMAGE_DETAIL=" in content
    assert "PPIO_RPM=" in content
