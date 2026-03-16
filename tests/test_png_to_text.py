from __future__ import annotations

import base64
import shutil
import threading
import uuid
from pathlib import Path

import pytest

from qmrkg.png_to_text import OCRProcessor, RollingRateLimiter, VLMSettings


class FakeResponse:
    def __init__(self, text: str):
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


def build_processor(monkeypatch, handler, **settings_overrides):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
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


def test_settings_require_api_key(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        VLMSettings.from_env()


def test_ocr_processor_accepts_legacy_constructor_args(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    processor = OCRProcessor(use_gpu=True, lang="en", show_log=True)

    assert processor.use_gpu is True
    assert processor.lang == "en"
    assert processor.show_log is True


def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    settings = VLMSettings.from_env()

    assert settings.base_url == "https://api.siliconflow.com/v1"
    assert settings.model == "Qwen/Qwen2-VL-72B-Instruct"


def test_extract_text_sends_openai_compatible_vision_request(scratch_dir, monkeypatch):
    image_path = write_image(scratch_dir, "page.png")

    processor = build_processor(monkeypatch, lambda **_: FakeResponse("recognized text"))

    text = processor.extract_text(image_path)

    assert text == "recognized text"
    assert processor._client.calls[0]["model"] == "Qwen/Qwen2-VL-72B-Instruct"
    assert processor._client.calls[0]["messages"][0]["role"] == "user"
    image_part = processor._client.calls[0]["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


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
        SILICONFLOW_MAX_RETRIES=2,
        SILICONFLOW_MAX_CONCURRENCY=1,
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
        SILICONFLOW_MAX_RETRIES=2,
        SILICONFLOW_MAX_CONCURRENCY=1,
    )

    result = processor.extract_text(image_path)

    assert result == "recovered text"
    assert attempts["count"] == 2


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
        SILICONFLOW_MAX_CONCURRENCY=3,
        SILICONFLOW_RPM=60,
    )

    results = processor.extract_from_images([page1, page2, page3])

    assert results == ["text for page1.png", "text for page2.png", "text for page3.png"]


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
        SILICONFLOW_MAX_RETRIES=2,
        SILICONFLOW_MAX_CONCURRENCY=1,
    )

    results = processor.extract_from_images([image_path])

    assert results == ["recovered text"]
    assert attempts["count"] == 2


def test_rate_limiter_blocks_requests_over_rpm():
    timeline = [0.0]
    fake_sleep = FakeSleep(timeline)
    limiter = RollingRateLimiter(rpm=2, time_fn=lambda: timeline[0], sleep_fn=fake_sleep)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert fake_sleep.calls == [60.0]


def test_process_and_save_skips_blank_pages_but_keeps_numbering_behavior(scratch_dir, monkeypatch):
    page1 = write_image(scratch_dir, "page1.png")
    page2 = write_image(scratch_dir, "page2.png")
    output_path = scratch_dir / "out.md"
    processor = build_processor(monkeypatch, lambda **_: FakeResponse("unused"))

    processor.extract_from_images = lambda image_paths, return_confidence=False: [
        "first page",
        "   ",
    ]

    saved_path = processor.process_and_save([page1, page2], output_path)

    assert saved_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "--- Page 1 ---" in content
    assert "first page" in content
    assert "--- Page 2 ---" not in content


def test_process_and_save_raises_configuration_errors(scratch_dir, monkeypatch):
    page1 = write_image(scratch_dir, "page1.png")
    output_path = scratch_dir / "out.md"
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    processor = OCRProcessor()

    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        processor.process_and_save([page1], output_path)


def test_env_example_lists_siliconflow_variables():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text(encoding="utf-8")

    assert "SILICONFLOW_API_KEY=" in content
    assert "SILICONFLOW_BASE_URL=" in content
    assert "SILICONFLOW_VLM_MODEL=" in content
    assert "SILICONFLOW_RPM=" in content
