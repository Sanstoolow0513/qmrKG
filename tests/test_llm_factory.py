from __future__ import annotations

from pathlib import Path

import pytest

from qmrkg import LLMContentPart, LLMMessage, MultimodalTaskProcessor, TextTaskProcessor
from qmrkg.llm_factory import LLMFactory


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


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_text_task_processor_uses_system_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
ner:
  provider:
    name: ppio
    base_url: https://api.ppio.com/openai
    model: qwen/qwen3-8b
    modality: text
    supports_thinking: false
  prompts:
    default: you are a ner assistant
  request:
    timeout_seconds: 12
    max_retries: 1
    thinking:
      enabled: false
  rate_limit:
    rpm: 20
    max_concurrency: 2
""".strip(),
    )
    fake_client = FakeClient(lambda **_: FakeResponse("entities"))

    processor = TextTaskProcessor("ner", config_path=config_path, client=fake_client)
    response = processor.run_text("extract entities")

    assert response.text == "entities"
    assert fake_client.calls[0]["messages"][0]["role"] == "system"
    assert fake_client.calls[0]["messages"][0]["content"] == "you are a ner assistant"
    assert fake_client.calls[0]["messages"][1]["content"] == "extract entities"


def test_text_task_processor_rejects_images(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
ner:
  provider:
    name: ppio
    model: qwen/qwen3-8b
    modality: text
    supports_thinking: false
  prompts: {}
  request:
    thinking:
      enabled: false
  rate_limit: {}
""".strip(),
    )
    processor = TextTaskProcessor("ner", config_path=config_path, client=FakeClient(lambda **_: None))

    with pytest.raises(ValueError, match="text-only"):
        processor.run_messages(
            [
                LLMMessage(
                    role="user",
                    content=[
                        LLMContentPart(type="text", text="hello"),
                        LLMContentPart(type="image_url", image_path=tmp_path / "missing.png"),
                    ],
                )
            ]
        )


def test_multimodal_processor_sends_reasoning_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-image")
    config_path = write_config(
        tmp_path,
        """
ocr:
  provider:
    name: ppio
    model: qwen/qwen3-vl-8b-instruct
    modality: multimodal
    supports_thinking: true
  prompts:
    default: OCR this page
  request:
    image_detail: high
    timeout_seconds: 10
    max_retries: 1
    thinking:
      enabled: true
  rate_limit:
    rpm: 20
    max_concurrency: 2
""".strip(),
    )
    fake_client = FakeClient(lambda **_: FakeResponse("vision result"))

    processor = MultimodalTaskProcessor("ocr", config_path=config_path, client=fake_client)
    response = processor.run_image("OCR this page", image_path)

    assert response.text == "vision result"
    assert fake_client.calls[0]["reasoning_enabled"] is True
    assert fake_client.calls[0]["messages"][0]["content"][1]["image_url"]["detail"] == "high"


def test_factory_loads_task_specific_models(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
ner:
  provider:
    name: ppio
    model: qwen/qwen3-8b
    modality: text
    supports_thinking: false
  prompts: {}
  request:
    thinking:
      enabled: false
  rate_limit: {}
re:
  provider:
    name: ppio
    model: deepseek/deepseek-r1-0528
    modality: text
    supports_thinking: false
  prompts: {}
  request:
    thinking:
      enabled: false
  rate_limit: {}
""".strip(),
    )
    factory = LLMFactory(config_path)

    ner_runner = factory.create("ner", client=FakeClient(lambda **_: FakeResponse("ner")))
    re_runner = factory.create("re", client=FakeClient(lambda **_: FakeResponse("re")))

    assert ner_runner.settings.model == "qwen/qwen3-8b"
    assert re_runner.settings.model == "deepseek/deepseek-r1-0528"
