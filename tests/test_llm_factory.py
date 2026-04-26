from __future__ import annotations

from pathlib import Path

import pytest

from qmrkg import LLMContentPart, LLMMessage, MultimodalTaskProcessor, TextTaskProcessor
from qmrkg.llm_factory import EmbeddingTaskProcessor, LLMFactory


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


class FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]], model: str = "fake-embedding-model"):
        self.model = model
        self.data = [type("EmbeddingItem", (), {"embedding": vector})() for vector in vectors]
        self.usage = type("Usage", (), {"prompt_tokens": 3, "total_tokens": 3})()


class FakeEmbeddings:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class FakeEmbeddingClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.embeddings = FakeEmbeddings(self._record_and_handle)

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
llm:
  profiles:
    ner_profile:
      provider:
        name: ppio
        base_url: https://api.ppio.com/openai
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
      request:
        timeout_seconds: 12
        max_retries: 1
        thinking:
          enabled: false
      rate_limit:
        rpm: 20
        max_concurrency: 2
ner:
  llm_profile: ner_profile
  prompts:
    default: you are a ner assistant
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
llm:
  profiles:
    ner_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
      request:
        thinking:
          enabled: false
      rate_limit: {}
ner:
  llm_profile: ner_profile
  prompts: {}
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
llm:
  profiles:
    ocr_profile:
      provider:
        name: ppio
        model: qwen/qwen3-vl-8b-instruct
        modality: multimodal
        supports_thinking: true
      request:
        image_detail: high
        timeout_seconds: 10
        max_retries: 1
        thinking:
          enabled: true
      rate_limit:
        rpm: 20
        max_concurrency: 2
ocr:
  llm_profile: ocr_profile
  prompts:
    default: OCR this page
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
llm:
  profiles:
    ner_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
      request:
        thinking:
          enabled: false
      rate_limit: {}
    re_profile:
      provider:
        name: ppio
        model: deepseek/deepseek-r1-0528
        modality: text
        supports_thinking: false
      request:
        thinking:
          enabled: false
      rate_limit: {}
ner:
  llm_profile: ner_profile
  prompts: {}
re:
  llm_profile: re_profile
  prompts: {}
""".strip(),
    )
    factory = LLMFactory(config_path)

    ner_runner = factory.create("ner", client=FakeClient(lambda **_: FakeResponse("ner")))
    re_runner = factory.create("re", client=FakeClient(lambda **_: FakeResponse("re")))

    assert ner_runner.settings.model == "qwen/qwen3-8b"
    assert re_runner.settings.model == "deepseek/deepseek-r1-0528"


def test_factory_requires_llm_profile_when_profiles_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    text_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
      request:
        thinking:
          enabled: false
ner:
  prompts: {}
""".strip(),
    )

    with pytest.raises(ValueError, match="must set non-empty 'llm_profile'"):
        LLMFactory(config_path).create("ner", client=FakeClient(lambda **_: FakeResponse("ner")))


def test_factory_rejects_unknown_llm_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    text_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
      request:
        thinking:
          enabled: false
ner:
  llm_profile: missing_profile
  prompts: {}
""".strip(),
    )

    with pytest.raises(ValueError, match="references unknown llm_profile"):
        LLMFactory(config_path).create("ner", client=FakeClient(lambda **_: FakeResponse("ner")))


def test_embedding_modality_no_prompt_required(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        base_url: https://api.ppinfra.com/openai
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        timeout_seconds: 30
        max_retries: 2
        encoding_format: float
        dimensions: 1024
      rate_limit:
        rpm: 100
        max_concurrency: 4
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    runner = LLMFactory(config_path).create("entity_embed", client=FakeClient(lambda **_: None))

    assert runner.settings.modality == "embedding"
    assert runner.settings.prompt == ""
    assert runner.settings.encoding_format == "float"
    assert runner.settings.embedding_dimensions == 1024


def test_embedding_modality_rejects_thinking(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        thinking:
          enabled: true
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    with pytest.raises(ValueError, match="request.thinking.enabled"):
        LLMFactory(config_path).create("entity_embed", client=FakeClient(lambda **_: None))


def test_run_embeddings_validates_modality(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    text_profile:
      provider:
        name: ppio
        model: qwen/qwen3-8b
        modality: text
        supports_thinking: false
ner:
  llm_profile: text_profile
  prompts: {}
""".strip(),
    )
    runner = LLMFactory(config_path).create(
        "ner",
        client=FakeEmbeddingClient(lambda **_: FakeEmbeddingResponse([[1.0, 0.0]])),
    )

    with pytest.raises(ValueError, match="modality=text"):
        runner.run_embeddings(["TCP"])


def test_run_embeddings_sends_request_options(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        timeout_seconds: 12
        max_retries: 1
        encoding_format: float
        dimensions: 1024
      rate_limit:
        rpm: 100
        max_concurrency: 4
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )
    fake_client = FakeEmbeddingClient(lambda **_: FakeEmbeddingResponse([[1.0, 0.0]]))
    runner = LLMFactory(config_path).create("entity_embed", client=fake_client)

    response = runner.run_embeddings(["protocol | TCP | 传输控制协议"])

    assert response.vectors == [[1.0, 0.0]]
    assert fake_client.calls[0]["model"] == "qwen/qwen3-embedding-8b"
    assert fake_client.calls[0]["input"] == ["protocol | TCP | 传输控制协议"]
    assert fake_client.calls[0]["encoding_format"] == "float"
    assert fake_client.calls[0]["dimensions"] == 1024


def test_embedding_task_processor_batches(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )

    def handler(**kwargs):
        vectors = [[float(index), 0.0] for index, _ in enumerate(kwargs["input"])]
        return FakeEmbeddingResponse(vectors)

    fake_client = FakeEmbeddingClient(handler)
    processor = EmbeddingTaskProcessor("entity_embed", config_path=config_path, client=fake_client)

    vectors = processor.embed(["a", "b", "c"], batch_size=2)

    assert vectors == [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    assert [call["input"] for call in fake_client.calls] == [["a", "b"], ["c"]]


def test_run_embeddings_retries_on_transient(monkeypatch, tmp_path):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    monkeypatch.setattr("qmrkg.llm_factory.time.sleep", lambda _: None)
    config_path = write_config(
        tmp_path,
        """
llm:
  profiles:
    embedding_profile:
      provider:
        name: ppio
        model: qwen/qwen3-embedding-8b
        modality: embedding
        supports_thinking: false
      request:
        max_retries: 1
entity_embed:
  llm_profile: embedding_profile
""".strip(),
    )
    attempts = {"count": 0}

    def handler(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            error = RuntimeError("rate limited")
            error.status_code = 429
            raise error
        return FakeEmbeddingResponse([[1.0, 0.0]])

    runner = LLMFactory(config_path).create("entity_embed", client=FakeEmbeddingClient(handler))

    response = runner.run_embeddings(["TCP"])

    assert response.vectors == [[1.0, 0.0]]
    assert attempts["count"] == 2
