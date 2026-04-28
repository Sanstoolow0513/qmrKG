"""Shared LLM factory and task processors for PPIO-compatible models."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_config import TaskLLMSettings
from .llm_types import LLMContentPart, LLMEmbeddingResponse, LLMMessage, LLMResponse
from .rate_limit import RollingRateLimiter

logger = logging.getLogger(__name__)


class TaskLLMRunner:
    """Execute task-scoped LLM requests with shared config, retries, and rate limiting."""

    def __init__(self, settings: TaskLLMSettings, *, client=None):
        self.settings = settings
        self._client = client
        self._rate_limiter: RollingRateLimiter | None = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError("openai not installed. Run: pip install openai") from exc
            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
            )
        return self._client

    @property
    def rate_limiter(self) -> RollingRateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = RollingRateLimiter(self.settings.rpm)
        return self._rate_limiter

    def run_text(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        messages: list[LLMMessage] = []
        effective_system_prompt = (
            system_prompt if system_prompt is not None else self.settings.prompt
        )
        if effective_system_prompt:
            messages.append(LLMMessage(role="system", content=effective_system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        return self.run_messages(messages)

    def run_image(self, prompt: str, image_path: Path) -> LLMResponse:
        self._ensure_multimodal()
        return self.run_messages(
            [
                LLMMessage(
                    role="user",
                    content=[
                        LLMContentPart(type="text", text=prompt),
                        LLMContentPart(
                            type="image_url",
                            image_path=Path(image_path),
                            detail=self.settings.image_detail,
                        ),
                    ],
                )
            ]
        )

    def run_messages(self, messages: list[LLMMessage]) -> LLMResponse:
        self._validate_messages(messages)
        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._invoke(messages)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Transient LLM failure for task=%s on attempt %s/%s: %s",
                    self.settings.task_name,
                    attempt,
                    attempts,
                    self._format_exception_summary(exc),
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("unreachable")

    def run_embeddings(self, inputs: list[str]) -> LLMEmbeddingResponse:
        if self.settings.modality != "embedding":
            raise ValueError(
                f"task '{self.settings.task_name}' is configured with modality={self.settings.modality}"
            )
        if not inputs:
            return LLMEmbeddingResponse(
                vectors=[],
                model=self.settings.model,
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._invoke_embeddings(inputs)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Transient embedding failure for task=%s on attempt %s/%s: %s",
                    self.settings.task_name,
                    attempt,
                    attempts,
                    self._format_exception_summary(exc),
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("unreachable")

    def _invoke(self, messages: list[LLMMessage]) -> LLMResponse:
        start_time = time.perf_counter()
        self.rate_limiter.acquire()

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[self._serialize_message(message) for message in messages],
            timeout=self.settings.timeout_seconds,
            **self._request_kwargs(),
        )

        duration = time.perf_counter() - start_time
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)
        reasoning_content, reasoning_details = self._extract_reasoning(response)
        return LLMResponse(
            text=self._extract_message_text(response).strip(),
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            model=getattr(response, "model", None) or self.settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_content=reasoning_content,
            reasoning_details=reasoning_details,
        )

    def _invoke_embeddings(self, inputs: list[str]) -> LLMEmbeddingResponse:
        start_time = time.perf_counter()
        self.rate_limiter.acquire()

        response = self.client.embeddings.create(
            model=self.settings.model,
            input=inputs,
            timeout=self.settings.timeout_seconds,
            **self._embedding_request_kwargs(),
        )

        duration = time.perf_counter() - start_time
        prompt_tokens, _, total_tokens = self._extract_usage(response)
        return LLMEmbeddingResponse(
            vectors=self._extract_embedding_vectors(response),
            model=getattr(response, "model", None) or self.settings.model,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if (
            self.settings.supports_thinking
            and self.settings.thinking_enabled
            and self.settings.reasoning_effort
        ):
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        return kwargs

    def _embedding_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"encoding_format": self.settings.encoding_format}
        if self.settings.embedding_dimensions is not None:
            kwargs["dimensions"] = self.settings.embedding_dimensions
        return kwargs

    def _validate_messages(self, messages: list[LLMMessage]) -> None:
        contains_images = False
        for message in messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.type == "image_url":
                        contains_images = True
                        if part.image_path is None:
                            raise ValueError("image_url content requires image_path")
        if contains_images and self.settings.modality != "multimodal":
            raise ValueError(
                f"task '{self.settings.task_name}' is configured for text-only use and cannot accept images"
            )

    def _ensure_multimodal(self) -> None:
        if self.settings.modality != "multimodal":
            raise ValueError(
                f"task '{self.settings.task_name}' is configured with modality={self.settings.modality}"
            )

    def _serialize_message(self, message: LLMMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        if isinstance(message.content, str):
            payload["content"] = message.content
        else:
            payload["content"] = [self._serialize_content_part(part) for part in message.content]
        return payload

    def _serialize_content_part(self, part: LLMContentPart) -> dict[str, Any]:
        if part.type == "text":
            return {"type": "text", "text": part.text or ""}
        if part.type == "image_url":
            if part.image_path is None:
                raise ValueError("image_url content requires image_path")
            return {
                "type": "image_url",
                "image_url": {
                    "url": self._build_image_data_url(part.image_path),
                    "detail": part.detail or self.settings.image_detail,
                },
            }
        raise ValueError(f"Unsupported content type: {part.type}")

    @staticmethod
    def _extract_message_text(response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        text_parts.append(text)
            return "\n".join(part for part in text_parts if part)
        return str(content or "")

    @staticmethod
    def _extract_reasoning(response) -> tuple[str | None, list[dict[str, Any]]]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None, []
        message = getattr(choices[0], "message", None)
        if message is None:
            return None, []
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_details = getattr(message, "reasoning_details", None) or []
        if isinstance(reasoning_details, list):
            normalized = [
                item if isinstance(item, dict) else getattr(item, "__dict__", {})
                for item in reasoning_details
            ]
        else:
            normalized = []
        return reasoning_content, normalized

    @staticmethod
    def _extract_embedding_vectors(response) -> list[list[float]]:
        vectors: list[list[float]] = []
        for item in getattr(response, "data", None) or []:
            if isinstance(item, dict):
                embedding = item.get("embedding")
            else:
                embedding = getattr(item, "embedding", None)
            if embedding is None:
                raise ValueError("embedding response item missing embedding")
            vectors.append([float(value) for value in embedding])
        return vectors

    @staticmethod
    def _build_image_data_url(image_path: Path) -> str:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime_type = mimetypes.guess_type(Path(image_path).name)[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (TimeoutError, ConnectionError)):
                return True

            exception_names = {cls.__name__ for cls in type(current).__mro__}
            if {"APITimeoutError", "APIConnectionError"} & exception_names:
                return True

            if TaskLLMRunner._has_transient_status_code(current):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _has_transient_status_code(exc: BaseException) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int) and (response_status == 429 or response_status >= 500):
            return True
        return False

    @staticmethod
    def _format_exception_summary(exc: BaseException) -> str:
        parts = [f"exception_type={type(exc).__name__}", f"message={exc}"]
        status_code = TaskLLMRunner._extract_status_code(exc)
        if status_code is not None:
            parts.append(f"status_code={status_code}")
        response_body = TaskLLMRunner._extract_response_body(exc)
        if response_body:
            parts.append(f"response_body={response_body}")
        return " ".join(parts)

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    @staticmethod
    def _extract_response_body(exc: BaseException) -> str | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return TaskLLMRunner._truncate_log_value(text.strip())
        json_loader = getattr(response, "json", None)
        if callable(json_loader):
            try:
                return TaskLLMRunner._truncate_log_value(
                    json.dumps(json_loader(), ensure_ascii=False, separators=(",", ":"))
                )
            except Exception:
                return None
        return None

    @staticmethod
    def _truncate_log_value(value: str, limit: int = 500) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit]}...(truncated)"

    @staticmethod
    def _extract_usage(response) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None
        if isinstance(usage, dict):
            return (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
        return (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )


class LLMFactory:
    """Create task-scoped runners backed by shared settings and transport behavior."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path

    def create(self, task_name: str, *, client=None) -> TaskLLMRunner:
        settings = TaskLLMSettings.from_env(task_name, self.config_path)
        return TaskLLMRunner(settings, client=client)


class TextTaskProcessor:
    """Convenience wrapper for text-only task execution."""

    def __init__(self, task_name: str, config_path: Path | None = None, client=None):
        self.task_name = task_name
        self._factory = LLMFactory(config_path)
        self._runner = self._factory.create(task_name, client=client)

    @property
    def settings(self) -> TaskLLMSettings:
        return self._runner.settings

    def run_text(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        return self._runner.run_text(prompt, system_prompt=system_prompt)

    def run_messages(self, messages: list[LLMMessage]) -> LLMResponse:
        return self._runner.run_messages(messages)


class MultimodalTaskProcessor:
    """Convenience wrapper for multimodal task execution."""

    def __init__(self, task_name: str, config_path: Path | None = None, client=None):
        self.task_name = task_name
        self._factory = LLMFactory(config_path)
        self._runner = self._factory.create(task_name, client=client)

    @property
    def settings(self) -> TaskLLMSettings:
        return self._runner.settings

    def run_image(self, prompt: str, image_path: Path) -> LLMResponse:
        return self._runner.run_image(prompt, image_path)

    def run_messages(self, messages: list[LLMMessage]) -> LLMResponse:
        return self._runner.run_messages(messages)


class EmbeddingTaskProcessor:
    """Convenience wrapper for embedding tasks with deterministic batching."""

    def __init__(self, task_name: str, config_path: Path | None = None, client=None):
        self.task_name = task_name
        self._factory = LLMFactory(config_path)
        self._runner = self._factory.create(task_name, client=client)

    @property
    def settings(self) -> TaskLLMSettings:
        return self._runner.settings

    def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            response = self._runner.run_embeddings(batch)
            if len(response.vectors) != len(batch):
                raise ValueError(
                    f"embedding response length mismatch: expected {len(batch)}, "
                    f"got {len(response.vectors)}"
                )
            vectors.extend(response.vectors)
        return vectors
