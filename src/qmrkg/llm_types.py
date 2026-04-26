"""Typed request/response structures for task-scoped LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


LLMRole = Literal["system", "user", "assistant", "tool"]
LLMModality = Literal["text", "multimodal", "embedding"]
LLMContentType = Literal["text", "image_url"]


@dataclass(slots=True)
class LLMContentPart:
    type: LLMContentType
    text: str | None = None
    image_path: Path | None = None
    detail: str | None = None


@dataclass(slots=True)
class LLMMessage:
    role: LLMRole
    content: str | list[LLMContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class LLMResponse:
    text: str
    processed_at: str
    duration_seconds: float
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: Literal["success", "failed"] = "success"
    error: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class LLMEmbeddingResponse:
    vectors: list[list[float]]
    model: str | None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float = 0.0
    processed_at: str = ""
