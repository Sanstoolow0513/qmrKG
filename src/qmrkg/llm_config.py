"""Task-scoped LLM configuration loading for PPIO-compatible models."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


DEFAULT_BASE_URL = "https://api.ppio.com/openai"
DEFAULT_IMAGE_DETAIL = "high"
DEFAULT_RPM = 60
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_PROMPT_KEY = "default"
DEFAULT_OCR_MODEL = "qwen/qwen3-vl-8b-instruct"
DEFAULT_OCR_PROMPT = (
    "Transcribe all visible text from this page exactly as written. Preserve reading order, "
    "line breaks, and heading hierarchy (e.g., # ## ###). For icons, symbols, diagrams, or "
    "non-text visual elements, describe them using fenced code blocks like ```icon: description```. "
    "Do not add commentary or summaries outside the transcribed content."
)

_API_KEY_ALIASES = ("PPIO_API_KEY", "SILICONFLOW_API_KEY")
_BASE_URL_ALIASES = ("PPIO_BASE_URL", "SILICONFLOW_BASE_URL")
_MODEL_ALIASES_BY_TASK = {
    "ocr": ("PPIO_VLM_MODEL", "PPIO_MODEL", "SILICONFLOW_VLM_MODEL", "SILICONFLOW_MODEL"),
}
_PROMPT_ALIASES_BY_TASK = {
    "ocr": ("PPIO_VLM_PROMPT", "PPIO_PROMPT", "SILICONFLOW_VLM_PROMPT", "SILICONFLOW_PROMPT"),
}
_PROMPT_KEY_ALIASES = ("PPIO_PROMPT_KEY", "SILICONFLOW_PROMPT_KEY")
_IMAGE_DETAIL_ALIASES = ("PPIO_IMAGE_DETAIL", "SILICONFLOW_IMAGE_DETAIL")
_RPM_ALIASES = ("PPIO_RPM", "SILICONFLOW_RPM")
_MAX_CONCURRENCY_ALIASES = ("PPIO_MAX_CONCURRENCY", "SILICONFLOW_MAX_CONCURRENCY")
_TIMEOUT_ALIASES = ("PPIO_TIMEOUT_SECONDS", "SILICONFLOW_TIMEOUT_SECONDS")
_MAX_RETRIES_ALIASES = ("PPIO_MAX_RETRIES", "SILICONFLOW_MAX_RETRIES")


def _load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, skipping YAML config loading")
        return {}

    if config_path is None:
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
    else:
        search_paths = [Path(config_path)]

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    logger.debug("Loaded config from %s", path)
                    return config or {}
            except Exception as exc:
                logger.warning("Failed to load config from %s: %s", path, exc)

    return {}


def _get_nested_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, {})
    return current if current != {} else default


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value.strip()
    return None


def _read_str(env_names: tuple[str, ...], config_value: Any, default: str) -> str:
    env_value = _read_env(*env_names)
    if env_value not in (None, ""):
        return env_value
    if isinstance(config_value, str) and config_value.strip():
        return config_value.strip()
    return default


def _read_int(env_names: tuple[str, ...], config_value: Any, default: int) -> int:
    env_value = _read_env(*env_names)
    if env_value not in (None, ""):
        parsed = int(env_value)
    elif config_value not in (None, ""):
        parsed = int(config_value)
    else:
        return default
    if parsed <= 0:
        raise ValueError(f"{env_names[0]} must be greater than 0")
    return parsed


def _read_float(env_names: tuple[str, ...], config_value: Any, default: float) -> float:
    env_value = _read_env(*env_names)
    if env_value not in (None, ""):
        parsed = float(env_value)
    elif config_value not in (None, ""):
        parsed = float(config_value)
    else:
        return default
    if parsed <= 0:
        raise ValueError(f"{env_names[0]} must be greater than 0")
    return parsed


def _read_image_detail(env_names: tuple[str, ...], config_value: Any, default: str) -> str:
    value = _read_str(env_names, config_value, default).lower()
    if value not in {"auto", "low", "high"}:
        raise ValueError(f"{env_names[0]} must be one of: auto, low, high")
    return value


def _read_bool(config_value: Any, *, field_name: str) -> bool:
    if config_value in (None, ""):
        return False
    if isinstance(config_value, bool):
        return config_value
    raise ValueError(f"{field_name} must be a boolean")


def _default_modality(task_name: str) -> str:
    return "multimodal" if task_name == "ocr" else "text"


def _default_model(task_name: str) -> str:
    if task_name == "ocr":
        return DEFAULT_OCR_MODEL
    return ""


def _default_prompt(task_name: str) -> str:
    if task_name == "ocr":
        return DEFAULT_OCR_PROMPT
    return ""


def _model_env_aliases(task_name: str) -> tuple[str, ...]:
    return _MODEL_ALIASES_BY_TASK.get(task_name, ("PPIO_MODEL", "SILICONFLOW_MODEL"))


def _prompt_env_aliases(task_name: str) -> tuple[str, ...]:
    return _PROMPT_ALIASES_BY_TASK.get(task_name, ("PPIO_PROMPT", "SILICONFLOW_PROMPT"))


def _get_task_config(config: dict[str, Any], task_name: str) -> dict[str, Any]:
    if not config:
        return {}

    if "openai" in config:
        raise ValueError("config.yaml must use task-scoped top-level sections instead of legacy 'openai'")

    task_config = config.get(task_name)
    if task_config is None:
        return {}
    if not isinstance(task_config, dict):
        raise ValueError(f"config.yaml top-level '{task_name}' must be a mapping")
    return task_config


@dataclass(slots=True)
class TaskLLMSettings:
    task_name: str
    api_key: str
    base_url: str
    model: str
    prompt: str
    modality: str
    supports_thinking: bool = False
    thinking_enabled: bool = False
    image_detail: str = DEFAULT_IMAGE_DETAIL
    rpm: int = DEFAULT_RPM
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls, task_name: str, config_path: Path | None = None) -> "TaskLLMSettings":
        load_dotenv()

        api_key = _read_env(*_API_KEY_ALIASES)
        if not api_key:
            raise ValueError(f"{_API_KEY_ALIASES[0]} is required")

        config = _load_yaml_config(config_path)
        task_config = _get_task_config(config, task_name)
        provider_config = _get_nested_value(task_config, "provider", default={}) or {}
        prompts_config = _get_nested_value(task_config, "prompts", default={}) or {}
        request_config = _get_nested_value(task_config, "request", default={}) or {}
        rate_config = _get_nested_value(task_config, "rate_limit", default={}) or {}

        prompt_key = _read_str(_PROMPT_KEY_ALIASES, None, DEFAULT_PROMPT_KEY)
        config_prompt = prompts_config.get(prompt_key, prompts_config.get(DEFAULT_PROMPT_KEY))

        base_url = _read_str(
            _BASE_URL_ALIASES,
            _get_nested_value(provider_config, "base_url"),
            DEFAULT_BASE_URL,
        )
        model = _read_str(
            _model_env_aliases(task_name),
            _get_nested_value(provider_config, "model"),
            _default_model(task_name),
        )
        if not model:
            raise ValueError(f"provider.model is required for task '{task_name}'")

        prompt = _read_str(
            _prompt_env_aliases(task_name),
            config_prompt,
            _default_prompt(task_name),
        )

        modality = _read_str(
            tuple(),
            _get_nested_value(provider_config, "modality"),
            _default_modality(task_name),
        ).lower()
        if modality not in {"text", "multimodal"}:
            raise ValueError("provider.modality must be one of: text, multimodal")

        supports_thinking = _read_bool(
            _get_nested_value(provider_config, "supports_thinking"),
            field_name="provider.supports_thinking",
        )
        thinking_enabled = _read_bool(
            _get_nested_value(request_config, "thinking", "enabled"),
            field_name="request.thinking.enabled",
        )
        if thinking_enabled and not supports_thinking:
            raise ValueError(
                f"task '{task_name}' enables request.thinking.enabled but provider.supports_thinking is false"
            )

        image_detail = _read_image_detail(
            _IMAGE_DETAIL_ALIASES,
            _get_nested_value(request_config, "image_detail"),
            DEFAULT_IMAGE_DETAIL,
        )
        rpm = _read_int(_RPM_ALIASES, _get_nested_value(rate_config, "rpm"), DEFAULT_RPM)
        max_concurrency = _read_int(
            _MAX_CONCURRENCY_ALIASES,
            _get_nested_value(rate_config, "max_concurrency"),
            DEFAULT_MAX_CONCURRENCY,
        )
        timeout_seconds = _read_float(
            _TIMEOUT_ALIASES,
            _get_nested_value(request_config, "timeout_seconds"),
            DEFAULT_TIMEOUT_SECONDS,
        )
        max_retries = _read_int(
            _MAX_RETRIES_ALIASES,
            _get_nested_value(request_config, "max_retries"),
            DEFAULT_MAX_RETRIES,
        )

        return cls(
            task_name=task_name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            modality=modality,
            supports_thinking=supports_thinking,
            thinking_enabled=thinking_enabled,
            image_detail=image_detail,
            rpm=rpm,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
