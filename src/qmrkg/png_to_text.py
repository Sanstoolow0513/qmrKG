"""PNG to text extraction backed by PPIO VLM OCR."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches for config.yaml
                     in project root and current working directory.

    Returns:
        Configuration dictionary. Empty dict if no config found.
    """
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, skipping YAML config loading")
        return {}

    if config_path is None:
        # Search for config.yaml in common locations
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
                    logger.debug(f"Loaded config from {path}")
                    return config or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

    return {}


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for environments without optional deps installed

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


DEFAULT_BASE_URL = "https://api.ppio.com/openai"
DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
DEFAULT_PROMPT = (
    "Transcribe all visible text from this page exactly as written. Preserve reading order, "
    "line breaks, and heading hierarchy (e.g., # ## ###). For icons, symbols, diagrams, or "
    "non-text visual elements, describe them using fenced code blocks like ```icon: description```. "
    "Do not add commentary or summaries outside the transcribed content."
)
DEFAULT_IMAGE_DETAIL = "high"
DEFAULT_RPM = 60
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_PROMPT_KEY = "default"


def _get_nested_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, {})
    return current if current != {} else default


def _read_str(name: str, config_value: Any, default: str) -> str:
    value = os.getenv(name)
    if value not in (None, ""):
        return value.strip()
    if isinstance(config_value, str) and config_value.strip():
        return config_value.strip()
    return default


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _read_image_detail(value: str) -> str:
    value = value.strip().lower()
    if value not in {"auto", "low", "high"}:
        raise ValueError(f"image_detail must be one of: auto, low, high")
    return value


def _read_int_with_config(name: str, config_value: Any, default: int) -> int:
    value = os.getenv(name)
    if value not in (None, ""):
        return _read_int(name, default)
    if config_value in (None, ""):
        return default
    parsed = int(config_value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _read_float_with_config(name: str, config_value: Any, default: float) -> float:
    value = os.getenv(name)
    if value not in (None, ""):
        return _read_float(name, default)
    if config_value in (None, ""):
        return default
    parsed = float(config_value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _read_image_detail_with_config(name: str, config_value: Any, default: str) -> str:
    value = os.getenv(name)
    if value not in (None, ""):
        detail = value
        env_error = name
    elif config_value not in (None, ""):
        detail = str(config_value)
        env_error = "image_detail"
    else:
        detail = default
        env_error = "image_detail"
    try:
        return _read_image_detail(detail)
    except ValueError as exc:
        raise ValueError(f"{env_error} must be one of: auto, low, high") from exc


def _get_ocr_config(config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        return {}

    ocr_config = config.get("ocr")
    if ocr_config is not None:
        if not isinstance(ocr_config, dict):
            raise ValueError("config.yaml top-level 'ocr' must be a mapping")
        return ocr_config

    if "openai" in config:
        raise ValueError("config.yaml must use top-level 'ocr' instead of legacy 'openai'")

    return {}


@dataclass(slots=True)
class VLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    image_detail: str = DEFAULT_IMAGE_DETAIL
    rpm: int = DEFAULT_RPM
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls, config_path: Path | None = None) -> "VLMSettings":
        load_dotenv()

        api_key = os.getenv("PPIO_API_KEY", "").strip()
        if not api_key:
            raise ValueError("PPIO_API_KEY is required")

        config = _load_yaml_config(config_path)
        ocr_config = _get_ocr_config(config)
        provider_config = _get_nested_value(ocr_config, "provider", default={})
        prompts_config = _get_nested_value(ocr_config, "prompts", default={})
        rate_config = _get_nested_value(ocr_config, "rate_limit", default={})
        request_config = _get_nested_value(ocr_config, "request", default={})

        prompt_key = os.getenv("PPIO_PROMPT_KEY", DEFAULT_PROMPT_KEY).strip()
        config_prompt = prompts_config.get(prompt_key, prompts_config.get(DEFAULT_PROMPT_KEY))

        base_url = _read_str(
            "PPIO_BASE_URL",
            _get_nested_value(provider_config, "base_url"),
            DEFAULT_BASE_URL,
        )
        model = _read_str(
            "PPIO_VLM_MODEL",
            _get_nested_value(provider_config, "model"),
            DEFAULT_MODEL,
        )
        prompt = _read_str("PPIO_VLM_PROMPT", config_prompt, DEFAULT_PROMPT)
        image_detail = _read_image_detail_with_config(
            "PPIO_IMAGE_DETAIL",
            _get_nested_value(request_config, "image_detail"),
            DEFAULT_IMAGE_DETAIL,
        )
        rpm = _read_int_with_config(
            "PPIO_RPM",
            _get_nested_value(rate_config, "rpm"),
            DEFAULT_RPM,
        )
        max_concurrency = _read_int_with_config(
            "PPIO_MAX_CONCURRENCY",
            _get_nested_value(rate_config, "max_concurrency"),
            DEFAULT_MAX_CONCURRENCY,
        )
        timeout_seconds = _read_float_with_config(
            "PPIO_TIMEOUT_SECONDS",
            _get_nested_value(request_config, "timeout_seconds"),
            DEFAULT_TIMEOUT_SECONDS,
        )
        max_retries = _read_int_with_config(
            "PPIO_MAX_RETRIES",
            _get_nested_value(request_config, "max_retries"),
            DEFAULT_MAX_RETRIES,
        )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            image_detail=image_detail,
            rpm=rpm,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


@dataclass(slots=True)
class OCRPageResult:
    image_path: Path
    page_number: int
    text: str
    processed_at: str
    duration_seconds: float
    confidence: float | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: Literal["success", "failed"] = "success"
    error: str | None = None


class RollingRateLimiter:
    """Enforce a rolling requests-per-minute cap across worker threads."""

    def __init__(
        self,
        rpm: int,
        *,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be greater than 0")
        self.rpm = rpm
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            wait_for = 0.0
            with self._lock:
                now = self._time_fn()
                self._trim(now)
                if len(self._requests) < self.rpm:
                    self._requests.append(now)
                    return
                wait_for = max(0.0, 60.0 - (now - self._requests[0]))
            if wait_for > 0:
                self._sleep_fn(wait_for)

    def _trim(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= 60.0:
            self._requests.popleft()


class OCRProcessor:
    """Extract text from images using PPIO's OpenAI-compatible VLM endpoint."""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        show_log: bool = False,
        config_path: Path | None = None,
    ):
        self.use_gpu = use_gpu
        self.lang = lang
        self.show_log = show_log
        self._config_path = config_path
        self._settings: VLMSettings | None = None
        self._client = None
        self._rate_limiter: RollingRateLimiter | None = None

    @property
    def settings(self) -> VLMSettings:
        if self._settings is None:
            self._settings = VLMSettings.from_env(self._config_path)
        return self._settings

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only in live runtime setup
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

    def extract_text(
        self,
        image_path: Path,
        return_confidence: bool = False,
    ) -> str | tuple[str, float]:
        result = self._extract_page_result_with_retries(Path(image_path), page_number=1)
        if return_confidence:
            return result.text, result.confidence or 1.0
        return result.text

    def extract_from_images(self, image_paths: list[Path]) -> list[OCRPageResult]:
        normalized_paths = [Path(image_path) for image_path in image_paths]
        if not normalized_paths:
            return []

        results_map: dict[int, OCRPageResult] = {}
        max_workers = min(len(normalized_paths), self.settings.max_concurrency)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_page_result_with_retries, image_path, index + 1
                ): index
                for index, image_path in enumerate(normalized_paths)
            }
            from tqdm import tqdm as _tqdm

            for future in _tqdm(
                as_completed(futures),
                total=len(futures),
                desc="OCR",
                unit="page",
                leave=False,
                dynamic_ncols=True,
            ):
                index = futures[future]
                try:
                    results_map[index] = future.result()
                except Exception as exc:
                    from datetime import datetime, timezone

                    image_path = normalized_paths[index]
                    logger.error(
                        "OCR failed for %s: %s",
                        image_path,
                        self._format_exception_summary(exc),
                    )
                    results_map[index] = OCRPageResult(
                        image_path=image_path,
                        page_number=index + 1,
                        text="",
                        processed_at=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=0.0,
                        status="failed",
                        error=self._format_exception_summary(exc),
                    )

        return [results_map[i] for i in range(len(normalized_paths))]

    def process_and_save(
        self,
        page_results: list[OCRPageResult],
        output_path: Path,
        pdf_source: str | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._render_markdown(page_results, pdf_source=pdf_source)

        output_path.write_text(content, encoding="utf-8")
        logger.info("Saved %s", output_path)
        return output_path

    def _render_markdown(
        self, page_results: list[OCRPageResult], pdf_source: str | None = None
    ) -> str:
        from datetime import datetime, timezone

        # Calculate document-level metadata
        total_pages = len(page_results)
        successful_pages = sum(1 for r in page_results if r.status == "success")
        failed_pages = total_pages - successful_pages

        # Get model from first successful result, fallback to settings
        ocr_model = None
        for r in page_results:
            if r.model:
                ocr_model = r.model
                break
        if not ocr_model:
            ocr_model = self.settings.model

        # Get last processed timestamp
        processed_at = ""
        for r in reversed(page_results):
            if r.processed_at:
                processed_at = r.processed_at
                break
        if not processed_at:
            processed_at = datetime.now(timezone.utc).isoformat()

        # Build frontmatter
        frontmatter_lines = [
            "---",
            f"source: {pdf_source or 'unknown'}",
            f"pages: {total_pages}",
            f"successful_pages: {successful_pages}",
            f"failed_pages: {failed_pages}",
            f"ocr_model: {ocr_model}",
            f"processed_at: {processed_at}",
            "---",
            "",
        ]

        # Build page sections
        page_sections: list[str] = []
        for result in page_results:
            section_lines = [f"## Page {result.page_number}", ""]

            # Metadata lines for this page
            section_lines.append(f"**Image:** `{result.image_path}`  ")
            section_lines.append(f"**Processed:** {result.processed_at}  ")
            section_lines.append(f"**Duration:** {result.duration_seconds:.2f}s  ")

            if result.model:
                section_lines.append(f"**Model:** `{result.model}`  ")

            if result.confidence is not None:
                section_lines.append(f"**Confidence:** {result.confidence:.2f}  ")

            if result.prompt_tokens is not None:
                section_lines.append(f"**Prompt Tokens:** {result.prompt_tokens}  ")

            if result.completion_tokens is not None:
                section_lines.append(f"**Completion Tokens:** {result.completion_tokens}  ")

            if result.total_tokens is not None:
                section_lines.append(f"**Total Tokens:** {result.total_tokens}  ")

            if result.status == "failed":
                section_lines.append(f"**Status:** {result.status}  ")
                if result.error:
                    section_lines.append(f"**Error:** {result.error}  ")

            section_lines.append("")

            # Content
            if result.status == "success" and result.text:
                section_lines.append(result.text)
            elif result.status == "failed":
                section_lines.append("_No text extracted._")
            else:
                section_lines.append("_Empty page._")

            page_sections.append("\n".join(section_lines))

        # Combine everything
        all_content = "\n\n---\n\n".join(page_sections)
        return "\n".join(frontmatter_lines) + "\n" + all_content

    def _extract_page_text_with_retries(self, image_path: Path) -> str:
        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._extract_page_text(image_path)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Transient OCR failure for %s on attempt %s/%s: %s",
                    image_path,
                    attempt,
                    attempts,
                    self._format_exception_summary(exc),
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("unreachable")

    def _extract_page_text(self, image_path: Path) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.rate_limiter.acquire()
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.settings.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._build_image_data_url(image_path),
                                "detail": self.settings.image_detail,
                            },
                        },
                    ],
                }
            ],
            timeout=self.settings.timeout_seconds,
        )
        return self._extract_message_text(response).strip()

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
            text_parts = []
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
    def _build_image_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
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

            if OCRProcessor._has_transient_status_code(current):
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
        parts = [
            f"exception_type={type(exc).__name__}",
            f"message={exc}",
        ]

        status_code = OCRProcessor._extract_status_code(exc)
        if status_code is not None:
            parts.append(f"status_code={status_code}")

        response_body = OCRProcessor._extract_response_body(exc)
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
            return OCRProcessor._truncate_log_value(text.strip())

        json_loader = getattr(response, "json", None)
        if callable(json_loader):
            try:
                return OCRProcessor._truncate_log_value(
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

    def _extract_page_result_with_retries(
        self, image_path: Path, page_number: int
    ) -> OCRPageResult:
        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._extract_page_result(image_path, page_number)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    from datetime import datetime, timezone

                    error_msg = self._format_exception_summary(exc)
                    logger.error("OCR failed for %s: %s", image_path, error_msg)
                    return OCRPageResult(
                        image_path=image_path,
                        page_number=page_number,
                        text="",
                        processed_at=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=0.0,
                        status="failed",
                        error=error_msg,
                    )
                backoff_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Transient OCR failure for %s on attempt %s/%s: %s",
                    image_path,
                    attempt,
                    attempts,
                    self._format_exception_summary(exc),
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("unreachable")

    def _extract_page_result(self, image_path: Path, page_number: int) -> OCRPageResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        from datetime import datetime, timezone

        start_time = time.perf_counter()
        self.rate_limiter.acquire()

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.settings.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._build_image_data_url(image_path),
                                "detail": self.settings.image_detail,
                            },
                        },
                    ],
                }
            ],
            timeout=self.settings.timeout_seconds,
        )

        duration = time.perf_counter() - start_time
        text = self._extract_message_text(response).strip()
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

        return OCRPageResult(
            image_path=image_path,
            page_number=page_number,
            text=text,
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            model=self.settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
