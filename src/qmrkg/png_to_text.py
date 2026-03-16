"""PNG to text extraction backed by SiliconFlow VLM OCR."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for environments without optional deps installed

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"
DEFAULT_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
DEFAULT_PROMPT = (
    "Transcribe all visible text from this page exactly as written. Preserve reading order and "
    "line breaks where possible. Do not add commentary, summaries, or markdown fences."
)
DEFAULT_RPM = 60
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3


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


@dataclass(slots=True)
class VLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    rpm: int = DEFAULT_RPM
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls) -> "VLMSettings":
        load_dotenv()

        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")

        return cls(
            api_key=api_key,
            base_url=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL).strip()
            or DEFAULT_BASE_URL,
            model=os.getenv("SILICONFLOW_VLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            prompt=os.getenv("SILICONFLOW_VLM_PROMPT", DEFAULT_PROMPT).strip() or DEFAULT_PROMPT,
            rpm=_read_int("SILICONFLOW_RPM", DEFAULT_RPM),
            max_concurrency=_read_int("SILICONFLOW_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY),
            timeout_seconds=_read_float(
                "SILICONFLOW_TIMEOUT_SECONDS",
                DEFAULT_TIMEOUT_SECONDS,
            ),
            max_retries=_read_int("SILICONFLOW_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        )


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
    """Extract text from images using SiliconFlow's OpenAI-compatible VLM endpoint."""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        show_log: bool = False,
    ):
        self.use_gpu = use_gpu
        self.lang = lang
        self.show_log = show_log
        self._settings: VLMSettings | None = None
        self._client = None
        self._rate_limiter: RollingRateLimiter | None = None

    @property
    def settings(self) -> VLMSettings:
        if self._settings is None:
            self._settings = VLMSettings.from_env()
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
        image_path = Path(image_path)
        text = self._extract_page_text_with_retries(image_path)
        if return_confidence:
            return text, 1.0
        return text

    def extract_from_images(
        self,
        image_paths: list[Path],
        return_confidence: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        normalized_paths = [Path(image_path) for image_path in image_paths]
        if not normalized_paths:
            return []

        results: list[str | tuple[str, float] | None] = [None] * len(normalized_paths)
        max_workers = min(len(normalized_paths), self.settings.max_concurrency)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._extract_page_text_with_retries, image_path): index
                for index, image_path in enumerate(normalized_paths)
            }
            for future in as_completed(futures):
                index = futures[future]
                image_path = normalized_paths[index]
                try:
                    text = future.result()
                    results[index] = (text, 1.0) if return_confidence else text
                except Exception as exc:
                    logger.error("OCR failed for %s: %s", image_path, exc)
                    results[index] = ("", 0.0) if return_confidence else ""

        return list(results)

    def process_and_save(
        self,
        image_paths: list[Path],
        output_path: Path,
        page_separator: str = "\n\n--- Page {page} ---\n\n",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extracted_texts = self.extract_from_images(image_paths)
        all_texts: list[str] = []
        for page_number, text in enumerate(extracted_texts, 1):
            if isinstance(text, str) and text.strip():
                all_texts.append(f"{page_separator.format(page=page_number)}{text}")

        output_path.write_text("".join(all_texts), encoding="utf-8")
        logger.info("Saved OCR text to: %s", output_path)
        return output_path

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
                    exc,
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
                            "image_url": {"url": self._build_image_data_url(image_path)},
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
