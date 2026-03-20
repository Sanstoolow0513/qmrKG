"""PNG to text extraction backed by PPIO VLM OCR."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .llm_config import TaskLLMSettings
from .llm_factory import LLMFactory, TaskLLMRunner
from .llm_types import LLMResponse
from .rate_limit import RollingRateLimiter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VLMSettings(TaskLLMSettings):
    """Backward-compatible OCR settings wrapper."""

    @classmethod
    def from_env(cls, config_path: Path | None = None) -> "VLMSettings":
        settings = TaskLLMSettings.from_env("ocr", config_path)
        return cls(**asdict(settings))


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


class OCRProcessor:
    """Extract text from images using task-scoped VLM settings."""

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
        self._runner: TaskLLMRunner | None = None

    @property
    def settings(self) -> VLMSettings:
        if self._settings is None:
            self._settings = VLMSettings.from_env(self._config_path)
        return self._settings

    @property
    def runner(self) -> TaskLLMRunner:
        if self._runner is None:
            self._runner = LLMFactory(self._config_path).create("ocr", client=self._client)
        return self._runner

    @property
    def client(self):
        return self.runner.client

    @property
    def rate_limiter(self) -> RollingRateLimiter:
        return self.runner.rate_limiter

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
                    logger.error("OCR failed for %s: %s", normalized_paths[index], exc)
                    results_map[index] = self._build_failed_page_result(
                        normalized_paths[index],
                        page_number=index + 1,
                        error_message=str(exc),
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

        total_pages = len(page_results)
        successful_pages = sum(1 for r in page_results if r.status == "success")
        failed_pages = total_pages - successful_pages

        ocr_model = next((r.model for r in page_results if r.model), None)
        if not ocr_model:
            ocr_model = self.settings.model

        processed_at = next((r.processed_at for r in reversed(page_results) if r.processed_at), "")
        if not processed_at:
            processed_at = datetime.now(timezone.utc).isoformat()

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

        page_sections: list[str] = []
        for result in page_results:
            section_lines = [f"## Page {result.page_number}", ""]
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

            if result.status == "success" and result.text:
                section_lines.append(result.text)
            elif result.status == "failed":
                section_lines.append("_No text extracted._")
            else:
                section_lines.append("_Empty page._")

            page_sections.append("\n".join(section_lines))

        all_content = "\n\n---\n\n".join(page_sections)
        return "\n".join(frontmatter_lines) + "\n" + all_content

    def _extract_page_result_with_retries(
        self, image_path: Path, page_number: int
    ) -> OCRPageResult:
        try:
            return self._extract_page_result(image_path, page_number)
        except Exception as exc:
            error_msg = self.runner._format_exception_summary(exc)
            logger.error("OCR failed for %s: %s", image_path, error_msg)
            return self._build_failed_page_result(image_path, page_number, error_msg)

    def _extract_page_result(self, image_path: Path, page_number: int) -> OCRPageResult:
        image_path = Path(image_path)
        response = self.runner.run_image(self.settings.prompt, image_path)
        return self._response_to_page_result(response, image_path=image_path, page_number=page_number)

    @staticmethod
    def _response_to_page_result(
        response: LLMResponse, *, image_path: Path, page_number: int
    ) -> OCRPageResult:
        return OCRPageResult(
            image_path=image_path,
            page_number=page_number,
            text=response.text,
            processed_at=response.processed_at,
            duration_seconds=response.duration_seconds,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            status=response.status,
            error=response.error,
        )

    def _build_failed_page_result(
        self, image_path: Path, page_number: int, error_message: str
    ) -> OCRPageResult:
        from datetime import datetime, timezone

        return OCRPageResult(
            image_path=image_path,
            page_number=page_number,
            text="",
            processed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0.0,
            status="failed",
            error=error_message,
        )
