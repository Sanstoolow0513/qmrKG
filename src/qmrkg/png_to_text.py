"""PNG to text extraction backed by PPIO VLM OCR."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .llm_config import TaskLLMSettings
from .llm_factory import LLMFactory, TaskLLMRunner
from .llm_types import LLMResponse
from .rate_limit import RollingRateLimiter

logger = logging.getLogger(__name__)

_BOOK_STEM_FROM_PAGE = re.compile(r"_page_\d+$")


def book_stem_from_image_stem(stem: str) -> str:
    """Strip trailing `_page_NNNN` from a PNG stem (same rule as cli_png_to_text)."""
    return _BOOK_STEM_FROM_PAGE.sub("", stem)


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

    def page_markdown_path(self, image_path: Path, text_dir: Path) -> Path:
        """Per-page markdown path: ``text_dir / {book_stem} / {image_stem}.md``."""
        image_path = Path(image_path)
        text_dir = Path(text_dir)
        book = book_stem_from_image_stem(image_path.stem)
        return text_dir / book / f"{image_path.stem}.md"

    def check_page_md_done(self, image_path: Path, text_dir: Path) -> bool:
        """Return True if the page's markdown exists and has non-whitespace content."""
        md_path = self.page_markdown_path(image_path, text_dir)
        if not md_path.is_file():
            return False
        try:
            return bool(md_path.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    @staticmethod
    def _extract_body_from_saved_page_markdown(content: str) -> str:
        """Recover OCR body text from a single-page markdown file written by this module."""
        body = content
        stripped = content.lstrip()
        if stripped.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2]
        lines = body.lstrip("\n").splitlines()
        i = 0
        while i < len(lines) and not lines[i].strip().startswith("## Page"):
            i += 1
        if i >= len(lines):
            return body.strip()
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("**")):
            i += 1
        return "\n".join(lines[i:]).strip()

    def _page_result_from_cached_md(
        self, image_path: Path, text_dir: Path, page_number: int
    ) -> OCRPageResult:
        from datetime import datetime, timezone

        md_path = self.page_markdown_path(image_path, text_dir)
        content = md_path.read_text(encoding="utf-8")
        body = self._extract_body_from_saved_page_markdown(content)
        if not body:
            body = content.strip()
        mtime = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).isoformat()
        return OCRPageResult(
            image_path=Path(image_path),
            page_number=page_number,
            text=body,
            processed_at=mtime,
            duration_seconds=0.0,
            status="success",
        )

    def extract_text(
        self,
        image_path: Path,
        return_confidence: bool = False,
    ) -> str | tuple[str, float]:
        result = self._extract_page_result_with_retries(Path(image_path), page_number=1)
        if return_confidence:
            return result.text, result.confidence or 1.0
        return result.text

    def extract_from_images(
        self,
        image_paths: list[Path],
        *,
        text_dir: Path | None = None,
        skip_existing_page_md: bool = False,
        show_progress: bool = True,
    ) -> list[OCRPageResult]:
        normalized_paths = [Path(image_path) for image_path in image_paths]
        if not normalized_paths:
            return []

        results_map: dict[int, OCRPageResult] = {}
        pending: list[tuple[int, Path]] = []

        for index, image_path in enumerate(normalized_paths):
            if (
                skip_existing_page_md
                and text_dir is not None
                and self.check_page_md_done(image_path, text_dir)
            ):
                logger.info("Skip OCR (existing page markdown): %s", image_path.name)
                results_map[index] = self._page_result_from_cached_md(
                    image_path, text_dir, index + 1
                )
            else:
                pending.append((index, image_path))

        if pending:
            max_workers = max(1, min(len(pending), self.settings.max_concurrency))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._extract_page_result_with_retries, image_path, index + 1
                    ): index
                    for index, image_path in pending
                }
                completed = as_completed(futures)
                if show_progress:
                    from tqdm import tqdm as _tqdm

                    completed = _tqdm(
                        completed,
                        total=len(futures),
                        desc="OCR",
                        unit="page",
                        leave=False,
                        dynamic_ncols=True,
                    )

                for future in completed:
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
        logger.debug("Saved %s", output_path)
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
