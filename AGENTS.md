# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `src/qmrkg/`. Keep PDF rendering in `pdf_to_png.py`, OCR logic in `png_to_text.py`, and orchestration in `pipeline.py`. `main.py` is the CLI entry point, and `examples.py` shows direct API usage. Runtime data stays under `data/`: place inputs in `data/pdf/`, intermediate images in `data/png/`, and OCR output in `data/markdown/`.

## Build, Test, and Development Commands
Use Python 3.13 with `uv` when possible.

- `uv pip install -e .` installs the package for local CLI and module development.
- `uv pip install -e ".[dev]"` adds `pytest`, `black`, and `ruff`.
- `python main.py` processes all PDFs in `data/pdf/`.
- `python main.py --pdf data/pdf/example.pdf --lang en` runs a single-file pipeline with explicit OCR settings.
- `python main.py --stats` prints counts for PDFs, generated images, and text outputs.
- `pytest` runs tests once a `tests/` suite exists.
- `ruff check .` and `black .` handle linting and formatting.

## Coding Style & Naming Conventions
Follow Black and Ruff defaults configured in `pyproject.toml`: 4-space indentation, 100-character line limit, and Python 3.13 syntax. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `PDFPipeline` and `OCRProcessor`. Prefer type hints and `pathlib.Path` for filesystem work. Keep modules focused on one stage of the pipeline.

## Testing Guidelines
Add tests under `tests/` with filenames like `test_pipeline.py` and test names like `test_process_pdf_returns_text_path`. Cover CLI argument handling, per-stage helpers, and failure cases such as missing files or OCR import errors. For OCR-heavy code, prefer fixtures or mocks over real model downloads so `pytest` stays fast and deterministic.

## Commit & Pull Request Guidelines
Current history uses Conventional Commits (`feat: implement PDF to PNG to Text pipeline for KG construction`), so keep using prefixes like `feat:`, `fix:`, and `docs:`. PRs should include a short summary, the commands used for verification, and sample output or screenshots when CLI behavior changes. Link related issues and call out any dependency, model-download, or GPU assumptions.

## Security & Configuration Tips
Do not commit PDFs with sensitive content or large generated artifacts from `data/png/` and `data/markdown/`. PaddleOCR may download models on first run, so note network requirements in reviews. Keep local paths configurable through CLI flags instead of hardcoding machine-specific directories.
