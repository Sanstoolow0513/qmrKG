# QmrKG (Knowledge Graph Pipeline)

## Project Overview

**QmrKG** is a graduation project focused on constructing a course knowledge graph using Large Language Models (LLMs). The project aims to automatically identify core knowledge points, extract relationships between concepts, and generate reusable knowledge graph structures from course materials (like syllabi, textbooks, and lecture notes).

### Architecture & Technologies

The workspace is organized as a monorepo consisting of:

1. **Python Data Pipeline (`src/qmrkg/`)**: The core extraction engine written in Python 3.13. It is a multi-stage CLI pipeline that converts PDF documents to a Knowledge Graph in Neo4j.
    *   **Technologies:** Python 3.13, `uv` (dependency management), PyMuPDF, OpenAI SDK (configured for PPIO API), Tiktoken, PyYAML, Neo4j Python driver.
    *   **Workflow:** PDF → PNG (`pdf_to_png.py`) → Markdown OCR via VLM (`png_to_text.py`) → JSON Chunks (`markdown_chunker.py`) → Triples Extraction via NER/RE (`kg_extractor.py`) → Triples Merging (`kg_merger.py`) → Neo4j Import (`kg_neo4j.py`).
2. **Next.js Frontend (`frontend/`)**: A web application likely intended to serve as the user interface, intelligent Q&A, or visualization platform for the extracted knowledge graph.
    *   **Technologies:** Next.js 16 (App Router), React 19, Tailwind CSS v4, TypeScript.
3. **Documentation & Data**: 
    *   `docs/reports/`: Contains the graduation thesis LaTeX templates and documentation.
    *   `docs/superpowers/`: Contains architectural design specs and development plans.
    *   `data/`: Runtime directory for storing inputs (`pdf/`), intermediate states (`png/`, `markdown/`, `chunks/`), and outputs (`triples/`).

## Building and Running

### Backend Pipeline

The Python backend uses `uv` for modern dependency management. Configurations are managed via `.env` (for secrets like `PPIO_API_KEY`) and `config.yaml` (for task-specific LLM parameters, prompts, and rate limits).

**Installation:**
```bash
# We recommend using uv
uv pip install -e ".[dev]"
```

**Running the Full Pipeline:**
```bash
# Process a single PDF
uv run python main.py --pdf data/pdf/example.pdf

# Process all PDFs in the data/pdf/ directory
uv run python main.py
```

**Running Specific Stages (CLI Tools):**
The project exposes several CLI commands mapped to specific pipeline stages. You can run them from anywhere in the project root:
```bash
uv run qmrkg --list                    # List all available commands
uv run pdftopng --pdf <path>           # Stage 1: PDF to PNG
uv run pngtotext --image <path>        # Stage 2: PNG to Markdown
uv run mdchunk --markdown <path>       # Stage 3: Markdown to Chunks
uv run kgextract --input data/chunks   # Stage 4: Extract KG Triples
uv run kgmerge                         # Stage 5: Merge Triples
uv run kgneo4j --import <path>         # Stage 6: Load into Neo4j
```

**Testing & Linting:**
```bash
pytest tests/ -v
ruff check .
black --check .
```

### Frontend

The Next.js frontend is located in the `frontend/` directory.

**Installation & Running:**
```bash
cd frontend
pnpm install
pnpm dev      # Starts the development server on localhost:3000
```

## Development Conventions

### Backend Code Style
*   **Python Version:** Strictly `>=3.13, <3.14`.
*   **Formatting & Linting:** Code is formatted with `black` (target `py313`, 100 character line limit) and linted with `ruff`.
*   **Naming:** `snake_case` for modules/functions/variables, `PascalCase` for classes.
*   **Type Hints:** Required and enforced throughout the codebase.
*   **Path Handling:** Always use `pathlib.Path` instead of the legacy `os.path`.

### Architecture & Config Conventions
*   **Task-Scoped LLM Factory:** All LLM interactions use the factory pattern (`llm_factory.py`).
*   **Configuration (`config.yaml`):** LLM configurations are strictly scoped by task (`ocr:`, `ner:`, `re:`, `extract:`). Avoid legacy `openai:` root sections. Each task configures its provider, prompts, request limits, and rate limits independently.
*   **Project Layout:** Source code must reside in `src/qmrkg/` rather than the project root. Each pipeline stage has a dedicated `cli_*.py` module to act as an entrypoint.

### Testing Practices
*   Tests should **never** make real API calls. Always use `FakeClient` or `FakeResponse` mocks.
*   Do not leave temporary files behind; properly utilize `pytest` fixtures (like `tmp_path`) for file operations.