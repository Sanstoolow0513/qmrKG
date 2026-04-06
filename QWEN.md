# QmrKG - Project Context

## Project Overview

**QmrKG** is a PDF-to-Knowledge-Graph pipeline built in Python 3.13. It converts PDF documents through OCR to Markdown, then extracts knowledge graph triples using LLMs via the PPIO API. The project is designed as a graduation project focused on Named Entity Recognition (NER) and Relation Extraction (RE) for computer networking course knowledge graphs.

### Core Data Flow
```
PDF → (pdftopng) → PNG → (pngtotext) → Markdown → (mdchunk) → JSON chunks → (kgextract) → Triples → (kgmerge) → Merged Triples → (kgneo4j) → Neo4j
```

### Key Capabilities
- **OCR**: PDF page rendering to images, then VLM-based OCR to Markdown (optimized for Chinese content)
- **Chunking**: Semantic Markdown chunking with token-aware splitting
- **KG Extraction**: LLM-driven NER + relation extraction producing structured triples
- **KG Merging**: Triple deduplication and consolidation
- **Neo4j Import**: Graph database bulk loading
- **Task-Scoped LLM Factory**: Unified interface for text and multimodal LLM tasks with per-task rate limiting, retries, and configuration

## Project Structure

```
.
├── src/qmrkg/              # Main source package (21 modules)
│   ├── pdf_to_png.py       # PDF to PNG conversion (pymupdf)
│   ├── png_to_text.py      # Image to Markdown via VLM OCR
│   ├── markdown_chunker.py # Markdown to JSON chunks
│   ├── kg_extractor.py     # Entity/relation extraction
│   ├── kg_merger.py        # Triple deduplication
│   ├── kg_neo4j.py         # Neo4j bulk import
│   ├── kg_schema.py        # Data models (Entity, Triple, etc.)
│   ├── llm_factory.py      # Task-scoped LLM processor creation
│   ├── llm_config.py       # YAML + env var config resolution
│   ├── llm_types.py        # LLM message/response types
│   ├── rate_limit.py       # Rolling rate limiter
│   ├── pipeline.py         # Full PDF-to-text pipeline
│   ├── cli_*.py            # 7 stage-specific CLI commands
│   └── tqdm_logging.py     # Progress bar utilities
├── tests/                  # Test suite (11 test files, pytest-based)
├── data/                   # Runtime data dirs (pdf, png, markdown, chunks, triples)
├── docs/                   # Documentation and specs
├── config.yaml             # Task-specific LLM configuration
├── examples.py             # Usage examples
└── pyproject.toml          # Project configuration
```

### Key Classes and Locations

| Class | Location | Role |
|-------|----------|------|
| `PDFPipeline` | `pipeline.py` | Main orchestration (full pipeline) |
| `PDFConverter` | `pdf_to_png.py` | PDF page to PNG image conversion |
| `OCRProcessor` | `png_to_text.py` | VLM-based OCR text extraction |
| `TextTaskProcessor` | `llm_factory.py` | Text-only LLM task wrapper |
| `MultimodalTaskProcessor` | `llm_factory.py` | Multimodal (image+text) LLM task wrapper |
| `KGExtractor` | `kg_extractor.py` | Entity and relation extraction from chunks |
| `KGMerger` | `kg_merger.py` | Triple deduplication and merging |
| `KGNeo4jLoader` | `kg_neo4j.py` | Neo4j graph database loader |
| `MarkdownChunker` | `markdown_chunker.py` | Semantic markdown chunking |
| `TaskLLMSettings` | `llm_config.py` | Configuration management (YAML + env) |

## Building and Running

### Prerequisites
- **Python 3.13** (strictly `>=3.13,<3.14`)
- **uv** package manager (recommended)
- PPIO API key (set in `.env`)

### Installation
```bash
# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Configuration
```bash
# Copy and configure .env
cp .env.example .env
# Edit .env with your PPIO_API_KEY
```

### CLI Commands
```bash
# List available commands
uv run qmrkg --list

# Stage-specific commands
uv run pdftopng --pdf data/pdf/example.pdf           # PDF → PNG
uv run pngtotext --image data/png/example.png         # PNG → Markdown (OCR)
uv run mdchunk --markdown data/markdown/example.md    # Markdown → JSON chunks
uv run kgextract --input data/chunks                   # Extract KG triples
uv run kgmerge                                         # Merge/deduplicate triples
uv run kgneo4j --import <path>                        # Load to Neo4j

# Full pipeline (via main.py)
uv run python main.py --pdf data/pdf/example.pdf
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_llm_factory.py -v
```

### Linting and Formatting
```bash
# Check with ruff
ruff check .

# Check formatting
black --check .

# Auto-fix
ruff check . --fix
black .
```

## Development Conventions

### Code Style
- **Line length:** 100 characters (configured in `pyproject.toml`)
- **Formatter:** Black with `py313` target version
- **Linter:** Ruff with `py313` target version
- **Type hints:** Required throughout
- **Naming:** `snake_case` for modules/functions, `PascalCase` for classes
- **Path handling:** Use `pathlib.Path` (not `os.path`)

### Architecture Patterns
- **src-layout:** All code lives in `src/qmrkg/`
- **CLI pattern:** Each pipeline stage has a dedicated `cli_*.py` module
- **Task factory:** All LLM interactions go through `llm_factory.py` — never call the API directly
- **Config-driven:** Task behavior is controlled via `config.yaml` sections
- **Rate limiting:** All LLM calls respect per-task rate limits using semaphore-based concurrency

### Configuration (config.yaml)
Tasks use scoped sections: `ocr:`, `extract:`, etc. Each specifies:
- `provider`: name, base_url, model, modality (`text` or `multimodal`)
- `prompts`: task-specific prompt templates
- `request`: timeout, retries, thinking flags
- `rate_limit`: rpm, max_concurrency

### Environment Variables
- **`.env`** file for sensitive config (e.g., `PPIO_API_KEY`)
- **`PPIO_*`** env vars for non-sensitive overrides of `config.yaml` values
- **Never** commit API keys or secrets to the repository

### Testing
- Use `pytest` fixtures — no real API calls in tests (use FakeClient/FakeResponse mocks)
- Tests live in `tests/` with `test_*.py` naming
- Always use `tmp_path` fixtures for file system test cleanup

### Git Conventions
- Conventional Commits preferred (`feat:`, `fix:`, `docs:`, etc.)
- Review changes with `git status && git diff HEAD && git log -n 3` before committing

## Anti-Patterns

### Config
- **Do NOT** use legacy `openai:` top-level section — use task-scoped sections
- **Do NOT** use `SILICONFLOW_*` env vars — use `PPIO_*` equivalents
- **Do NOT** use invalid `image_detail` values — must be `"low"`, `"high"`, or `"auto"`
- **Do NOT** pass images to text-only tasks — the task runner will reject them

### Testing
- **Do NOT** use real API calls — use mocks (FakeClient/FakeResponse)
- **Do NOT** skip `tmp_path` cleanup — use proper pytest fixtures

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | PDF rendering (no poppler needed) |
| `openai` | PPIO OpenAI-compatible API client |
| `pillow` | Image processing |
| `python-dotenv` | `.env` file loading |
| `tiktoken` | Token counting for chunking |
| `tqdm` | Progress bars |
| `pyyaml` | YAML config parsing |
| `neo4j` | Neo4j graph database driver |
