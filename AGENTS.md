# QmrKG Project Knowledge Base

**Generated:** 2026-04-05  
**Commit:** 9aa151b  
**Branch:** master  

## OVERVIEW

PDF-to-Knowledge Graph pipeline using Python 3.13. Converts PDF documents through OCR to markdown, then extracts knowledge graph triples using LLMs (PPIO API). Supports OCR, NER, and RE tasks with a task-scoped LLM factory pattern.

## STRUCTURE

```
.
├── src/qmrkg/          # Main source package (21 modules)
├── tests/              # Test suite (11 test files)
├── data/               # Runtime data (pdf, png, markdown, chunks, triples)
├── docs/               # Documentation and specs
├── config.yaml         # Task-specific LLM configuration
├── examples.py         # Usage examples
└── pyproject.toml      # Project configuration
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| PDF processing | `src/qmrkg/pdf_to_png.py` | PDF to PNG conversion |
| OCR/VLM | `src/qmrkg/png_to_text.py` | Image to markdown via LLM |
| Chunking | `src/qmrkg/markdown_chunker.py` | Markdown to JSON chunks |
| KG extraction | `src/qmrkg/kg_extractor.py` | Entity/relation extraction |
| KG merging | `src/qmrkg/kg_merger.py` | Triple consolidation |
| Neo4j import | `src/qmrkg/kg_neo4j.py` | Graph database loader |
| LLM factory | `src/qmrkg/llm_factory.py` | Task-scoped processor creation |
| CLI entry | `src/qmrkg/cli_*.py` | 7 stage-specific CLI commands |
| Pipeline orchestration | `src/qmrkg/pipeline.py` | Full PDF-to-text pipeline |
| Config loading | `src/qmrkg/llm_config.py` | YAML + env var resolution |
| Tests | `tests/test_*.py` | pytest-based test suite |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PDFPipeline | class | pipeline.py | Main orchestration |
| PDFConverter | class | pdf_to_png.py | PDF rendering |
| OCRProcessor | class | png_to_text.py | VLM-based OCR |
| TextTaskProcessor | class | llm_factory.py | Text LLM wrapper |
| MultimodalTaskProcessor | class | llm_factory.py | VLM wrapper |
| KGExtractor | class | kg_extractor.py | Entity/relation extraction |
| KGMerger | class | kg_merger.py | Triple deduplication |
| KGNeo4jLoader | class | kg_neo4j.py | Neo4j bulk import |
| MarkdownChunker | class | markdown_chunker.py | Semantic chunking |
| LLMConfig | class | llm_config.py | Configuration management |

## CONVENTIONS

### Code Style
- **Line length:** 100 chars (not 88) - configured in pyproject.toml
- **Python version:** 3.13 only (`>=3.13,<3.14`)
- **Formatter:** Black with py313 target
- **Linter:** Ruff with py313 target
- **Naming:** snake_case for modules/functions, PascalCase for classes
- **Type hints:** Required throughout
- **Path handling:** Use `pathlib.Path` (not os.path)

### Project Structure
- **src-layout:** Code lives in `src/qmrkg/`, not at root
- **CLI pattern:** Each stage has dedicated `cli_*.py` module
- **Task factory:** All LLM interactions go through `llm_factory.py`
- **Config-driven:** Task behavior controlled via `config.yaml` sections

### Task Configuration (config.yaml)
Tasks use scoped sections: `ocr:`, `ner:`, `re:`, `extract:`
Each task specifies:
- `provider`: name, base_url, model, modality
- `prompts`: task-specific prompt templates
- `request`: timeout, retries, thinking flags
- `rate_limit`: rpm, max_concurrency

## ANTI-PATTERNS

### Rejected Configurations
- **Legacy `openai:` section** → Use task-scoped sections (`ocr:`, `extract:`)
- **SILICONFLOW_* env vars** → Use `PPIO_*` equivalents
- **Invalid image_detail** → Must be "low", "high", or "auto"
- **Images to text-only tasks** → Text tasks reject image input

### Testing
- Don't use real API calls in tests → Use FakeClient/FakeResponse mocks
- Don't skip tmp_path cleanup → Tests use pytest fixtures properly

## CLI COMMANDS

```bash
# Installation
uv pip install -e ".[dev]"

# Full pipeline
uv run python main.py --pdf data/pdf/example.pdf

# Stage commands
uv run qmrkg --list                    # List commands
uv run pdftopng --pdf <path>           # PDF → PNG
uv run pngtotext --image <path>        # PNG → Markdown
uv run mdchunk --markdown <path>       # Markdown → Chunks
uv run kgextract --input data/chunks   # Extract triples
uv run kgmerge                         # Merge triples
uv run kgneo4j --import <path>         # Load to Neo4j

# Testing
pytest tests/ -v

# Linting
ruff check .
black --check .
```

## NOTES

### Data Flow
```
PDF → (pdftopng) → PNG → (pngtotext) → Markdown → (mdchunk) → JSON chunks → (kgextract) → Triples → (kgmerge) → Merged → (kgneo4j) → Neo4j
```

### Rate Limiting
All LLM calls respect per-task rate limits from `config.yaml`. Uses semaphore-based concurrency control in `llm_factory.py`.

### Chinese Content
OCR prompts optimized for Chinese text with specific heading detection rules (第X章, 第X节, etc.).

### Environment Variables
Sensitive config in `.env` (PPIO_API_KEY). Non-sensitive overrides via `PPIO_*` env vars matching config.yaml paths.

### CI/CD
- Uses GitHub Actions for opencode AI assistant only
- No automated test/build workflows (run locally)
- Conventional Commits preferred (`feat:`, `fix:`, `docs:`)
