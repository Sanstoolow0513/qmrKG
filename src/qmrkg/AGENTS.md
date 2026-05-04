# QmrKG Python Package

**Location:** `src/qmrkg/`

## OVERVIEW

Core Python pipeline for PDF-to-Knowledge-Graph processing. 25 modules handling document conversion, OCR, chunking, entity extraction, embedding canonicalization, and Neo4j import.

## STRUCTURE

```
src/qmrkg/
├── __init__.py             # Package exports
├── pipeline.py             # PDFPipeline: PDF → PNG → Markdown
├── pdf_to_png.py           # PDFConverter: PyMuPDF rendering
├── png_to_text.py          # OCRProcessor: VLM-based OCR
├── markdown_chunker.py     # MarkdownChunker: token-aware splitting
├── kg_extractor.py         # KGExtractor: entity/relation extraction
├── kg_merger.py            # KGMerger: triple dedup + embedding canonicalization
├── kg_neo4j.py             # KGNeo4jLoader: graph import
├── evaluation.py           # Evaluation: precision/recall/F1 against gold triples
├── kg_schema.py            # Data models (Entity, Triple)
├── llm_factory.py          # TaskLLMFactory: rate-limited LLM calls (text+multimodal+embedding)
├── llm_config.py           # TaskLLMSettings: YAML + env config
├── llm_types.py            # LLM message/response types
├── rate_limit.py           # RollingRateLimiter
├── config.py               # Pipeline run config loader
├── tqdm_logging.py         # Progress bar utilities
└── cli_*.py                # 10 CLI entry points (incl. kgeval)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| End-to-end pipeline | `pipeline.py` | `PDFPipeline` class orchestrates stages |
| PDF rendering | `pdf_to_png.py` | `PDFConverter` uses PyMuPDF |
| OCR extraction | `png_to_text.py` | `OCRProcessor` with multimodal LLM |
| Text chunking | `markdown_chunker.py` | Token-based with overlap |
| Markdown merging | `cli_kg_md_combine.py` | Per-page → book MD |
| Triple extraction | `kg_extractor.py` | NER + RE via LLM (zs/fs modes) |
| Triple merging | `kg_merger.py` | Entity resolution, relation dedup, embedding canonicalization |
| Neo4j loading | `kg_neo4j.py` | Cypher-based bulk import |
| LLM orchestration | `llm_factory.py` | Factory for text/multimodal/embedding tasks |
| Configuration | `llm_config.py` | YAML parsing + env overrides |
| Rate limiting | `rate_limit.py` | `RollingRateLimiter` (NOTE: not `RollingWindowRateLimiter`) |
| Evaluation | `evaluation.py` | Precision, recall, F1 against gold triples |
| Run defaults | `config.py` | Pipeline stage defaults |

## KEY CLASSES

| Class | File | Purpose |
|-------|------|---------|
| `PDFPipeline` | pipeline.py | Orchestrates PDF→PNG→Markdown flow |
| `PDFConverter` | pdf_to_png.py | Renders PDF pages to PNG images |
| `OCRProcessor` | png_to_text.py | VLM OCR for image→text |
| `MarkdownChunker` | markdown_chunker.py | Splits markdown into JSON chunks |
| `KGExtractor` | kg_extractor.py | Extracts entities/triples from chunks (zs/fs) |
| `KGMerger` | kg_merger.py | Merges, deduplicates, canonicalizes entities |
| `KGNeo4jLoader` | kg_neo4j.py | Imports triples to Neo4j |
| `TextTaskProcessor` | llm_factory.py | Text-only LLM tasks |
| `MultimodalTaskProcessor` | llm_factory.py | Image+text LLM tasks |
| `EmbeddingTaskProcessor` | llm_factory.py | Embedding generation tasks |
| `TaskLLMSettings` | llm_config.py | Configuration management |
| `RunConfig` | config.py | Pipeline stage default settings |
| `RollingRateLimiter` | rate_limit.py | Per-task rate limiting |
| `TripleEvaluator` | evaluation.py | Merged triple evaluation against gold |

## CONVENTIONS

### Class Design
- **Factory Pattern:** All LLM interactions go through `llm_factory.py`
- **Pipeline Stages:** Each stage has dedicated processor class
- **CLI Separation:** Each CLI command in separate `cli_*.py` file
- **Settings Classes:** Config-driven via `TaskLLMSettings`

### Type Hints
- Required throughout codebase
- Use `pathlib.Path` for paths
- Use `Optional[]` and `List[]` from typing

### Error Handling
- Custom exceptions where appropriate
- Validation in `llm_config.py` raises descriptive errors
- CLI commands catch and log errors with user-friendly messages

### Rate Limiting
All LLM calls use `RollingRateLimiter`:
```python
async with self.rate_limiter.acquire():
    response = await self._call_llm(...)
```

## ANTI-PATTERNS (DO NOT)

- ❌ **Don't call OpenAI directly** - always use `llm_factory.py`
- ❌ **Don't skip rate limiting** - all LLM calls must use `RollingRateLimiter`
- ❌ **Don't hardcode paths** - use `pathlib.Path` everywhere
- ❌ **Don't use `os.path`** - use `Path` methods instead
- ❌ **Don't exceed 100 char lines** - black/ruff enforces this
- ❌ **Don't use Python <3.13** - type hints require 3.13 features
- ❌ **Don't import from `llm_factory` directly in tests** - mock via `monkeypatch`
- ❌ **Don't use deprecated `openai:` top-level key** in config.yaml — raises error
- ❌ **Don't use `SILICONFLOW_*` env vars** — use `PPIO_*` equivalents
- ❌ **Don't omit `evidence` field** in extracted triples

## TESTING

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module tests
uv run pytest tests/test_kg_extractor.py -v
uv run pytest tests/test_llm_factory.py -v
```

### Test Patterns
- Use `FakeClient`/`FakeResponse` for LLM mocking
- Use `monkeypatch` for environment/config mocking
- Use `tmp_path` fixture for temp files
- Use `@pytest.fixture` for reusable test setup

## NOTES

- **Task Factory:** All LLM tasks go through factory - never direct API calls
- **Concurrency:** Uses `asyncio` with semaphore-based rate limiting
- **Chunking:** Token-based with configurable overlap
- **Entity Types:** protocol, concept, mechanism, metric (defined in kg_schema.py)
- **Relation Types:** contains, depends_on, compared_with, applied_to
- **Extraction Modes:** zero-shot / few-shot via `extract.prompts` in config.yaml
- **Embedding Canonicalization:** Optional, configured in `kg_merge.embedding` (config.yaml)
- **kgmdcombine:** OCR outputs per-page files; kgmdcombine merges them into book-level MD before chunking
- **Evaluation:** `kgeval` CLI → `evaluation.py` → precision/recall/F1 against gold triples
- **Rate limiter class:** `RollingRateLimiter` (not `RollingWindowRateLimiter` as stated in older docs)
