# src/qmrkg/ Package Knowledge

**Package:** qmrkg  
**Purpose:** PDF-to-Knowledge Graph pipeline stages  
**Files:** 21 Python modules  

## STRUCTURE

```
src/qmrkg/
├── __init__.py           # Public API exports
├── pipeline.py           # PDFPipeline orchestration
├── pdf_to_png.py         # PDF rendering (pymupdf)
├── png_to_text.py        # OCR via VLM
├── markdown_chunker.py   # Semantic text chunking
├── kg_extractor.py       # LLM entity/relation extraction
├── kg_merger.py          # Triple deduplication
├── kg_neo4j.py           # Neo4j bulk loader
├── kg_schema.py          # Graph data models
├── llm_factory.py        # Task-scoped LLM processors
├── llm_config.py         # Config loading (YAML + env)
├── llm_types.py          # Type definitions
├── cli_*.py (7 files)    # Stage-specific CLI entry points
└── rate_limit.py         # Concurrency limiting
```

## WHERE TO LOOK

| Task | File | Key Classes |
|------|------|-------------|
| Full pipeline | `pipeline.py` | PDFPipeline |
| PDF → PNG | `pdf_to_png.py` | PDFConverter |
| PNG → Text | `png_to_text.py` | OCRProcessor |
| Text → Chunks | `markdown_chunker.py` | MarkdownChunker, HeaderNode |
| Extract triples | `kg_extractor.py` | KGExtractor, Triple, Entity |
| Merge graphs | `kg_merger.py` | KGMerger |
| Load to Neo4j | `kg_neo4j.py` | KGNeo4jLoader |
| Create LLM client | `llm_factory.py` | TextTaskProcessor, MultimodalTaskProcessor |
| Load config | `llm_config.py` | LLMConfig, TaskSettings |
| CLI entry | `cli_*.py` | main() functions |

## CONVENTIONS

### Module Organization
- Each stage has `cli_<stage>.py` for CLI and `<stage>.py` for library
- CLI modules parse args and call library functions
- Library modules define classes with `process()` or `run_*()` methods
- All LLM interaction flows through `llm_factory.py` (never direct API calls)

### Class Patterns
```python
# Task processors (from llm_factory.py)
processor = TextTaskProcessor("ner")  # or "re", "extract"
result = processor.run_text("input text")

# Pipeline stages
converter = PDFConverter(dpi=200)
images = converter.convert("doc.pdf")

ocr = OCRProcessor()
text = ocr.extract_text("page.png")
```

### Error Handling
- Use custom exceptions for validation errors
- Config errors raise with descriptive messages (see `llm_config.py`)
- API errors bubble up with retry logic handled in factory

### Rate Limiting
All LLM processors use semaphore-based concurrency:
```python
async with self._semaphore:
    response = await self._client.chat.completions.create(...)
```

## ANTI-PATTERNS

- **Never instantiate LLM client directly** → Always use `llm_factory.create_task_processor()`
- **Never pass images to text tasks** → Use `MultimodalTaskProcessor` for images
- **Don't hardcode config paths** → Use `LLMConfig` with env var overrides
- **Don't skip rate limiting** → All API calls must respect `rate_limit` config

## TESTING

Tests mirror module structure:
- `test_pdf_to_png.py` for `pdf_to_png.py`
- `test_kg_extractor.py` for `kg_extractor.py`
- etc.

Use `monkeypatch` for mocking, `tmp_path` for file operations.
