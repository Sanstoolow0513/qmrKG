# QmrKG Project Knowledge Base

**Generated:** 2026-04-26
**Commit:** 58ea922
**Branch:** master

## OVERVIEW

QmrKG is a PDF-to-Knowledge-Graph pipeline for computer networking course materials. Converts PDF/PPT → Markdown (OCR) → JSON chunks → Knowledge Graph triples → Neo4j, with a Next.js frontend for interactive visualization.

**Tech Stack:** Python 3.13 + Next.js 16 + Neo4j

## STRUCTURE

```
qmrkg/
├── src/qmrkg/              # Python pipeline (24 modules)
│   ├── pipeline.py         # PDFPipeline: PDF → PNG → Markdown
│   ├── cli_*.py            # 9 CLI entry points
│   ├── llm_factory.py      # Task-scoped LLM processor (text + multimodal + embedding)
│   ├── kg_*.py             # Knowledge graph extraction/merge/import
│   └── *_chunker.py        # Markdown chunking
├── frontend/               # Next.js 16 visualization
│   ├── app/                # App Router
│   ├── app/components/     # GraphVisualizer, GraphCanvas
│   └── app/api/graph/      # Neo4j graph data API
├── data/                   # Runtime data (pdf/png/markdown/chunks/triples)
├── tests/                  # pytest suite (15 files)
├── config.yaml             # LLM profiles + prompt config
└── pyproject.toml          # Python package config
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| PDF → PNG conversion | `src/qmrkg/pdf_to_png.py` | PyMuPDF-based |
| OCR/VLM text extraction | `src/qmrkg/png_to_text.py` | qwen3-vl-8b model |
| Per-page MD → book MD | `src/qmrkg/cli_kg_md_combine.py` | Merges page files after OCR |
| Markdown chunking | `src/qmrkg/markdown_chunker.py` | Token-aware splitting |
| Entity/relation extraction | `src/qmrkg/kg_extractor.py` | deepseek-v4-flash model |
| Triple merging | `src/qmrkg/kg_merger.py` | Dedup + embedding canonicalization |
| Entity canonicalization | `src/qmrkg/kg_merger.py` | Embedding-based entity resolution |
| Neo4j import | `src/qmrkg/kg_neo4j.py` | Bulk loading |
| LLM task factory | `src/qmrkg/llm_factory.py` | Rate limiting, retries |
| Graph visualization | `frontend/app/page.tsx` | react-force-graph-2d |

## ENTRY POINTS

### CLI Commands (via `uv run`)
```bash
uv run qmrkg --list                     # List all commands
uv run pdftopng --pdf-dir data/pdf      # PDF → PNG
uv run pngtotext --image-dir data/png   # OCR → Markdown
uv run kgmdcombine --markdown-dir data/markdown  # Merge page files → book MD
uv run mdchunk --markdown-dir data/markdown   # → JSON chunks
uv run kgextract --input data/chunks    # → Raw triples (zs/fs modes)
uv run kgmerge                          # → Merged triples
uv run kgneo4j --import data/triples/merged/merged_triples.json  # → Neo4j
uv run qmr                              # Full pipeline (PDF → Neo4j, single command)
```

### Python API
```python
from qmrkg import PDFPipeline, KGExtractor
pipeline = PDFPipeline(pdf_dir=Path("data/pdf"), ...)
```

### Frontend
```bash
cd frontend && pnpm dev   # localhost:3000
```

## CONVENTIONS

### Python Code Style
- **Version:** Python 3.13 only (`>=3.13,<3.14`)
- **Package Manager:** `uv` (modern, fast)
- **Line Length:** 100 characters (black + ruff)
- **Formatting:** Black target `py313`
- **Paths:** Always use `pathlib.Path` (not `os.path`)
- **Naming:** `snake_case` for functions/vars, `PascalCase` for classes

### Configuration
- **Task-scoped:** Use `ocr:`, `extract:`, `ner:`, `re:` sections in `config.yaml`
- **No legacy keys:** Never use top-level `openai:` key (deprecated)
- **Secrets in .env:** `PPIO_API_KEY` in `.env`, never in YAML
- **Provider:** PPIO infrastructure (not OpenAI directly)

### Frontend
- **Package Manager:** pnpm
- **Framework:** Next.js 16 with App Router
- **Styling:** Tailwind CSS v4
- **Path Alias:** `@/*` maps to `./*`

## ANTI-PATTERNS (DO NOT)

### Configuration
- ❌ **Never use** deprecated top-level `openai:` key in config.yaml
- ❌ **Never commit** API keys - use `.env` file
- ❌ **Don't use** `SILICONFLOW_*` env vars - use `PPIO_*` equivalents
- ❌ `provider.modality` must be `text` or `multimodal` only
- ❌ `image_detail` must be `auto`, `low`, or `high` only

### LLM Tasks
- ❌ **Don't pass images** to text-only tasks (raises ValueError)
- ❌ **Don't enable** `request.thinking.enabled` if `provider.supports_thinking` is false

### OCR Output (per config.yaml prompts)
- ❌ Don't translate or paraphrase - preserve original text
- ❌ Don't add commentary or summaries outside content
- ❌ Don't skip title markers (`# `) even when uncertain
- ❌ Don't put `#` in middle of sentences
- ❌ Title format: `# Title` (space after # required)

### Knowledge Extraction
- ❌ **Don't fabricate** entities - only extract from given text
- ❌ **Don't omit** evidence field in triples (required)
- ❌ Entity names must use **original text** (not normalized)
- ❌ Output **strict JSON only** - no markdown, no extra text

### Testing
- ❌ **Never use** real API calls in tests
- ❌ **Always use** `FakeClient`/`FakeResponse` mocks
- ❌ **Don't skip** `tmp_path` cleanup - use pytest fixtures

### Version Control
- ❌ Don't commit: `.env`, `data/`, `__pycache__/`, `node_modules/`, `.next/`

## COMMANDS

### Backend
```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Linting
ruff check .
black --check .
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev        # Development
pnpm build      # Production build
pnpm lint       # ESLint
```

### Neo4j (Docker)
```bash
make neo4j-up      # Start Neo4j
make neo4j-down    # Stop Neo4j
make neo4j-down-v  # Stop and remove volumes
```

## NOTES

- **Chinese-first:** LLM prompts in config.yaml are Chinese
- **Entity types:** protocol, concept, mechanism, metric
- **Relation types:** contains, depends_on, compared_with, applied_to
- **Rate limiting:** Per-task rpm/max_concurrency in config.yaml
- **Not a monorepo:** Two independent projects (Python CLI + Next.js frontend)
- **Pipeline stages:** PDF → PNG → Markdown (per-page) → kgmdcombine (book MD) → Chunks → Triples → Merged → Neo4j
- **kgextract modes:** `--mode zero-shot` / `few-shot` switches config.yaml prompts; use separate output dirs
- **Embedding canonicalization:** kgmerge optionally uses embedding profile for entity dedup
- **LLM profiles:** qwen3-vl-8b (OCR), deepseek-v4-flash (extract), embedding_qwen3_8b (embed)
