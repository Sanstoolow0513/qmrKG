# QmrKG Test Suite

**Location:** `tests/`

## OVERVIEW

pytest suite (17 files, ~4,700 lines) covering all pipeline stages and CLI commands. No real API calls — all LLM endpoints are mocked.

## STRUCTURE

```
tests/
├── fixtures/
│   └── eval/
│       ├── gold_triples.json     # Gold standard for evaluation tests
│       └── pred_merged.json      # Predicted merged triples for eval tests
├── _scratch/                     # Runtime temp dir (gitignored)
├── test_llm_factory.py           # LLM factory unit tests (FakeClient/FakeResponse origin)
├── test_png_to_text.py           # OCR processor tests (including multimodal)
├── test_pdf_to_png.py            # PDF/PPT converter tests
├── test_pipeline.py              # End-to-end pipeline tests
├── test_markdown_chunker.py      # Chunker tests (class-based Test* organization)
├── test_kg_extractor.py          # KG extraction tests (_RecordingRunner pattern)
├── test_kg_merger.py             # Triple merge tests
├── test_kg_merger_embedding.py   # Embedding canonicalization tests (780 lines)
├── test_kg_neo4j.py              # Neo4j import tests (only file using unittest.mock)
├── test_kg_schema.py             # Entity/Triple model validation
├── test_evaluation.py            # Precision/recall/F1 tests (@parametrize)
├── test_config.py                # Config loading
├── test_main.py                  # tqdm/logging
├── test_cli_eval.py              # kgeval CLI integration
├── test_cli_kg_extract.py        # kgextract CLI integration
├── test_cli_qmr.py               # Full pipeline CLI
└── test_cli_stage_commands.py    # pdftopng/pngtotext/mdchunk/kgmdcombine CLI (547 lines)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Core mock utilities | `test_llm_factory.py` lines 11-60 | FakeClient, FakeResponse, FakeEmbeddingClient |
| PDF mocking | `test_pdf_to_png.py` lines 15-44 | FakePixmap, FakePage, FakeDoc |
| Rate limiter mocking | `test_png_to_text.py` line 114+ | FakeSleep with mutable timeline |
| Embedding mocking | `test_kg_merger_embedding.py` lines 11-58 | FakeEmbeddingProcessor, FakeFaissIndexFlatIP |
| CLI stub pattern | `test_cli_stage_commands.py` | StubConverter, StubProcessor, StubChunker |
| Legacy config rejection | `test_png_to_text.py` lines 224-236 | Tests deprecated `openai:` key rejection |
| Eval fixtures | `fixtures/eval/` | Deterministic gold/pred JSON for eval tests |

## CONVENTIONS

### Mocking Strategy
- **No real API calls** — enforced by convention
- **FakeClient/FakeResponse** pattern for LLM endpoints (defined inline per file, not shared)
- **monkeypatch** for environment/config injection (`monkeypatch.setenv(...)`)
- **tmp_path** fixture for filesystem isolation
- **unittest.mock.patch** used only in `test_kg_neo4j.py` (the exception)
- Fakes defined at module level in test files, configurable via lambda handlers:
  ```python
  FakeClient(lambda **_: FakeResponse("ok"))
  ```

### Naming
- Files: `test_<module>.py`
- Functions: `test_<scenario>` in snake_case
- Classes: `Test<Feature>` in PascalCase (used in `test_markdown_chunker.py`)
- Helpers: snake_case module-level functions (`write_config()`, `build_processor()`)

### Fixtures
- **No conftest.py** — only one local `@pytest.fixture`: `scratch_dir` in `test_png_to_text.py`
- `tmp_path` is the universal temp-dir fixture
- Eval fixtures committed as JSON in `fixtures/eval/`

### Parametrize
- `@pytest.mark.parametrize` used in `test_evaluation.py` and `test_png_to_text.py`

## ANTI-PATTERNS (DO NOT)

- ❌ **Never use real API calls** — all LLM endpoints must be mocked
- ❌ **Never skip** `tmp_path` cleanup — use pytest fixtures
- ❌ **Don't use** `unittest.mock` when `monkeypatch` or `FakeClient` suffice
- ❌ **Don't commit** real API keys or test files that depend on live services
- ❌ **Don't skip** `evidence` field validation in KG tests
- ❌ **Don't import from** `llm_factory` directly — use public package API (`from qmrkg import ...`)

## COMMANDS

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module
uv run pytest tests/test_kg_extractor.py -v

# Run single test function
uv run pytest tests/test_llm_factory.py::test_text_task_processor -v

# Run with markers
uv run pytest tests/ -v -k "eval"
```

## NOTES

- **Flat structure** — no nested test directories (besides fixtures/)
- **No pytest plugins** — no pytest-cov, no pytest-xdist
- **Dependency versions**: `pytest>=9.0.2` (dev group), `pytest>=8.0.0` (optional dev)
- **`sys.path` hack** in `test_markdown_chunker.py` line 7 — legacy, should be removed
- **`write_config()` helper** duplicated across 4+ files — candidate for conftest.py
- **Rate limiter class** is `RollingRateLimiter` (not `RollingWindowRateLimiter`)
