# kgextract Zero/Few Shot Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `uv run kgextract` 中新增 `--mode zero-shot|few-shot`，并通过 `config.yaml` 的两套 prompt 支持可复现的手动对照实验。

**Architecture:** 在 CLI 层负责参数约束与透传，在 `KGExtractor` 层负责 prompt 选择与回退链路。抽取、解析、校验、落盘逻辑保持不变，确保实验对照只有 prompt 变量变化。配置层通过 `extract.prompts.zero_shot/few_shot/default` 提供模式与兼容性。

**Tech Stack:** Python 3.13, argparse, pathlib, pytest, uv, existing `qmrkg` package modules

---

### Task 1: 配置先行（新增 zero_shot/few_shot prompt）

**Files:**
- Modify: `config.yaml`
- Test: `tests/test_kg_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from qmrkg.kg_extractor import KGExtractor


def test_resolve_prompt_prefers_mode_specific_prompt(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3.2"
    modality: "text"
    supports_thinking: false
  prompts:
    default: "DEFAULT_PROMPT"
    zero_shot: "ZERO_PROMPT"
    few_shot: "FEW_PROMPT"
  request:
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false
  rate_limit:
    rpm: 50
    max_concurrency: 4
""",
        encoding="utf-8",
    )
    extractor = KGExtractor(config_path=cfg, mode="few-shot")
    assert extractor._extract_prompt == "FEW_PROMPT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kg_extractor.py::test_resolve_prompt_prefers_mode_specific_prompt -v`  
Expected: FAIL，原因是 `KGExtractor` 还没有 `mode` 或 `_extract_prompt` 行为。

- [ ] **Step 3: Write minimal implementation**

```python
# config.yaml 中 extract.prompts 扩展为：
extract:
  prompts:
    default: |
      # 与 zero_shot 等价的兼容 prompt
    zero_shot: |
      # zero-shot 联合抽取 prompt
    few_shot: |
      # few-shot 联合抽取 prompt（含统一 entities+triples 示例）
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kg_extractor.py::test_resolve_prompt_prefers_mode_specific_prompt -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml tests/test_kg_extractor.py src/qmrkg/kg_extractor.py
git commit -m "feat(extract): add mode-specific prompt config for zero/few shot"
```

### Task 2: 提取器支持 mode 与回退链路

**Files:**
- Modify: `src/qmrkg/kg_extractor.py`
- Test: `tests/test_kg_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from qmrkg.kg_extractor import KGExtractor, EXTRACT_PROMPT


def test_resolve_prompt_fallback_chain(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3.2"
    modality: "text"
    supports_thinking: false
  prompts:
    default: "DEFAULT_PROMPT"
""",
        encoding="utf-8",
    )
    few = KGExtractor(config_path=cfg, mode="few-shot")
    assert few._extract_prompt == "DEFAULT_PROMPT"

    cfg.write_text(
        """
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3.2"
    modality: "text"
    supports_thinking: false
""",
        encoding="utf-8",
    )
    zero = KGExtractor(config_path=cfg, mode="zero-shot")
    assert zero._extract_prompt == EXTRACT_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kg_extractor.py::test_resolve_prompt_fallback_chain -v`  
Expected: FAIL，原因是尚未实现 `mode -> default -> EXTRACT_PROMPT` 回退链路。

- [ ] **Step 3: Write minimal implementation**

```python
# src/qmrkg/kg_extractor.py
from .llm_config import _load_yaml_config


class KGExtractor:
    def __init__(
        self,
        runner: TaskLLMRunner | None = None,
        config_path: Path | None = None,
        mode: str = "zero-shot",
    ):
        self._mode = mode
        self._config_path = config_path
        self._extract_prompt = self._resolve_extract_prompt(mode)
        # ... existing runner init stays unchanged

    def _resolve_extract_prompt(self, mode: str) -> str:
        mode_key_map = {"zero-shot": "zero_shot", "few-shot": "few_shot"}
        mode_key = mode_key_map.get(mode, "zero_shot")
        cfg = _load_yaml_config(self._config_path)
        prompts = ((cfg.get("extract") or {}).get("prompts") or {})
        mode_prompt = prompts.get(mode_key)
        if isinstance(mode_prompt, str) and mode_prompt.strip():
            return mode_prompt.strip()
        default_prompt = prompts.get("default")
        if isinstance(default_prompt, str) and default_prompt.strip():
            return default_prompt.strip()
        return EXTRACT_PROMPT

    def extract_from_chunk(self, chunk: dict) -> ChunkExtractionResult:
        # ...
        response = self._runner.run_text(content, system_prompt=self._extract_prompt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kg_extractor.py -v`  
Expected: PASS（包含既有解析相关测试）

- [ ] **Step 5: Commit**

```bash
git add src/qmrkg/kg_extractor.py tests/test_kg_extractor.py
git commit -m "feat(extract): add mode-aware prompt resolution with fallback chain"
```

### Task 3: CLI 增加 --mode 并透传到 KGExtractor

**Files:**
- Modify: `src/qmrkg/cli_kg_extract.py`
- Test: `tests/test_cli_kg_extract.py`

- [ ] **Step 1: Write the failing test**

```python
import argparse
from qmrkg.cli_kg_extract import build_parser


def test_cli_mode_argument_defaults_to_zero_shot():
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args([])
    assert args.mode == "zero-shot"


def test_cli_mode_argument_accepts_few_shot():
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args(["--mode", "few-shot"])
    assert args.mode == "few-shot"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_kg_extract.py -v`  
Expected: FAIL，原因是当前 CLI 没有 `build_parser` 和 `--mode`。

- [ ] **Step 3: Write minimal implementation**

```python
# src/qmrkg/cli_kg_extract.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract KG triples from markdown chunks")
    # ... existing args
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompt mode for extraction (default: zero-shot)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    print(f"kgextract mode: {args.mode}")
    extractor = KGExtractor(mode=args.mode)
    # ... existing flow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_kg_extract.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qmrkg/cli_kg_extract.py tests/test_cli_kg_extract.py
git commit -m "feat(cli): add --mode zero-shot|few-shot for kgextract"
```

### Task 4: 端到端回归与使用文档补充

**Files:**
- Modify: `README.md`
- Modify: `docs/pipeline-task-gap-list.md`
- Test: `tests/test_kg_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from qmrkg.kg_extractor import KGExtractor


def test_extract_prompt_uses_builtin_when_no_prompts_in_config(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3.2"
    modality: "text"
    supports_thinking: false
""",
        encoding="utf-8",
    )
    extractor = KGExtractor(config_path=cfg, mode="zero-shot")
    assert "知识图谱构建专家" in extractor._extract_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kg_extractor.py::test_extract_prompt_uses_builtin_when_no_prompts_in_config -v`  
Expected: FAIL（在实现回退链路之前，`_extract_prompt` 可能为空或不包含内置 prompt 关键词）。

- [ ] **Step 3: Write minimal implementation**

```markdown
# README.md 追加示例
uv run kgextract --mode zero-shot --input data/chunks --output-dir data/triples/raw/zs
uv run kgextract --mode few-shot --input data/chunks --output-dir data/triples/raw/fs
```

```markdown
# docs/pipeline-task-gap-list.md 更新状态
- [x] 将抽取 prompt 模板参数化（extract.prompts.zero_shot / few_shot）
- [x] kgextract 支持 --mode zero-shot|few-shot 运行
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kg_extractor.py tests/test_cli_kg_extract.py -v`  
Expected: PASS

Run: `uv run kgextract --help`  
Expected: 输出中包含 `--mode {zero-shot,few-shot}`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/pipeline-task-gap-list.md tests/test_kg_extractor.py tests/test_cli_kg_extract.py
git commit -m "docs: document zero-shot/few-shot kgextract usage and status"
```

### Task 5: 最终验证（实现完成后的统一验收）

**Files:**
- Test: `tests/test_kg_extractor.py`
- Test: `tests/test_cli_kg_extract.py`

- [ ] **Step 1: Write the failing test**

```python
from qmrkg.kg_extractor import KGExtractor


def test_mode_string_is_preserved_for_logging_and_debug():
    extractor = KGExtractor(runner=None, mode="few-shot")
    assert extractor._mode == "few-shot"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kg_extractor.py tests/test_cli_kg_extract.py -v`  
Expected: FAIL（新增模式状态断言尚未实现时失败）。

- [ ] **Step 3: Write minimal implementation**

```bash
# 验证命令（真实输入路径按你的机器数据为准）
uv run kgextract --mode zero-shot --input data/chunks --output-dir data/triples/raw/zs
uv run kgextract --mode few-shot --input data/chunks --output-dir data/triples/raw/fs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kg_extractor.py tests/test_cli_kg_extract.py -v`  
Expected: PASS

Run: `uv run kgextract --mode zero-shot --input data/chunks --output-dir data/triples/raw/zs --no-skip`  
Expected: 成功生成/覆盖 zero-shot 输出

Run: `uv run kgextract --mode few-shot --input data/chunks --output-dir data/triples/raw/fs --no-skip`  
Expected: 成功生成/覆盖 few-shot 输出

- [ ] **Step 5: Commit**

```bash
git add src/qmrkg/cli_kg_extract.py src/qmrkg/kg_extractor.py config.yaml tests/test_kg_extractor.py tests/test_cli_kg_extract.py README.md docs/pipeline-task-gap-list.md
git commit -m "test: verify kgextract zero/few-shot mode behavior end-to-end"
```

## Self-Review Checklist (Completed)

- Spec coverage: 已覆盖配置结构、CLI 参数、KGExtractor 回退链路、容错、测试与验收。
- Placeholder scan: 无 `TBD/TODO/implement later` 占位文本。
- Type consistency: `mode` 命名、`zero-shot/few-shot` 字面值与 `zero_shot/few_shot` 配置键映射保持一致。
