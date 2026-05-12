# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

QmrKG 是 PDF → 知识图谱 → 评估的端到端 pipeline。仓库包含两个**互相独立**的项目（不是 monorepo）：

- `src/qmrkg/` — Python 3.13 包，10 个用户级 CLI（含评估闭环）+ 统一 LLM 工厂
- `frontend/` — Next.js 16 + React 19 + Neo4j driver 的力导向图可视化（pnpm 管理）

更细的目录约定写在 `AGENTS.md`、`src/qmrkg/AGENTS.md`、`frontend/AGENTS.md`，本文件不重复。

## Pipeline Architecture

数据流分 8 个阶段（含评估），每个阶段都有独立 CLI：

```
PDF/PPT → pdftopng → PNG → pngtotext → per-page MD (data/markdown/<book>/*_page_*.md)
        → kgmdcombine → 整书 MD (data/markdown/<book>.md)
        → mdchunk → JSON chunks → kgextract → raw triples (data/triples/raw/{zs,fs}/)
        → kgmerge → merged_triples.json → kgneo4j → Neo4j → frontend
                                       ↘ kgeval / kgevalraw → 评估报告 (vs data/eval/gold_triples.json)
```

中间产物落盘在 `data/{pdf,png,markdown,chunks,triples/{raw,merged},eval}/`，每个阶段都可独立重跑。CLI 入口在 `src/qmrkg/cli_*.py`，对应类在同名模块（`pdf_to_png.py`、`png_to_text.py`、`evaluation.py`、`eval_raw.py` 等）。`pipeline.py` 的 `PDFPipeline` 仅编排前三阶段（PDF → Markdown），不覆盖合并、KG 与评估阶段；端到端编排走 `cli_qmr.py`（命令名 `qmr`）。

### kgmdcombine 是 OCR 与分块之间的衔接阶段

`pngtotext` 把每本书拆为 `data/markdown/<book>/<book>_page_N.md`；`mdchunk` 期望按文档分块，因此先用 `kgmdcombine` 把每个书目录下的 `*_page_*.md` 合并为 `data/markdown/<book>.md`（合并通配符在 `config.yaml` `run.kg_md_combine.page_glob`）。**注意**：旧版本的 `mdchunk --merge` 已废弃并删除，合并逻辑统一改走独立 CLI；CI / 脚本里若仍调用 `--merge` 会失败。

### LLM 工厂是唯一入口

**所有** LLM 调用都必须经 `src/qmrkg/llm_factory.py` 的 `TextTaskProcessor` / `MultimodalTaskProcessor` / `EmbeddingTaskProcessor`。不要直接 `import openai` 调用接口——工厂负责：

- 按任务名（`ocr` / `extract` / `ner` / `re` / `kg_merge.embedding` 等）从 `config.yaml` 读取所引用 profile 的 provider、prompt、超时、重试
- 通过 `rate_limit.py` 的 `RollingRateLimiter` 做 per-profile RPM + 并发限流
- 校验 modality（`text` profile 任务收到 image 会 `ValueError`）
- 校验 thinking 开关（`provider.supports_thinking=false` 时不允许 `request.thinking.enabled=true`）

**配置结构（重要）**：模型/限流/超时全部集中在顶层 `llm.profiles.<name>:` 下；任务段（如 `ocr:` / `extract:`）只放 `llm_profile: <name>` 引用 + 该任务专属的 prompts。**不要**在任务段里复制 provider/rate_limit/request 字段——工厂从 profile 读取。

新增 LLM 任务的方法：复用已有 profile 时，在 `config.yaml` 加新任务段写 `llm_profile: <已有 profile>` 加 prompts，在调用方 `TextTaskProcessor("<新任务名>")`；需要新模型时再在 `llm.profiles` 下加一个 profile。

### 配置与密钥

- `config.yaml` 按任务分段（`ocr:` / `extract:` 等），**严禁**用已废弃的顶层 `openai:` 键，会触发配置加载错误
- 密钥（`PPIO_API_KEY`）只放 `.env`，YAML 里不要写密钥
- 环境变量统一用 `PPIO_*` 前缀（`PPIO_BASE_URL` / `PPIO_MODEL` / `PPIO_RPM` 等可覆盖 YAML）；旧的 `SILICONFLOW_*` 已弃用
- Neo4j 使用独立变量：`NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`，前端在 `frontend/.env.local` 单独配置

### KG schema（在 `kg_schema.py` + `config.py`）

- 实体类型仅 4 类：`protocol`、`concept`、`mechanism`、`metric`
- 关系类型仅 4 类：`contains`、`depends_on`、`compared_with`、`applied_to`
- `triple.evidence` 字段必填（原文支撑句），`entity.name` 必须使用原文出现形式，禁止改写或翻译
- 合并阶段可选开启 embedding 实体对齐（`config.yaml` `kg_merge.embedding.enabled`）

### kgextract 的 zs/fs 双模式

`kgextract --mode zero-shot|few-shot` 切换 `config.yaml` 中 `extract.prompts.zero_shot` / `few_shot`。做对照实验时务必把 `--output-dir` 也分到 `data/triples/raw/{zs,fs}/`，并让下游 `kgmerge --input-dir` / `--output` 同样分目录，否则两组结果互相覆盖。

### 评估阶段（kgeval / kgevalraw）

两个独立 CLI，不要混用：

- **`kgeval`**（`cli_eval.py` → `evaluation.py`）评估**合并后**的 triples，输入 `merged_triples.json` vs `data/eval/gold_triples.json`，输出 precision / recall / F1 与 Markdown 报告。所有参数从 `config.yaml` `run.kg_eval` 读取，CLI 不接位置参数，只接 `--config`。
- **`kgevalraw`**（`cli_eval_raw.py` → `eval_raw.py`）评估**抽取层原始**输出，对比 `data/triples/raw/{zs,fs}/` 两套结果，用于报告 zs/fs 模式差异。配置位于 `run.kg_eval_raw`。

Gold 数据放在 `data/eval/`：`gold_triples.json`（最终）、`gold_triples.500_reviewed.json`（人工复核版）、`sample_manifest.json`（采样清单）、`_rejection_log.json`（拒绝记录）。**修改 gold 文件**前请先看 `docs/reports/gold-generation-summary.md` 的方法学。

## Common Commands

### Backend (Python, 必须 Python 3.13)

```bash
uv sync --extra dev                      # 安装依赖（包括 dev 工具）
uv run qmrkg --list                      # 列出所有 CLI（kgmrkg 是 helper，不是 pipeline 阶段）
uv run qmr                               # 端到端编排（PDF → Neo4j 全流程）
uv run kgeval --config config.yaml       # 合并三元组 vs gold（无位置参数）
uv run kgevalraw --config config.yaml    # zs/fs 原始输出对比
uv run pytest tests/ -v                  # 跑全部测试
uv run pytest tests/test_kg_extractor.py::test_name -v   # 跑单个测试
ruff check .                             # lint（行宽 100）
black --check .                          # 格式检查
make neo4j-up / neo4j-down / neo4j-down-v   # Docker 启停 Neo4j（含数据卷清理）
```

Pipeline 单阶段示例参见 `README.md`。

### Frontend（pnpm，不是 npm）

```bash
cd frontend && pnpm install
pnpm dev      # localhost:3000
pnpm build && pnpm start
pnpm lint
```

## Testing Conventions

- **不能**发真实 API 请求。LLM 调用统一用 `FakeClient` / `FakeResponse` mock，环境/配置用 `monkeypatch`，临时文件用 `tmp_path`
- 测试文件命名 `tests/test_*.py`，与被测模块一一对应
- 运行单测时通过 pytest 的 `::` 选择具体函数；无需 `cd tests`

## Anti-patterns（容易踩的坑）

- ❌ 直接 `import openai` 调用 API — 必须走 `llm_factory`
- ❌ 在 `config.yaml` 写顶层 `openai:` 键 — 已废弃
- ❌ 在任务段（`ocr:` / `extract:` 等）里复制 provider/rate_limit/request — 这些字段属于 `llm.profiles.<name>`，任务段只放 `llm_profile` 引用 + prompts
- ❌ 给 `text` modality 任务传图像 — 会 `ValueError`
- ❌ `provider.supports_thinking=false` 时打开 `request.thinking.enabled` — 会被工厂拒绝
- ❌ 用 `os.path` — 全仓库统一 `pathlib.Path`
- ❌ 在前端用 npm — 锁文件是 pnpm
- ❌ kgextract zs/fs 共用同一 output 目录 — 后跑的会覆盖先跑的
- ❌ 抽取阶段省略 `evidence` 字段或编造原文外的实体
- ❌ 调用已删除的 `mdchunk --merge` — 改用独立的 `kgmdcombine` CLI
- ❌ 给 `kgeval` / `kgevalraw` 传位置参数 — 它们只接 `--config`，所有路径走 `run.kg_eval{,_raw}`

## 架构决策记录（ADR）

- ADR 存放于 `docs/adr/`，索引见 `docs/adr/README.md`，模板见 `docs/adr/TEMPLATE.md`
- 涉及架构、依赖、API 形态、数据模型、认证、部署、基础设施或 LLM/KG schema 的重大改动前，先查阅相关 ADR
- 规划这类改动时使用 `/adr <任务>`；做出新决策时使用 `/record-adr <决策摘要>`
- 历史 ADR 不得覆盖或重写。决策变更时新增 ADR，并在新旧两份的 `Related` 中互相链接（旧的标记为 `superseded by ADR-NNNN`）

## Cross-references

- 项目级别的目录索引、anti-patterns 清单：`AGENTS.md`
- Python 包内部约定与类清单：`src/qmrkg/AGENTS.md`
- 前端依赖与样式约定：`frontend/AGENTS.md`
- 用户视角的快速开始与 FAQ：`README.md`
- 任务级 LLM 配置（包含 OCR / 抽取的提示词全文）：`config.yaml`
- 架构决策记录索引：`docs/adr/README.md`
