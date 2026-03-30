# QmrKG - PDF To Text And Task-Scoped LLM Pipeline

PDF -> PNG -> Text 转换流水线，并提供面向 `ocr`、`ner`、`re` 等任务的 PPIO LLM Factory。

## 📁 项目结构

```
qmrkg/
├── src/qmrkg/              # 主包
│   ├── pdf_to_png.py       # PDF 转图片
│   ├── png_to_text.py      # OCR / VLM 文字识别
│   ├── markdown_chunker.py # Markdown 分块
│   ├── kg_extractor.py   # 大模型实体与关系抽取
│   ├── kg_merger.py        # 三元组合并与去重
│   ├── kg_neo4j.py         # Neo4j 导入
│   ├── ner_prompts.py      # 零样本 / 少样本抽取提示（实验用）
│   └── ...
├── data/
│   ├── pdf/                # 输入 PDF
│   ├── png/                # 中间图片
│   ├── markdown/           # OCR 输出 Markdown
│   ├── chunks/             # mdchunk 输出的 JSON 分块（kgextract 输入）
│   └── triples/
│       ├── raw/            # 每 chunk 一条 JSON；默认按提示策略分子目录（见下文）
│       └── merged/         # kgmerge 输出的合并图 JSON
├── main.py                 # CLI 入口
├── examples.py             # 使用示例
└── pyproject.toml          # 项目配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 uv (推荐)
uv pip install -e .

# 或使用 pip
pip install -e .
```

### 2. 配置 PPIO API

复制 `.env.example` 为 `.env`，填写你的 API Key：

```dotenv
PPIO_API_KEY=your_api_key_here
```

其他配置请在 `config.yaml` 中修改。如需临时覆盖某些配置，可通过环境变量实现（见下文配置选项）。

### 3. 放置 PDF 文件

将 PDF 文件放入 `data/pdf/` 目录。

### 4. 运行流水线

```bash
# 方式 A：完整流程（PDF -> PNG -> OCR -> Markdown）
uv run python main.py

# 处理所有 PDF
python main.py

# 处理单个 PDF
python main.py --pdf data/pdf/example.pdf

# 不保存中间图片
python main.py --no-images

# 兼容保留参数，当前由 PPIO OCR 忽略
python main.py --gpu

# 兼容保留参数，当前由 PPIO OCR 忽略
python main.py --lang en
```

### 5. 阶段化 CLI（根目录直达）

```bash
# 查看当前可执行命令
uv run qmrkg --list

# 仅 PDF -> PNG
uv run pdftopng --pdf data/pdf/example.pdf

# 仅 PNG -> Markdown（OCR）
uv run pngtotext --image data/png/example_page_0001.png --output data/markdown/example.md

# 仅 Markdown -> Chunks(JSON)
uv run mdchunk --markdown data/markdown/example.md --chunk-dir data/chunks
```

### 6. 知识图谱：抽取 → 合并 → Neo4j

LLM 抽取使用 `config.yaml` 中 **`extract`** 任务（与 `ocr` 并列）的模型与速率限制；提示词由 CLI **`--prompt-kind`** 选择（见下）。

**从分块 JSON 抽取实体与关系（`kgextract`）**

```bash
# 处理 data/chunks 下所有 *.json（默认提示策略 legacy，与历史行为一致）
uv run kgextract --input data/chunks -v

# 零样本 / 少样本提示（用于对比实验）；不写 --output-dir 时结果分目录存放，避免互相覆盖
uv run kgextract --input data/chunks --prompt-kind zero_shot -v
uv run kgextract --input data/chunks --prompt-kind few_shot -v

# 强制重跑已存在的 chunk 结果
uv run kgextract --input data/chunks --prompt-kind few_shot --no-skip -v

# 自定义输出目录（三种策略可写到同一目录，需自行避免文件名冲突）
uv run kgextract --input data/chunks --prompt-kind few_shot --output-dir data/triples/raw -v
```

默认输出路径（未指定 `--output-dir` 时）：

| `--prompt-kind` | 原始三元组目录 |
|-----------------|----------------|
| `legacy`        | `data/triples/raw/legacy/` |
| `zero_shot`     | `data/triples/raw/zero_shot/` |
| `few_shot`      | `data/triples/raw/few_shot/` |

每条 chunk 结果 JSON 中含 `extraction_meta.prompt_kind`，便于追溯实验配置。

**合并去重（`kgmerge`）**

`kgmerge` 只读取**指定目录下的一层** `*.json`，因此需与上表目录一致。为与 `kgextract` 对齐，可使用 **`--prompt-kind`** 自动选择输入 / 输出文件：

```bash
# 读取 data/triples/raw/zero_shot/，写入 data/triples/merged/merged_triples_zero_shot.json
uv run kgmerge --prompt-kind zero_shot -v

# 读取 data/triples/raw/few_shot/，写入 merged_triples_few_shot.json
uv run kgmerge --prompt-kind few_shot -v
```

仍使用「扁平」`data/triples/raw/*.json` 时，不传 `--prompt-kind` 即可（输入默认 `data/triples/raw`，输出默认 `data/triples/merged/merged_triples.json`）。也可用 `--input-dir`、`--output` 完全手动指定。

**导入 Neo4j（`kgneo4j`）**

```bash
uv run kgneo4j --import data/triples/merged/merged_triples_zero_shot.json -v
# 或合并文件名与 --output 一致即可
```

说明：`pdftopng`、`pngtotext`、`mdchunk`、`kgextract`、`kgmerge`、`kgneo4j` 均通过 `pyproject.toml` 的 `project.scripts` 注册，可直接 `uv run <命令>`。

## ⚙️ Task Configuration (`config.yaml`)

`config.yaml` 按任务（`ocr`/`ner`/`re`）分组配置能力。每个任务都可以通过统一的 factory 获取自己的模型、prompt、请求参数和速率限制。示例结构如下：

```yaml
ocr:
  provider:
    name: ppio
    base_url: "https://api.ppio.com/openai"
    model: "qwen/qwen3-vl-8b-instruct"
    modality: "multimodal"
    supports_thinking: false

  prompts:
    default: |
      你是一个 OCR 文字识别系统。

      输出要求：
      - 输出必须是有效的 Markdown。
      - 输出语言必须与图片语言一致（中文图片就输出中文）。
      - 尽可能保留阅读顺序与换行。
      - 不要翻译或改写文字：识别到什么就输出什么（保持原样）。
      - 对“看起来像标题/章节名”的行，尽量输出 Markdown 标题（优先加 `#`，不要把标题当普通段落丢掉）。

      标题识别与格式（非常重要）：
      - 当一行很可能是标题/章节名时，把它输出为 Markdown 标题：使用 `#`、`##` 或 `###` 作为前缀。
      - 标题行格式必须严格为：`# <标题内容>`（`## <...>`、`### <...>` 同理）。
        `#` 后面必须紧跟一个空格；不要输出 `#` 后面没有空格的形式。
      - 标题必须占据自己的行/行组：不要在句子中间插入 `#`。
      - 标题层级选择规则（按证据强弱）：
        - 不确定时：也要用 `#`（顶层），不要省略标题标记。
        - 更像二级标题时：用 `##`（例如“第 X 节”、`X.Y`、以 `1.` 开头且后面跟较短标题）。
        - 更像三级/更小单元时：用 `###`（例如“第 X 条”、“（一）（二）”、`2.1`、`1)`）。

      中文标题识别偏好（你的输入主要是中文）：
      - 当行中包含常见中文结构关键词或典型编号样式时，把它当标题：`第X章`、`第X节`、`第X条`、`第X款`、`第X项`、`章`、`节`、`条`、`款`、`小节`、`附录`、`附件`、`前言`、`目录`、`参考文献`、`致谢`、`结论`。
      - 或者当它符合编号模式时：`1. ...`、`1) ...`、`1）...`、`（一）...`、`（1）...`、`一、...`。
      - 另外：如果该行在版面上很像标题（上下有明显留白、居中、加粗、字体更大），也应输出为标题。
      - 如果该行看起来更像“标题而不是句子”（例如短、末尾不像句子那样带 `。` 等标点，或包含明确章节词），优先转换为标题。

      文字版 Few-shot（演示如何加标题标记）：
      - 输入行：`第一章 绪论`
        输出：`# 第一章 绪论`
      - 输入行：`2.1 相关工作`
        输出：`## 2.1 相关工作`
      - 输入行：`第十条 义务`
        输出：`### 第十条 义务`
      - 输入行：`（一）定义`
        输出：`### （一）定义`

      对于图标、符号、图表或其他非文字视觉元素：
      用围栏代码块描述，例如 ```icon: description```。

      不要在转录内容之外添加任何评论或总结。

  request:
    image_detail: "high"
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false

  rate_limit:
    rpm: 1000
    max_concurrency: 20

ner:
  provider:
    name: ppio
    base_url: "https://api.ppio.com/openai"
    model: "qwen/qwen3-8b"
    modality: "text"
    supports_thinking: false
  prompts:
    default: |
      你是一个命名实体识别助手。
  request:
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false
  rate_limit:
    rpm: 300
    max_concurrency: 8

re:
  provider:
    name: ppio
    base_url: "https://api.ppio.com/openai"
    model: "qwen/qwen3-8b"
    modality: "text"
    supports_thinking: false
  prompts:
    default: |
      你是一个关系抽取助手。
  request:
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false
  rate_limit:
    rpm: 300
    max_concurrency: 8
```

关键字段说明：

- `provider.modality`: `text` 或 `multimodal`
- `provider.supports_thinking`: 标记该任务使用的模型是否支持显式 thinking 开关
- `request.thinking.enabled`: 为支持的模型显式开启或关闭 thinking
- `request.image_detail`: 仅对 `multimodal` 任务有效

注意不要把任何配置重命名为旧的 `openai` 顶层段。敏感凭据仍应保留在 `.env` 中，尤其 `PPIO_API_KEY`，不要写入 `config.yaml`。

## 📖 API 使用

### 完整流水线

```python
from qmrkg import PDFPipeline

pipeline = PDFPipeline(
    pdf_dir="data/pdf",
    image_dir="data/png",
    text_dir="data/markdown",
    dpi=200,
    ocr_lang="ch",  # 兼容保留参数
)

# 处理所有 PDF
results = pipeline.process_all()

# 处理单个 PDF
image_paths, text_path = pipeline.process_pdf("data/pdf/example.pdf")
```

### 分步使用

```python
from qmrkg import PDFConverter, OCRProcessor

# PDF -> PNG
converter = PDFConverter(dpi=200)
image_paths = converter.convert("document.pdf")

# PNG -> Text
ocr = OCRProcessor(lang="ch")
text = ocr.extract_text("page_1.png")
```

### 通用文本任务

```python
from qmrkg import TextTaskProcessor

ner = TextTaskProcessor("ner")
response = ner.run_text("从这段文本中抽取关键实体：张三于2024年加入派欧云。")

print(response.text)
```

## ⚙️ 配置选项

### `.env` 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PPIO_API_KEY` | PPIO API Key | 必填 |
| `PPIO_BASE_URL` | 全局覆盖 `provider.base_url` | `https://api.ppio.com/openai` |
| `PPIO_MODEL` | 全局覆盖当前任务 `provider.model` | 空 |
| `PPIO_PROMPT` | 全局覆盖当前任务 prompt | 空 |
| `PPIO_PROMPT_KEY` | 选择 `prompts` 中的 prompt key | `default` |

### `config.yaml` 配置（主要配置）

任务级运行参数都在 `config.yaml` 对应段中配置，例如 `ocr`、`ner`、`re`。

### 环境变量覆盖（可选）

如需临时覆盖 `config.yaml` 中的配置，可通过以下环境变量实现：

| 环境变量 | 对应配置项 |
|----------|-----------|
| `PPIO_BASE_URL` | `<task>.provider.base_url` |
| `PPIO_MODEL` | `<task>.provider.model` |
| `PPIO_PROMPT` | `<task>.prompts.default` |
| `PPIO_PROMPT_KEY` | `<task>.prompts.<key>` |
| `PPIO_VLM_MODEL` | `ocr.provider.model` |
| `PPIO_VLM_PROMPT` | `ocr.prompts.default` |
| `PPIO_IMAGE_DETAIL` | `ocr.request.image_detail` |
| `PPIO_RPM` | `<task>.rate_limit.rpm` |
| `PPIO_MAX_CONCURRENCY` | `<task>.rate_limit.max_concurrency` |
| `PPIO_TIMEOUT_SECONDS` | `<task>.request.timeout_seconds` |
| `PPIO_MAX_RETRIES` | `<task>.request.max_retries` |

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dpi` | 图片分辨率 | 200 |
| `--lang` | 兼容保留参数，当前被 PPIO OCR 忽略 | ch |
| `--gpu` | 兼容保留参数，当前被 PPIO OCR 忽略 | False |
| `--no-images` | 不保存中间图片 | False |
| `--recursive` | 递归搜索子目录 | False |

## 🔧 依赖说明

- **pymupdf**: PDF 渲染，无需额外安装 poppler
- **openai**: PPIO OpenAI-compatible client
- **python-dotenv**: 加载 `.env` 配置

## 📝 输出格式

生成的 OCR 文本文件格式：

```
--- Page 1 ---
第一页识别出的文字内容...

--- Page 2 ---
第二页识别出的文字内容...
```

## 🐛 常见问题

**Q: 为什么第一次调用 API 失败？**  
A: 先确认 `.env` 中的 `PPIO_API_KEY` 正确，且当前网络可以访问 PPIO API。

**Q: 如何提高识别准确率？**  
  A: 提高 `--dpi` 参数 (300-400)，或调整 `config.yaml` 中 `ocr.prompts.default`。

**Q: 如何控制调用速率？**  
A: 修改 `config.yaml` 中的 `ocr.rate_limit.rpm` 和 `ocr.rate_limit.max_concurrency`。

**Q: reasoning 模型的 thinking 怎么控制？**  
A: 在对应任务段设置 `provider.supports_thinking: true`，然后用 `request.thinking.enabled: true/false` 控制是否开启。

## 📄 License

MIT License
