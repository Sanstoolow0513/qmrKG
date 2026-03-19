# QmrKG - PDF to Text Pipeline

PDF -> PNG -> Text 转换流水线，用于知识图谱构建。

## 📁 项目结构

```
qmrkg/
├── src/qmrkg/              # 主包
│   ├── __init__.py
│   ├── pdf_to_png.py       # PDF 转图片
│   ├── png_to_text.py      # OCR 文字识别
│   └── pipeline.py         # 完整流水线
├── data/
│   ├── pdf/                # 输入 PDF 文件
│   ├── png/                # 中间图片文件
│   └── markdown/           # 输出文本文件
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
# 处理所有 PDF
python main.py

# 处理单个 PDF
python main.py --pdf data/pdf/example.pdf

# 不保存中间图片
python main.py --no-images

# 兼容保留参数，当前由 SiliconFlow OCR 忽略
python main.py --gpu

# 兼容保留参数，当前由 SiliconFlow OCR 忽略
python main.py --lang en
```

## ⚙️ Task Configuration (`config.yaml`)

`config.yaml` 按任务（ocr/ner/re）分组配置能力。OCR 运行时通过 `ocr` 这一顶层段读取 provider、prompts、request 和 rate_limit；其他段目前只是占位符，运行时不会使用它们。示例结构如下：

```yaml
ocr:
  provider:
    name: siliconflow
    base_url: "https://api.siliconflow.cn/v1"
    model: "Qwen/Qwen3-VL-8B-Instruct"

  prompts:
    default: |
      Transcribe all visible text from this page exactly as written. Preserve reading order and
      line breaks where possible. Do not add commentary, summaries, or markdown fences.

    detailed: |
      Perform OCR on this image and extract all visible text content. Preserve the original
      formatting, line breaks, and reading order as accurately as possible. Include all text
      elements: headings, body text, captions, footnotes, and any other visible text. Do not
      add any commentary, summaries, or explanations.

    structured: |
      Extract text from this document image. Identify and preserve the document structure
      including headings, paragraphs, lists, and tables. Format the output with appropriate
      markdown syntax to represent the structure. Transcribe all visible text exactly as
      written without adding commentary.

    with_headers: |
      Transcribe all visible text from this page exactly as written. Preserve reading order,
      line breaks, and especially HEADING STRUCTURE. If the document contains titles or
      headings (marked with #, ##, ### or similar), KEEP the # symbols and markdown heading
      format in your output. Do not add commentary or summaries.

    chinese: |
      请精确转录图片中的所有可见文字。保持原有的阅读顺序和换行。不要添加任何
      评论、总结或格式标记。请确保识别所有中文字符和标点符号。

  request:
    image_detail: "high"
    timeout_seconds: 60.0
    max_retries: 3

  rate_limit:
    rpm: 1000
    max_concurrency: 20

ner:
  provider:
    name: ""
    base_url: ""
    model: ""
  prompts: {}
  request: {}
  rate_limit: {}

re:
  provider:
    name: ""
    base_url: ""
    model: ""
  prompts: {}
  request: {}
  rate_limit: {}
```

当前运行时只解析 `ocr` 段并调用 SiliconFlow 的 OpenAI-compatible 客户端；`ner`/`re` 仍为预留结构。注意不要把任何配置重命名为旧的 `openai` 顶层段，加载器会因找不到 `ocr` 而报错。

敏感凭据仍应保留在 `.env` 中，尤其 `SILICONFLOW_API_KEY`，不要写入 `config.yaml`。

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

## ⚙️ 配置选项

### `.env` 配置（仅密钥）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PPIO_API_KEY` | PPIO API Key | 必填 |

### `config.yaml` 配置（主要配置）

所有运行参数（base_url、model、rate_limit 等）均在 `config.yaml` 的 `ocr` 段中配置。

### 环境变量覆盖（可选）

如需临时覆盖 `config.yaml` 中的配置，可通过以下环境变量实现：

| 环境变量 | 对应配置项 |
|----------|-----------|
| `PPIO_BASE_URL` | `ocr.provider.base_url` |
| `PPIO_VLM_MODEL` | `ocr.provider.model` |
| `PPIO_IMAGE_DETAIL` | `ocr.request.image_detail` |
| `PPIO_RPM` | `ocr.rate_limit.rpm` |
| `PPIO_MAX_CONCURRENCY` | `ocr.rate_limit.max_concurrency` |
| `PPIO_TIMEOUT_SECONDS` | `ocr.request.timeout_seconds` |
| `PPIO_MAX_RETRIES` | `ocr.request.max_retries` |

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dpi` | 图片分辨率 | 200 |
| `--lang` | 兼容保留参数，当前被 SiliconFlow OCR 忽略 | ch |
| `--gpu` | 兼容保留参数，当前被 SiliconFlow OCR 忽略 | False |
| `--no-images` | 不保存中间图片 | False |
| `--recursive` | 递归搜索子目录 | False |

## 🔧 依赖说明

- **pymupdf**: PDF 渲染，无需额外安装 poppler
- **openai**: SiliconFlow OpenAI-compatible client
- **python-dotenv**: 加载 `.env` 配置

## 📝 输出格式

生成的文本文件格式：

```
--- Page 1 ---
第一页识别出的文字内容...

--- Page 2 ---
第二页识别出的文字内容...
```

## 🐛 常见问题

**Q: 为什么第一次调用 API 失败？**  
A: 先确认 `.env` 中的 `SILICONFLOW_API_KEY` 正确，且当前网络可以访问 SiliconFlow API。

**Q: 如何提高识别准确率？**  
A: 提高 `--dpi` 参数 (300-400)，或调整 `config.yaml` 中 `ocr.prompts.*`。

**Q: 如何控制调用速率？**  
A: 修改 `config.yaml` 中的 `ocr.rate_limit.rpm` 和 `ocr.rate_limit.max_concurrency`。

## 📄 License

MIT License
