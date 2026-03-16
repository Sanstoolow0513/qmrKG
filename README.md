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

### 2. 配置 SiliconFlow API

复制 `.env.example` 为 `.env`，并至少填写 `SILICONFLOW_API_KEY`：

```dotenv
SILICONFLOW_API_KEY=your_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.com/v1
SILICONFLOW_VLM_MODEL=Qwen/Qwen2-VL-72B-Instruct
SILICONFLOW_RPM=60
SILICONFLOW_MAX_CONCURRENCY=4
SILICONFLOW_TIMEOUT_SECONDS=60
SILICONFLOW_MAX_RETRIES=3
```

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
A: 提高 `--dpi` 参数 (300-400)，或调整 `SILICONFLOW_VLM_PROMPT` 让模型更偏向逐字转录。

**Q: 如何控制调用速率？**  
A: 通过 `.env` 中的 `SILICONFLOW_RPM` 和 `SILICONFLOW_MAX_CONCURRENCY` 调整并发与速率限制。

## 📄 License

MIT License
