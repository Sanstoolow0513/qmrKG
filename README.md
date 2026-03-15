# QmrKG - PDF to Text Pipeline

PDF → PNG → Text 转换流水线，用于知识图谱构建。

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

**注意**: PaddleOCR 首次运行会自动下载模型文件 (~100MB)。

### 2. 放置 PDF 文件

将 PDF 文件放入 `data/pdf/` 目录。

### 3. 运行流水线

```bash
# 处理所有 PDF
python main.py

# 处理单个 PDF
python main.py --pdf data/pdf/example.pdf

# 不保存中间图片
python main.py --no-images

# 使用 GPU 加速 OCR
python main.py --gpu

# 英文文档 (更快)
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
    ocr_lang="ch",  # 中文+英文
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
| `--lang` | OCR 语言 (ch/en/korean/japan) | ch |
| `--gpu` | 使用 GPU 加速 | False |
| `--no-images` | 不保存中间图片 | False |
| `--recursive` | 递归搜索子目录 | False |

## 🔧 依赖说明

- **pymupdf**: PDF 渲染，无需额外安装 poppler
- **paddleocr**: OCR 引擎，支持中英文
  - 首次使用会自动下载模型
  - CPU 模式较慢但无需 CUDA
  - GPU 模式需要 paddlepaddle-gpu

## 📝 输出格式

生成的文本文件格式：

```
--- Page 1 ---
第一页识别出的文字内容...

--- Page 2 ---
第二页识别出的文字内容...
```

## 🐛 常见问题

**Q: 第一次运行很慢？**  
A: PaddleOCR 需要下载模型文件 (~100MB)，请保持网络通畅。

**Q: 如何提高识别准确率？**  
A: 提高 `--dpi` 参数 (300-400)，或使用更高质量的 PDF。

**Q: 纯英文文档如何加速？**  
A: 使用 `--lang en`，识别速度提升 2-3 倍。

## 📄 License

MIT License
