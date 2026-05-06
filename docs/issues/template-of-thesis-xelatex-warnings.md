# 论文模板 XeLaTeX 编译日志问题记录

**路径**：`papers/template-of-thesis/main.tex`  
**环境**：TeX Live 2026，XeLaTeX  
**记录日期**：2026-05-06  

本文档根据一次 `xelatex main.tex` 的完整终端日志整理，便于后续逐项消除警告与版式问题。

---

## 总体结论

- **产物**：已成功生成 `main.pdf`（日志中为 65 页）。
- **需优先处理**：文末 `(\end occurred when \ifx ... was incomplete)` 与 `hyperref` 对 `\ifdefstring` 的 PDF 字符串警告，与致谢标题宏定义相关（见下文「严重」一节）。

---

## 严重：条件展开不完整 + hyperref 书签

### 现象

```text
Package hyperref Warning: Token not allowed in a PDF string (Unicode):
(hyperref)                removing `\ifdefstring' on input line 1.

(\end occurred when \ifx on line 41 was incomplete)
```

### 原因（概要）

`body/acknowledgement.tex` 第 1 行调用 `\acknowledgement`。该宏在 `template/hust.cls` 中将 `\ifdefstring{...}` 作为无编号节标题传入 `\sectionUnnumbered`，进而进入 `\section*{...}`。`hyperref` 生成 PDF 书签时会把标题改成合法 PDF 字符串，并移除不允许的 token；移除 `\ifdefstring` 后，etoolbox 内部依赖的 `\ifx` 结构可能无法正确闭合，在 `\end{document}` 附近报错。

### 建议修复

- 在印刷标题仍为条件文字的前提下，用 `\texorpdfstring{<TeX 标题>}{<纯文本书签>}` 包装；或
- thesis 模式下与中文摘要一致，直接使用固定标题如 `\sectionUnnumbered{致谢}`（若版式允许）。

修改位置：`papers/template-of-thesis/template/hust.cls` 中 `\newcommand{\acknowledgement}{...}`。

---

## 轻微：文档类名称不一致

```text
LaTeX Warning: You have requested document class `template/hust',
               but the document class provides `hust'.
```

`\documentclass{template/hust}` 与 `\ProvidesClass{hust}` 名称不一致的提示，一般不影响排版。若要消除警告，可统一 `ProvidesClass` 与引用路径的命名（可选）。

---

## 版心：geometry 水平方向过约束

```text
Package geometry Warning: Over-specification in `h'-direction.
    `width' (400.0pt) is ignored.
```

水平方向参数同时指定导致冲突，`width` 被忽略。需在 `hust.cls` / `packages.sty` 等处检查 `\geometry{...}`，去掉重复或矛盾的键。

---

## 字体（xeCJK / Times / mathrsfs）

| 日志要点 | 说明 |
|----------|------|
| `xeCJK Warning: Redefining CJKfamily ...` | 中文字族被再次设置，常见，可忽略除非排查字体冲突。 |
| `Font shape 'U/rsfs/m/n' in size ... not available` | `mathrsfs` 请求的字号无对应曲线，用邻近字号替代。 |
| `Font shape 'TU/TimesNewRoman(0)/m/sc' undefined` | Times New Roman 无 small caps，回退为正常体；某处使用了 `\textsc` 或小 caps。 |
| 文末 `Size substitutions` / `Some font shapes were not available` | 上述字体的汇总提示。 |

---

## 版式：Overfull / Underfull

- **Overfull `\hbox`（`body/method.tex`、`body/experiments.tex` 等）**：行宽超出版心，PDF 中可能溢出。可通过断行、`url`/`xurl`、公式/表格缩放、`allowbreak` 等处理。
- **Underfull `\hbox`（`body/references.tex`）**：参考文献行长、断行不佳；多数可接受。

---

## 其它包提示

- **`transparent`**：`pdfTeX is not running in PDF mode` —— XeLaTeX 下常见；未使用透明效果时可忽略。
- **`nameref`**：`The definition of \label has changed` —— 与宏包对 `\label` 的补丁顺序有关；交叉引用正常则可暂缓。
- **`pgfplots`**：建议 `\pgfplotsset{compat=1.18}`（可选）。
- **`rerunfilecheck`**：`main.out` 已变化 —— 需再运行一次或多次 XeLaTeX（或使用 `latexmk`）以稳定书签与引用。

---

## 建议处理顺序

1. 修复致谢 `\acknowledgement` 与 `hyperref` 书签（消除尾部 `\ifx` 不完整与相关警告）。
2. 连续编译 2～3 次（biblatex、目录、`rerunfilecheck`）。
3. 按需调整 geometry 与 Overfull 段落。

---

## 参考命令

在 `papers/template-of-thesis` 目录下：

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

或使用 `latexmk -xelatex main.tex` 自动处理多次运行。
