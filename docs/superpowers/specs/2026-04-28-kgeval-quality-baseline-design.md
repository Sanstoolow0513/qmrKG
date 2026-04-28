# kgeval 最小稳定质量基线设计

## 1. 背景与目标

当前 QmrKG 从 `kgextract` 到 `kgmerge` 已能生成 `merged_triples.json`，但缺少一条可提交、可复现、可解释的质量评估闭环。旧的 `docs/pipeline-task-gap-list.md` 已经滞后：本地存在 `data/eval/gold_triples.json` 和评估规范文档，但 `data/` 被 `.gitignore` 忽略，不能作为稳定交付；同时仓库还没有 `kgeval` CLI、评估核心模块和可测试的报告生成能力。

本设计目标是先建立最小稳定质量基线：

- 评估对象固定为 `kgmerge` 后的 `merged_triples.json`。
- gold 数据使用纯金标格式，而不是抽样审计表格式。
- 匹配规则使用 strict matching，保证指标定义清楚。
- 输出实体与三元组的 Precision / Recall / F1，以及 evidence 覆盖率。
- 提交轻量 fixture，真实大规模 gold set 继续作为本地数据管理。

## 2. 非目标

第一版不做以下事项：

- 不评估 `kgextract` raw chunk 输出。
- 不兼容当前本地审计表格式中的 `correct` / `corrected_*` 字段。
- 不做别名、大小写、空格、后缀或 embedding 归一后的宽松匹配。
- 不做 macro F1。第一版样本较小，按类型或关系拆分后的指标波动大。
- 不做 zero-shot / few-shot 实验编排。后续实验脚本复用本评估能力。

## 3. 架构

新增一个边界清晰的评估子系统：

1. `src/qmrkg/evaluation.py`
   - 负责读取 predicted/gold JSON。
   - 校验输入 schema。
   - 生成 strict matching key。
   - 计算实体和三元组指标。
   - 生成 JSON 可序列化报告结构。
   - 生成 Markdown 报告文本。

2. `src/qmrkg/cli_eval.py`
   - 提供 `kgeval` CLI。
   - 只处理参数解析、调用核心逻辑、写报告和退出码。

3. `pyproject.toml`
   - 注册命令：`kgeval = "qmrkg.cli_eval:main"`。

4. `src/qmrkg/cli_qmrkg.py`
   - 在 `qmrkg --list` 中展示 `kgeval`。

5. `tests/fixtures/eval/`
   - 提交最小 `pred_merged.json` 和 `gold_triples.json`。
   - 用于测试、格式示例和文档引用。

6. `docs/evaluation/`
   - 更新评估说明，解释 gold 格式、strict matching 和推荐命令。

## 4. 输入格式

### 4.1 Predicted

`--pred` 输入为现有 `kgmerge` 输出，即 `merged_triples.json` schema：

```json
{
  "entities": [
    {"name": "HTTP", "type": "protocol", "description": "HTTP协议", "frequency": 3}
  ],
  "triples": [
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "depends_on",
      "tail": "TCP",
      "tail_type": "protocol",
      "frequency": 2,
      "evidences": ["HTTP 使用 TCP 作为传输层协议"]
    }
  ],
  "stats": {}
}
```

`description`、`frequency`、`stats` 不参与匹配。

### 4.2 Gold

`--gold` 使用纯金标格式：

```json
{
  "meta": {
    "name": "qmrkg-minimal-eval",
    "schema_version": 1
  },
  "entities": [
    {"name": "HTTP", "type": "protocol"}
  ],
  "triples": [
    {
      "head": "HTTP",
      "head_type": "protocol",
      "relation": "depends_on",
      "tail": "TCP",
      "tail_type": "protocol",
      "evidence": "HTTP 使用 TCP 作为传输层协议"
    }
  ]
}
```

`entities` 可显式提供。若 gold 未提供 `entities` 或为空，`kgeval` 从 gold triples 派生实体集合：

- `(head, head_type)`
- `(tail, tail_type)`

`evidence` 在 gold 中可保留为人工核对依据，但不参与三元组 strict matching。

## 5. Strict Matching

实体匹配键：

```python
(name, type)
```

三元组匹配键：

```python
(head, head_type, relation, tail, tail_type)
```

匹配前只做安全性的 `strip()`，不做语义归一、大小写折叠、别名映射或后缀剥离。这样指标可以明确解释为“当前最终图谱 JSON 中的结构化字段是否精确命中金标”。

## 6. 指标

第一版输出 micro 指标。

### 6.1 Entity

- `pred_count`
- `gold_count`
- `tp`
- `fp`
- `fn`
- `precision`
- `recall`
- `f1`

### 6.2 Triple

- `pred_count`
- `gold_count`
- `tp`
- `fp`
- `fn`
- `precision`
- `recall`
- `f1`

### 6.3 Evidence

预测 evidence 覆盖率：

```text
pred triples with non-empty evidences / total pred triples
```

TP evidence 覆盖率：

```text
true positive pred triples with non-empty evidences / total true positive triples
```

`merged_triples.json` 中 `evidences` 是列表；只要列表中存在非空字符串，即视为该预测三元组有 evidence。

### 6.4 Error Samples

报告中保留前 N 条错误样例：

- `false_positives`
- `false_negatives`

默认 `N=10`，由 CLI `--top-errors` 控制。

## 7. CLI

命令示例：

```bash
uv run kgeval \
  --pred data/triples/merged/merged_triples.json \
  --gold data/eval/gold_triples.json \
  --output-json docs/reports/eval-baseline.json \
  --output-md docs/reports/eval-baseline.md
```

参数：

- `--pred PATH`：必填，预测 `merged_triples.json`。
- `--gold PATH`：必填，纯金标 JSON。
- `--output-json PATH`：可选，写 JSON 报告；不传时将 JSON 摘要打印到 stdout。
- `--output-md PATH`：可选，写 Markdown 报告。
- `--top-errors N`：可选，默认 `10`。
- `-v/--verbose`：可选，启用详细日志。

错误处理：

- 输入文件不存在：返回非零退出码。
- JSON 无法解析：返回非零退出码。
- 缺少 `triples` 或关键字段缺失：返回非零退出码，并说明缺失字段。
- 非法实体类型或关系类型：返回非零退出码。

## 8. Markdown 报告

Markdown 报告包含：

1. Evaluation Inputs
   - pred 路径
   - gold 路径
   - 运行时间
   - gold schema version

2. Summary
   - Entity P/R/F1 表格
   - Triple P/R/F1 表格

3. Evidence
   - predicted evidence coverage
   - TP evidence coverage

4. Error Samples
   - false positives
   - false negatives

5. Notes
   - 说明 strict matching 规则。
   - 明确 evidence 只统计覆盖率，不参与三元组匹配。

## 9. 测试

新增测试覆盖：

- 完全匹配时实体和三元组 F1 均为 `1.0`。
- 预测多一条、少一条时 FP/FN 计数正确。
- gold 未显式提供 `entities` 时能从 triples 派生实体集合。
- evidence 覆盖率按 `evidences` 非空列表计算。
- CLI 能生成 JSON 和 Markdown 文件。
- 输入字段缺失时失败，而不是静默给出假指标。
- `qmrkg --list` 包含 `kgeval`。

## 10. 验收标准

实现完成后应满足：

- `uv run kgeval --pred ... --gold ...` 可运行。
- JSON 报告和 Markdown 报告均可生成。
- 指标定义可在报告和文档中解释清楚。
- 测试套件通过。
- 后续 zero-shot / few-shot、验证器、embedding 对比实验可以复用同一个评估入口。
