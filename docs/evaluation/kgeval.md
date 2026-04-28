# kgeval 质量基线

`kgeval` 用于评估 `kgmerge` 生成的 `merged_triples.json`。第一版只评估最终合并图谱，不评估 `kgextract` 的 raw chunk 输出。

## 命令

```bash
uv run kgeval \
  --pred data/triples/merged/merged_triples.json \
  --gold data/eval/gold_triples.json \
  --output-json docs/reports/eval-baseline.json \
  --output-md docs/reports/eval-baseline.md
```

如果不传 `--output-json`，命令会把 JSON 报告打印到 stdout。

## Gold 格式

Gold 文件使用纯金标格式：

```json
{
  "meta": {
    "name": "qmrkg-minimal-eval",
    "schema_version": 1
  },
  "entities": [
    {"name": "HTTP", "type": "protocol"},
    {"name": "TCP", "type": "protocol"}
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

`entities` 可以省略或留空。省略时，`kgeval` 会从 gold triples 的 `head/head_type` 和 `tail/tail_type` 派生实体集合。

## Strict Matching

实体匹配键：

```text
(name, type)
```

三元组匹配键：

```text
(head, head_type, relation, tail, tail_type)
```

匹配前只去掉字段两端空白，不做别名归一、大小写折叠、后缀剥离或 embedding 语义匹配。

## 指标

报告输出实体和三元组的 micro Precision、Recall、F1，并输出 evidence 覆盖率：

- predicted evidence coverage：预测三元组中 `evidences` 非空的比例。
- true-positive evidence coverage：命中 gold 的预测三元组中 `evidences` 非空的比例。

Evidence 只用于覆盖率统计，不参与 strict triple matching。

## Fixture

仓库提交了轻量示例：

- `tests/fixtures/eval/pred_merged.json`
- `tests/fixtures/eval/gold_triples.json`

真实评估集仍建议放在本地 `data/eval/`，因为 `data/` 是运行数据目录，不进入 git。
