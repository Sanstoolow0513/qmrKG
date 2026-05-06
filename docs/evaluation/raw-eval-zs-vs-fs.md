# Raw Extraction Evaluation: ZS vs FS

**Date:** 2026-05-06
**Gold:** `data/eval/gold_triples.json` (500 triples, 594 entities, schema v1, 186 chunks, 88 source files)
**Evaluator:** `src/qmrkg/eval_raw.py` → `kgevalraw` CLI

## Methodology

直接对 raw extraction（不经过 merge）per-chunk 评估 zs/fs 抽取质量。

1. Gold triples 按 `(source_file, chunk_index)` 分组
2. 在 raw 目录找到对应的 per-chunk JSON，从 entities 列表构建 `{name → type}` lookup
3. 用 lookup 给 raw triple 注入 `head_type` / `tail_type`
4. 严格 exace match 计算 P/R/F1
5. FP 只计 gold-覆盖的 chunk；FN = gold 有但 pred 无

### 评估维度

| 维度 | 匹配 key | 说明 |
|------|----------|------|
| Entity F1 | `(name, type)` | 实体名 + 类型 |
| Triple F1 | `(head, head_type, relation, tail, tail_type)` | 完整三元组 |
| Relation F1 | `(head_type, relation, tail_type)` | 忽略实体名 |
| head_type acc | `head_type` 是否匹配 gold | 头实体类型准确率 |
| tail_type acc | `tail_type` 是否匹配 gold | 尾实体类型准确率 |
| relation acc | `relation` 是否匹配 gold | 关系分类准确率 |

## Results

### Scope-level

| Scope | Metric | ZS | FS | Δ (FS−ZS) |
| --- | --- | ---: | ---: | ---: |
| Entity | Precision | 0.4000 | 0.3690 | −0.0310 |
| Entity | Recall | 0.2862 | 0.2727 | −0.0135 |
| **Entity** | **F1** | **0.3337** | 0.3136 | −0.0200 |
| Triple | Precision | 0.1644 | 0.1335 | −0.0309 |
| Triple | Recall | 0.1200 | 0.0980 | −0.0220 |
| **Triple** | **F1** | **0.1387** | 0.1130 | −0.0257 |
| Relation | Precision | 0.5500 | 0.5385 | −0.0115 |
| Relation | Recall | 0.8049 | 0.6829 | −0.1220 |
| **Relation** | **F1** | **0.6535** | 0.6022 | −0.0513 |

### Attribute Accuracy

| Attribute | ZS | FS | Δ (FS−ZS) |
| --- | ---: | ---: | ---: |
| head_type | **0.8667** | 0.8493 | −0.0174 |
| tail_type | 0.7524 | **0.8904** | +0.1380 |
| relation | 0.8571 | **0.8630** | +0.0059 |

### Coverage Stats

| | ZS | FS |
| --- | ---: | ---: |
| Gold-covered chunks | 186 | 186 |
| Chunks with raw match | 182 | 176 |
| Chunks missing raw | 4 | 10 |
| Raw triples in gold chunks | 397 | 392 |

## Analysis

### ZS 总体优于 FS

Zero-shot 在所有 scope-level F1 上全面超过 few-shot：Entity F1 高 2%，Triple F1 高 2.6%，Relation F1 高 5.1%。Few-shot 示例可能引入了过拟合 bias，导致 LLM 更保守或产生模式化输出。

### Triple F1 低但正常

Triple F1（ZS 14%, FS 11%）远低于 Relation F1（ZS 65%, FS 60%），差距来自实体名的严格匹配。LLM 抽取的实体名使用原文表述（如"传输控制协议"），而 gold 使用标准缩写（"TCP"），导致大量 false negative。这不反映抽取能力差——Relation F1 证明 LLM 的关系分类能力很强。

### 属性准确率接近

三个属性（head_type / tail_type / relation）的准确率都在 75-89%，差距不大。**FS 在 tail_type 上显著优于 ZS（+13.8%）**——few-shot 示例中的标准分类可能帮助 LLM 更准确判断尾实体的概念类别。

### FS 缺失更多 chunk

FS 有 10 个 chunk 找不到对应 raw 文件，ZS 只有 4 个。这表明 FS 模式下某些文件的抽取可能完全失败（无输出），也可能是文件命名不一致导致。

## Usage

```bash
uv run kgevalraw \
  --zs-dir data/triples/raw-zs-recheck \
  --fs-dir data/triples/raw-fs-recheck \
  --gold data/eval/gold_triples.json \
  --output-json data/eval/raw_results/eval.json \
  --output-md data/eval/raw_results/comparison.md
```

## Files

| File | Description |
|------|-------------|
| `src/qmrkg/eval_raw.py` | Core evaluator |
| `src/qmrkg/cli_eval_raw.py` | CLI entry point |
| `data/eval/raw_results/zs_report.json` | ZS detailed metrics |
| `data/eval/raw_results/fs_report.json` | FS detailed metrics |
| `data/eval/raw_results/comparison.json` | Comparison data |
| `data/eval/raw_results/comparison.md` | Comparison Markdown report |
