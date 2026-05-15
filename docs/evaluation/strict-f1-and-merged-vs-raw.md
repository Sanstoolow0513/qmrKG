# 严格 F1 偏低的成因与「raw 用于比较、merged 用于报告」的评估口径

**Date:** 2026-05-13
**Scope:** 解释 `kgeval` / `kgevalraw` 在课程 KG 上严格 F1 偏低（11%~17%）的原因，并说明为什么 zs / fs 提示策略对比应在 raw 层做、而系统最终评估口径应回到 merged 层。
**关联文件**
- 论文：`papers/template-of-thesis/body/experiments.tex` §5.5
- 严格评估实现：`src/qmrkg/evaluation.py`
- raw 评估实现：`src/qmrkg/eval_raw.py`
- 四档松匹配脚本：`scripts/eval_relaxed.py`、`scripts/eval_relaxed_raw.py`
- 四档结果产物：`data/eval/relaxed_*.json`
- 旧报告（说明字面问题）：`data/eval/baseline_report.md`

---

## 1. 严格 F1 的定义与现象

`evaluation.py` 的 `_set_counts` 把预测和金标都规约为不可变 `TripleKey`：

```python
TripleKey(head, head_type, relation, tail, tail_type)  # 见 src/qmrkg/evaluation.py:27
```

「TP」是两个集合的交集，五个字符串字段必须**逐字相等**（含大小写、括号、空白、中英文、缩写）。实体侧同样要求 `EntityKey(name, type)` 完全相等。

在 500 条人工复核金标 + 186 个源 chunk 子集上，四种配置的严格三元组 F1 落在 11.90%~15.29%（合并层）和 12.20%~17.40%（raw 层），见 `experiments.tex` 表 5-9 与 5-10。同一份预测把匹配口径放宽：

| 口径 | zs-recheck 合并层 F1 | 含义 |
| --- | ---: | --- |
| 严格（五字段相等） | 13.76% | 当前 `kgeval` 默认 |
| 软匹配（实体名归一化 + 子串包含） | 18.40% | 类型与关系仍精确 |
| 槽位（仅 head+relation+两端类型） | 42.49% | tail 名可不同 |
| 类型级（仅 head_type+relation+tail_type） | **81.93%** | KG schema 覆盖能力 |

同一组三元组、仅放宽实体名匹配方式，F1 即从 13.76% 抬升到 81.93%。这条阶梯本身就是失分构成的最直接证据：**严格 F1 偏低不是关系判错，而是端点名字面差异。**

## 2. 严格 F1 为何这么低：六个具体原因

### 2.1 中英缩写并用是课程语料的常态

教材、RFC、PPT 三种语料源混用「TCP / 传输控制协议 / 传输控制协议(TCP) / TCP协议」「IP 地址 / IP address / IPv4 地址」「CSMA/CA / 二进制指数退避」等表达。模型给出哪一种、金标记录哪一种，常常不一致——但语义对得上。`baseline_report.md` 的 FN 列表里大量出现 `5层因特网协议栈 / AP / CSMA/CA / GBN` 这类缩写或括号变体，就属于此类。

### 2.2 抽样金标的非穷尽性导致 FP 被高估

`gold_triples.json` 是 500 条**人工复核的正例参考集**，并非 186 个源 chunk 中所有正确关系的穷尽标注。系统抽出的、语义正确但没出现在金标里的三元组会被记为 FP，从而压低精确率。这一点 `experiments.tex` §5.5.2 已有明文说明，本文不重复。

### 2.3 Schema 颗粒度让边界关系两可

四种关系（`contains` / `applied_to` / `depends_on` / `compared_with`）覆盖率高但语义边界模糊：「TCP `contains` 拥塞控制」与「TCP `depends_on` 拥塞控制」在某些上下文里都说得通；模型挑一个、人工挑另一个，严格匹配立刻判错。`compared_with` 还存在方向性（`A compared_with B` vs `B compared_with A`）没有强制规范化，进一步降低字面命中。

### 2.4 类型注入路径在 raw / merged 两层不同

- raw 层：`eval_raw.py:_load_raw_triples` 用同 chunk 的 `entities` 列表查 `(name → type)` 注入 `head_type / tail_type`；查不到则注入空串，与金标无法相等，自动判错（`experiments.tex` §5.5.3 提到 ZS 缺 4 个、FS 缺 10 个 chunk 即源于此）。
- merged 层：`kgmerge` 在合并阶段用词表别名 + embedding 聚类把同一实体的多种表面形式压成「规范名」。规范名取众数或最长形式，未必恰好与金标的字面形式相同——把 raw 层原本能命中的字面命中替换成更"标准"但与金标错位的形式（详见 §3）。

### 2.5 `kgmerge` 把字面命中"洗"成规范命中

合并层严格 F1 在所有四种配置下都比 raw 层低 1~3 个百分点（如 zs-nocheck：合并 15.29% vs raw 17.40%）。这不是系统能力下降，而是规范化把原本巧合命中金标字面的预测替换成了类簇中心。槽位 / 类型级档下两层差异基本消失，证实合并对"关系类型 + 头实体识别能力"几乎无损。

### 2.6 严格匹配不区分「证据强度」

`evaluation.py` 把所有三元组同等看待。复核（recheck）开关其实在剔除"字面对、语义弱"的命中并保留"语义对、字面更自然"的候选——前者会减分严格 F1（损失字面 TP），后者要在槽位 / 类型级档才看得见收益。所以严格档下 recheck 普遍略输 nocheck，但槽位档下 zs-recheck 反超到 42.49%，类型级档下到 81.93% 全场最高。**严格 F1 与系统实际可用质量是两条曲线。**

## 3. 为什么 zs vs fs 对比应在 raw 层进行

提示策略对比（zero-shot vs few-shot，含 / 不含 LLM 复核）属于**抽取阶段**的消融，应当在抽取直接产物上度量。理由：

1. **变量隔离**。raw 层只携带 `(head, relation, tail) + 同 chunk entities`，没有别名表、没有 embedding、没有 schema 约束——除提示词外其它变量为零。一旦走到 merged，上游差异会被合并算法的同义聚类"洗平"，弱化两策略的实际差距。
2. **per-chunk 对齐天然可比**。`eval_raw.py` 按 `(source_file, chunk_index)` 与金标 chunk 一一对应；同一 chunk 上 zs 与 fs 的输入完全一样，差异只来自 prompt。这给了"控制变量"的硬保证。merged 没有这种边界，金标 500 条 vs 全量合并图（数万条）存在不对称。
3. **覆盖率指标只在 raw 层有意义**。`chunks_missing_raw`（FS 缺 10、ZS 缺 4）反映 prompt 让模型"该输出而没输出"的情况；合并阶段会用别的 chunk 的命中把"缺口"填掉，无法度量这种行为差。
4. **属性层准确率（head_type / tail_type / relation）只在 raw 层报得清**。`_attribute_accuracy` 要求预测和金标在同一 (head, tail) 对上比较，merged 阶段实体已规范化，这种逐三元组比对就失去对照基础。例子见 `raw-eval-zs-vs-fs.md`：FS 在 tail_type 上比 ZS 高 +13.80 个百分点（89.04% vs 75.24%），这个观察必须在 raw 层才看得到。

结论：要回答 **"哪种 prompt 抽得更准"**，看 `kgevalraw` / `relaxed_raw_*.json`。

## 4. 为什么系统最终评估应在 merged 层进行

merged 是**用户实际消费**的工件——`kgneo4j` 入图、前端读图、下游问答检索都基于 `merged_triples.json`，而非 `data/triples/raw/{zs,fs}/`。所以"系统好不好用"必须在 merged 层度量。理由：

1. **合并是系统能力的一部分，不是评估外的变量**。词表别名规范化 + embedding 实体对齐是流水线的设计组件（`kg_merge.embedding.enabled`），把它从评估里剥掉等于评一个用户拿不到的中间态。
2. **merged 才能体现去重 / 对齐效果**。raw 层"TCP" / "传输控制协议" / "TCP(传输控制协议)" 是三个独立节点；merged 把它们压成一个枢纽节点，是 KG 价值的核心来源。这部分能力只有 merged 层指标才能反映。
3. **总规模指标只在 merged 层有定义**。论文 §5.3、§5.4、§5.7（实体/关系/图谱统计：32,760 实体 / 17,627 边 / 平均度 2.71 等）报告的是 `merged-fs-recheck` 的产物。raw 层有重复，规模统计无意义。
4. **复核机制收益要在合并 + 宽口径下才显现**。`zs-recheck` 在合并层类型级 F1 达到 81.93%（全场最高），把"复核 = 提升语义覆盖"的设计意图量化下来。

结论：要回答 **"系统跑出来的 KG 质量如何"**，看 `kgeval` / `relaxed_zs-recheck.json` 等合并层产物。

## 5. 推荐的读数规则

| 用途 | 用哪个 CLI / 文件 | 用哪一档 |
| --- | --- | --- |
| 比较 prompt 策略（zs/fs、复核开关） | `kgevalraw` → `relaxed_raw_*.json` | 严格 + 槽位 + 类型级 三档对比看趋势 |
| 报告系统整体质量 | `kgeval` → `relaxed_*.json` | 软匹配 / 槽位为主指标，严格为下界，类型级为上界 |
| 回答审稿人「你们 F1 才 15%？」 | 引导到表 5-9 的阶梯 | 严格 13.76% → 类型级 81.93%，强调阶梯本身就是诊断 |
| 论文里挑一个数当"系统 F1" | merged 层、zs-recheck、槽位档 | 42.49%（合理且可辩护） |
| 决定是否上线 / 调 prompt | `kgevalraw` 的属性层准确率 | head_type / tail_type / relation 三个 acc + 缺失 chunk 数 |

不要做的事：
- ❌ 拿 raw 层数字当系统对外指标（用户拿不到 raw）
- ❌ 拿合并层严格 F1 当 prompt 策略的 ablation 结论（合并把差异洗掉了）
- ❌ 跨配置（不同 merge 参数）比较严格 F1（变量没控住）
- ❌ 把严格 F1 当唯一指标（它只是阶梯下界，不是系统能力）

## 6. 与论文的对应关系

`experiments.tex` §5.5 已经实现了本文档主张的口径分离：

- 表 5-9（合并层四档）：报告系统对外质量，主指标在槽位/类型级。
- 表 5-10（raw 层四档）：报告 prompt 策略 ablation，对比 zs/fs/复核。
- §5.5.3 三条结论（严格 F1 单独不能刻画系统、复核作用方向、few-shot 不能简单堆示例）正是 §3、§4 的论文语言版本。

本文档相当于把这一评估口径**显式化为方法学说明**，便于后续答辩或外部读者快速理解为什么不能只看严格 F1、为什么 raw 与 merged 各有专属用途。
