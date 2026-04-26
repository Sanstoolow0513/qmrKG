# Pipeline 对照 `task.md` 补齐清单

## 1. 目标

将当前 QmrKG 的工程实现，从“核心流程可运行”补齐到“满足毕业设计任务要求（方法 + 验证 + 评估）”。

## 2. 当前结论（简版）

- 已满足：自动化抽取主流程（文本输入 -> 联合抽取（entities + triples）-> 融合 -> 入库）。
- **已满足（抽取侧）**：`config.yaml` 中 `extract.prompts.zero_shot` / `few_shot` 已参数化；`KGExtractor` 通过 `_mode_to_prompt_key()` 解析 mode 参数并从 `config.yaml` 加载对应 prompt；`kgextract --mode zero-shot|few-shot` 可按模式选用对应 prompt，并配合不同 `--output-dir` 分目录产出便于对照。README.md 已更新 zs/fs 分目录工作流。
- 部分满足：知识验证（目前以规则过滤为主，位于 `kg_merger.py` 内联逻辑）。
- **仍未完成**：评估闭环（评测数据、自动评估脚本、评估 CLI、评估报告模板）、实验脚本与对照报告、验证器模块化、embedding 语义建模。

## 3. 必做项（按优先级）

### P0：评估闭环（先做）

- [ ] 新增评测数据与标注规范文档
  - 交付物：`docs/evaluation/annotation-guideline.md`
  - 交付物：`data/eval/`（样例或说明，不提交敏感/大体积原始数据）
- [ ] 新增自动评估脚本（实体、关系、三元组）
  - 交付物：`src/qmrkg/evaluation.py` 或 `src/qmrkg/eval/*`
  - 指标：Precision / Recall / F1（至少 micro；最好增加 macro）
- [ ] 新增评估 CLI
  - 交付物：`src/qmrkg/cli_eval.py`
  - 示例命令：`uv run kgeval --pred ... --gold ...`
- [ ] 形成评估报告模板
  - 交付物：`docs/reports/eval-report-template.md`
  - 内容：覆盖率、准确率、可解释性（evidence 命中率）、人工修订成本（每 100 条修订时长）

验收标准：
- 能一键输出指标 JSON/Markdown 报告。
- 能清晰展示“当前方案”在实体和关系抽取上的量化结果。

### P1：实验设计（zero-shot/few-shot）

- [x] 将抽取 prompt 模板参数化
  - 交付物：`config.yaml` 中 `extract.prompts.zero_shot`、`extract.prompts.few_shot`
  - 实现：`KGExtractor._resolve_system_prompt()` → `_load_extract_prompts()` → `_mode_to_prompt_key()` 链
- [x] `kgextract` 支持按模式选用 prompt（`--mode zero-shot` / `--mode few-shot`，默认 `zero-shot`）
  - 实现：`cli_kg_extract.py` argparse + `KGExtractor(mode=...)` 透传
- [x] README 文档已更新 zs/fs 分目录工作流示例
- [ ] 🔴 增加实验运行脚本（**下一优先级 — 依赖 P0 评估**）
  - 交付物：`scripts/run_experiments.py`
  - 维度：mode、temperature、retry、chunk 策略
- [ ] 🔴 产出对比实验结果（**下一优先级 — 依赖 P0 评估**）
  - 交付物：`docs/reports/experiment-zeroshot-vs-fewshot.md`
  - 指标：实体 F1、关系 F1、三元组 F1、成本（token/时间）

验收标准：
- 至少完成一组 zero-shot vs few-shot 对照实验并可复现。
- 报告中给出“推荐默认配置”的依据。

### P1：知识验证模块化

- [ ] 从 `kg_merger` 中抽离验证器（规则可配置）
  - 交付物：`src/qmrkg/kg_validator.py`
- [ ] 增加验证维度
  - 自环、实体类型合法性、长度阈值、关系白名单、证据长度阈值、证据覆盖率
- [ ] 输出验证统计
  - 交付物：`validation_stats`（保留/剔除数量及原因分布）

验收标准：
- 每次 `kgmerge` 后可看到结构化验证报告。
- 能解释“为什么删除了哪些三元组”。

### P2：Embedding 语义关联（补足课题要求）

- [ ] 增加向量生成流程（实体名或 chunk 级）
  - 交付物：`src/qmrkg/kg_embedding.py`
- [ ] 增加语义关联策略
  - 用途：同义实体发现、弱关系候选、结果校验辅助
- [ ] 增加可开关配置
  - 交付物：`config.yaml` 的 `embedding` 段
- [ ] 形成效果报告
  - 交付物：`docs/reports/embedding-impact.md`

验收标准：
- 可重复运行 embedding 流程。
- 报告说明 embedding 是否提升关系发现或降噪效果。

## 4. 推荐两周排期

### Week 1

- [ ] D1-D2：评测数据规范 + 标注样例
- [ ] D3-D4：`kgeval` CLI 与指标计算
- [x] D5：zero/few-shot 配置化（`extract.prompts` + `kgextract --mode`）✅ 已完成
- [ ] D5 剩余：实验脚本与首轮对比报告（依赖 D1-D4 评估闭环）

### Week 2

- [ ] D1-D2：验证模块化 + 验证统计
- [ ] D3-D4：embedding 原型接入
- [ ] D5：汇总实验报告与答辩图表素材

## 5. 最小可答辩交付包（建议）

- [ ] 一张端到端流程图（现有可复用）
- [ ] 一张 zero-shot vs few-shot 对比表
- [ ] 一张“是否加入 embedding”的增益对比表
- [ ] 一份评估报告（含指标与样例误差分析）
- [ ] 一份系统演示脚本（从输入文档到 Neo4j 可视化）

## 6. 风险与规避

- 风险：标注成本高
  - 规避：先做小规模高质量 gold set（例如 200~500 三元组）
- 风险：few-shot 提升不稳定
  - 规避：固定随机种子、固定抽样集、多次运行取均值
- 风险：embedding 引入复杂度但收益不明显
  - 规避：先做离线对照，不直接耦合主链路

## 7. 完成定义（DoD）

满足以下条件可判定“已达到 `task.md` 要求”：

- [ ] 有可复现的自动化抽取与融合 pipeline；
- [ ] 有 zero-shot/few-shot 方法对比与结论；
- [ ] 有 embedding 语义建模或充分论证其取舍；
- [ ] 有覆盖率、准确率、可解释性、人工干预成本的量化评估；
- [ ] 有演示与文档可支持毕业设计答辩。

---

## 8. 下一步计划（按执行顺序）

### 步骤 0：现状确认 ✅

| 模块 | 状态 | 关键文件 |
|------|------|----------|
| 抽取主流程（entities + triples） | ✅ 完成 | `kg_extractor.py`, `kg_merger.py`, `kg_neo4j.py` |
| zero-shot prompt + CLI mode | ✅ 完成 | `config.yaml` L111-143, `cli_kg_extract.py` L33-37 |
| few-shot prompt + CLI mode | ✅ 完成 | `config.yaml` L146-179, `kg_extractor.py` L56-62 |
| README zs/fs 工作流文档 | ✅ 完成 | `README.md` |
| 评估标注规范与数据 | ❌ 未开始 | 待建 `docs/evaluation/`, `data/eval/` |
| 自动评估脚本 | ❌ 未开始 | 待建 `src/qmrkg/evaluation.py` |
| 评估 CLI | ❌ 未开始 | 待建 `src/qmrkg/cli_eval.py` |
| 实验脚本 | ❌ 未开始 | 待建 `scripts/run_experiments.py` |
| 对照实验报告 | ❌ 未开始 | 待建 `docs/reports/experiment-zeroshot-vs-fewshot.md` |
| 验证器模块化 | ❌ 未开始 | 待建 `src/qmrkg/kg_validator.py` |
| Embedding 模块 | ❌ 未开始 | 待建 `src/qmrkg/kg_embedding.py` |

### 步骤 1：P0 评估闭环（当前 🔴 阻塞项）

**为什么先做？** 没有评估能力，实验脚本和对照报告无量化指标支撑，后续所有 "对比" 工作都无法量化。

#### 1a. 评测规范文档 + 标注数据

```
docs/evaluation/annotation-guideline.md   # 标注规范
data/eval/gold_triples.json               # gold set (200-500 三元组)
```

**内容要点：**
- 实体标注规则（类型边界、别名处理）
- 关系标注规则（方向性、负例定义）
- 三元组评估粒度（严格匹配 vs 部分匹配）
- 建议来源：从已有 `data/triples/merged/` 产出中人工筛选 + 校订

#### 1b. 自动评估脚本

```
src/qmrkg/evaluation.py   # 核心评估逻辑
src/qmrkg/cli_eval.py     # CLI 入口
```

**指标实现：**
- Entity: Micro/Macro Precision, Recall, F1
- Relation: Micro/Macro Precision, Recall, F1（按 (head, relation, tail) 匹配）
- 统计：总抽取量、evidence 覆盖率、空响应比例
- 输出：JSON 报告 + Markdown 表格

**CLI 接口：**
```bash
uv run kgeval --pred data/triples/raw/zs --gold data/eval/gold_triples.json
uv run kgeval --pred data/triples/raw/fs --gold data/eval/gold_triples.json
```

#### 1c. 评估报告模板

```
docs/reports/eval-report-template.md
```

### 步骤 2：P1 实验脚本 + 对照报告（依赖步骤 1）

```
scripts/run_experiments.py                # 一键 zs/fs 对照
docs/reports/experiment-zeroshot-vs-fewshot.md
```

**实验脚本功能：**
1. 对同一 `--input` 目录，自动运行 `kgextract --mode zero-shot` 和 `--mode few-shot`
2. 分别 `kgmerge` 到不同输出目录
3. 调用 `kgeval` 输出指标对比
4. 汇总 token 消耗和时间成本

**报告内容：**
- 方法说明（zero-shot vs few-shot 设计差异）
- 定量对比表（Entity F1 / Relation F1 / Triple F1 / Token / Time）
- 定性分析（抽样 case study，误差类型分布）
- 推荐默认配置及依据

### 步骤 3：P1 验证器模块化（可并行）

```
src/qmrkg/kg_validator.py
```

从 `kg_merger.py` 中抽离验证逻辑，增加可配置规则：
- 自环检测 (head == tail)
- 实体类型白名单
- 关系白名单
- 名称长度阈值
- evidence 长度阈值
- 统计输出（filtered 数量 + 原因分布）

### 步骤 4：P2 Embedding 语义关联（可选增项）

```
src/qmrkg/kg_embedding.py
config.yaml → embedding 段
docs/reports/embedding-impact.md
```

**策略：**
- 使用轻量 embedding 模型（如 text2vec 或 bge-small-zh）
- 对实体名称做向量化 → 余弦相似度发现同义实体候选
- 离线对照：对比 "有 embedding 辅助去重" vs "纯规则去重" 效果
- 不直接耦合主链路，通过开关控制
