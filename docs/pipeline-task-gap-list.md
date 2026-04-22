# Pipeline 对照 `task.md` 补齐清单

## 1. 目标

将当前 QmrKG 的工程实现，从“核心流程可运行”补齐到“满足毕业设计任务要求（方法 + 验证 + 评估）”。

## 2. 当前结论（简版）

- 已满足：自动化抽取主流程（文本输入 -> NER/RE -> 三元组 -> 融合 -> 入库）。
- 已满足（抽取侧）：`config.yaml` 中 `extract.prompts.zero_shot` / `few_shot` 已参数化；`kgextract --mode zero-shot|few-shot` 可按模式选用对应 prompt，并配合不同 `--output-dir` 分目录产出便于对照。
- 部分满足：知识验证（目前以规则过滤为主）。
- 未满足或证据不足：embedding 语义建模、zero-shot/few-shot **系统化对照实验与评估闭环**、完整实验脚本与报告。

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
- [x] `kgextract` 支持按模式选用 prompt（`--mode zero-shot` / `--mode few-shot`，默认 `zero-shot`）
- [ ] 增加实验运行脚本
  - 交付物：`scripts/run_experiments.py`（或等价实现）
  - 维度：prompt 版本、温度、重试策略、chunk 策略
- [ ] 产出对比实验结果
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
- [x] D5：zero/few-shot 配置化（`extract.prompts` + `kgextract --mode`）；对照实验脚本与首轮报告仍待办

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
