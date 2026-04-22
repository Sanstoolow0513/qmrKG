# kgextract 支持 Zero-shot / Few-shot 模式设计

## 1. 背景与目标

当前 `QmrKG` 已具备可运行的知识抽取主流程，但根据任务清单仍需补齐 `zero-shot / few-shot` 方法对比能力。  
本设计目标是在不破坏现有链路的前提下，使 `uv run kgextract` 直接支持：

- `--mode zero-shot`
- `--mode few-shot`

并通过手动分别运行两次命令完成可复现实验对比。

## 2. 约束与边界

- 抽取方式为联合抽取（同一 prompt 同时输出 `entities` + `triples`）。
- few-shot 示例写入 `config.yaml`（不拆外部示例文件）。
- 本次仅做 `kgextract` 模式切换，不引入自动实验编排脚本。
- 对照实验应保持公平：除 prompt 外，不修改模型、请求参数、后处理逻辑。

## 3. 方案选择

已对比三种方案并选择方案 1：

- 方案 1（采用）：在 `extract.prompts` 下新增 `zero_shot`、`few_shot` 两套完整 prompt，由 `--mode` 选择。
- 方案 2（未采用）：`base_prompt + examples` 动态拼接，复杂度更高，稳定性风险更大。
- 方案 3（未采用）：在任务工厂层做任务变体，改动面过大，不符合“最小可用改造”。

选择理由：方案 1 实现简单、风险低、与当前配置风格一致，最适合当前毕业设计阶段交付。

## 4. 配置设计（config.yaml）

在 `extract.prompts` 下新增并保留以下键：

- `zero_shot`：完整 zero-shot 联合抽取提示词。
- `few_shot`：完整 few-shot 联合抽取提示词（内含 1~3 组示例）。
- `default`：兼容保留，语义对齐到 `zero_shot`（避免旧逻辑受影响）。

约定：

- `zero_shot` 与 `few_shot` 使用同一输出 JSON schema：
  - `entities: [{name, type, description}]`
  - `triples: [{head, relation, tail, evidence}]`
- few-shot 示例覆盖计网课程术语，降低示例分布偏移风险。

## 5. CLI 设计（kgextract）

在 `kgextract` 命令新增参数：

- `--mode {zero-shot,few-shot}`
- 默认值：`zero-shot`

运行行为：

- CLI 解析 `mode` 后传给 `KGExtractor(mode=...)`。
- 启动日志打印当前模式（用于实验留痕）。
- 输出目录仍由 `--output-dir` 显式控制，不做隐式自动分目录。

推荐实验运行方式（手动两次）：

- `uv run kgextract --mode zero-shot --input data/chunks --output-dir data/triples/raw/zs`
- `uv run kgextract --mode few-shot --input data/chunks --output-dir data/triples/raw/fs`

## 6. 代码改动范围

仅修改以下位置：

1. `src/qmrkg/cli_kg_extract.py`
   - 增加 `--mode` 参数（含默认值与 choices）。
   - 传递 mode 到 `KGExtractor`。
   - 打印运行模式日志。

2. `src/qmrkg/kg_extractor.py`
   - `KGExtractor.__init__` 增加 `mode` 入参。
   - 增加 prompt 选择与回退逻辑。
   - `extract_from_chunk` 使用选中的 prompt。

3. `config.yaml`
   - 增加 `extract.prompts.zero_shot` 与 `extract.prompts.few_shot`。
   - 保留 `extract.prompts.default` 做兼容锚点。

本次不改动：

- `llm_factory.py`
- `kg_merge`、`evaluation` 相关模块
- 自动化实验脚本

## 7. 数据流与容错

数据流：

`CLI --mode -> KGExtractor(mode) -> 解析 prompt -> run_text(..., system_prompt=selected_prompt) -> JSON 解析 -> schema 校验`

容错策略：

- 非法 mode：由 CLI 参数 choices 提前拦截。
- prompt 缺失：按回退链路处理：
  1. `extract.prompts.<mode_key>`
  2. `extract.prompts.default`
  3. 内置 `EXTRACT_PROMPT`
- LLM 非 JSON：记录 warning，返回空结果，继续流程。
- 单 chunk 异常：记录 error，继续后续 chunk，保证批处理鲁棒性。

## 8. 测试与验收

最小测试集：

- mode 选择测试：
  - `mode=zero-shot` 使用 `zero_shot` prompt
  - `mode=few-shot` 使用 `few_shot` prompt
- 回退链路测试：
  - 缺 `mode prompt` -> 回退 `default`
  - 缺 `default` -> 回退内置 prompt
- CLI 兼容测试：
  - 不传 `--mode` 时默认 `zero-shot`

验收标准：

- `uv run kgextract --mode zero-shot ...` 与 `--mode few-shot ...` 均可运行。
- 两次运行产物可通过不同 `--output-dir` 分离保存。
- 除 prompt 外，处理链路与校验逻辑一致，可用于公平对照。
- 日志可追溯模式，配置缺失时不崩溃。

## 9. 风险与对策

- 风险：few-shot 示例质量不足导致提升不稳定。  
  对策：优先保证示例高质量、强约束输出格式、覆盖典型关系类型。

- 风险：提示词改动造成旧流程行为漂移。  
  对策：保留 `default` 并设置默认 mode 为 `zero-shot`，确保兼容。

- 风险：实验对照不公平（混入其他变量变化）。  
  对策：文档明确“仅切换 mode”，其余参数固定。

## 10. 结论

该设计以最小改动实现 `kgextract` 的 zero-shot / few-shot 可切换能力，满足当前毕业设计阶段的可复现实验需求，并为后续实验报告（实体 F1、关系 F1、三元组 F1、成本）提供可追溯输入基础。
