# QmrKG 论文答辩 PPT 文本

> 一份纯文本 PPT 大纲,可直接迁移到 Keynote / PPT。每页布局统一为:**标题 → 图示/表格 → 要点(3-5 条) → 设计动机一句话**。共 12 页,核心模块占 5 页。
>
> 全部实验数字以仓库 `data/eval/raw_results/comparison.md` 与 `data/eval/gold_triples.json` 的 meta 为准。

---

## 第 1 页 · 封面

**标题**:QmrKG

**副标题**:面向计算机网络教材的 PDF → 知识图谱端到端 Pipeline

**关键标签**:`知识抽取` ｜ `实体归一化` ｜ `可视化浏览`

**作者 / 指导老师 / 答辩日期**:_(占位)_

---

## 第 2 页 · 毕设要求简介

```
┌────────────────────────────────────────────────────┐
│  非结构化教学资料  ──►  结构化知识图谱  ──►  可视化  │
│  (PDF / PPT / RFC)                                  │
└────────────────────────────────────────────────────┘
```

**任务来源**:面向计算机网络领域的教学资料,把分散在教材、PPT 与协议文档中的知识点抽出,建成可查询、可浏览的知识图谱。

**三个硬约束(任务书明确要求)**:
- **自动化** — 不依赖人工标注
- **可复现** — 同输入两次跑结果一致
- **可量化** — 必须给出精确率 / 召回率 / F1

**三个目标**:
1. 端到端跑通:PDF → OCR → 抽取 → 归一化 → 入库 → 前端
2. 自建评测体系:gold standard + 双层评估
3. 在真实教材规模上验证(非 demo 级)

**设计动机**:让"质量好不好"成为可比较、可追责的工程指标,而不是肉眼判断。

---

## 第 3 页 · 设计思路与系统总览

```
PDF/PPT ─► PNG ─► per-page MD ─► 整书 MD ─► chunks
                                                │
                                                ▼
                       raw triples (zs/fs)  ◄── 抽取 + 自审核
                                │
                                ▼
                          merged triples   ◄── 三层归一化
                                │
                                ▼
                            Neo4j  ─►  前端力导向图
                                │
                                ▼
                        kgeval / kgevalraw (双层评估)
```

**三条设计原则**:
- **独立可重跑** — 7 个阶段产物全部落盘,任一环节挂掉单独重跑
- **配置驱动** — 模型 / Prompt / 阈值全部在 `config.yaml`,代码不动
- **全过程留痕** — 拒审日志、评估报告、embedding 缓存均可追溯

**真实处理规模**:
- 输入:200+ PDF 教材 + 数十份 PPT
- 中间:数万张 PNG → 数千万字符 Markdown → 万级 chunks
- 输出:`merged_triples.json` 约 **2.36 万实体**

**设计动机**:把整个"PDF → 图谱"过程切成可独立验证的工程阶段,任何指标退化都能定位到具体一环。

---

## 第 4 页 · 关键技术选型与理由

| 环节 | 选型 | 关键理由 |
|------|------|----------|
| OCR | `qwen3-vl-8b` 多模态大模型 | 直接出 Markdown,保留章节层级、公式与表格 |
| 三元组抽取 | `deepseek-v3.2` / `v4-flash` + schema 约束 | thinking 模式 + 中文表现好 + 成本低 |
| 实体归一化 | 规则 + Embedding + LLM 三层级联 | 单层都不够;成本递增、精度递增,互补 |
| 图存储 | Neo4j + Cypher | 多跳遍历与子图查询是图库强项 |
| 前端 | Next.js 16 + `react-force-graph-2d` | App Router 原生 SSR + 力导向开箱即用 |
| 横切支撑 | 自建 LLM 工厂 + 滚动窗限流器 | 限流 / 重试 / modality 校验统一收口 |

**关于"横切支撑"**:所有 LLM 调用必须经 `llm_factory.py`,**禁止直接 `import openai`**。它把 provider、模型、限流、重试和 prompt 全部绑定到 `config.yaml` 的 profile,成为单一事实源。

**设计动机**:每一项选型都对应一个具体的失败模式 — 传统 OCR 丢结构、开放抽取无 schema 不可控、单一归一化方法各自有盲区、关系型库做不了多跳查询。

---

## 第 5 页 · 核心模块 ① — LLM 工厂(横切基础设施)

```
            ┌─────────────────────────────────────┐
            │           config.yaml               │
            │  llm.profiles.<name>:               │
            │   provider · model · modality       │
            │   thinking · RPM · max_concurrency  │
            │   timeout · max_retries             │
            └─────────────────────────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────────┐
            │   TextTaskProcessor                 │
            │   MultimodalTaskProcessor           │
            │   EmbeddingTaskProcessor            │
            └─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       RollingRateLimiter          参数校验
       (RPM 滚动窗 60s              modality 匹配
        + 并发信号量)                thinking 开关
```

**要点**:
- 所有 LLM 调用走 `TextTaskProcessor` / `MultimodalTaskProcessor` / `EmbeddingTaskProcessor` 三类入口
- profile 是单一事实源:换模型只改 `llm_profile: <name>` 引用,任务段不重复 provider 字段
- **限流双重**:RPM 滚动 60s 窗 + `max_concurrency` 并发上限(信号量),抗 429
- **校验拒绝**:text profile 收到 image → `ValueError`;`supports_thinking=false` 但请求开启 → 拒绝
- **重试**:HTTP 429 / 5xx / timeout 走指数退避,默认 3 次,最长等 30s

**设计动机**:让"换模型 / 换 prompt / 调限流"变成**改配置不改代码**,同时把 LLM 调用易踩的雷(限流、modality 不匹配、thinking 误开)统一拦截在工厂层。

---

## 第 6 页 · 核心模块 ② — OCR 与文档分块

```
PDF / PPT / PPTX
       │
       ▼  PDFConverter / PPTConverter
   PNG 序列  (data/png/)
       │
       ▼  MultimodalTaskProcessor.run_image  (qwen3-vl-8b)
   per-page Markdown  (data/markdown/<book>/*_page_N.md)
       │
       ▼  kgmdcombine
   整书 Markdown  (data/markdown/<book>.md)
       │
       ▼  MarkdownChunker (token-aware, 默认 4000 token/chunk)
   chunks JSON  (chunk_index, text, titles)
```

**要点**:
- **不用传统 OCR**:Tesseract 类工具丢标题层级、表格结构与公式 — 教材没法用
- **VLM Prompt 强约束**:`第一章 绪论` → `# 第一章 绪论`;`2.1 相关工作` → `## 2.1 相关工作`,公式与图表用围栏代码块
- **多线程并发**调用 OCR,限流由工厂层托管,不会被 API 打挂
- **token-aware 分块**:用 `tiktoken` 切,不会把一句话切成两半,保留章节标题作为 metadata 传给下游

**设计动机**:把"非结构化像素"变成"结构化文本块",**为下游抽取提供干净、带结构信号的输入**。

---

## 第 7 页 · 核心模块 ③ — 三元组抽取与自审核闭环

```
chunk
  │
  ▼  抽取 LLM(zero-shot / few-shot prompt 二选一)
raw {entities, triples, evidence}
  │
  ▼  审核 LLM(独立的二次调用)
检查四项:
  ① evidence 是否为 chunk 原文连续子串
  ② head / tail 实体是否出现在 evidence 中
  ③ 实体类型 + 关系类型是否在 schema 内
  ④ 是否自环(head == tail)
  │
  ┌─────┴─────┐
  ▼           ▼
keep       drop / revise  ──►  _rejection_log.json
```

**要点**:
- **schema 强约束**:实体仅 4 类(`protocol` / `concept` / `mechanism` / `metric`),关系仅 4 类(`contains` / `depends_on` / `compared_with` / `applied_to`)
- **evidence 必填**:每条 triple 必须带原文支撑句,缺失即被审核 drop
- **zs / fs 双模式**:CLI `--mode zero-shot|few-shot` + `--output-dir` 分目录(`raw-zs` / `raw-fs`),不互相覆盖
- **审核是独立的二次 LLM 调用**(不是同一回答的 self-check),用不同 prompt 仅做证据/类型校验 — 这是降低幻觉的关键
- `reason_code` 枚举:`SUPPORTED` / `EVIDENCE_NOT_IN_CHUNK` / `SPAN_MISMATCH` / `SELF_LOOP` …,事后归因清晰

**设计动机**:把"LLM 编造"这件事用"另一个 LLM 监督"接住,所有判决留痕,**让幻觉变成可统计、可追责的事件**而不是黑箱。

---

## 第 8 页 · 核心模块 ④ — 实体归一化(三层级联)

```
所有候选实体
     │
     ▼
┌──────────────────────────────────────────────┐
│ ① 规则别名表 ALIAS_MAP(27 条)               │   零成本
│   TCP / UDP / HTTP / DNS / FTP / SMTP /       │   命中即合并
│   ICMP / IP / ARP / RARP / RTT / MTU / MSS /  │
│   QoS / OSI / NAT / DHCP / SNMP / RSVP /      │
│   MPLS / BGP / IGP / OSPF / RIP / …           │
└──────────────────────────────────────────────┘
     │ 未命中
     ▼
┌──────────────────────────────────────────────┐
│ ② Embedding + FAISS 近邻                     │   低成本批量
│   qwen3-embedding-8b (1024-dim, 走工厂)       │
│   IndexFlatIP + 向量归一 ≈ 余弦相似度          │
│   top_k = 50 召回                             │
│   candidate_threshold = 0.78  (初筛 / 扩召回)  │
│   similarity_threshold = 0.85 (终筛 / 提精度)  │
└──────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│ ③ LLM 复核 EntityMergeJudge                  │   高精度
│   候选对 + 双方原文上下文 → LLM 仲裁           │
│   决策:merge / keep_separate / unsure         │
└──────────────────────────────────────────────┘
     │
     ▼
Union-Find 传递闭包(_UnionFind 路径压缩)
A=B, B=C  ⇒  A=C
     │
     ▼
归一化实体节点  ──►  merged_triples.json
```

**要点**:
- **三层成本-精度阶梯**:规则零成本兜常见、Embedding 批量筛候选、LLM 只判最难的边缘对
- **双阈值设计**:0.78 召回放大 + 0.85 终筛保精度,把 LLM 调用量压在数量级最低的那一档
- FAISS `IndexFlatIP` + 单位化向量 = 余弦,O(n²) 在万级实体上仍可接受
- `_UnionFind` 处理传递闭包(路径压缩,均摊近 O(1)),保证"A 等于 B 等于 C 时三者必属同一簇"
- embedding 计算结果落盘缓存(`.embed_cache.npy` + `.meta.json`),重跑零额外开销

**设计动机**:**避免"全 LLM"的成本爆炸,也避免"全规则"的覆盖盲区**;让每个候选对都被它"配得上"的方法处理。

---

## 第 9 页 · 核心模块 ⑤ — 评估闭环(双层评估)

```
┌─────  data/eval/gold_triples.json  ────────────────┐
│  500 条人工审核三元组                                │
│  594 个实体  /  88 源文件  /  186 (file, chunk) 对   │
│                                                     │
│  关系分布:contains 304 / depends_on 127 /          │
│            applied_to  48 / compared_with 21        │
│  实体分布:concept 411 / protocol 342 /             │
│            mechanism 198 / metric 49                │
└─────────────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   kgevalraw                       kgeval
   (评 raw 抽取层)                (评合并后 KG)
   对比 zs vs fs                  对比 merged vs gold
        │                             │
        ▼                             ▼
   raw_results/comparison.md     evaluation_report.md
   六维指标(Triple / Entity /    P / R / F1
   Relation / 类型准确率)         + evidence 覆盖率
```

**要点**:
- **双层解耦**:raw 层定位"抽错没抽对",merged 层定位"合并合错了没",任一指标退化都能立刻定位环节
- **严格匹配**:`EntityKey` / `TripleKey` 用 `frozenset` 直接相等,不做模糊匹配 — 保证任何人重跑结果一致
- **gold 标注规范**:每条 evidence 必须是原文连续子串;500 条逐条核实,修正了 17 条
- **拒审留痕**:`_rejection_log.json` 记录拒绝原因,gold 生成方法学落在 `docs/reports/gold-generation-summary.md`

**设计动机**:**让质量讨论从"图好不好看"变成"P/R/F1 改善了多少"**,可比可证,具备工程意义。

---

## 第 10 页 · 系统集成与前端可视化

**端到端编排(单命令)**:
```
uv run qmr                            # 7 阶段一键编排
uv run kgeval     --config config.yaml   # merged 三元组 vs gold
uv run kgevalraw  --config config.yaml   # zs vs fs 抽取层对比
```

**前端栈**:
```
Next.js 16  (App Router, 默认 SSR)
    │
    ├─ /api/graph    (Route Handler, neo4j-driver v6)
    │     查询:Top-K 节点(按 frequency DESC)
    │           + 诱导子图(只取这些节点之间的边)
    │     可调:NEO4J_GRAPH_NODE_LIMIT / REL_LIMIT
    │
    └─ <GraphCanvas/>  (CSR, react-force-graph-2d)
          力导向布局 + 4 类实体着色
```

**要点**:
- **GraphCanvas 必须 CSR**(D3 依赖 DOM),其余路由保持 SSR
- **诱导子图查询**:默认 1000 节点 + 4000 边,保证连通性;`*_LIMIT=0` 切全量
- 节点颜色由 Neo4j 标签决定,与 schema 4 类实体一一对应
- 后端 → 前端只走一个 API 端点,接口面收敛

**设计动机**:可视化要"能用",不只是"好看" — **频率排序 + 诱导子图避免一打开就上万节点卡死**;一键命令避免手动串接 7 个 CLI 出错。

---

## 第 11 页 · 实验结果

**Zero-shot vs Few-shot 抽取层对比**(数据来源:`data/eval/raw_results/comparison.md`,严格匹配):

| 维度 | Zero-shot | Few-shot |
|------|-----------|----------|
| Triple F1 | **0.1387** | 0.1130 |
| Triple Precision | 0.1644 | 0.1335 |
| Triple Recall | 0.1200 | 0.0980 |
| Entity F1 | **0.3337** | 0.3136 |
| Entity Precision | 0.4000 | 0.3690 |
| Entity Recall | 0.2862 | 0.2727 |

**三条关键观察**:
1. **Zero-shot 整体优于 Few-shot** — 与直觉相反,推测 fs 示例过拟合特定表述,引入 bias
2. **Triple F1 远低于 Entity F1**(13.87% vs 33.37%) — 实体能找到但完整三元组难,**根因是表述不一**(`TCP` vs `传输控制协议`),正是归一化模块要解决的
3. **Precision 始终高于 Recall** — 模型"宁缺毋滥",对不确定的实体倾向不抽,这对"低噪声 KG"的目标是友好的

**归一化效果**:`merged_triples.json` 落到约 **2.36 万实体**,远低于原始抽取的累计实体数 — 三层级联实际把大量同义实体合并到同一节点。

**设计动机**:**严格匹配下的低 F1 不等于系统不可用** — 它揭示了"表述差异"这一可被归一化模块吸收的损失,而不是"模型抽错了"。

---

## 第 12 页 · 总结与展望

**完成的工作**:
1. **端到端 7 阶段 pipeline** — 单命令 `uv run qmr` 跑全程,所有阶段独立可重跑
2. **三层级联实体归一化** — 规则 / Embedding+FAISS / LLM 复核,成本可控、精度可控
3. **自建 500 条 gold + 双层评估** — raw 层与 merged 层解耦,严格匹配保可复现
4. **真实教材规模 KG + 力导向可视化** — 2.36 万实体落到 Neo4j,前端诱导子图浏览

**不足与下一步**:
1. Gold 单标注者 → 加多人标注 + Cohen's Kappa 一致性检验
2. 严格匹配偏严 → 引入"归一化感知"的评估匹配策略,把"TCP / 传输控制协议"判为同一
3. 仅覆盖计网领域 → 通过换 Prompt + Schema 适配其他学科
4. 公式与图表理解偏弱 → 引入 LaTeX OCR / 表格抽取等专门模块

**一句话总结**:把"PDF → 知识图谱"做成**可重跑、可量化、可扩展**的工程系统,而不是一次性 demo。

---
