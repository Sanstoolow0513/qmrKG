# 知识图谱抽取管线设计

## 概述

在现有 PDF → PNG → Markdown → Chunks 管线的基础上，新增知识抽取管线，实现从课程教材文本中自动抽取命名实体与关系，生成三元组并导入 Neo4j 图数据库。

**技术选型：**
- LLM 平台：PPIO
- 抽取模型：DeepSeek V3.2（纯文本）
- 抽取策略：联合抽取 + 后验证（一次 LLM 调用同时识别实体和关系，再做去重融合）
- 存储与可视化：Neo4j 图数据库 + 浏览器查询

**实体类型（4 类）：**

| 英文标识 | 中文名 | 示例 |
|---|---|---|
| `protocol` | 协议名称 | TCP、HTTP、DNS、ARP |
| `concept` | 概念术语 | 拥塞控制、子网掩码、路由表 |
| `mechanism` | 机制算法 | 三次握手、慢启动算法、CSMA/CD |
| `metric` | 性能指标 | 吞吐量、RTT、丢包率、带宽 |

**关系类型（4 类）：**

| 英文标识 | 中文名 | 示例 |
|---|---|---|
| `contains` | 包含关系 | 传输层 包含 TCP |
| `depends_on` | 依赖关系 | HTTP 依赖 TCP |
| `compared_with` | 对比关系 | TCP 对比 UDP |
| `applied_to` | 应用关系 | 慢启动算法 应用于 拥塞控制 |

## 数据流

```
data/chunks/*.json                   (已有，Markdown 切块输出)
    ↓  kg_extractor（LLM 联合抽取）
data/triples/raw/<stem>_chunk_<N>.json   (原始三元组，每 chunk 一个文件)
    ↓  kg_merger（去重 + 归一化 + 过滤）
data/triples/merged/merged_triples.json  (融合后三元组)
    ↓  kg_neo4j（导入）
Neo4j 数据库                              (浏览器可视化查询)
```

## 新增模块

### 1. `src/qmrkg/kg_extractor.py` — 联合抽取

读取 chunks JSON，对每个 chunk 调用 LLM 联合抽取实体和关系，输出原始三元组。

**Prompt 设计：**

```text
你是一个计算机网络课程知识图谱构建专家。

任务：从给定的教材文本中，识别命名实体并抽取实体间的关系，生成知识三元组。

## 实体类型（4类）
- protocol: 协议名称（如 TCP、HTTP、DNS、ARP）
- concept: 概念术语（如 拥塞控制、子网掩码、路由表）
- mechanism: 机制算法（如 三次握手、慢启动算法、CSMA/CD）
- metric: 性能指标（如 吞吐量、RTT、丢包率、带宽）

## 关系类型（4类）
- contains: 包含关系（A 包含 B）
- depends_on: 依赖关系（A 依赖 B）
- compared_with: 对比关系（A 与 B 对比）
- applied_to: 应用关系（A 应用于 B）

## 输出格式
严格输出 JSON，不要输出任何其他内容：
{
  "entities": [
    {"name": "TCP", "type": "protocol", "description": "传输控制协议"}
  ],
  "triples": [
    {"head": "TCP", "relation": "compared_with", "tail": "UDP", "evidence": "原文依据"}
  ]
}

## 规则
1. 只从给定文本中抽取，不要编造
2. entity.name 使用文本中出现的原始名称
3. 每个 triple 必须附带 evidence（原文中支持该关系的关键句）
4. 如果文本中没有可抽取的实体或关系，返回空列表
```

**输出数据格式：**

每个 chunk 一个 JSON 文件，存放于 `data/triples/raw/`：

```json
{
  "chunk_index": 0,
  "source_file": "xxx.md",
  "titles": ["第5章 运输层", "5.3 TCP协议"],
  "entities": [
    {"name": "TCP", "type": "protocol", "description": "传输控制协议"}
  ],
  "triples": [
    {
      "head": "TCP",
      "relation": "compared_with",
      "tail": "UDP",
      "evidence": "TCP是面向连接的，而UDP是无连接的"
    }
  ]
}
```

**实现要点：**
- 复用 `TextTaskProcessor` / `TaskLLMRunner`，config.yaml 新增 `extract` 任务段
- 并发处理多个 chunk（利用现有 `rate_limit` 和 `max_concurrency`）
- LLM 返回的 JSON 做解析校验（类型不在预定义范围内则丢弃）
- 每个 chunk 结果单独存文件，支持断点续跑（已存在的文件跳过）

### 2. `src/qmrkg/kg_merger.py` — 去重、归一化与融合

读取 `data/triples/raw/` 下所有原始三元组文件，合并为干净的知识图谱数据集。

**步骤 1：实体归一化**

同一实体在不同 chunk 中可能有不同表述（如"TCP协议"与"TCP"），需要归一化：

- 去除常见后缀词（"协议""算法""机制"）得到核心名
- 中英文别名映射表（手动维护小字典，如 `{"传输控制协议": "TCP", "往返时延": "RTT"}`）
- 模糊匹配兜底：编辑距离或 Jaccard 相似度 > 0.85 才合并

**步骤 2：三元组去重**

相同 `(head, relation, tail)` 只保留一条，累计出现次数和所有 evidence：

```json
{
  "head": "TCP",
  "relation": "compared_with",
  "tail": "UDP",
  "frequency": 3,
  "evidences": [
    "TCP是面向连接的，而UDP是无连接的",
    "TCP提供可靠交付，UDP不保证",
    "TCP有拥塞控制机制，UDP没有"
  ]
}
```

`frequency` 作为天然的置信度指标。

**步骤 3：低质量过滤**

移除以下情况：
- `head == tail`（自环）
- `head` 或 `tail` 不在已知实体列表中
- 实体名过短（< 2 字符）或过长（> 30 字符）
- 实体类型不在 4 种预定义类型中

**输出格式** (`data/triples/merged/merged_triples.json`)：

```json
{
  "entities": [
    {"name": "TCP", "type": "protocol", "description": "传输控制协议", "frequency": 15}
  ],
  "triples": [
    {
      "head": "TCP",
      "head_type": "protocol",
      "relation": "compared_with",
      "tail": "UDP",
      "tail_type": "protocol",
      "frequency": 3,
      "evidences": ["..."]
    }
  ],
  "stats": {
    "total_entities": 120,
    "total_triples": 350,
    "entities_by_type": {"protocol": 40, "concept": 45, "mechanism": 20, "metric": 15},
    "triples_by_relation": {"contains": 150, "depends_on": 80, "compared_with": 60, "applied_to": 60}
  }
}
```

### 3. `src/qmrkg/kg_neo4j.py` — Neo4j 导入

读取融合后的 `merged_triples.json`，导入 Neo4j 图数据库。

**图模型：**

节点标签（按实体类型）：

| 标签 | 实体类型 |
|---|---|
| `Protocol` | protocol |
| `Concept` | concept |
| `Mechanism` | mechanism |
| `Metric` | metric |

节点属性：`name`（主键）、`description`、`frequency`

关系类型：

| Neo4j 关系名 | 含义 |
|---|---|
| `CONTAINS` | 包含 |
| `DEPENDS_ON` | 依赖 |
| `COMPARED_WITH` | 对比 |
| `APPLIED_TO` | 应用 |

关系属性：`frequency`、`evidences`（JSON 字符串）

**导入逻辑：**
1. 创建节点：`MERGE` 按 name 去重
2. 创建关系：`MATCH` 头尾节点，`MERGE` 关系
3. 使用 `MERGE` 保证幂等性，可重复导入

**Neo4j 连接配置（`.env`）：**

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 4. CLI 入口

**`src/qmrkg/cli_kg_extract.py`** — 抽取 CLI

```bash
kgextract                                    # 处理 data/chunks/ 下所有文件
kgextract --input data/chunks/xxx.json       # 处理单个 chunk 文件
kgextract --output-dir data/triples/raw      # 指定输出目录
kgextract --skip-existing                    # 跳过已存在的输出文件
```

**`src/qmrkg/cli_kg_neo4j.py`** — Neo4j 导入 CLI

```bash
kgneo4j --import data/triples/merged/merged_triples.json
kgneo4j --import ... --uri bolt://localhost:7687 --user neo4j --password xxx
kgneo4j --import ... --clear                 # 清空后重新导入
kgneo4j --stats                              # 打印统计
```

## config.yaml 变更

新增 `extract` 任务段（替代原有的 `ner` 和 `re`）：

```yaml
extract:
  provider:
    name: ppio
    base_url: "https://api.ppinfra.com/openai"
    model: "deepseek/deepseek-v3-0324"
    modality: "text"
    supports_thinking: false
  prompts:
    default: |
      （完整 prompt 见上文）
  request:
    timeout_seconds: 60.0
    max_retries: 3
    thinking:
      enabled: false
  rate_limit:
    rpm: 30
    max_concurrency: 4
```

## pyproject.toml 变更

新增依赖：`neo4j>=5.0.0`

新增 CLI 入口：
```toml
[project.scripts]
kgextract = "qmrkg.cli_kg_extract:main"
kgmerge = "qmrkg.cli_kg_merge:main"
kgneo4j = "qmrkg.cli_kg_neo4j:main"
```

## 示例查询（Neo4j Browser）

```cypher
-- 查看所有协议及其关系
MATCH (p:Protocol)-[r]->(n) RETURN p, r, n

-- 查看 TCP 的所有关联知识
MATCH (n {name: "TCP"})-[r]-(m) RETURN n, r, m

-- 包含关系最多的实体
MATCH (n)-[r:CONTAINS]->(m) RETURN n.name, count(r) AS cnt ORDER BY cnt DESC LIMIT 10
```
