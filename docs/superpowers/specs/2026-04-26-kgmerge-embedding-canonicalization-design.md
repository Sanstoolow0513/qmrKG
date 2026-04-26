# kgmerge 引入 embedding 同义实体合并设计

## 1. 背景与目标

`kgmerge` 当前依赖 `ALIAS_MAP`（28 条手工同义词）+ 后缀剥离（`协议|算法|机制|方法|技术|方式`）做实体名归一，再按字符串相等分组。该机制的问题：

- 维护成本高：换一本书/换一个学科就要重写 alias map。
- 召回不足：同义但表述不同的实体（如 "三次握手" / "TCP 三次握手" / "三路握手"，"客户机/服务器" / "客户端服务器模式"）合不进同一个 canonical。
- 现有 `merged_triples.json` 实测有 23 600 个实体、44 526 条三元组，主因之一是同义簇没收拢。

本设计目标：在 `kgmerge` 阶段叠加一层 embedding 驱动的同义实体合并，让规则没覆盖的尾部实体也能正确归一。

## 2. 约束与非目标

**约束**：

- LLM 调用一律走 `llm_factory`，禁止直接 `import openai`。
- 配置遵循 `task → llm_profile → provider/request/rate_limit` 三层结构。
- `merged_triples.json` 的 schema **不变**（仅同义簇被合并掉，实体数 / 三元组数下降）。
- 下游 `kgneo4j` 与前端无须改动。

**非目标**：

- 不做关系归一（`relation` 已是 `Literal` 4 类枚举）。
- 不做三元组级 embedding 去重（head/tail 已 canonical 化后精确去重已足够）。
- 不引入 ANN / faiss（数据规模直接 numpy 硬算可行）。
- 不做跨阶段 embedding 缓存外的高级缓存层（YAGNI）。

## 3. 方案选择

候选三个，已确定采用方案 A：

- **方案 A（采用）**：分桶（按 `entity.type`）+ 桶内 pairwise cosine + 阈值并查集。
- 方案 B（未采用）：HDBSCAN 聚类，免阈值但中文短词稳定性差。
- 方案 C（未采用）：embedding 召回 + LLM 仲裁。准确率最高，但复杂度与调用成本均高，作为后续可选增强。

选择理由：

- 数据规模适中（最大桶 ~6k 实体），pairwise 直接算 → numpy 即可。
- 零额外 LLM 调用（仅 embedding 一类调用），可重跑、确定性、便于调阈值。
- 与现有 `_merge_entities` / `_merge_triples` 的字符串相等去重路径完全兼容，只需扩一层归一映射。

## 4. 总体数据流

```
raw/*.json
   │
   ▼
load entities / triples
   │
   ▼  (现有 ALIAS_MAP + 后缀剥离作为 strong prior)
normalize_entity_name(name) → canonical_v1
   │
   ▼  (新增：embedding 同义簇合并)
EmbeddingCanonicalizer:
  1. 按 type 分桶（protocol / concept / mechanism / metric）
  2. 桶内对所有 canonical_v1 调用 entity_embed → 1024 维向量
  3. 桶内计算 pairwise cosine 矩阵
  4. cosine ≥ τ 的对加入并查集
  5. 每簇选 canonical_v2（频率最高，并列取字符最短）
  6. 产出 mapping: canonical_v1 → canonical_v2
   │
   ▼
对所有 entity / triple.head / triple.tail 套用 mapping
   │
   ▼  (现有逻辑不变)
_merge_entities → frequency 累加 / description 取首个非空
_merge_triples  → (h, r, t) 字符串相等去重，frequency 累加，evidence 累计
   │
   ▼
merged_triples.json （schema 不变）
```

## 5. 配置变更（config.yaml）

### 5.1 新增 embedding profile

```yaml
llm:
  profiles:
    embedding_qwen3_8b:
      provider:
        name: ppio
        base_url: "https://api.ppinfra.com/openai"
        model: "qwen/qwen3-embedding-8b"
        modality: "embedding"            # 新 modality
        supports_thinking: false
      request:
        timeout_seconds: 60.0
        max_retries: 3
        encoding_format: "float"
        dimensions: 1024                 # 若 PPIO 不支持此参数则去掉，使用原生 4096
      rate_limit:
        rpm: 100
        max_concurrency: 4
```

### 5.2 新增 LLM 任务段

```yaml
entity_embed:
  llm_profile: embedding_qwen3_8b
```

`entity_embed` 任务无 prompt 字段；`llm_config.py` 中 `modality="embedding"` 时 prompt 不再为必需。

### 5.3 `run.kg_merge` 新增 embedding 子段

```yaml
run:
  kg_merge:
    input_dir: "data/triples/raw"
    output: "data/triples/merged/merged_triples.json"
    embedding:
      enabled: true
      task_name: "entity_embed"
      encode_fields: ["type", "name", "description"]   # 拼接顺序
      similarity_threshold: 0.85
      bucket_by_type: true
      batch_size: 1024                                  # PPIO 数组 input 上限 2048，留余量
      cache_path: "data/triples/merged/.embed_cache.json"
```

`embedding.enabled: false` 时完全走旧逻辑，向后兼容。

## 6. 代码变更

### 6.1 `src/qmrkg/llm_types.py`

- `LLMModality` 扩展为 `Literal["text", "multimodal", "embedding"]`。
- 新增数据类：

```python
@dataclass(slots=True)
class LLMEmbeddingResponse:
    vectors: list[list[float]]
    model: str | None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float = 0.0
    processed_at: str = ""
```

### 6.2 `src/qmrkg/llm_config.py`

- `_default_modality(task_name)`：当 task 名属于 embedding 类（如 `entity_embed`）或 profile 显式指定 `modality: "embedding"` 时返回 `"embedding"`。
- `TaskLLMSettings`：新增字段 `encoding_format: str = "float"`、`embedding_dimensions: int | None = None`。
- 校验：modality 取值集合扩展为 `{"text", "multimodal", "embedding"}`。
- 当 `modality == "embedding"` 时不要求 `prompt`，但仍要求 `model`、`base_url`。

### 6.3 `src/qmrkg/llm_factory.py`

- `TaskLLMRunner` 新增方法：

```python
def run_embeddings(self, inputs: list[str]) -> LLMEmbeddingResponse:
    """call client.embeddings.create with rate limiting and retry."""
```

  - 强制要求 `self.settings.modality == "embedding"`，否则 `ValueError`。
  - 走与 `run_messages` 同一套重试 + RPM 限流。
  - 请求 kwargs 由 `_embedding_request_kwargs()` 构造：仅在 `embedding_dimensions` 非空时携带 `dimensions`。

- 新增 `EmbeddingTaskProcessor`：

```python
class EmbeddingTaskProcessor:
    def __init__(self, task_name: str, config_path: Path | None = None, client=None): ...
    def embed(self, inputs: list[str], batch_size: int = 1024) -> list[list[float]]:
        """Auto-batch over PPIO array input limit; returns vectors aligned with inputs."""
```

### 6.4 `src/qmrkg/kg_merger.py`

新增 `EmbeddingCanonicalizer` 类，在 `KGMerger.merge_directory` 中按配置启用。

```python
class EmbeddingCanonicalizer:
    def __init__(
        self,
        *,
        task_name: str,
        encode_fields: list[str],
        similarity_threshold: float,
        bucket_by_type: bool,
        batch_size: int,
        cache_path: Path | None,
        config_path: Path | None,
    ): ...

    def build_canonical_map(
        self, entities: list[Entity]
    ) -> dict[str, str]:
        """
        输入：经过 normalize_entity_name 之后的 Entity 列表
        输出：name (canonical_v1) -> canonical_v2 的映射
        步骤：
          1. 去重得到唯一 (name, type, description) 列表
          2. 按 type 分桶（若 bucket_by_type=True）
          3. 桶内调 EmbeddingTaskProcessor.embed(...)
          4. numpy 计算 cosine 矩阵；遍历上三角，cosine ≥ τ 则 union
          5. 每簇按 frequency desc, len(name) asc 选 canonical
          6. 写入 cache_path（按 (model_id, encode_text) hash）
        """
```

`KGMerger.merge_directory` 修改点：

- 接受 `embedding_config: dict | None`（来自 `run.kg_merge.embedding`）。
- 当 `embedding_config and embedding_config.get("enabled")`：
  - 调 `_merge_entities` 拿到 v1 list
  - 调 `EmbeddingCanonicalizer.build_canonical_map(v1)` 拿 mapping
  - 用 mapping 把 entity.name / triple.head / triple.tail 改写一遍
  - 再走一次 `_merge_entities` / `_merge_triples` 做最终去重
- 否则走旧路径，行为完全不变。

### 6.5 `src/qmrkg/cli_kg_merge.py`

- 新增 `--no-embedding` 旗标（强制关闭 embedding 步骤），默认从 config 读。
- 新增 `--embedding-task` 覆盖 `entity_embed`（便于实验）。
- `--similarity-threshold` 覆盖 τ（便于调参）。

## 7. 算法细节

### 7.1 编码文本

```
text = " | ".join(
    field_value for field in encode_fields
    if (field_value := getattr(entity, field, ""))
)
# 示例: "protocol | TCP | 传输控制协议"
```

空字段跳过；仅 `name` 字段为空时跳过该实体。

### 7.2 桶内 pairwise cosine

- 把桶内 N 个 1024 维向量摆成 `M ∈ R^{N×1024}`，先 L2 归一化（`M / np.linalg.norm(M, axis=1, keepdims=True)`）。
- `S = M @ M.T`，得到 `N×N` cosine 矩阵。
- 遍历上三角 `i < j` 且 `S[i,j] >= τ`：`union(i, j)`。
- 内存：N=6000、float32 → 144 MB，可承受。若桶 > 8000 退化为分块计算（实现可后置）。

### 7.3 canonical 选取

簇内按以下优先级排序，取第一项：

1. `frequency` 降序（频率高的更可能是规范名）
2. `len(name)` 升序（短名通常是缩写/标准名，如 "TCP" 优于 "TCP 协议"）
3. `name` 字典序升序（确定性 tiebreak）

### 7.4 缓存

- `cache_path` 是 JSON 文件：`{ "<model>::<sha1(text)>": [vec...] }`。
- 重跑时先读 cache，cache miss 才发请求。删除 cache 文件即可强制重算。

## 8. 测试

新增 `tests/test_kg_merger_embedding.py`：

- `FakeEmbeddingProcessor`：mock 掉 `EmbeddingTaskProcessor.embed`，按预设把 `"TCP"` / `"传输控制协议"` 编成同向量、`"UDP"` 编成正交向量。
- 测试用例：
  - `test_canonicalizer_merges_synonyms`：两条实体 cosine 高 → 合并为同 canonical，frequency 累加。
  - `test_canonicalizer_respects_type_bucket`：`"TCP" (protocol)` vs `"TCP" (concept)` 即便向量一致也不跨桶合并。
  - `test_canonicalizer_threshold`：阈值 0.95 时不合，0.7 时合。
  - `test_canonical_pick_rule`：频率高者胜出；并列时短者胜出。
  - `test_merger_with_embedding_disabled`：`embedding.enabled=false` 时输出与旧逻辑逐字一致（回归保护）。
  - `test_merger_writes_cache_and_reuses`：第二次跑不再调用 `embed`。

`tests/test_llm_factory.py` 扩展：

- `test_run_embeddings_validates_modality`：text task 调 `run_embeddings` → `ValueError`。
- `test_embedding_task_processor_batches`：超过 batch_size 自动切批。
- `test_run_embeddings_retries_on_transient`：429/5xx 进入指数退避。

`tests/test_llm_config.py`（如不存在则新增）：

- `test_embedding_modality_no_prompt_required`。
- `test_embedding_dimensions_field_passed_through`。

## 9. 向后兼容与回退

- 默认 `embedding.enabled` 由开发者在 `config.yaml` 中显式打开（建议先 `false`，验证 `llm_factory` 改动不破坏现有路径，再切 `true`）。
- 旧 `ALIAS_MAP` + 后缀剥离继续作为前置 strong prior，不删除。
- 回退方法：`embedding.enabled: false` 即可恢复旧行为，无需 revert 代码。

## 10. 待澄清项（实施前需对账）

- PPIO 的 `qwen/qwen3-embedding-8b` 是否支持 `dimensions` 参数（MRL 截断）。若不支持，profile 去掉该字段，使用原生 4096 维（多耗 4× embedding 内存，但仍在可接受范围）。
- PPIO embedding 端点的实际 RPM 上限。
- batch input 数组上限是否对该模型仍是 2048（bge-m3 文档明确，qwen3-embedding-8b 待验证）。

## 11. 工作量估算

- `llm_types.py` / `llm_config.py` / `llm_factory.py`：~150 行新增 + 局部修改。
- `kg_merger.py` / `cli_kg_merge.py`：~120 行新增。
- 测试：~250 行。
- 配置：YAML 新增 ~15 行。
- 总计：1 个开发日内可完成实现 + 测试。
