# 金标三元组标注规范

## 判定标准

每条三元组从 5 个维度判定：

| 维度 | 正确条件 |
|------|----------|
| head | 实体名称准确反映原文所指 |
| head_type | 类型正确：protocol / concept / mechanism / metric |
| relation | 关系正确：contains / depends_on / compared_with / applied_to |
| tail | 实体名称准确反映原文所指 |
| evidence | 原文关键句足以支撑该关系成立 |

**correct = true**：上述 5 项全部正确。

**correct = false**：任意一项有误，同时填写对应的 `corrected_*` 字段。

## 实体类型判断细则

| 类型 | 判定标准 | 正例 | 负例 |
|------|---------|------|------|
| protocol | 标准协议名称或缩写 | TCP, HTTP, BGP, OSPF | "可靠传输" (概念), "三次握手" (机制) |
| concept | 抽象术语、理论概念 | 拥塞控制, 子网掩码, 路由表 | TCP (协议), RTT (指标) |
| mechanism | 具体算法、机制、流程 | 慢启动, CSMA/CD, 滑动窗口 | "拥塞控制" (概念, 太抽象) |
| metric | 可量化指标 | 吞吐量, 丢包率, 带宽, RTT | "性能" (概念, 不可量化) |

**边界处理**：
- "RTT" 既是 protocol 名称缩写也是 metric → 按上下文判断，教材中通常作为 metric
- "OSPF" 既是 protocol 也是 algorithm → 按 type 字段已有标注判断

## 关系类型判断细则

| 关系 | 含义 | 正例 |
|------|------|------|
| contains | A 包含/由 B 组成 | TCP contains 拥塞控制 |
| depends_on | A 依赖/需要 B | HTTP depends_on TCP |
| compared_with | A 与 B 对比 | TCP compared_with UDP |
| applied_to | A 应用于 B | 拥塞控制 applied_to TCP |

## 标注步骤

1. 读 evidence 字段，在原文上下文中理解该关系
2. 判断 head 和 tail 是否确实是所述实体
3. 判断 head_type、tail_type 是否正确
4. 判断 relation 方向是否正确（head_rel_tail 不能反过来）
5. 判断 evidence 是否有足够的支持力度
6. 如全部正确 → `correct: true`，其余 corrected 字段保持 null
7. 如有误 → `correct: false`，填写对应 `corrected_*` 字段，可选填 `notes` 说明理由

## 标注示例

### ✅ 正确示例

```json
{
  "id": 1,
  "head": "TCP",
  "head_type": "protocol",
  "relation": "compared_with",
  "tail": "UDP",
  "tail_type": "protocol",
  "evidence": "TCP 是面向连接的，UDP 是无连接的",
  "correct": true,
  "corrected_head": null,
  "corrected_head_type": null,
  "corrected_relation": null,
  "corrected_tail": null,
  "corrected_tail_type": null,
  "corrected_evidence": null,
  "notes": null
}
```

### ❌ 错误示例（含修正）

```json
{
  "id": 2,
  "head": "TCP",
  "head_type": "concept",
  "relation": "contains",
  "tail": "IP",
  "tail_type": "protocol",
  "evidence": "TCP 提供可靠传输",
  "correct": false,
  "corrected_head": "TCP",
  "corrected_head_type": "protocol",
  "corrected_relation": "depends_on",
  "corrected_tail": "IP",
  "corrected_tail_type": "protocol",
  "corrected_evidence": "TCP 运行在 IP 之上，依赖 IP 提供主机间通信",
  "notes": "head_type 应为 protocol 非 concept；relation 应为 depends_on 非 contains（TCP 不包含 IP，而是依赖 IP）；evidence 未体现关系"
}
```

## 时间预估

- 单条标注：1.5 ~ 4 分钟（视 ambiguity 程度）
- 60 条总量：约 2 ~ 4 小时

## 注意事项

- 不确定时标注 false + 最佳修正 + notes 解释，不要猜测
- 保持实体名称与原文一致，不要自行改写
- evidence 不足但 relation 方向正确 → 标注 false + 补充更好的 evidence
- 建议分批标注：每 15 条休息一次，保持判断一致性
