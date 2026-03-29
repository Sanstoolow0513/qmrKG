"""课程领域实体/关系抽取的系统提示：零样本与少样本对照（Prompt Engineering 实验用）。"""

from __future__ import annotations

from enum import Enum
from typing import Literal

_JSON_SPEC = """\
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
"""

_RULES = """\
## 规则
1. 只从给定文本中抽取，不要编造
2. entity.name 使用文本中出现的原始名称（与原文一致）
3. 每个 triple 必须附带 evidence（原文中支持该关系的关键句）
4. 如果文本中没有可抽取的实体或关系，返回空列表"""

_TYPE_GUIDE = """\
## 实体类型（4类）
- protocol: 协议名称
- concept: 概念术语
- mechanism: 机制算法
- metric: 性能指标

## 关系类型（4类）
- contains: 包含关系（A 包含 B）
- depends_on: 依赖关系（A 依赖 B）
- compared_with: 对比关系（A 与 B 对比）
- applied_to: 应用关系（A 应用于 B）
"""

# 零样本：仅类型说明 + JSON 规范 + 规则，不包含完整「文本→JSON」示例
ZERO_SHOT_EXTRACTION_PROMPT = f"""\
你是一个面向课程教材的知识图谱构建助手。从用户给出的文本中识别命名实体并抽取关系。

{_TYPE_GUIDE}

{_JSON_SPEC}

{_RULES}
"""

# 少样本：在零样本基础上增加若干完整输入-输出示例，便于模型对齐标注粒度与关系选择
_FEW_SHOT_EXAMPLES = """\
## 标注示例（下列仅为格式演示；待处理文本由用户在对话中提供）

### 示例 1
文本：「UDP 是无连接的运输层协议，常与 TCP 对比；其实时应用优于 TCP。」
输出：
{"entities": [{"name": "UDP", "type": "protocol", "description": "无连接运输层协议"}, {"name": "TCP", "type": "protocol", "description": "运输层协议"}], "triples": [{"head": "UDP", "relation": "compared_with", "tail": "TCP", "evidence": "常与 TCP 对比"}]}

### 示例 2
文本：「慢启动是 TCP 拥塞控制中的机制，用于在连接初期探测可用带宽。」
输出：
{"entities": [{"name": "慢启动", "type": "mechanism", "description": "TCP 拥塞控制机制"}, {"name": "TCP", "type": "protocol", "description": ""}, {"name": "拥塞控制", "type": "concept", "description": ""}], "triples": [{"head": "慢启动", "relation": "depends_on", "tail": "TCP", "evidence": "TCP 拥塞控制中的机制"}]}

### 示例 3（性能指标）
文本：「该链路的 RTT 为 40ms，端到端吞吐量受拥塞窗口限制。」
输出：
{"entities": [{"name": "RTT", "type": "metric", "description": "往返时延"}, {"name": "吞吐量", "type": "metric", "description": ""}, {"name": "拥塞窗口", "type": "concept", "description": ""}], "triples": [{"head": "吞吐量", "relation": "depends_on", "tail": "拥塞窗口", "evidence": "端到端吞吐量受拥塞窗口限制"}]}
"""

FEW_SHOT_EXTRACTION_PROMPT = (
    "你是一个面向课程教材的知识图谱构建助手。从用户给出的文本中识别命名实体并抽取关系。\n\n"
    + _TYPE_GUIDE
    + "\n"
    + _FEW_SHOT_EXAMPLES
    + "\n"
    + _JSON_SPEC
    + "\n"
    + _RULES
)


class ExtractionPromptKind(str, Enum):
    """与 CLI / KGExtractor 共用的提示策略枚举。"""

    LEGACY = "legacy"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


def get_extraction_system_prompt(
    kind: Literal["legacy", "zero_shot", "few_shot"] | ExtractionPromptKind | str,
    *,
    legacy_prompt: str,
) -> str:
    """返回对应策略下的 system prompt。legacy 使用调用方传入的原有提示（保持行为不变）。"""
    key = kind.value if isinstance(kind, ExtractionPromptKind) else str(kind)
    if key == ExtractionPromptKind.ZERO_SHOT.value:
        return ZERO_SHOT_EXTRACTION_PROMPT
    if key == ExtractionPromptKind.FEW_SHOT.value:
        return FEW_SHOT_EXTRACTION_PROMPT
    if key == ExtractionPromptKind.LEGACY.value:
        return legacy_prompt
    raise ValueError(f"unknown extraction prompt kind: {kind!r}")
