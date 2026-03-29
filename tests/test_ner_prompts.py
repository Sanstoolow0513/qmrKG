from qmrkg.kg_extractor import EXTRACT_PROMPT
from qmrkg.ner_prompts import (
    FEW_SHOT_EXTRACTION_PROMPT,
    ZERO_SHOT_EXTRACTION_PROMPT,
    ExtractionPromptKind,
    get_extraction_system_prompt,
)


def test_zero_shot_has_no_few_shot_example_section():
    assert "### 示例" not in ZERO_SHOT_EXTRACTION_PROMPT
    assert "## 标注示例" not in ZERO_SHOT_EXTRACTION_PROMPT
    assert "输出格式" in ZERO_SHOT_EXTRACTION_PROMPT


def test_few_shot_contains_worked_examples():
    assert "## 标注示例" in FEW_SHOT_EXTRACTION_PROMPT
    assert "UDP" in FEW_SHOT_EXTRACTION_PROMPT
    assert "慢启动" in FEW_SHOT_EXTRACTION_PROMPT


def test_get_extraction_system_prompt_legacy_uses_passed_string():
    sentinel = "LEGACY_SENTINEL_XYZ"
    assert get_extraction_system_prompt("legacy", legacy_prompt=sentinel) == sentinel
    assert (
        get_extraction_system_prompt(ExtractionPromptKind.LEGACY, legacy_prompt=sentinel) == sentinel
    )


def test_get_extraction_system_prompt_zero_and_few_differ_from_legacy():
    z = get_extraction_system_prompt("zero_shot", legacy_prompt=EXTRACT_PROMPT)
    f = get_extraction_system_prompt("few_shot", legacy_prompt=EXTRACT_PROMPT)
    assert z != EXTRACT_PROMPT
    assert f != EXTRACT_PROMPT
    assert len(f) > len(z)
