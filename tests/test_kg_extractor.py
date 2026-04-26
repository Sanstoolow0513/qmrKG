from pathlib import Path

from qmrkg.kg_extractor import EXTRACT_PROMPT, KGExtractor
from qmrkg.kg_schema import Triple
from qmrkg.llm_types import LLMResponse


class _RecordingRunner:
    """Minimal runner that records the last run_text system_prompt (Task2 contract)."""

    def __init__(self):
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def run_text(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        self.last_user_prompt = prompt
        self.last_system_prompt = system_prompt
        return LLMResponse(
            text='{"entities": [], "triples": []}',
            processed_at="2026-01-01T00:00:00Z",
            duration_seconds=0.0,
        )


def _write_extract_config(tmp_path: Path, prompts_block: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
llm:
  profiles:
    extract_profile:
      provider:
        name: ppio
        base_url: https://api.ppio.com/openai
        model: deepseek/deepseek-v3.2
        modality: text
        supports_thinking: false
      request:
        timeout_seconds: 60.0
        max_retries: 1
        thinking:
          enabled: false
      rate_limit:
        rpm: 50
        max_concurrency: 4
extract:
  llm_profile: extract_profile
  prompts:
{prompts_block}
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_resolve_prompt_prefers_mode_specific_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = _write_extract_config(
        tmp_path,
        '    default: "DEFAULT_SYSTEM"\n    zero_shot: "ZERO_SYSTEM"\n    few_shot: "FEW_PROMPT"',
    )
    ex = KGExtractor(config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == "FEW_PROMPT"


def test_resolve_prompt_prefers_zero_shot(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = _write_extract_config(
        tmp_path,
        '    default: "DEFAULT_SYSTEM"\n    zs: "ZERO_PROMPT"',
    )
    ex = KGExtractor(config_path=config_path, mode="zero-shot")
    assert ex.resolve_prompt() == "ZERO_PROMPT"


def test_resolve_prompt_falls_back_to_default_when_mode_prompt_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = _write_extract_config(
        tmp_path,
        '    default: "DEFAULT_SYSTEM"\n    zero_shot: "ZERO_PROMPT"',
    )
    ex = KGExtractor(config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == "DEFAULT_SYSTEM"


def test_resolve_prompt_falls_back_to_builtin_when_default_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    config_path = _write_extract_config(
        tmp_path,
        '    zero_shot: "ZERO_PROMPT"',
    )
    ex = KGExtractor(config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == EXTRACT_PROMPT


def test_resolve_prompt_uses_discovered_config_when_config_path_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    _write_extract_config(
        tmp_path,
        '    default: "DEFAULT_SYSTEM"\n    few_shot: "FEW_FROM_DISCOVERED_CONFIG"',
    )
    monkeypatch.chdir(tmp_path)
    ex = KGExtractor(config_path=None, mode="few-shot")
    assert ex.resolve_prompt() == "FEW_FROM_DISCOVERED_CONFIG"


def test_resolve_prompt_discovers_repo_config_when_cwd_has_no_config(tmp_path, monkeypatch):
    """With no local config, fall back to the repository config.yaml (real discovery)."""
    monkeypatch.chdir(tmp_path)
    ex = KGExtractor(runner=_RecordingRunner(), config_path=None, mode="few-shot")
    prompt = ex.resolve_prompt()
    assert prompt != EXTRACT_PROMPT
    assert "## 示例" in prompt


def test_ancestor_cwd_config_does_not_hijack_repo_few_shot(tmp_path, monkeypatch):
    """A config.yaml only in a parent of cwd must not be used (no cwd.parent walk)."""
    outer = tmp_path / "with_decoy"
    work = outer / "nested_work"
    work.mkdir(parents=True)
    (outer / "config.yaml").write_text(
        """
extract:
  prompts:
    few_shot: "DECOY_FROM_PARENT_CWD"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)
    ex = KGExtractor(runner=_RecordingRunner(), config_path=None, mode="few-shot")
    prompt = ex.resolve_prompt()
    assert "DECOY_FROM_PARENT_CWD" not in prompt
    assert "## 示例" in prompt


def test_extract_from_chunk_passes_resolved_system_prompt_to_run_text(tmp_path, monkeypatch):
    """extract_from_chunk must use run_text(..., system_prompt=<resolved for mode>), not a stale or wrong prompt."""
    config_path = _write_extract_config(
        tmp_path,
        '    default: "DEFAULT_SYSTEM"\n    few_shot: "FEW_SHOT_VIA_EXTRACT"',
    )
    runner = _RecordingRunner()
    ex = KGExtractor(runner=runner, config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == "FEW_SHOT_VIA_EXTRACT"
    ex.extract_from_chunk(
        {
            "chunk_index": 0,
            "source_file": "s.md",
            "titles": [],
            "content": "some text for extraction",
        }
    )
    assert runner.last_system_prompt == "FEW_SHOT_VIA_EXTRACT"
    assert runner.last_user_prompt == "some text for extraction"


def test_extract_from_chunk_system_prompt_falls_back_to_builtin_like_resolve(tmp_path, monkeypatch):
    config_path = _write_extract_config(
        tmp_path,
        '    zero_shot: "ONLY_ZERO"',
    )
    runner = _RecordingRunner()
    ex = KGExtractor(runner=runner, config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == EXTRACT_PROMPT
    ex.extract_from_chunk(
        {
            "chunk_index": 0,
            "source_file": "s.md",
            "titles": [],
            "content": "x",
        }
    )
    assert runner.last_system_prompt == EXTRACT_PROMPT


def test_parse_json_response_plain():
    text = '{"entities": [{"name": "TCP", "type": "protocol", "description": "x"}], "triples": []}'
    result = KGExtractor._parse_json_response(text)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "TCP"


def test_parse_json_response_fenced():
    text = '```json\n{"entities": [], "triples": []}\n```'
    result = KGExtractor._parse_json_response(text)
    assert result == {"entities": [], "triples": []}


def test_parse_json_response_fenced_no_lang():
    text = '```\n{"entities": [{"name": "HTTP", "type": "protocol"}], "triples": []}\n```'
    result = KGExtractor._parse_json_response(text)
    assert len(result["entities"]) == 1


def test_parse_json_response_invalid():
    result = KGExtractor._parse_json_response("not json at all")
    assert result == {"entities": [], "triples": []}


def test_parse_json_response_with_surrounding_text():
    text = 'Here is the result:\n```json\n{"entities": [], "triples": []}\n```\nDone.'
    result = KGExtractor._parse_json_response(text)
    assert result == {"entities": [], "triples": []}


def test_parse_entities_valid():
    raw = [
        {"name": "TCP", "type": "protocol", "description": "传输控制协议"},
        {"name": "拥塞控制", "type": "concept", "description": ""},
    ]
    entities = KGExtractor._parse_entities(raw)
    assert len(entities) == 2


def test_parse_entities_filters_invalid():
    raw = [
        {"name": "TCP", "type": "protocol", "description": "x"},
        {"name": "X", "type": "protocol", "description": "too short"},
        {"name": "TCP", "type": "unknown", "description": "bad type"},
        "not a dict",
    ]
    entities = KGExtractor._parse_entities(raw)
    assert len(entities) == 1
    assert entities[0].name == "TCP"


def test_parse_triples_valid():
    raw = [
        {"head": "TCP", "relation": "compared_with", "tail": "UDP", "evidence": "x"},
        {"head": "HTTP", "relation": "depends_on", "tail": "TCP", "evidence": "y"},
    ]
    triples = KGExtractor._parse_triples(raw)
    assert len(triples) == 2


def test_parse_triples_filters_invalid():
    raw = [
        {"head": "TCP", "relation": "compared_with", "tail": "UDP", "evidence": "x"},
        {"head": "TCP", "relation": "contains", "tail": "TCP", "evidence": "self loop"},
        {"head": "TCP", "relation": "bad_rel", "tail": "UDP", "evidence": "bad"},
        "not a dict",
    ]
    triples = KGExtractor._parse_triples(raw)
    assert len(triples) == 1
    assert triples[0].head == "TCP"
    assert triples[0].tail == "UDP"


def test_parse_entities_normalizes_type_case():
    raw = [{"name": "TCP", "type": "PROTOCOL", "description": ""}]
    entities = KGExtractor._parse_entities(raw)
    assert len(entities) == 1
    assert entities[0].type == "protocol"


def test_parse_triples_normalizes_relation_case():
    raw = [{"head": "TCP", "relation": "COMPARED_WITH", "tail": "UDP", "evidence": ""}]
    triples = KGExtractor._parse_triples(raw)
    assert len(triples) == 1
    assert triples[0].relation == "compared_with"


def test_apply_gate_rejects_non_substring_evidence():
    chunk_content = "TCP 通过 IP 层转发数据报。"
    candidate = Triple(
        head="TCP",
        relation="depends_on",
        tail="IP",
        evidence="TCP 依赖 IP",
        evidence_span={"start": 0, "end": 8},
        review_decision="keep",
        review_reason_code="SUPPORTED",
        review_reason="",
    )
    kept, dropped = KGExtractor._apply_gate([candidate], chunk_content)
    assert len(kept) == 0
    assert len(dropped) == 1
    assert dropped[0]["review"]["reason_code"] == "EVIDENCE_NOT_IN_CHUNK"


def test_apply_gate_rejects_span_mismatch():
    chunk_content = "TCP 依赖 IP 进行寻址。"
    evidence = "TCP 依赖 IP"
    candidate = Triple(
        head="TCP",
        relation="depends_on",
        tail="IP",
        evidence=evidence,
        evidence_span={"start": 1, "end": 1 + len(evidence)},
        review_decision="keep",
        review_reason_code="SUPPORTED",
        review_reason="",
    )
    kept, dropped = KGExtractor._apply_gate([candidate], chunk_content)
    assert len(kept) == 0
    assert dropped[0]["review"]["reason_code"] == "SPAN_MISMATCH"


def test_apply_gate_rejects_when_head_tail_not_in_evidence():
    chunk_content = "IP 提供寻址，路由表提供转发信息。"
    evidence = "路由表提供转发信息"
    start = chunk_content.find(evidence)
    candidate = Triple(
        head="IP",
        relation="depends_on",
        tail="路由表",
        evidence=evidence,
        evidence_span={"start": start, "end": start + len(evidence)},
        review_decision="keep",
        review_reason_code="SUPPORTED",
        review_reason="",
    )
    kept, dropped = KGExtractor._apply_gate([candidate], chunk_content)
    assert len(kept) == 0
    assert dropped[0]["review"]["reason_code"] == "HEAD_NOT_IN_EVIDENCE"


def test_apply_gate_keeps_valid_triple():
    chunk_content = "TCP 依赖 IP 进行端到端通信。"
    evidence = "TCP 依赖 IP"
    start = chunk_content.find(evidence)
    candidate = Triple(
        head="TCP",
        relation="depends_on",
        tail="IP",
        evidence=evidence,
        evidence_span={"start": start, "end": start + len(evidence)},
        review_decision="keep",
        review_reason_code="SUPPORTED",
        review_reason="",
    )
    kept, dropped = KGExtractor._apply_gate([candidate], chunk_content)
    assert len(kept) == 1
    assert len(dropped) == 0
