from pathlib import Path

from qmrkg.kg_extractor import KGExtractor


def test_resolve_prompt_prefers_mode_specific_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "test-key")
    few = "FEW_PROMPT"
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
extract:
  provider:
    name: ppio
    base_url: https://api.ppio.com/openai
    model: deepseek/deepseek-v3.2
    modality: text
    supports_thinking: false
  prompts:
    default: "DEFAULT_SYSTEM"
    zero_shot: "ZERO_SYSTEM"
    few_shot: |
      {few}
  request:
    timeout_seconds: 60.0
    max_retries: 1
    thinking:
      enabled: false
  rate_limit:
    rpm: 50
    max_concurrency: 4
""".strip(),
        encoding="utf-8",
    )
    ex = KGExtractor(config_path=config_path, mode="few-shot")
    assert ex.resolve_prompt() == few


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
