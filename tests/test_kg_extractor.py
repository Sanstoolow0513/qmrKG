from qmrkg.kg_extractor import EXTRACT_PROMPT, KGExtractor
from qmrkg.llm_types import LLMResponse
from qmrkg.ner_prompts import ZERO_SHOT_EXTRACTION_PROMPT


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


def test_extract_from_chunk_zero_shot_uses_expected_system_prompt():
    captured: dict[str, str | None] = {}

    class FakeRunner:
        def run_text(self, prompt: str, *, system_prompt: str | None = None):
            captured["system_prompt"] = system_prompt
            return LLMResponse(
                text='{"entities": [], "triples": []}',
                processed_at="",
                duration_seconds=0.0,
            )

    ex = KGExtractor(runner=FakeRunner(), prompt_kind="zero_shot")
    chunk = {"chunk_index": 0, "source_file": "x.md", "titles": [], "content": "TCP 与 UDP。"}
    ex.extract_from_chunk(chunk)
    assert captured["system_prompt"] == ZERO_SHOT_EXTRACTION_PROMPT


def test_extract_from_chunk_legacy_uses_original_prompt():
    captured: dict[str, str | None] = {}

    class FakeRunner:
        def run_text(self, prompt: str, *, system_prompt: str | None = None):
            captured["system_prompt"] = system_prompt
            return LLMResponse(
                text='{"entities": [], "triples": []}',
                processed_at="",
                duration_seconds=0.0,
            )

    ex = KGExtractor(runner=FakeRunner(), prompt_kind="legacy")
    chunk = {"chunk_index": 0, "source_file": "x.md", "titles": [], "content": "test"}
    ex.extract_from_chunk(chunk)
    assert captured["system_prompt"] == EXTRACT_PROMPT
