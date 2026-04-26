"""Entity and relation type definitions for knowledge graph extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityType = Literal["protocol", "concept", "mechanism", "metric"]
RelationType = Literal["contains", "depends_on", "compared_with", "applied_to"]

ENTITY_TYPES: set[str] = {"protocol", "concept", "mechanism", "metric"}
RELATION_TYPES: set[str] = {"contains", "depends_on", "compared_with", "applied_to"}

ENTITY_TYPE_LABELS: dict[str, str] = {
    "protocol": "Protocol",
    "concept": "Concept",
    "mechanism": "Mechanism",
    "metric": "Metric",
}

RELATION_TYPE_LABELS: dict[str, str] = {
    "contains": "CONTAINS",
    "depends_on": "DEPENDS_ON",
    "compared_with": "COMPARED_WITH",
    "applied_to": "APPLIED_TO",
}


@dataclass(slots=True)
class Entity:
    name: str
    type: str
    description: str = ""
    frequency: int = 1

    def is_valid(self) -> bool:
        return (
            self.type in ENTITY_TYPES
            and 2 <= len(self.name) <= 30
            and bool(self.name.strip())
        )


@dataclass(slots=True)
class Triple:
    head: str
    relation: str
    tail: str
    evidence: str = ""
    evidence_span: dict[str, int] | None = None
    frequency: int = 1
    evidences: list[str] = field(default_factory=list)
    review_decision: str = "keep"
    review_reason_code: str = "SUPPORTED"
    review_reason: str = ""

    def is_valid(self) -> bool:
        return (
            self.relation in RELATION_TYPES
            and self.head != self.tail
            and bool(self.head.strip())
            and bool(self.tail.strip())
        )


@dataclass(slots=True)
class ChunkExtractionResult:
    chunk_index: int
    source_file: str
    titles: list[str]
    entities: list[Entity]
    triples: list[Triple]
    dropped: list[dict[str, Any]] = field(default_factory=list)
