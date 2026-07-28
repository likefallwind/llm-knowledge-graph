from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ENTITY_TYPES = frozenset(
    {"resource", "criterion", "data", "task", "solution", "concept"}
)
RELATIONS = frozenset({"is_a", "part_of", "prerequisite_of"})
POLARITIES = frozenset({"support", "oppose"})


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    source_type: str
    uri: str = ""
    path: Path | None = None
    version: str = ""
    language: str = ""


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    content: str
    content_hash: str
    version: str


@dataclass(frozen=True)
class SourcePassage:
    passage_id: str
    text: str
    location: str
    content_hash: str
    start: int
    end: int


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    location: str
    content_hash: str
    passages: tuple[SourcePassage, ...]


@dataclass(frozen=True)
class EntityObservation:
    name: str
    definition: str
    entity_type: str
    model_quote: str
    source_text: str
    passage_ids: tuple[str, ...]
    location: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimObservation:
    subject: str
    relation: str
    object: str
    model_quote: str
    source_text: str
    passage_ids: tuple[str, ...]
    location: str
    polarity: str = "support"


@dataclass(frozen=True)
class ExtractionBatch:
    entities: tuple[EntityObservation, ...]
    claims: tuple[ClaimObservation, ...]
    rejected: tuple[str, ...] = ()


@dataclass(frozen=True)
class Resolution:
    entity_id: int
    outcome: str
    reason: str = ""
    candidates: tuple[int, ...] = ()


@dataclass
class ChunkResult:
    entities: int = 0
    claims: int = 0
    evidence: int = 0
    rejected: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "claims": self.claims,
            "evidence": self.evidence,
            "rejected": self.rejected,
        }
