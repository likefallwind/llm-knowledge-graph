from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LEGACY_ENTITY_TYPES = frozenset(
    {"resource", "criterion", "data", "task", "solution", "concept"}
)
CORE_RELATION_KINDS = frozenset(
    {"is_a", "part_of", "prerequisite_of", "other"}
)
# Compatibility exports for the old ontology documentation and callers.  The
# vNext extractors no longer use either set as an acceptance whitelist.
ENTITY_TYPES = LEGACY_ENTITY_TYPES
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
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    location: str
    content_hash: str
    passages: tuple[SourcePassage, ...]
    section_path: tuple[str, ...] = ()


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
    type_labels: tuple[str, ...] = ()


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
    raw_relation: str = ""
    relation_kind: str = "other"
    relation_type_id: int | None = None
    statement_text: str = ""
    scope_text: str = ""
    scope_is_restrictive: bool = False
    normalized_statement: str = ""


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
    assertions: int = 0
    evidence: int = 0
    entity_observations: int = 0
    claim_observations: int = 0
    entity_cap_hit: bool = False
    rejected: list[str] = field(default_factory=list)
    rejection_details: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    not_materialized: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "claims": self.claims,
            "assertions": self.assertions,
            "evidence": self.evidence,
            "entity_observations": self.entity_observations,
            "claim_observations": self.claim_observations,
            "entity_cap_hit": self.entity_cap_hit,
            "rejected": self.rejected,
            "rejection_details": self.rejection_details,
            "pending": self.pending,
            "not_materialized": self.not_materialized,
        }
