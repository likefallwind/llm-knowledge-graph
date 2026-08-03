from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import store


def graph_dict(conn: sqlite3.Connection) -> dict[str, Any]:
    entities = []
    for row in conn.execute("SELECT * FROM entities ORDER BY id"):
        aliases = [
            str(item["name"])
            for item in conn.execute(
                "SELECT name FROM entity_aliases WHERE entity_id=? ORDER BY id",
                (row["id"],),
            )
        ]
        synthesis = conn.execute(
            """
            SELECT * FROM entity_definition_syntheses
            WHERE entity_id=? ORDER BY id DESC LIMIT 1
            """,
            (row["id"],),
        ).fetchone()
        item = {
                "id": int(row["id"]),
                "canonical_name": str(row["canonical_name"]),
                "aliases": aliases,
                "definition": str(row["definition"]),
                "type_profile": store.type_profile(conn, int(row["id"])),
            }
        if synthesis is not None:
            item["definition_synthesis"] = {
                "model": str(synthesis["synthesizer_model"]),
                "prompt_version": str(synthesis["prompt_version"]),
                "observation_fingerprint": str(
                    synthesis["observation_fingerprint"]
                ),
                "supporting_observations": json.loads(
                    str(synthesis["supporting_observations"])
                ),
                "rejected_candidates": json.loads(
                    str(synthesis["rejected_candidates"])
                ),
                "limitation": str(synthesis["limitation"]),
            }
        entities.append(item)
    claims = []
    for row in conn.execute(
        """
        SELECT c.id,c.subject_id,c.relation_type_id,c.relation,c.object_id,
               r.relation_kind,r.description AS relation_description,
               s.canonical_name AS subject_name,
               o.canonical_name AS object_name
        FROM claims c
        JOIN relation_types r ON r.id=c.relation_type_id
        JOIN entities s ON s.id=c.subject_id
        JOIN entities o ON o.id=c.object_id
        ORDER BY c.id
        """
    ):
        claims.append(
            {
                "id": int(row["id"]),
                "subject_id": int(row["subject_id"]),
                "subject": str(row["subject_name"]),
                "relation": str(row["relation"]),
                "relation_type_id": int(row["relation_type_id"]),
                "relation_kind": str(row["relation_kind"]),
                "relation_description": str(row["relation_description"]),
                "object_id": int(row["object_id"]),
                "object": str(row["object_name"]),
            }
        )
    evidence = [
        {
            "id": int(row["id"]),
            "target": str(row["target_key"]),
            "source_id": int(row["source_id"]),
            "source": {
                "name": str(row["source_name"]),
                "uri": str(row["source_uri"]),
                "version": str(row["source_version"]),
                "content_hash": str(row["source_content_hash"]),
            },
            "passage_ids": json.loads(str(row["passage_ids"])),
            "passage_version": str(row["passage_version"]),
            "source_text": str(row["excerpt"]),
            "model_quote": str(row["model_quote"]),
            "location": str(row["location"]),
            "polarity": str(row["polarity"]),
            "extraction": {
                "model": str(row["extraction_model"]),
                "prompt_version": str(row["extraction_prompt_version"]),
            },
            "validation": {
                "model": str(row["validator_model"]),
                "prompt_version": str(row["validator_prompt_version"]),
                "verdict": str(row["validator_verdict"]),
                "reason": str(row["validator_reason"]),
            },
        }
        for row in conn.execute(
            """
            SELECT e.*,s.name AS source_name,s.uri AS source_uri,
                   s.version AS source_version,
                   s.content_hash AS source_content_hash
            FROM evidence e
            JOIN sources s ON s.id=e.source_id
            ORDER BY e.id
            """
        )
    ]
    sources = [
        {
            "id": int(row["id"]),
            "key": str(row["source_key"]),
            "name": str(row["name"]),
            "type": str(row["source_type"]),
            "uri": str(row["uri"]),
            "version": str(row["version"]),
            "content_hash": str(row["content_hash"]),
            "language": str(row["language"]),
        }
        for row in conn.execute("SELECT * FROM sources ORDER BY id")
    ]
    entity_observations = [
        {
            "id": int(row["id"]),
            "entity_id": (
                int(row["entity_id"])
                if row["entity_id"] is not None
                else None
            ),
            "name": str(row["name"]),
            "definition": str(row["definition"]),
            "observed_entity_type": str(row["observed_entity_type"]),
            "raw_type_labels": json.loads(str(row["raw_type_labels"])),
            "aliases": json.loads(str(row["aliases"])),
            "source_id": int(row["source_id"]),
            "source_name": str(row["source_name"]),
            "chunk_index": int(row["chunk_index"]),
            "passage_ids": json.loads(str(row["passage_ids"])),
            "location": str(row["location"]),
            "source_text": str(row["source_text"]),
            "model_quote": str(row["model_quote"]),
            "resolution": {
                "outcome": str(row["resolution_outcome"]),
                "reason": str(row["resolution_reason"]),
                "candidate_entity_ids": json.loads(
                    str(row["candidate_entity_ids"])
                ),
                "model": str(row["resolver_model"]),
                "prompt_version": str(row["resolver_prompt_version"]),
            },
        }
        for row in conn.execute(
            """
            SELECT o.*,s.name AS source_name
            FROM entity_observations o
            JOIN sources s ON s.id=o.source_id
            ORDER BY o.id
            """
        )
    ]
    relation_types = [
        {
            "id": int(row["id"]),
            "canonical_name": str(row["canonical_name"]),
            "relation_kind": str(row["relation_kind"]),
            "description": str(row["description"]),
        }
        for row in conn.execute("SELECT * FROM relation_types ORDER BY id")
    ]
    sections = [
        {
            "id": int(row["id"]),
            "source_id": int(row["source_id"]),
            "parent_id": int(row["parent_id"]) if row["parent_id"] is not None else None,
            "title": str(row["title"]),
            "depth": int(row["depth"]),
            "ordinal": int(row["ordinal"]),
            "path": json.loads(str(row["path_json"])),
            "summary": str(row["summary"] or ""),
            "entity_ids": json.loads(str(row["entity_ids"] or "[]")),
        }
        for row in conn.execute(
            """SELECT s.*,
                      (SELECT summary FROM section_summaries x
                       WHERE x.section_id=s.id ORDER BY x.id DESC LIMIT 1) AS summary,
                      (SELECT json_group_array(entity_id) FROM (
                         SELECT DISTINCT entity_id FROM entity_observations o
                         WHERE o.section_id=s.id AND entity_id IS NOT NULL
                         ORDER BY entity_id
                       )) AS entity_ids
               FROM source_sections s ORDER BY source_id,depth,ordinal,id"""
        )
    ]
    return {
        "sources": sources,
        "entities": entities,
        "claims": claims,
        "evidence": evidence,
        "entity_observations": entity_observations,
        "relation_types": relation_types,
        "sections": sections,
    }


def write_json(conn: sqlite3.Connection, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph_dict(conn), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
