from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any


def rejection_category(message: str) -> str:
    if message.startswith("Claim 端点无法唯一解析:"):
        return "endpoint_unresolved"
    if message.startswith("Claim 证据裁决为 insufficient:"):
        return "insufficient"
    if message.startswith("Claim 证据裁决为 contradicts:"):
        return "contradicts"
    if message.startswith("Claim 未写入:") and "不允许自环" in message:
        return "self_loop_after_resolution"
    if "passage_ids" in message or "不存在的段落" in message:
        return "invalid_passage"
    if "超过上限" in message:
        return "limit_truncated"
    if message.startswith(("entity[", "claim[")):
        return "invalid_extraction"
    return "other"


def rejection_report(
    conn: sqlite3.Connection, *, sample_limit: int = 20
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    for row in _latest_progress(conn):
        try:
            result = json.loads(str(row["result"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            result = {}
        rejected = result.get("rejected", [])
        if not isinstance(rejected, list):
            rejected = []
        for raw in rejected:
            message = str(raw)
            category = rejection_category(message)
            categories[category] += 1
            bucket = samples.setdefault(category, [])
            if len(bucket) < sample_limit:
                bucket.append(
                    {
                        "source_id": int(row["source_id"]),
                        "chunk": int(row["chunk_index"]),
                        "message": message,
                    }
                )
        raw_details = result.get("rejection_details", [])
        if isinstance(raw_details, list):
            for raw in raw_details:
                if not isinstance(raw, dict):
                    continue
                detail = dict(raw)
                detail["source_id"] = int(row["source_id"])
                detail["chunk"] = int(row["chunk_index"])
                details.append(detail)

    algorithmic_categories = {
        "endpoint_unresolved",
        "invalid_passage",
        "limit_truncated",
        "invalid_extraction",
    }
    semantic_categories = {
        "insufficient",
        "contradicts",
        "self_loop_after_resolution",
    }
    return {
        "total": sum(categories.values()),
        "categories": dict(sorted(categories.items())),
        "algorithmic_loss": sum(
            count
            for category, count in categories.items()
            if category in algorithmic_categories
        ),
        "semantic_rejection": sum(
            count
            for category, count in categories.items()
            if category in semantic_categories
        ),
        "unclassified": sum(
            count
            for category, count in categories.items()
            if category not in algorithmic_categories | semantic_categories
        ),
        "samples": samples,
        "details": details[: sample_limit * 5],
    }


def graph_report(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        str(row["name"]): int(row["count"])
        for row in conn.execute(
            """
            SELECT 'sources' AS name,COUNT(*) AS count FROM sources
            UNION ALL SELECT 'source_sections',COUNT(*) FROM source_sections
            UNION ALL SELECT 'entities',COUNT(*) FROM entities
            UNION ALL SELECT 'entity_observations',COUNT(*) FROM entity_observations
            UNION ALL SELECT 'relation_types',COUNT(*) FROM relation_types
            UNION ALL SELECT 'claims',COUNT(*) FROM claims
            UNION ALL SELECT 'assertions',COUNT(*) FROM assertions
            UNION ALL SELECT 'claim_observations',COUNT(*) FROM claim_observations
            UNION ALL SELECT 'evidence',COUNT(*) FROM evidence
            """
        )
    }
    degree = {
        int(row["entity_id"]): int(row["degree"])
        for row in conn.execute(
            """
            SELECT entity_id,COUNT(*) AS degree FROM (
              SELECT subject_id AS entity_id FROM claims
              UNION ALL
              SELECT object_id AS entity_id FROM claims
            ) GROUP BY entity_id
            """
        )
    }
    entity_ids = [
        int(row["id"]) for row in conn.execute("SELECT id FROM entities")
    ]
    adjacency = {entity_id: set() for entity_id in entity_ids}
    for row in conn.execute("SELECT subject_id,object_id FROM claims"):
        left, right = int(row["subject_id"]), int(row["object_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    component_sizes: list[int] = []
    for entity_id in entity_ids:
        if entity_id in seen:
            continue
        pending = [entity_id]
        seen.add(entity_id)
        size = 0
        while pending:
            current = pending.pop()
            size += 1
            for neighbor in adjacency[current] - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        component_sizes.append(size)
    component_sizes.sort(reverse=True)
    relations = {
        str(row["relation"]): int(row["count"])
        for row in conn.execute(
            "SELECT relation,COUNT(*) AS count FROM claims GROUP BY relation"
        )
    }
    relation_kinds = {
        str(row["relation_kind"]): int(row["count"])
        for row in conn.execute(
            """SELECT r.relation_kind,COUNT(*) AS count FROM claims c
               JOIN relation_types r ON r.id=c.relation_type_id
               GROUP BY r.relation_kind"""
        )
    }
    section_total = counts.get("source_sections", 0)
    sections_with_entities = int(
        conn.execute(
            "SELECT COUNT(DISTINCT section_id) FROM entity_observations WHERE section_id IS NOT NULL"
        ).fetchone()[0]
    )
    sections_with_summaries = int(
        conn.execute("SELECT COUNT(DISTINCT section_id) FROM section_summaries").fetchone()[0]
    )
    entity_evidence = int(
        conn.execute("SELECT COUNT(*) FROM evidence WHERE entity_id IS NOT NULL").fetchone()[0]
    )
    claim_evidence = int(
        conn.execute("SELECT COUNT(*) FROM evidence WHERE claim_id IS NOT NULL").fetchone()[0]
    )
    assertion_evidence = int(
        conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE assertion_id IS NOT NULL"
        ).fetchone()[0]
    )
    return {
        "counts": counts,
        "relations": relations,
        "relation_kinds": relation_kinds,
        "document_coverage": {
            "sections": section_total,
            "sections_with_entities": sections_with_entities,
            "sections_with_summaries": sections_with_summaries,
        },
        "evidence_density": {
            "per_entity": round(entity_evidence / max(1, counts.get("entities", 0)), 3),
            "per_claim": round(claim_evidence / max(1, counts.get("claims", 0)), 3),
            "per_assertion": round(
                assertion_evidence / max(1, counts.get("assertions", 0)), 3
            ),
        },
        "isolated_entities": sum(
            1 for entity_id in entity_ids if degree.get(entity_id, 0) == 0
        ),
        "leaf_entities": sum(1 for value in degree.values() if value == 1),
        "components": len(component_sizes),
        "largest_component": component_sizes[0] if component_sizes else 0,
    }


def _latest_progress(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT source_id,chunk_index,status,result,error,updated_at
        FROM (
          SELECT rowid AS progress_rowid,*,
                 ROW_NUMBER() OVER (
                   PARTITION BY source_id,chunk_index
                   ORDER BY updated_at DESC,rowid DESC
                 ) AS position
          FROM source_progress
        )
        WHERE position=1
        ORDER BY source_id,chunk_index
        """
    ).fetchall()
