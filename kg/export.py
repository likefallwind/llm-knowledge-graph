from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


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
        entities.append(
            {
                "id": int(row["id"]),
                "canonical_name": str(row["canonical_name"]),
                "aliases": aliases,
                "definition": str(row["definition"]),
                "entity_type": str(row["entity_type"]),
            }
        )
    claims = []
    for row in conn.execute(
        """
        SELECT c.id,c.subject_id,c.relation,c.object_id,
               s.canonical_name AS subject_name,
               o.canonical_name AS object_name
        FROM claims c
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
                "object_id": int(row["object_id"]),
                "object": str(row["object_name"]),
            }
        )
    evidence = [
        {
            "id": int(row["id"]),
            "target": str(row["target_key"]),
            "source_id": int(row["source_id"]),
            "excerpt": str(row["excerpt"]),
            "location": str(row["location"]),
            "polarity": str(row["polarity"]),
        }
        for row in conn.execute("SELECT * FROM evidence ORDER BY id")
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
    return {
        "sources": sources,
        "entities": entities,
        "claims": claims,
        "evidence": evidence,
    }


def write_json(conn: sqlite3.Connection, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph_dict(conn), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
