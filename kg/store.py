from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

from .models import EntityObservation, LoadedSource, RELATIONS


def normalize_name(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", folded)


def reference_key(value: str) -> str:
    """Identity lookup key: preserve punctuation, ignore whitespace only."""
    return re.sub(r"\s+", "", normalize_name(value))


def add_source(conn: sqlite3.Connection, loaded: LoadedSource) -> tuple[int, bool]:
    row = conn.execute(
        "SELECT id FROM sources WHERE source_key=? AND content_hash=?",
        (loaded.spec.key, loaded.content_hash),
    ).fetchone()
    if row:
        return int(row["id"]), False
    cursor = conn.execute(
        """
        INSERT INTO sources
        (source_key,name,source_type,uri,version,content,content_hash,language)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            loaded.spec.key,
            loaded.spec.name,
            loaded.spec.source_type,
            loaded.spec.uri,
            loaded.version,
            loaded.content,
            loaded.content_hash,
            loaded.spec.language,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid), True


def get_entity(conn: sqlite3.Connection, entity_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()


def exact_entity_ids(conn: sqlite3.Connection, name: str) -> list[int]:
    normalized = normalize_name(name)
    rows = conn.execute(
        """
        SELECT id FROM entities WHERE normalized_name=?
        UNION
        SELECT entity_id AS id FROM entity_aliases WHERE normalized_name=?
        ORDER BY id
        """,
        (normalized, normalized),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def reference_entity_ids(conn: sqlite3.Connection, name: str) -> list[int]:
    """Resolve a canonical name or verified alias only when the key is unique."""
    key = reference_key(name)
    rows = conn.execute(
        """
        SELECT id FROM entities
        WHERE REPLACE(normalized_name,' ','')=?
        UNION
        SELECT entity_id AS id FROM entity_aliases
        WHERE REPLACE(normalized_name,' ','')=?
        ORDER BY id
        """,
        (key, key),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def list_entities(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM entities ORDER BY id").fetchall()


def aliases_for(conn: sqlite3.Connection, entity_id: int) -> list[str]:
    return [
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM entity_aliases WHERE entity_id=? ORDER BY id",
            (entity_id,),
        )
    ]


def evidence_for_entity(
    conn: sqlite3.Connection, entity_id: int, *, limit: int = 3
) -> list[str]:
    return [
        str(row["excerpt"])
        for row in conn.execute(
            """
            SELECT excerpt FROM evidence
            WHERE entity_id=? AND polarity='support'
            ORDER BY id DESC LIMIT ?
            """,
            (entity_id, limit),
        )
    ]


def type_profile(conn: sqlite3.Connection, entity_id: int) -> list[dict[str, Any]]:
    """Entity 层的类型表示：各类型分别有多少条观察、来自多少个独立来源。

    这不是投票——不取 argmax，也不折叠成单一类型。一个词确实可能同时属于
    多个类型（「深度学习」既是一族做法也是一个研究方向），profile 如实保留
    这一点。observations 反映语料分布，sources 反映有多少独立来源这样判，
    两者含义不同，因此都给出。
    """
    return [
        {
            "entity_type": str(row["entity_type"]),
            "observations": int(row["observations"]),
            "sources": int(row["sources"]),
        }
        for row in conn.execute(
            """
            SELECT observed_entity_type AS entity_type,
                   COUNT(*) AS observations,
                   COUNT(DISTINCT source_id) AS sources
            FROM evidence
            WHERE entity_id=? AND polarity='support' AND observed_entity_type<>''
            GROUP BY observed_entity_type
            ORDER BY sources DESC, observations DESC, observed_entity_type
            """,
            (entity_id,),
        )
    ]


def create_entity(
    conn: sqlite3.Connection,
    observation: EntityObservation,
    *,
    canonical_name: str = "",
) -> int:
    canonical = canonical_name.strip() or observation.name
    normalized = normalize_name(canonical)
    existing = conn.execute(
        "SELECT id FROM entities WHERE normalized_name=?", (normalized,)
    ).fetchone()
    if existing:
        entity_id = int(existing["id"])
    else:
        cursor = conn.execute(
            """
            INSERT INTO entities(canonical_name,normalized_name,definition)
            VALUES (?,?,?)
            """,
            (canonical, normalized, observation.definition),
        )
        entity_id = int(cursor.lastrowid)
    for alias in (observation.name, *observation.aliases):
        add_alias(conn, entity_id, alias)
    return entity_id


def add_alias(conn: sqlite3.Connection, entity_id: int, name: str) -> bool:
    value = name.strip()
    if not value:
        return False
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO entity_aliases(entity_id,name,normalized_name)
        VALUES (?,?,?)
        """,
        (entity_id, value, normalize_name(value)),
    )
    return cursor.rowcount > 0


def add_evidence(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    source_text: str,
    model_quote: str,
    passage_ids: Iterable[str] = (),
    passage_version: str = "source-passages-2",
    location: str,
    polarity: str,
    extraction_model: str = "",
    extraction_prompt_version: str = "",
    validator_model: str = "",
    validator_prompt_version: str = "",
    validator_verdict: str = "",
    validator_reason: str = "",
    observed_entity_type: str = "",
    entity_id: int | None = None,
    claim_id: int | None = None,
) -> bool:
    if (entity_id is None) == (claim_id is None):
        raise ValueError("Evidence 必须且只能指向一个 Entity 或 Claim")
    passage_list = list(passage_ids)
    excerpt_hash = hashlib.sha256(
        "\0".join(
            (
                source_text,
                model_quote,
                json.dumps(passage_list, ensure_ascii=False),
            )
        ).encode("utf-8")
    ).hexdigest()
    target_key = f"entity:{entity_id}" if entity_id is not None else f"claim:{claim_id}"
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO evidence
        (target_key,entity_id,claim_id,source_id,excerpt,model_quote,
         observed_entity_type,passage_ids,passage_version,extraction_model,
         extraction_prompt_version,validator_model,validator_prompt_version,
         validator_verdict,validator_reason,excerpt_hash,location,polarity)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            target_key,
            entity_id,
            claim_id,
            source_id,
            source_text,
            model_quote,
            observed_entity_type,
            json.dumps(passage_list, ensure_ascii=False),
            passage_version,
            extraction_model,
            extraction_prompt_version,
            validator_model,
            validator_prompt_version,
            validator_verdict,
            validator_reason,
            excerpt_hash,
            location,
            polarity,
        ),
    )
    return cursor.rowcount > 0


def find_claim(
    conn: sqlite3.Connection, subject_id: int, relation: str, object_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM claims WHERE subject_id=? AND relation=? AND object_id=?",
        (subject_id, relation, object_id),
    ).fetchone()


def _path_exists(
    conn: sqlite3.Connection, start_id: int, target_id: int, relation: str
) -> bool:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for row in conn.execute(
        "SELECT subject_id,object_id FROM claims WHERE relation=?", (relation,)
    ):
        adjacency[int(row["subject_id"])].add(int(row["object_id"]))
    pending = [start_id]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current == target_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, set()) - seen)
    return False


def upsert_claim(
    conn: sqlite3.Connection, subject_id: int, relation: str, object_id: int
) -> tuple[int | None, bool, str]:
    if relation not in RELATIONS:
        return None, False, f"不支持的关系: {relation}"
    if subject_id == object_id:
        return None, False, "不允许自环"
    existing = find_claim(conn, subject_id, relation, object_id)
    if existing:
        return int(existing["id"]), False, ""
    if relation in {"is_a", "prerequisite_of"} and _path_exists(
        conn, object_id, subject_id, relation
    ):
        return None, False, f"{relation} 会形成循环"
    cursor = conn.execute(
        "INSERT INTO claims(subject_id,relation,object_id) VALUES (?,?,?)",
        (subject_id, relation, object_id),
    )
    return int(cursor.lastrowid), True, ""


def progress_done(
    conn: sqlite3.Connection, source_id: int, chunk_index: int, chunk_hash: str
) -> bool:
    row = conn.execute(
        """
        SELECT status FROM source_progress
        WHERE source_id=? AND chunk_index=? AND chunk_hash=?
        """,
        (source_id, chunk_index, chunk_hash),
    ).fetchone()
    return bool(row and row["status"] == "done")


def save_progress(
    conn: sqlite3.Connection,
    source_id: int,
    chunk_index: int,
    chunk_hash: str,
    *,
    status: str,
    result: dict | None = None,
    error: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO source_progress
        (source_id,chunk_index,chunk_hash,status,result,error)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(source_id,chunk_index,chunk_hash) DO UPDATE SET
          status=excluded.status,
          result=excluded.result,
          error=excluded.error,
          updated_at=CURRENT_TIMESTAMP
        """,
        (
            source_id,
            chunk_index,
            chunk_hash,
            status,
            json.dumps(result or {}, ensure_ascii=False),
            error,
        ),
    )
    conn.commit()


def merge_entities(
    conn: sqlite3.Connection, source_id: int, target_id: int
) -> int:
    """Merge source into target and deduplicate Claims/Evidence transactionally."""
    if source_id == target_id:
        return target_id
    source = get_entity(conn, source_id)
    target = get_entity(conn, target_id)
    if not source or not target:
        raise ValueError("待合并实体不存在")

    with conn:
        conn.execute(
            "UPDATE entity_observations SET entity_id=? WHERE entity_id=?",
            (target_id, source_id),
        )
        conn.execute(
            "UPDATE claim_observations SET subject_entity_id=? WHERE subject_entity_id=?",
            (target_id, source_id),
        )
        conn.execute(
            "UPDATE claim_observations SET object_entity_id=? WHERE object_entity_id=?",
            (target_id, source_id),
        )
        add_alias(conn, target_id, str(source["canonical_name"]))
        for alias in aliases_for(conn, source_id):
            add_alias(conn, target_id, alias)

        evidence_rows = conn.execute(
            "SELECT * FROM evidence WHERE entity_id=?", (source_id,)
        ).fetchall()
        for row in evidence_rows:
            add_evidence(
                conn,
                source_id=int(row["source_id"]),
                source_text=str(row["excerpt"]),
                model_quote=str(row["model_quote"]),
                passage_ids=json.loads(str(row["passage_ids"])),
                passage_version=str(row["passage_version"]),
                location=str(row["location"]),
                polarity=str(row["polarity"]),
                extraction_model=str(row["extraction_model"]),
                extraction_prompt_version=str(
                    row["extraction_prompt_version"]
                ),
                validator_model=str(row["validator_model"]),
                validator_prompt_version=str(row["validator_prompt_version"]),
                validator_verdict=str(row["validator_verdict"]),
                validator_reason=str(row["validator_reason"]),
                observed_entity_type=str(row["observed_entity_type"]),
                entity_id=target_id,
            )
        conn.execute("DELETE FROM evidence WHERE entity_id=?", (source_id,))

        claim_rows = conn.execute(
            "SELECT * FROM claims WHERE subject_id=? OR object_id=? ORDER BY id",
            (source_id, source_id),
        ).fetchall()
        for row in claim_rows:
            old_claim_id = int(row["id"])
            new_subject = target_id if int(row["subject_id"]) == source_id else int(row["subject_id"])
            new_object = target_id if int(row["object_id"]) == source_id else int(row["object_id"])
            evidence = conn.execute(
                "SELECT * FROM evidence WHERE claim_id=?", (old_claim_id,)
            ).fetchall()
            conn.execute("DELETE FROM evidence WHERE claim_id=?", (old_claim_id,))
            conn.execute("DELETE FROM claims WHERE id=?", (old_claim_id,))
            if new_subject == new_object:
                continue
            new_claim_id, _, _ = upsert_claim(
                conn, new_subject, str(row["relation"]), new_object
            )
            if new_claim_id is None:
                continue
            for item in evidence:
                add_evidence(
                    conn,
                    source_id=int(item["source_id"]),
                    source_text=str(item["excerpt"]),
                    model_quote=str(item["model_quote"]),
                    passage_ids=json.loads(str(item["passage_ids"])),
                    passage_version=str(item["passage_version"]),
                    location=str(item["location"]),
                    polarity=str(item["polarity"]),
                    extraction_model=str(item["extraction_model"]),
                    extraction_prompt_version=str(
                        item["extraction_prompt_version"]
                    ),
                    validator_model=str(item["validator_model"]),
                    validator_prompt_version=str(
                        item["validator_prompt_version"]
                    ),
                    validator_verdict=str(item["validator_verdict"]),
                    validator_reason=str(item["validator_reason"]),
                    claim_id=new_claim_id,
                )

        conn.execute("DELETE FROM entities WHERE id=?", (source_id,))
    return target_id


def set_canonical_name(
    conn: sqlite3.Connection, entity_id: int, canonical_name: str
) -> bool:
    value = canonical_name.strip()
    if not value:
        return False
    normalized = normalize_name(value)
    conflict = conn.execute(
        "SELECT id FROM entities WHERE normalized_name=? AND id!=?",
        (normalized, entity_id),
    ).fetchone()
    if conflict:
        return False
    current = get_entity(conn, entity_id)
    if not current:
        return False
    if str(current["canonical_name"]) == value:
        return True
    with conn:
        add_alias(conn, entity_id, str(current["canonical_name"]))
        conn.execute(
            """
            UPDATE entities
            SET canonical_name=?,normalized_name=?,updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (value, normalized, entity_id),
        )
    return True


def integrity_report(conn: sqlite3.Connection) -> dict:
    fk = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    cycles: dict[str, list[int]] = {}
    for relation in ("is_a", "prerequisite_of"):
        bad: list[int] = []
        for row in conn.execute(
            "SELECT id,subject_id,object_id FROM claims WHERE relation=?", (relation,)
        ):
            if _path_exists(
                conn, int(row["object_id"]), int(row["subject_id"]), relation
            ):
                bad.append(int(row["id"]))
        if bad:
            cycles[relation] = bad
    no_evidence = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT e.id FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM evidence v WHERE v.entity_id=e.id)
            """
        )
    ]
    claim_no_evidence = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT c.id FROM claims c
            WHERE NOT EXISTS (
              SELECT 1 FROM evidence v
              WHERE v.claim_id=c.id AND v.polarity='support'
            )
            """
        )
    ]
    broken_observations = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT o.id FROM claim_observations o
            LEFT JOIN claims c ON c.id=o.claim_id
            WHERE o.claim_id IS NOT NULL AND (
              c.id IS NULL OR c.subject_id<>o.subject_entity_id
              OR c.object_id<>o.object_entity_id OR c.relation<>o.relation
            )
            """
        )
    ]
    broken_entity_observations = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT o.id FROM entity_observations o
            LEFT JOIN entities e ON e.id=o.entity_id
            WHERE o.entity_id IS NOT NULL AND e.id IS NULL
            """
        )
    ]
    return {
        "ok": not fk and not cycles and not no_evidence and not claim_no_evidence
        and not broken_observations and not broken_entity_observations,
        "foreign_key_errors": fk,
        "cycles": cycles,
        "entities_without_evidence": no_evidence,
        "claims_without_support": claim_no_evidence,
        "broken_claim_observations": broken_observations,
        "broken_entity_observations": broken_entity_observations,
    }


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = ("sources", "entities", "claims", "evidence")
    return {
        name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        for name in names
    }
