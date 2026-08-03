from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from itertools import combinations
from typing import Any

from . import extraction, observations, store, validation, vocabulary
from .llm import JSONLLM
from .models import ClaimObservation


EXPANSION_PROMPT_VERSION = "toc-relation-expansion-1"
SYSTEM = """你是教材关系补全器，不是知识来源。
目录距离只能用来筛选候选，不能证明关系。只有给出的原始 Passage 明确表达了两个
实体之间的有向关系时才能返回 relation；否则必须返回 null。只输出 JSON 对象。"""


def expand_relations(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    limit: int = 50,
    simple_llm: JSONLLM | None = None,
) -> dict[str, Any]:
    """Try close TOC neighbours and persist only passage-supported relations."""
    if limit < 0:
        raise ValueError("limit 不能为负")
    model = _model_name(llm)
    fast_llm = simple_llm or llm
    stats: dict[str, Any] = {
        "candidates": 0,
        "relations": 0,
        "none": 0,
        "failed": [],
    }
    for candidate in _candidate_pairs(conn)[:limit]:
        source_id, subject_id, object_id = candidate[:3]
        passages = _context_passages(conn, *candidate)
        fingerprint = hashlib.sha256(
            json.dumps(
                [candidate, passages, {"simple_model": _model_name(fast_llm)}],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        cached = conn.execute(
            """SELECT 1 FROM relation_expansion_attempts
               WHERE source_id=? AND subject_id=? AND object_id=?
                 AND context_fingerprint=? AND model=? AND prompt_version=?""",
            (
                source_id,
                subject_id,
                object_id,
                fingerprint,
                model,
                EXPANSION_PROMPT_VERSION,
            ),
        ).fetchone()
        if cached:
            continue
        stats["candidates"] += 1
        try:
            outcome, observation_id, reason = _attempt(
                conn, llm, candidate, passages, model=model, simple_llm=fast_llm
            )
            conn.execute(
                """INSERT INTO relation_expansion_attempts
                   (source_id,subject_id,object_id,context_fingerprint,model,
                    prompt_version,outcome,observation_id,reason)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    source_id,
                    subject_id,
                    object_id,
                    fingerprint,
                    model,
                    EXPANSION_PROMPT_VERSION,
                    outcome,
                    observation_id,
                    reason,
                ),
            )
            conn.commit()
            stats["relations" if outcome == "relation" else "none"] += 1
        except Exception as exc:
            conn.rollback()
            stats["failed"].append(
                {"subject_id": subject_id, "object_id": object_id, "error": str(exc)}
            )
            conn.execute(
                """INSERT OR IGNORE INTO relation_expansion_attempts
                   (source_id,subject_id,object_id,context_fingerprint,model,
                    prompt_version,outcome,reason) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    source_id,
                    subject_id,
                    object_id,
                    fingerprint,
                    model,
                    EXPANSION_PROMPT_VERSION,
                    "failed",
                    str(exc),
                ),
            )
            conn.commit()
    return stats


def _candidate_pairs(conn: sqlite3.Connection) -> list[tuple[int, int, int, int, int]]:
    placements: dict[int, list[tuple[int, int, int | None]]] = {}
    for row in conn.execute(
        """SELECT DISTINCT o.source_id,o.entity_id,o.section_id,s.parent_id
           FROM entity_observations o
           JOIN source_sections s ON s.id=o.section_id
           WHERE o.entity_id IS NOT NULL AND o.section_id IS NOT NULL"""
    ):
        placements.setdefault(int(row["source_id"]), []).append(
            (int(row["entity_id"]), int(row["section_id"]),
             int(row["parent_id"]) if row["parent_id"] is not None else None)
        )
    ranked: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for source_id, items in placements.items():
        for left, right in combinations(items, 2):
            left_id, left_section, left_parent = left
            right_id, right_section, right_parent = right
            if left_id == right_id:
                continue
            distance = 0 if left_section == right_section else (
                1 if left_parent is not None and left_parent == right_parent else 99
            )
            if distance > 1:
                continue
            a, b = sorted((left_id, right_id))
            key = (source_id, a, b)
            if key in seen:
                continue
            seen.add(key)
            existing = conn.execute(
                """SELECT 1 FROM claims WHERE
                   (subject_id=? AND object_id=?) OR
                   (subject_id=? AND object_id=?) LIMIT 1""",
                (a, b, b, a),
            ).fetchone()
            if not existing:
                ranked.append((distance, source_id, a, b, left_section, right_section))
    ranked.sort()
    return [item[1:] for item in ranked]


def _context_passages(
    conn: sqlite3.Connection,
    source_id: int,
    _subject_id: int,
    _object_id: int,
    left_section: int,
    right_section: int,
) -> list[dict[str, str]]:
    section_ids = {left_section, right_section}
    for section_id in tuple(section_ids):
        row = conn.execute(
            "SELECT parent_id FROM source_sections WHERE id=?", (section_id,)
        ).fetchone()
        if row and row["parent_id"] is not None:
            section_ids.add(int(row["parent_id"]))
    placeholders = ",".join("?" for _ in section_ids)
    source = conn.execute(
        "SELECT content FROM sources WHERE id=?", (source_id,)
    ).fetchone()
    content = str(source["content"])
    rows = conn.execute(
        f"""SELECT passage_id,start_offset,end_offset,location
            FROM source_passages WHERE source_id=? AND section_id IN ({placeholders})
            ORDER BY start_offset LIMIT 24""",
        (source_id, *sorted(section_ids)),
    ).fetchall()
    return [
        {
            "passage_id": str(row["passage_id"]),
            "location": str(row["location"]),
            "text": content[int(row["start_offset"]):int(row["end_offset"])],
        }
        for row in rows
    ]


def _attempt(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    candidate: tuple[int, int, int, int, int],
    passages: list[dict[str, str]],
    *,
    model: str,
    simple_llm: JSONLLM,
) -> tuple[str, int | None, str]:
    source_id, left_id, right_id, _left_section, _right_section = candidate
    entities = []
    for entity_id in (left_id, right_id):
        row = conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        entities.append(
            {"id": entity_id, "name": str(row["canonical_name"]),
             "definition": str(row["definition"])}
        )
    payload = llm.complete_json(
        SYSTEM,
        """判断候选实体之间是否存在原文明确表达的关系，方向可任选。
返回 {"relation":null,"subject_id":null,"object_id":null,"predicate":"",
"passage_ids":[],"quote":"","reason":""}。relation 有证据时写 true，否则 null；
predicate 是开放谓词。passage_ids 必须从 inputs 中选 1-3 个。
entities=%s
inputs=%s"""
        % (json.dumps(entities, ensure_ascii=False), json.dumps(passages, ensure_ascii=False)),
    )
    if payload.get("relation") is not True:
        return "none", None, str(payload.get("reason", "证据不足"))
    try:
        subject_id = int(payload.get("subject_id"))
        object_id = int(payload.get("object_id"))
    except (TypeError, ValueError):
        return "none", None, "非法端点"
    if {subject_id, object_id} != {left_id, right_id}:
        return "none", None, "端点不属于候选对"
    predicate = str(payload.get("predicate", "")).strip()
    quote = str(payload.get("quote", "")).strip()
    cited = payload.get("passage_ids", [])
    by_id = {item["passage_id"]: item for item in passages}
    if (
        not predicate
        or not quote
        or not isinstance(cited, list)
        or not 1 <= len(cited) <= 3
        or any(str(item) not in by_id for item in cited)
    ):
        return "none", None, "缺少可验证的谓词或 Passage 引用"
    cited_ids = tuple(dict.fromkeys(str(item) for item in cited))
    if not 1 <= len(cited_ids) <= 3:
        return "none", None, "Passage 引用重复"
    names = {item["id"]: item["name"] for item in entities}
    selected = [by_id[item] for item in cited_ids]
    claim = ClaimObservation(
        subject=names[subject_id],
        relation=predicate,
        raw_relation=predicate,
        object=names[object_id],
        model_quote=quote,
        source_text="\n\n".join(item["text"] for item in selected),
        passage_ids=cited_ids,
        location="; ".join(item["location"] for item in selected),
    )
    normalized = vocabulary.resolve_relation(conn, simple_llm, claim)
    claim = replace(
        claim,
        relation=normalized.canonical_name,
        relation_kind=normalized.relation_kind,
        relation_type_id=normalized.relation_type_id,
    )
    observation_id, _ = observations.add_claim_observation(
        conn,
        source_id=source_id,
        chunk_index=-1,
        claim=claim,
        extraction_model=model,
        extraction_prompt_version=EXPANSION_PROMPT_VERSION,
    )
    vocabulary.save_relation_resolution(
        conn, observation_id, predicate, normalized, model=_model_name(simple_llm)
    )
    conn.execute(
        """UPDATE claim_observations SET subject_entity_id=?,object_entity_id=?
           WHERE id=?""",
        (subject_id, object_id, observation_id),
    )
    verdict, reason = validation.judge_claim(llm, claim)
    observations.save_judgment(
        conn, observation_id, validator_model=model, verdict=verdict, reason=reason
    )
    result = observations.materialize(conn, observation_id, validator_model=model)
    if result.get("outcome") != "materialized":
        return "none", observation_id, f"验证未入图: {result.get('outcome')}"
    return "relation", observation_id, reason


def _model_name(llm: JSONLLM) -> str:
    config = getattr(llm, "config", None)
    return str(getattr(config, "model", "") or llm.__class__.__name__)
