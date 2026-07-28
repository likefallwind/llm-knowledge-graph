from __future__ import annotations

import json
import sqlite3
from difflib import SequenceMatcher
from typing import Any

from . import store
from .llm import JSONLLM
from .models import EntityObservation, Resolution


RESOLUTION_SYSTEM = """你是实体身份裁判，不是知识来源。
只能根据给出的语料观察与候选实体判断身份，禁止补充外部知识。
宁可 uncertain，也不要错误合并。只输出 JSON 对象。"""


def candidate_entities(
    conn: sqlite3.Connection,
    name: str,
    *,
    limit: int = 5,
    threshold: float = 0.35,
    exclude_id: int | None = None,
) -> list[dict[str, Any]]:
    query = store.normalize_name(name)
    candidates: list[dict[str, Any]] = []
    for row in store.list_entities(conn):
        entity_id = int(row["id"])
        if entity_id == exclude_id:
            continue
        names = [str(row["canonical_name"]), *store.aliases_for(conn, entity_id)]
        score = max(
            SequenceMatcher(None, query, store.normalize_name(value)).ratio()
            for value in names
        )
        compact_query = query.replace(" ", "")
        compact_names = [store.normalize_name(value).replace(" ", "") for value in names]
        if any(
            compact_query in value or value in compact_query for value in compact_names
        ):
            score = max(score, 0.55)
        if score >= threshold:
            candidates.append(
                {
                    "id": entity_id,
                    "canonical_name": str(row["canonical_name"]),
                    "aliases": names[1:],
                    "definition": str(row["definition"]),
                    "entity_type": str(row["entity_type"]),
                    "evidence": store.evidence_for_entity(conn, entity_id),
                    "score": round(score, 4),
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["id"])))
    return candidates[:limit]


def resolve_observation(
    conn: sqlite3.Connection, llm: JSONLLM, observation: EntityObservation
) -> Resolution:
    exact = store.exact_entity_ids(conn, observation.name)
    if len(exact) == 1:
        entity_id = exact[0]
        for alias in observation.aliases:
            store.add_alias(conn, entity_id, alias)
        return Resolution(entity_id=entity_id, outcome="same", reason="exact name/alias")

    candidates = candidate_entities(conn, observation.name)
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """判断新观察与候选是否指向同一个知识对象。
返回：
{
  "decision": "same | new | uncertain",
  "candidate_id": 仅 same 时填写候选 id，否则为 null,
  "canonical_name": new/uncertain 时给出规范正式名称；只能规范化观察名，不能创造新知识,
  "reason": "简短理由"
}

新观察：
%s

候选：
%s"""
        % (
            json.dumps(
                {
                    "name": observation.name,
                    "definition": observation.definition,
                    "entity_type": observation.entity_type,
                    "model_quote": observation.model_quote,
                    "source_text": observation.source_text,
                    "passage_ids": observation.passage_ids,
                },
                ensure_ascii=False,
            ),
            json.dumps(candidates, ensure_ascii=False),
        ),
    )
    decision = str(payload.get("decision", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    candidate_ids = tuple(int(item["id"]) for item in candidates)
    if decision == "same":
        try:
            selected = int(payload.get("candidate_id"))
        except (TypeError, ValueError):
            selected = -1
        if selected in candidate_ids:
            for alias in (observation.name, *observation.aliases):
                store.add_alias(conn, selected, alias)
            return Resolution(
                entity_id=selected,
                outcome="same",
                reason=reason,
                candidates=candidate_ids,
            )
        decision = "uncertain"
        reason = reason or "same 返回了非法 candidate_id"

    canonical = str(payload.get("canonical_name", "")).strip()
    if decision not in {"new", "uncertain"}:
        decision = "uncertain"
        reason = reason or "resolver 返回了非法 decision"
    # A `new` or `uncertain` answer must never collapse into an existing row
    # merely because the model suggested an already-used canonical spelling.
    if canonical and store.exact_entity_ids(conn, canonical):
        canonical = observation.name
    entity_id = store.create_entity(
        conn, observation, canonical_name=canonical or observation.name
    )
    return Resolution(
        entity_id=entity_id,
        outcome=decision,
        reason=reason,
        candidates=candidate_ids,
    )


def reconcile(
    conn: sqlite3.Connection, llm: JSONLLM, *, limit: int = 20
) -> dict[str, Any]:
    """Revisit similar existing entities; merge only explicit `same` decisions."""
    pairs: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    entities = [_entity_context(conn, int(row["id"])) for row in store.list_entities(conn)]
    by_id = {int(item["id"]): item for item in entities}
    for entity in entities:
        source_id = int(entity["id"])
        for candidate in candidate_entities(
            conn,
            str(entity["canonical_name"]),
            exclude_id=source_id,
            threshold=0.55,
        ):
            pair = tuple(sorted((source_id, int(candidate["id"]))))
            if pair not in pairs:
                pairs[pair] = (by_id[pair[0]], by_id[pair[1]])
    examined = 0
    merged: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for (left_id, right_id), (left, right) in list(pairs.items())[:limit]:
        if not store.get_entity(conn, left_id) or not store.get_entity(conn, right_id):
            continue
        examined += 1
        payload = llm.complete_json(
            RESOLUTION_SYSTEM,
            """两个已有实体是否指向同一个知识对象？
返回 {"decision":"same|new|uncertain","canonical_name":"若 same 给出更规范名称","reason":"..."}。
实体 A：%s
实体 B：%s"""
            % (
                json.dumps(left, ensure_ascii=False),
                json.dumps(right, ensure_ascii=False),
            ),
        )
        decision = str(payload.get("decision", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if decision == "same":
            target_id, source_id = min(left_id, right_id), max(left_id, right_id)
            store.merge_entities(conn, source_id, target_id)
            store.set_canonical_name(
                conn, target_id, str(payload.get("canonical_name", ""))
            )
            merged.append(
                {"source_id": source_id, "target_id": target_id, "reason": reason}
            )
        elif decision == "uncertain":
            uncertain.append({"ids": [left_id, right_id], "reason": reason})
    return {"examined": examined, "merged": merged, "uncertain": uncertain}


def _entity_context(
    conn: sqlite3.Connection, entity_id: int
) -> dict[str, Any]:
    row = store.get_entity(conn, entity_id)
    if not row:
        raise ValueError(f"实体不存在: {entity_id}")
    return {
        "id": entity_id,
        "canonical_name": str(row["canonical_name"]),
        "aliases": store.aliases_for(conn, entity_id),
        "definition": str(row["definition"]),
        "entity_type": str(row["entity_type"]),
        "evidence": store.evidence_for_entity(conn, entity_id),
    }
