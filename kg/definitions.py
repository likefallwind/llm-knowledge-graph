from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable

from .llm import JSONLLM


DEFINITION_PROMPT_VERSION = "entity-definition-observations-2"

SYSTEM_PROMPT = """你是语料约束的知识定义整理器。
只能使用用户提供的 EntityObservation，不得使用模型记忆补充任何事实。
目标是从同一 Entity 的全部观察中形成一个规范、简洁、能回答“它是什么”的定义。
宣传性评价、地位、流行程度、影响力、仅说明用途、仅说明出现位置或仅与其他对象
比较的文字不能进入 definition；即使原文支持，也只能列入 rejected_candidates。
只输出 JSON 对象。"""

USER_PROMPT = """请为下面这个 Entity 合成规范定义。

要求：
1. definition 只回答“它是什么”，采用“上位类别 + 区分性特征”的形式，使用一至两句话。
2. definition 中每个实质性陈述都必须被引用 Observation 的 source_text 直接支持。
3. 不得因为常识上正确就补充语料没有表达的特征。
4. supporting_observations 返回一至五项，每项包含 observation_id、passage_ids、support。
5. passage_ids 必须属于对应 Observation。
6. 地位、流行度、重要性、影响、宣传性形容和应用成绩不得写入 definition；
   rejected_candidates 简要指出没有采用的较弱定义及原因。
7. 如果证据存在局限，写入 limitation，不要猜测。

输出格式：
{{
  "definition": "规范定义",
  "supporting_observations": [
    {{"observation_id": 1, "passage_ids": ["P000001"], "support": "支持内容"}}
  ],
  "rejected_candidates": ["较弱定义及原因"],
  "limitation": "无则为空字符串"
}}

当前 Entity：
{entity}

全部 EntityObservation：
{observations}
"""


def synthesize_pending(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    entity_ids: Iterable[int] | None = None,
    limit: int | None = None,
    min_observations: int = 2,
) -> dict[str, Any]:
    """Synthesize definitions whose complete Observation set has changed."""
    if min_observations < 1:
        raise ValueError("min_observations 必须至少为 1")
    if limit is not None and limit < 0:
        raise ValueError("limit 不能为负数")

    candidates = _candidate_entity_ids(
        conn, entity_ids=entity_ids, min_observations=min_observations
    )
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending: list[tuple[int, list[dict[str, Any]], str]] = []
    model = _model_name(llm)
    for entity_id in candidates:
        observations = _observations(conn, entity_id)
        fingerprint = observation_fingerprint(observations)
        if _cached(conn, entity_id, fingerprint, model):
            skipped.append({"entity_id": entity_id, "reason": "unchanged"})
            continue
        pending.append((entity_id, observations, fingerprint))
    selected = pending if limit is None else pending[:limit]
    for entity_id, observations, fingerprint in selected:
        try:
            processed.append(
                synthesize_entity_definition(
                    conn,
                    llm,
                    entity_id,
                    observations=observations,
                    fingerprint=fingerprint,
                )
            )
        except Exception as exc:
            conn.rollback()
            row = conn.execute(
                "SELECT canonical_name FROM entities WHERE id=?", (entity_id,)
            ).fetchone()
            failures.append(
                {
                    "entity_id": entity_id,
                    "entity": str(row["canonical_name"]) if row else "",
                    "error": str(exc),
                }
            )
    return {
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "remaining": len(pending) - len(selected),
    }


def synthesize_entity_definition(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    entity_id: int,
    *,
    observations: list[dict[str, Any]] | None = None,
    fingerprint: str = "",
) -> dict[str, Any]:
    entity = conn.execute(
        "SELECT id,canonical_name,definition FROM entities WHERE id=?",
        (entity_id,),
    ).fetchone()
    if entity is None:
        raise ValueError(f"Entity 不存在: {entity_id}")
    items = observations if observations is not None else _observations(conn, entity_id)
    if not items:
        raise ValueError(f"Entity #{entity_id} 没有 EntityObservation")
    current_fingerprint = fingerprint or observation_fingerprint(items)
    model = _model_name(llm)
    cached = _cached(conn, entity_id, current_fingerprint, model)
    if cached:
        return {
            "entity_id": entity_id,
            "entity": str(entity["canonical_name"]),
            "definition": str(cached["definition"]),
            "observations": len(items),
            "cached": True,
        }

    result = llm.complete_json(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            entity=json.dumps(dict(entity), ensure_ascii=False),
            observations=json.dumps(items, ensure_ascii=False),
        ),
        validate=lambda payload: _validate_payload(payload, items),
    )
    with conn:
        conn.execute(
            """
            INSERT INTO entity_definition_syntheses
            (entity_id,observation_fingerprint,synthesizer_model,prompt_version,
             definition,supporting_observations,rejected_candidates,limitation)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(entity_id,observation_fingerprint,synthesizer_model,prompt_version)
            DO UPDATE SET definition=excluded.definition,
                          supporting_observations=excluded.supporting_observations,
                          rejected_candidates=excluded.rejected_candidates,
                          limitation=excluded.limitation
            """,
            (
                entity_id,
                current_fingerprint,
                model,
                DEFINITION_PROMPT_VERSION,
                result["definition"],
                json.dumps(result["supporting_observations"], ensure_ascii=False),
                json.dumps(result["rejected_candidates"], ensure_ascii=False),
                result["limitation"],
            ),
        )
        conn.execute(
            """
            UPDATE entities
            SET definition=?,updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (result["definition"], entity_id),
        )
    return {
        "entity_id": entity_id,
        "entity": str(entity["canonical_name"]),
        "definition": result["definition"],
        "observations": len(items),
        "supporting_observations": result["supporting_observations"],
        "cached": False,
    }


def observation_fingerprint(observations: list[dict[str, Any]]) -> str:
    material = [
        {
            "id": int(item["id"]),
            "name": str(item["name"]),
            "definition": str(item["definition"]),
            "source_text": str(item["source_text"]),
            "passage_ids": list(item["passage_ids"]),
        }
        for item in observations
    ]
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_entity_ids(
    conn: sqlite3.Connection,
    *,
    entity_ids: Iterable[int] | None,
    min_observations: int,
) -> list[int]:
    requested = sorted({int(item) for item in entity_ids or ()})
    where = ""
    params: list[Any] = []
    if requested:
        placeholders = ",".join("?" for _ in requested)
        where = f"AND e.id IN ({placeholders})"
        params.extend(requested)
    params.append(min_observations)
    rows = conn.execute(
        f"""
        SELECT e.id
        FROM entities e
        JOIN entity_observations o ON o.entity_id=e.id
        WHERE 1=1 {where}
        GROUP BY e.id
        HAVING COUNT(*)>=?
        ORDER BY e.id
        """,
        params,
    ).fetchall()
    return [int(row["id"]) for row in rows]


def _observations(
    conn: sqlite3.Connection, entity_id: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT o.id,o.source_id,s.name AS source_name,o.chunk_index,o.name,
               o.definition,o.observed_entity_type,o.aliases,o.source_text,
               o.model_quote,o.passage_ids,o.location,o.extraction_model,
               o.extraction_prompt_version,o.resolution_outcome,
               o.resolution_reason
        FROM entity_observations o
        JOIN sources s ON s.id=o.source_id
        WHERE o.entity_id=?
        ORDER BY o.chunk_index,o.id
        """,
        (entity_id,),
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["aliases"] = json.loads(str(item["aliases"]))
        item["passage_ids"] = json.loads(str(item["passage_ids"]))
        result.append(item)
    return result


def _cached(
    conn: sqlite3.Connection,
    entity_id: int,
    fingerprint: str,
    model: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM entity_definition_syntheses
        WHERE entity_id=? AND observation_fingerprint=?
          AND synthesizer_model=? AND prompt_version=?
        ORDER BY id DESC LIMIT 1
        """,
        (entity_id, fingerprint, model, DEFINITION_PROMPT_VERSION),
    ).fetchone()


def _validate_payload(
    payload: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    definition = str(payload.get("definition", "")).strip()
    if len(definition) < 4:
        raise ValueError("definition 缺少实质内容")
    citations = payload.get("supporting_observations")
    if not isinstance(citations, list) or not 1 <= len(citations) <= 5:
        raise ValueError("supporting_observations 必须包含 1 至 5 项")
    by_id = {int(item["id"]): item for item in observations}
    normalized = []
    for raw in citations:
        if not isinstance(raw, dict):
            raise ValueError("supporting_observations 项必须是对象")
        try:
            observation_id = int(raw.get("observation_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("supporting_observations 缺少 observation_id") from exc
        if observation_id not in by_id:
            raise ValueError(f"引用了不属于当前 Entity 的 Observation: {observation_id}")
        passage_ids = raw.get("passage_ids")
        if not isinstance(passage_ids, list) or not passage_ids:
            raise ValueError(f"Observation #{observation_id} 缺少 passage_ids")
        actual = {str(item) for item in by_id[observation_id]["passage_ids"]}
        cited = [str(item).strip() for item in passage_ids if str(item).strip()]
        if not cited or not set(cited).issubset(actual):
            raise ValueError(f"Observation #{observation_id} 的 Passage 引用无效")
        support = str(raw.get("support", "")).strip()
        if not support:
            raise ValueError(f"Observation #{observation_id} 缺少 support")
        normalized.append(
            {
                "observation_id": observation_id,
                "passage_ids": cited,
                "support": support,
            }
        )
    rejected = payload.get("rejected_candidates", [])
    if not isinstance(rejected, list) or not all(
        isinstance(item, str) for item in rejected
    ):
        raise ValueError("rejected_candidates 必须是字符串数组")
    limitation = payload.get("limitation", "")
    if not isinstance(limitation, str):
        raise ValueError("limitation 必须是字符串")
    return {
        "definition": definition,
        "supporting_observations": normalized,
        "rejected_candidates": [item.strip() for item in rejected if item.strip()],
        "limitation": limitation.strip(),
    }


def _model_name(llm: JSONLLM) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", "")
    return str(model or llm.__class__.__name__)
