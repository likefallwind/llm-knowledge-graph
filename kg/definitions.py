from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable

from .llm import JSONLLM


DEFINITION_PROMPT_VERSION = "entity-definition-observations-3-knowledge-assisted"

SYSTEM_PROMPT = """你是语料辅助的知识概念整理器。
这里的 definition 是用于帮助识别 Entity 的规范概念解释，不要求是严格的词典定义。
你可以使用可靠的通用知识理解术语的通常含义，并结合用户提供的全部
EntityObservation 确定当前 Entity 实际指向哪个义项。

EntityObservation 和 source_text 是必要依据：definition 必须与全部原始语料一致；
原文中特有的事实、数字、版本、历史事件和应用结果必须有 Observation 直接支持。
可靠通用知识只用于明确通常含义、上位类别和跨场景稳定特征，不得虚构不确定或有争议的
具体事实，也不得仅因概念首先出现在特定模型、章节或应用场景中，就把该场景写成概念
本身的身份边界。当前 Entity.definition 可能只是首次观察留下的局部解释，不是权威边界。
只输出 JSON 对象。"""

USER_PROMPT = """请为下面这个 Entity 合成规范定义。

要求：
1. definition 使用一至两句话。第一句优先说明“它通常是什么”；第二句可以补充有助于
   识别它的用途、性质、实现方式或典型比较。
2. 可以使用可靠通用知识补全上位类别、通常含义和跨场景稳定特征，但 definition 必须与
   全部 Observation 一致。来自原文的具体事实必须由 source_text 直接支持。
3. 原文中的用途、性质、实现方式、比较关系和应用场景可以进入解释，但不得让一次局部
   使用遮蔽通常含义，或把应用场景误写成概念身份边界。
4. supporting_observations 返回一至五项，每项包含 observation_id、passage_ids、support；
   support 说明该 Observation 如何锚定当前义项，或支持解释中的原文具体事实。
5. passage_ids 必须属于对应 Observation。
6. rejected_candidates 简要记录与语料冲突、明显过窄或会误导身份判断的候选解释；没有则
   返回空数组。
7. 如果语料只提供了局部信息、不同观察存在张力，或解释使用了原文未完整重述的通用知识，
   在 limitation 中如实说明，不要猜测。

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
    min_observations: int = 1,
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
