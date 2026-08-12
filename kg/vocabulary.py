from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from . import store
from .llm import JSONLLM
from .models import CORE_RELATION_KINDS, ClaimObservation, EntityObservation


RELATION_NORMALIZER_VERSION = "open-relation-normalizer-4-direct-projection"
TYPE_NORMALIZER_VERSION = "open-type-normalizer-1"
SYSTEM = """你是开放知识词表的归一裁判，不是知识来源。
只能根据给出的原始标签、Source 证据和已有词表判断是否同义。相近但不相同必须
new 或 uncertain；宁可保留重复，也不要错误合并。只输出 JSON 对象。"""


@dataclass(frozen=True)
class RelationResolution:
    relation_type_id: int | None
    canonical_name: str
    relation_kind: str
    outcome: str
    reason: str
    register_alias: bool = False
    candidates: tuple[int, ...] = ()
    description: str = ""
    projection_statement: str = ""


def _relation_exact(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    normalized = store.normalize_name(name)
    return conn.execute(
        """SELECT r.* FROM relation_types r
           WHERE r.normalized_name=? OR r.id IN (
             SELECT relation_type_id FROM relation_aliases WHERE normalized_name=?
           ) ORDER BY r.id LIMIT 1""",
        (normalized, normalized),
    ).fetchone()


def _relation_candidates(conn: sqlite3.Connection, name: str) -> list[dict]:
    """Return the open relation catalog that has graph evidence.

    Exact canonical/alias matches are included so they can be judged in
    context, but never bypass the judge.  No relation kind receives a special
    slot and lexical similarity is deliberately not used for open predicates.
    """
    normalized = store.normalize_name(name)
    rows = conn.execute(
        """
        SELECT DISTINCT r.id,r.canonical_name,r.relation_kind,r.description
        FROM relation_types r
        LEFT JOIN relation_aliases a ON a.relation_type_id=r.id
        WHERE EXISTS (SELECT 1 FROM claims c WHERE c.relation_type_id=r.id)
           OR r.normalized_name=? OR a.normalized_name=?
        ORDER BY r.id
        """,
        (normalized, normalized),
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "canonical_name": str(row["canonical_name"]),
            "relation_kind": str(row["relation_kind"]),
            "description": str(row["description"]),
        }
        for row in rows
    ]


def resolve_relation(
    conn: sqlite3.Connection, llm: JSONLLM, claim: ClaimObservation
) -> RelationResolution:
    raw = claim.raw_relation or claim.relation
    candidates = _relation_candidates(conn, raw)
    payload = llm.complete_json(
        SYSTEM,
        """归一开放关系谓词。候选是图中已有证据支撑的开放关系，以及名称精确命中的
关系；没有任何固定的“核心关系”享有优先权。只有语义和方向都相同才是 same。
relation_kind 只能是 is_a、part_of、prerequisite_of、other；它只是导航类别，
不能把任意开放关系强塞进前三类。
第一步必须直接把当前 subject→predicate→object 依次口头化，并核对 statement 中真正
承担关系的主语、关系和宾语。若真实主语/宾语其实是端点的参数、输出、组成部分、作者等
第三个对象，不得把它藏进谓词；这种三元组不能忠实投影，返回 non_projectable。条件、
工具和方式可以省略的前提是删去后，subject→predicate→object 命题本身仍然为真。

只有确认可投影后，才判断 same/new/uncertain。new 只是关系类型提案，并不会立即写入
全局词表；canonical_name 必须简洁、可复用、明确表达固定方向，不能夹带当前实体名。

decision 只判断当前 observation 是否能映射到候选关系。register_alias 是另一项独立
判断：只有 raw_relation 脱离当前 subject、object 和上下文后，仍稳定表达完全相同的
语义与方向，才可为 true。当前命题映射为 same，不自动证明 raw_relation 是全局 alias。

返回 {"decision":"same|new|uncertain|non_projectable","candidate_id":null,
"canonical_name":"简洁可复用谓词","relation_kind":"other",
"description":"关系含义和固定方向",
"projection_statement":"用候选关系口头化 subject→object 后得到的命题",
"register_alias":false,"reason":"..."}。
观察=%s
候选=%s"""
        % (
            json.dumps(
                {
                    "subject": claim.subject,
                    "raw_relation": raw,
                    "object": claim.object,
                    "statement": claim.statement_text,
                    "scope": claim.scope_text,
                    "scope_is_restrictive": claim.scope_is_restrictive,
                    "model_quote": claim.model_quote,
                    "source_text": claim.source_text,
                }, ensure_ascii=False,
            ),
            json.dumps(candidates, ensure_ascii=False),
        ),
        validate=_validate_relation_payload,
    )
    decision = _text(payload.get("decision")).lower()
    reason = _text(payload.get("reason"))
    register_alias = payload.get("register_alias") is True
    candidate_ids = tuple(int(item["id"]) for item in candidates)
    projection = _text(payload.get("projection_statement"))
    description = _text(payload.get("description"))
    if decision == "same":
        try:
            selected = int(payload.get("candidate_id"))
        except (TypeError, ValueError):
            selected = -1
        if selected in candidate_ids:
            row = conn.execute("SELECT * FROM relation_types WHERE id=?", (selected,)).fetchone()
            return RelationResolution(
                selected, str(row["canonical_name"]), str(row["relation_kind"]),
                "same", reason, register_alias, candidate_ids,
                str(row["description"]), projection,
            )
        decision = "uncertain"
        reason = reason or "same 返回非法 candidate_id"
    if decision in {"uncertain", "non_projectable"}:
        # An uncertain judgment has not established a reusable predicate
        # identity.  Keep the grounded observation pending instead of creating
        # a global RelationType or poisoning the alias table.
        return RelationResolution(
            None, raw, "other", decision, reason, False, candidate_ids,
            description, projection,
        )
    canonical = _text(payload.get("canonical_name"))
    kind = _text(payload.get("relation_kind")) or "other"
    if kind not in CORE_RELATION_KINDS:
        kind = "other"
    return RelationResolution(
        None, canonical, kind, decision, reason,
        register_alias, candidate_ids, description, projection,
    )


def save_relation_resolution(
    conn: sqlite3.Connection,
    observation_id: int,
    raw_relation: str,
    result: RelationResolution,
    *,
    model: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO relation_resolution_attempts
           (observation_id,raw_relation,outcome,candidate_relation_ids,
            matched_relation_type_id,canonical_name,relation_kind,description,
            projection_statement,register_alias,normalizer_model,prompt_version,reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            observation_id, raw_relation, result.outcome,
            json.dumps(result.candidates), result.relation_type_id,
            result.canonical_name, result.relation_kind, result.description,
            result.projection_statement, int(result.register_alias), model,
            RELATION_NORMALIZER_VERSION, result.reason,
        ),
    )


def finalize_relation_resolution(
    conn: sqlite3.Connection,
    observation_id: int,
    raw_relation: str,
    result: RelationResolution,
    *,
    model: str,
) -> int | None:
    """Promote a same/new proposal only after the final judge supports it."""
    if result.outcome not in {"same", "new"}:
        return None
    relation_id = result.relation_type_id
    outcome = result.outcome
    if relation_id is None:
        collision = _relation_exact(conn, result.canonical_name)
        if collision is not None:
            relation_id = int(collision["id"])
            outcome = "same"
        else:
            cursor = conn.execute(
                """INSERT INTO relation_types
                   (canonical_name,normalized_name,relation_kind,description)
                   VALUES (?,?,?,?)""",
                (
                    result.canonical_name,
                    store.normalize_name(result.canonical_name),
                    result.relation_kind,
                    result.description,
                ),
            )
            relation_id = int(cursor.lastrowid)
    row = conn.execute(
        "SELECT canonical_name,relation_kind FROM relation_types WHERE id=?",
        (relation_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        result.register_alias
        and store.normalize_name(raw_relation)
        != store.normalize_name(str(row["canonical_name"]))
    ):
        conn.execute(
            """INSERT OR IGNORE INTO relation_aliases
               (relation_type_id,name,normalized_name) VALUES (?,?,?)""",
            (relation_id, raw_relation, store.normalize_name(raw_relation)),
        )
    conn.execute(
        """UPDATE claim_observations
           SET relation_type_id=?,relation=?,relation_kind=?,
               updated_at=CURRENT_TIMESTAMP WHERE id=?""",
        (
            relation_id,
            str(row["canonical_name"]),
            str(row["relation_kind"]),
            observation_id,
        ),
    )
    conn.execute(
        """INSERT OR REPLACE INTO relation_resolutions
           (observation_id,raw_relation,relation_type_id,outcome,
            candidate_relation_ids,normalizer_model,prompt_version,reason)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            observation_id, raw_relation, relation_id, outcome,
            json.dumps(result.candidates), model, RELATION_NORMALIZER_VERSION,
            result.reason,
        ),
    )
    return relation_id


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_relation_payload(payload: dict) -> dict:
    decision = _text(payload.get("decision")).lower()
    if decision not in {"same", "new", "uncertain", "non_projectable"}:
        raise ValueError("relation normalizer decision 非法或缺失")
    if decision == "new":
        canonical = _text(payload.get("canonical_name"))
        if store.normalize_name(canonical) in {
            "",
            "none",
            "null",
            "n/a",
            "unknown",
            "未知",
            "无",
        }:
            raise ValueError("new relation 缺少有效 canonical_name")
    if decision in {"same", "new"} and not _text(
        payload.get("projection_statement")
    ):
        raise ValueError("relation normalizer 缺少 projection_statement")
    # Missing means “do not promote”. This is safe for interrupted runs and old
    # test fixtures while the new prompt requires an explicit boolean.
    if not isinstance(payload.get("register_alias"), bool):
        payload = dict(payload)
        payload["register_alias"] = False
    return payload


def _type_exact(conn: sqlite3.Connection, label: str) -> sqlite3.Row | None:
    normalized = store.normalize_name(label)
    return conn.execute(
        """SELECT t.* FROM entity_type_vocab t
           WHERE t.normalized_name=? OR t.id IN (
             SELECT type_id FROM entity_type_aliases WHERE normalized_name=?
           ) ORDER BY t.id LIMIT 1""",
        (normalized, normalized),
    ).fetchone()


def resolve_observation_types(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    observation_id: int,
    observation: EntityObservation,
    *,
    model: str,
) -> None:
    for raw in observation.type_labels:
        existing = conn.execute(
            """SELECT 1 FROM entity_observation_types
               WHERE observation_id=? AND raw_label=? AND normalizer_model=?
                 AND prompt_version=?""",
            (observation_id, raw, model, TYPE_NORMALIZER_VERSION),
        ).fetchone()
        if existing:
            continue
        exact = _type_exact(conn, raw)
        if exact:
            type_id, outcome, reason = int(exact["id"]), "same", "exact type/alias"
        else:
            candidates = [dict(row) for row in conn.execute(
                "SELECT id,canonical_name,description FROM entity_type_vocab ORDER BY id LIMIT 80"
            )]
            payload = llm.complete_json(
                SYSTEM,
                """归一开放实体类型标签。返回
{"decision":"same|new|uncertain","candidate_id":null,
 "canonical_name":"简洁类别词","description":"类别含义","reason":"..."}。
实体观察=%s
候选=%s"""
                % (
                    json.dumps({"name": observation.name, "definition": observation.definition,
                                "raw_type": raw, "source_text": observation.source_text},
                               ensure_ascii=False),
                    json.dumps(candidates, ensure_ascii=False),
                ),
            )
            decision = str(payload.get("decision", "uncertain")).strip().lower()
            ids = {int(item["id"]) for item in candidates}
            try:
                selected = int(payload.get("candidate_id"))
            except (TypeError, ValueError):
                selected = -1
            if decision == "same" and selected in ids:
                type_id, outcome = selected, "same"
                reason = str(payload.get("reason", ""))
                conn.execute(
                    "INSERT OR IGNORE INTO entity_type_aliases(type_id,name,normalized_name) VALUES (?,?,?)",
                    (type_id, raw, store.normalize_name(raw)),
                )
            else:
                outcome = decision if decision in {"new", "uncertain"} else "uncertain"
                canonical = str(payload.get("canonical_name", "")).strip() or raw
                collision = _type_exact(conn, canonical)
                if collision:
                    type_id, outcome = int(collision["id"]), "same"
                else:
                    cursor = conn.execute(
                        "INSERT INTO entity_type_vocab(canonical_name,normalized_name,description) VALUES (?,?,?)",
                        (canonical, store.normalize_name(canonical),
                         str(payload.get("description", "")).strip()),
                    )
                    type_id = int(cursor.lastrowid)
                reason = str(payload.get("reason", ""))
        conn.execute(
            """INSERT INTO entity_observation_types
               (observation_id,raw_label,type_id,outcome,normalizer_model,prompt_version,reason)
               VALUES (?,?,?,?,?,?,?)""",
            (observation_id, raw, type_id, outcome, model, TYPE_NORMALIZER_VERSION, reason),
        )
