from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher

from . import store
from .llm import JSONLLM
from .models import CORE_RELATION_KINDS, ClaimObservation, EntityObservation


RELATION_NORMALIZER_VERSION = "open-relation-normalizer-3-contextual-alias"
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


def _relation_exact(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    normalized = store.normalize_name(name)
    return conn.execute(
        """SELECT r.* FROM relation_types r
           WHERE r.normalized_name=? OR r.id IN (
             SELECT relation_type_id FROM relation_aliases WHERE normalized_name=?
           ) ORDER BY r.id LIMIT 1""",
        (normalized, normalized),
    ).fetchone()


def _relation_candidates(conn: sqlite3.Connection, name: str, limit: int = 8) -> list[dict]:
    query = store.normalize_name(name)
    values = []
    for row in conn.execute("SELECT * FROM relation_types ORDER BY id"):
        score = SequenceMatcher(None, query, str(row["normalized_name"])).ratio()
        values.append(
            {
                "id": int(row["id"]),
                "canonical_name": str(row["canonical_name"]),
                "relation_kind": str(row["relation_kind"]),
                "description": str(row["description"]),
                "score": round(score, 4),
            }
        )
    values.sort(key=lambda item: (-item["score"], item["id"]))
    return values[:limit]


def resolve_relation(
    conn: sqlite3.Connection, llm: JSONLLM, claim: ClaimObservation
) -> RelationResolution:
    raw = claim.raw_relation or claim.relation
    exact = _relation_exact(conn, raw)
    if exact:
        return RelationResolution(
            int(exact["id"]), str(exact["canonical_name"]),
            str(exact["relation_kind"]), "same", "exact relation/alias",
        )
    candidates = _relation_candidates(conn, raw)
    payload = llm.complete_json(
        SYSTEM,
        """归一开放关系谓词。只有语义和方向都相同才是 same。
relation_kind 只能是 is_a、part_of、prerequisite_of、other；它只是导航类别，
不能把任意开放关系强塞进前三类。
先把候选关系按其 description 和方向口头化为“subject 通过候选谓词指向 object”的
完整命题，再与 observation.statement 逐项比较。statement 中仅仅出现了两个端点，
不等于它们真的构成该有向关系；若实际关系涉及“subject 的参数/输出/组成部分”等第三个
对象，不得把这个第三个对象偷换成 subject。

decision 只判断当前 observation 是否能映射到候选关系。register_alias 是另一项独立
判断：只有 raw_relation 脱离当前 subject、object 和上下文后，仍稳定表达完全相同的
语义与方向，才可为 true。当前命题映射为 same，不自动证明 raw_relation 是全局 alias。

返回 {"decision":"same|new|uncertain","candidate_id":null,
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
    if decision == "same":
        try:
            selected = int(payload.get("candidate_id"))
        except (TypeError, ValueError):
            selected = -1
        if selected in candidate_ids:
            row = conn.execute("SELECT * FROM relation_types WHERE id=?", (selected,)).fetchone()
            if register_alias:
                conn.execute(
                    "INSERT OR IGNORE INTO relation_aliases(relation_type_id,name,normalized_name) VALUES (?,?,?)",
                    (selected, raw, store.normalize_name(raw)),
                )
            return RelationResolution(
                selected, str(row["canonical_name"]), str(row["relation_kind"]),
                "same", reason, register_alias, candidate_ids,
            )
        decision = "uncertain"
        reason = reason or "same 返回非法 candidate_id"
    if decision == "uncertain":
        # An uncertain judgment has not established a reusable predicate
        # identity.  Keep the grounded observation pending instead of creating
        # a global RelationType or poisoning the alias table.
        return RelationResolution(
            None, raw, "other", "uncertain", reason, False, candidate_ids
        )
    canonical = _text(payload.get("canonical_name"))
    kind = _text(payload.get("relation_kind")) or "other"
    if kind not in CORE_RELATION_KINDS:
        kind = "other"
    collision = _relation_exact(conn, canonical)
    if collision:
        canonical = raw
        collision = _relation_exact(conn, canonical)
    if collision:
        return RelationResolution(
            int(collision["id"]), str(collision["canonical_name"]),
            str(collision["relation_kind"]), "same", "canonical collision",
            False, candidate_ids,
        )
    cursor = conn.execute(
        "INSERT INTO relation_types(canonical_name,normalized_name,relation_kind,description) VALUES (?,?,?,?)",
        (canonical, store.normalize_name(canonical), kind,
         _text(payload.get("description"))),
    )
    relation_id = int(cursor.lastrowid)
    if (
        register_alias
        and store.normalize_name(raw) != store.normalize_name(canonical)
    ):
        conn.execute(
            "INSERT OR IGNORE INTO relation_aliases(relation_type_id,name,normalized_name) VALUES (?,?,?)",
            (relation_id, raw, store.normalize_name(raw)),
        )
    return RelationResolution(
        relation_id, canonical, kind, decision, reason,
        register_alias, candidate_ids,
    )


def save_relation_resolution(
    conn: sqlite3.Connection,
    observation_id: int,
    raw_relation: str,
    result: RelationResolution,
    *,
    model: str,
) -> None:
    if result.relation_type_id is None:
        return
    conn.execute(
        """INSERT OR IGNORE INTO relation_resolutions
           (observation_id,raw_relation,relation_type_id,outcome,
            candidate_relation_ids,normalizer_model,prompt_version,reason)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            observation_id, raw_relation, result.relation_type_id, result.outcome,
            json.dumps(result.candidates), model, RELATION_NORMALIZER_VERSION,
            result.reason,
        ),
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_relation_payload(payload: dict) -> dict:
    decision = _text(payload.get("decision")).lower()
    if decision not in {"same", "new", "uncertain"}:
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
