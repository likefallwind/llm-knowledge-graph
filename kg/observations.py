from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from typing import Any, Iterable

from . import extraction, ontology, resolution, store, structure, validation
from .llm import JSONLLM
from .models import ClaimObservation, EntityObservation, Resolution


PROMOTION_REVIEW_VERSION = "endpoint-promotion-1"
PROMOTION_SYSTEM = """你是待定实体审核器，不是知识来源。
只能使用给出的 Source 原文判断这些反复出现的端点是否稳定指向一个可独立学习的知识对象。
禁止用模型记忆补充定义；宁可 uncertain，也不要制造空壳实体或错误别名。只输出 JSON 对象。"""


def entity_observation_key(
    *,
    source_id: int,
    chunk_index: int,
    observation: EntityObservation,
    extraction_model: str,
    extraction_prompt_version: str,
) -> str:
    payload = [
        source_id,
        chunk_index,
        store.reference_key(observation.name),
        observation.definition,
        observation.entity_type,
        list(observation.type_labels),
        list(observation.aliases),
        observation.source_text,
        observation.model_quote,
        list(observation.passage_ids),
        extraction_model,
        extraction_prompt_version,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def add_entity_observation(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    chunk_index: int,
    observation: EntityObservation,
    extraction_model: str,
    extraction_prompt_version: str = extraction.EXTRACTION_PROMPT_VERSION,
) -> tuple[int, bool]:
    """Persist one grounded Entity sighting before identity resolution."""
    key = entity_observation_key(
        source_id=source_id,
        chunk_index=chunk_index,
        observation=observation,
        extraction_model=extraction_model,
        extraction_prompt_version=extraction_prompt_version,
    )
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO entity_observations
        (observation_key,source_id,section_id,chunk_index,name,reference_key,definition,
         observed_entity_type,raw_type_labels,aliases,source_text,model_quote,passage_ids,
         passage_version,location,extraction_model,extraction_prompt_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            key,
            source_id,
            structure.section_id_for_passages(
                conn, source_id, observation.passage_ids
            ),
            chunk_index,
            observation.name,
            store.reference_key(observation.name),
            observation.definition,
            observation.entity_type,
            json.dumps(list(observation.type_labels), ensure_ascii=False),
            json.dumps(list(observation.aliases), ensure_ascii=False),
            observation.source_text,
            observation.model_quote,
            json.dumps(list(observation.passage_ids), ensure_ascii=False),
            extraction.PASSAGE_VERSION,
            observation.location,
            extraction_model,
            extraction_prompt_version,
        ),
    )
    row = conn.execute(
        "SELECT id FROM entity_observations WHERE observation_key=?", (key,)
    ).fetchone()
    if not row:
        raise RuntimeError("EntityObservation 保存失败")
    return int(row["id"]), cursor.rowcount > 0


def save_entity_resolution(
    conn: sqlite3.Connection,
    observation_id: int,
    resolution_result: Resolution,
    *,
    resolver_model: str,
    resolver_prompt_version: str = resolution.RESOLUTION_PROMPT_VERSION,
) -> None:
    conn.execute(
        """
        UPDATE entity_observations
        SET entity_id=?,resolution_outcome=?,resolution_reason=?,
            candidate_entity_ids=?,resolver_model=?,resolver_prompt_version=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            resolution_result.entity_id,
            resolution_result.outcome,
            resolution_result.reason,
            json.dumps(list(resolution_result.candidates)),
            resolver_model,
            resolver_prompt_version,
            observation_id,
        ),
    )


def observation_key(
    *,
    source_id: int,
    chunk_index: int,
    claim: ClaimObservation,
    extraction_model: str,
    extraction_prompt_version: str,
) -> str:
    payload = [
        source_id,
        chunk_index,
        store.reference_key(claim.subject),
        claim.relation,
        claim.raw_relation,
        claim.relation_kind,
        store.reference_key(claim.object),
        claim.polarity,
        claim.statement_text,
        claim.scope_text,
        claim.scope_is_restrictive,
        claim.source_text,
        claim.model_quote,
        list(claim.passage_ids),
        extraction_model,
        extraction_prompt_version,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def add_claim_observation(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    chunk_index: int,
    claim: ClaimObservation,
    extraction_model: str,
    extraction_prompt_version: str = extraction.EXTRACTION_PROMPT_VERSION,
) -> tuple[int, bool]:
    relation_type_id = claim.relation_type_id
    relation_kind = claim.relation_kind
    relation_name = claim.relation
    if relation_type_id is None:
        relation_row = conn.execute(
            """SELECT id,canonical_name,relation_kind FROM relation_types
               WHERE normalized_name=?""",
            (store.normalize_name(claim.relation),),
        ).fetchone()
        if relation_row:
            relation_type_id = int(relation_row["id"])
            relation_name = str(relation_row["canonical_name"])
            relation_kind = str(relation_row["relation_kind"])
    key = observation_key(
        source_id=source_id,
        chunk_index=chunk_index,
        claim=claim,
        extraction_model=extraction_model,
        extraction_prompt_version=extraction_prompt_version,
    )
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO claim_observations
        (observation_key,source_id,section_id,chunk_index,subject_name,
         subject_reference_key,raw_relation,relation,relation_type_id,relation_kind,
         object_name,object_reference_key,
         polarity,statement_text,scope_text,scope_is_restrictive,
         source_text,model_quote,passage_ids,passage_version,location,
         extraction_model,extraction_prompt_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            key,
            source_id,
            structure.section_id_for_passages(conn, source_id, claim.passage_ids),
            chunk_index,
            claim.subject,
            store.reference_key(claim.subject),
            claim.raw_relation or claim.relation,
            relation_name,
            relation_type_id,
            relation_kind,
            claim.object,
            store.reference_key(claim.object),
            claim.polarity,
            claim.statement_text,
            claim.scope_text,
            int(claim.scope_is_restrictive),
            claim.source_text,
            claim.model_quote,
            json.dumps(list(claim.passage_ids), ensure_ascii=False),
            extraction.PASSAGE_VERSION,
            claim.location,
            extraction_model,
            extraction_prompt_version,
        ),
    )
    row = conn.execute(
        "SELECT id FROM claim_observations WHERE observation_key=?", (key,)
    ).fetchone()
    if not row:
        raise RuntimeError("ClaimObservation 保存失败")
    return int(row["id"]), cursor.rowcount > 0


def get_observation(
    conn: sqlite3.Connection, observation_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM claim_observations WHERE id=?", (observation_id,)
    ).fetchone()


def as_claim(conn: sqlite3.Connection, row: sqlite3.Row) -> ClaimObservation:
    subject = str(row["subject_name"])
    object_name = str(row["object_name"])
    if row["subject_entity_id"] is not None:
        entity = store.get_entity(conn, int(row["subject_entity_id"]))
        if entity is not None:
            subject = str(entity["canonical_name"])
    if row["object_entity_id"] is not None:
        entity = store.get_entity(conn, int(row["object_entity_id"]))
        if entity is not None:
            object_name = str(entity["canonical_name"])
    return ClaimObservation(
        subject=subject,
        relation=str(row["relation"]),
        object=object_name,
        model_quote=str(row["model_quote"]),
        source_text=str(row["source_text"]),
        passage_ids=tuple(json.loads(str(row["passage_ids"]))),
        location=str(row["location"]),
        polarity=str(row["polarity"]),
        raw_relation=str(row["raw_relation"]),
        relation_kind=str(row["relation_kind"]),
        relation_type_id=(
            int(row["relation_type_id"])
            if row["relation_type_id"] is not None
            else None
        ),
        statement_text=str(row["statement_text"]),
        scope_text=str(row["scope_text"]),
        scope_is_restrictive=bool(row["scope_is_restrictive"]),
        normalized_statement=str(row["normalized_statement"]),
    )


def resolve_endpoint_ids(
    conn: sqlite3.Connection,
    observation_ids: Iterable[int] | None = None,
    *,
    local: dict[str, int] | None = None,
) -> int:
    ids = list(observation_ids or ())
    if observation_ids is not None and not ids:
        return 0
    where = "WHERE id IN (%s)" % ",".join("?" for _ in ids) if ids else ""
    rows = conn.execute(
        f"SELECT * FROM claim_observations {where} ORDER BY id", ids
    ).fetchall()
    changed = 0
    local = local or {}
    for row in rows:
        updates: dict[str, int] = {}
        for side in ("subject", "object"):
            if row[f"{side}_entity_id"] is not None:
                continue
            key = str(row[f"{side}_reference_key"])
            entity_id = local.get(key)
            if entity_id is None:
                matches = store.reference_entity_ids(
                    conn, str(row[f"{side}_name"])
                )
                if len(matches) == 1:
                    entity_id = matches[0]
            if entity_id is not None:
                updates[f"{side}_entity_id"] = entity_id
        if not updates:
            continue
        assignments = ",".join(f"{name}=?" for name in updates)
        conn.execute(
            f"UPDATE claim_observations SET {assignments},"
            "updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (*updates.values(), int(row["id"])),
        )
        changed += len(updates)
    return changed


def _replace_endpoint(text: str, raw_name: str, canonical_name: str) -> str:
    if not text or not raw_name or raw_name == canonical_name:
        return text
    # Canonical names can contain the observed form, for example
    # ``随机梯度下降（SGD）`` contains ``随机梯度下降``.  Preserve already
    # canonical occurrences so repeated replay/materialization stays idempotent.
    return canonical_name.join(
        part.replace(raw_name, canonical_name)
        for part in text.split(canonical_name)
    )


def prepare_assertions(
    conn: sqlite3.Connection,
    observation_ids: Iterable[int] | None = None,
) -> int:
    """Build the exact proposition that the final judge will evaluate.

    Extraction preserves a complete source-grounded statement.  This step runs
    only after both endpoints and the relation have canonical identities, then
    substitutes canonical endpoint labels and fingerprints the final meaning.
    A changed endpoint/relation therefore invalidates an older cached judgment.
    """
    ids = list(observation_ids or ())
    if observation_ids is not None and not ids:
        return 0
    where = "WHERE o.id IN (%s)" % ",".join("?" for _ in ids) if ids else ""
    rows = conn.execute(
        f"""
        SELECT o.*,s.canonical_name AS canonical_subject,
               t.canonical_name AS canonical_object
        FROM claim_observations o
        LEFT JOIN entities s ON s.id=o.subject_entity_id
        LEFT JOIN entities t ON t.id=o.object_entity_id
        {where}
        ORDER BY o.id
        """,
        ids,
    ).fetchall()
    changed = 0
    for row in rows:
        if (
            row["subject_entity_id"] is None
            or row["object_entity_id"] is None
            or row["relation_type_id"] is None
        ):
            continue
        subject = str(row["canonical_subject"])
        object_name = str(row["canonical_object"])
        statement = str(row["statement_text"]).strip()
        scope = str(row["scope_text"]).strip()
        statement = _replace_endpoint(
            statement, str(row["subject_name"]), subject
        )
        statement = _replace_endpoint(
            statement, str(row["object_name"]), object_name
        )
        scope = _replace_endpoint(scope, str(row["subject_name"]), subject)
        scope = _replace_endpoint(scope, str(row["object_name"]), object_name)
        fingerprint_payload = {
            "subject_id": int(row["subject_entity_id"]),
            "relation_type_id": int(row["relation_type_id"]),
            "object_id": int(row["object_entity_id"]),
            "statement": statement,
            "scope": scope,
            "scope_is_restrictive": bool(row["scope_is_restrictive"]),
            "polarity": str(row["polarity"]),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            statement == str(row["normalized_statement"])
            and fingerprint == str(row["assertion_fingerprint"])
        ):
            continue
        conn.execute(
            """
            UPDATE claim_observations
            SET normalized_statement=?,scope_text=?,assertion_fingerprint=?,
                claim_id=NULL,assertion_id=NULL,
                materialization_error='',updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (statement, scope, fingerprint, int(row["id"])),
        )
        changed += 1
    return changed


def current_judgment(
    conn: sqlite3.Connection,
    observation_id: int,
    *,
    validator_model: str,
    validator_prompt_version: str = validation.VALIDATION_PROMPT_VERSION,
) -> sqlite3.Row | None:
    row = get_observation(conn, observation_id)
    if row is None or not str(row["assertion_fingerprint"]):
        return None
    return conn.execute(
        """
        SELECT * FROM claim_observation_judgments
        WHERE observation_id=? AND validator_model=?
          AND validator_prompt_version=? AND assertion_fingerprint=?
        """,
        (
            observation_id,
            validator_model,
            validator_prompt_version,
            str(row["assertion_fingerprint"]),
        ),
    ).fetchone()


def latest_judgment(
    conn: sqlite3.Connection, observation_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM claim_observation_judgments
        WHERE observation_id=? ORDER BY id DESC LIMIT 1
        """,
        (observation_id,),
    ).fetchone()


def save_judgment(
    conn: sqlite3.Connection,
    observation_id: int,
    *,
    validator_model: str,
    verdict: str,
    reason: str,
    validator_prompt_version: str = validation.VALIDATION_PROMPT_VERSION,
) -> None:
    row = get_observation(conn, observation_id)
    fingerprint = (
        str(row["assertion_fingerprint"])
        if row is not None and str(row["assertion_fingerprint"])
        else f"unresolved:{row['observation_key'] if row is not None else observation_id}"
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO claim_observation_judgments
        (observation_id,validator_model,validator_prompt_version,
         assertion_fingerprint,verdict,reason)
        VALUES (?,?,?,?,?,?)
        """,
        (
            observation_id,
            validator_model,
            validator_prompt_version,
            fingerprint,
            verdict,
            reason,
        ),
    )


def materialize(
    conn: sqlite3.Connection,
    observation_id: int,
    *,
    validator_model: str,
    validator_prompt_version: str = validation.VALIDATION_PROMPT_VERSION,
) -> dict[str, Any]:
    row = get_observation(conn, observation_id)
    if not row:
        return {"outcome": "missing"}
    if row["claim_id"] is not None and row["assertion_id"] is not None:
        existing_assertion = conn.execute(
            "SELECT id FROM assertions WHERE id=?", (int(row["assertion_id"]),)
        ).fetchone()
        if existing_assertion:
            return {
                "outcome": "already_materialized",
                "claim_id": int(row["claim_id"]),
                "assertion_id": int(row["assertion_id"]),
            }
    subject_id = row["subject_entity_id"]
    object_id = row["object_entity_id"]
    if subject_id is None or object_id is None:
        return {"outcome": "pending_endpoint"}
    if row["relation_type_id"] is None:
        return {"outcome": "pending_relation"}
    prepare_assertions(conn, [observation_id])
    row = get_observation(conn, observation_id)
    if row is None or not str(row["assertion_fingerprint"]):
        return {"outcome": "pending_assertion"}
    judgment = current_judgment(
        conn,
        observation_id,
        validator_model=validator_model,
        validator_prompt_version=validator_prompt_version,
    )
    if not judgment:
        return {"outcome": "pending_judgment"}
    expected = "supports" if row["polarity"] == "support" else "contradicts"
    if str(judgment["verdict"]) != expected:
        return {
            "outcome": "not_supported",
            "verdict": str(judgment["verdict"]),
            "reason": str(judgment["reason"]),
        }
    relation_type_id = row["relation_type_id"]
    relation_name = str(row["relation"])
    relation_kind = str(row["relation_kind"])
    if relation_type_id is None:
        relation_row = conn.execute(
            """SELECT id,canonical_name,relation_kind FROM relation_types
               WHERE normalized_name=?""",
            (store.normalize_name(str(row["relation"])),),
        ).fetchone()
        if not relation_row:
            return {"outcome": "pending_relation"}
        relation_type_id = int(relation_row["id"])
        relation_name = str(relation_row["canonical_name"])
        relation_kind = str(relation_row["relation_kind"])
        conn.execute(
            """UPDATE claim_observations
               SET relation_type_id=?,relation=?,relation_kind=? WHERE id=?""",
            (
                relation_type_id,
                str(relation_row["canonical_name"]),
                str(relation_row["relation_kind"]),
                observation_id,
            ),
        )
    existing = store.find_claim(
        conn,
        int(subject_id),
        relation_name,
        int(object_id),
        relation_type_id=int(relation_type_id),
    )
    if row["polarity"] == "oppose" and existing is None:
        error = "反对证据对应的 Claim 尚不存在"
        _save_materialization(conn, observation_id, claim_id=None, error=error)
        return {"outcome": "blocked", "error": error}
    if existing is None:
        claim_id, created, error = store.upsert_claim(
            conn,
            int(subject_id),
            relation_name,
            int(object_id),
            relation_type_id=int(relation_type_id),
            relation_kind=relation_kind,
        )
        if claim_id is None:
            _save_materialization(conn, observation_id, claim_id=None, error=error)
            return {"outcome": "blocked", "error": error}
    else:
        claim_id, created = int(existing["id"]), False

    assertion_id, assertion_created = store.upsert_assertion(
        conn,
        claim_id,
        str(row["normalized_statement"]),
        scope_text=str(row["scope_text"]),
        scope_is_restrictive=bool(row["scope_is_restrictive"]),
        polarity=str(row["polarity"]),
    )

    evidence_created = store.add_evidence(
        conn,
        source_id=int(row["source_id"]),
        source_text=str(row["source_text"]),
        model_quote=str(row["model_quote"]),
        passage_ids=json.loads(str(row["passage_ids"])),
        passage_version=str(row["passage_version"]),
        location=str(row["location"]),
        polarity=str(row["polarity"]),
        extraction_model=str(row["extraction_model"]),
        extraction_prompt_version=str(row["extraction_prompt_version"]),
        validator_model=str(judgment["validator_model"]),
        validator_prompt_version=str(judgment["validator_prompt_version"]),
        validator_verdict=str(judgment["verdict"]),
        validator_reason=str(judgment["reason"]),
        claim_id=claim_id,
        assertion_id=assertion_id,
    )
    _save_materialization(
        conn,
        observation_id,
        claim_id=claim_id,
        assertion_id=assertion_id,
        error="",
    )
    return {
        "outcome": "materialized",
        "claim_id": claim_id,
        "assertion_id": assertion_id,
        "claim_created": created,
        "assertion_created": assertion_created,
        "evidence_created": evidence_created,
    }


def _save_materialization(
    conn: sqlite3.Connection,
    observation_id: int,
    *,
    claim_id: int | None,
    assertion_id: int | None = None,
    error: str,
) -> None:
    conn.execute(
        """
        UPDATE claim_observations
        SET claim_id=?,assertion_id=?,materialization_error=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (claim_id, assertion_id, error, observation_id),
    )


def resolve_and_materialize_cached(conn: sqlite3.Connection) -> dict[str, int]:
    resolved = resolve_endpoint_ids(conn)
    prepare_assertions(conn)
    materialized = 0
    for row in conn.execute(
        """
        WITH latest AS (
          SELECT observation_id,validator_model,validator_prompt_version
          FROM (
            SELECT observation_id,validator_model,validator_prompt_version,
                   ROW_NUMBER() OVER (
                     PARTITION BY observation_id ORDER BY id DESC
                   ) AS position
            FROM claim_observation_judgments
          ) WHERE position=1
        )
        SELECT o.id,j.validator_model,j.validator_prompt_version
        FROM claim_observations o
        JOIN latest j ON j.observation_id=o.id
        WHERE o.claim_id IS NULL
        ORDER BY o.id
        """
    ):
        result = materialize(
            conn,
            int(row["id"]),
            validator_model=str(row["validator_model"]),
            validator_prompt_version=str(row["validator_prompt_version"]),
        )
        materialized += result.get("outcome") == "materialized"
    conn.commit()
    return {"resolved_endpoints": resolved, "materialized": materialized}


def replay_pending(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    limit: int | None = None,
    promote_threshold: int = 3,
) -> dict[str, Any]:
    model = _model_name(llm)
    resolved_before = resolve_endpoint_ids(conn)
    prepare_assertions(conn)
    promotion = promote_candidates(
        conn,
        llm,
        threshold=promote_threshold,
        limit=limit,
    )
    resolved_after = resolve_endpoint_ids(conn)
    prepare_assertions(conn)
    rows = conn.execute(
        "SELECT * FROM claim_observations ORDER BY id"
    ).fetchall()
    pending_judgment = [
        row
        for row in rows
        if row["claim_id"] is None
        and str(row["assertion_fingerprint"])
        and current_judgment(
            conn, int(row["id"]), validator_model=model
        ) is None
    ]
    if limit is not None:
        pending_judgment = pending_judgment[: max(0, limit)]
    judged = 0
    for row in pending_judgment:
        verdict, reason = validation.judge_claim(llm, as_claim(conn, row))
        save_judgment(
            conn,
            int(row["id"]),
            validator_model=model,
            verdict=verdict,
            reason=reason,
        )
        judged += 1
    conn.commit()
    materialized = 0
    blocked = 0
    for row in conn.execute(
        "SELECT id FROM claim_observations WHERE claim_id IS NULL ORDER BY id"
    ):
        result = materialize(conn, int(row["id"]), validator_model=model)
        materialized += result.get("outcome") == "materialized"
        blocked += result.get("outcome") == "blocked"
    conn.commit()
    return {
        "resolved_endpoints": resolved_before + resolved_after,
        "judged": judged,
        "promotion": promotion,
        "materialized": materialized,
        "blocked": blocked,
        "pending": observation_report(conn),
    }


def promotion_candidates(
    conn: sqlite3.Connection, *, threshold: int = 3
) -> list[dict[str, Any]]:
    grouped = _unresolved_reference_groups(conn)
    candidates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        units = _evidence_units(group["rows"])
        explicit = any(
            key in store.reference_key(str(row["source_text"]))
            for row in group["rows"]
        )
        if len(units) < threshold or not explicit:
            continue
        candidates.append(
            {
                "reference_key": key,
                "name": group["names"][0],
                "names": group["names"],
                "passage_count": len(units),
                "source_count": len({source_id for source_id, _ in units}),
                "rows": group["rows"],
            }
        )
    candidates.sort(
        key=lambda item: (-int(item["passage_count"]), str(item["reference_key"]))
    )
    return candidates


def promote_candidates(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    threshold: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    if threshold < 1:
        raise ValueError("promote_threshold 必须至少为 1")
    candidates = promotion_candidates(conn, threshold=threshold)
    if limit is not None:
        candidates = candidates[: max(0, limit)]
    model = _model_name(llm)
    created: list[int] = []
    linked: list[int] = []
    uncertain = 0
    skipped_cached = 0
    for candidate in candidates:
        rows = candidate["rows"]
        fingerprint = _candidate_fingerprint(rows)
        cached = conn.execute(
            """
            SELECT * FROM entity_candidate_reviews
            WHERE reference_key=? AND evidence_fingerprint=?
              AND reviewer_model=? AND reviewer_version=?
            """,
            (
                candidate["reference_key"],
                fingerprint,
                model,
                PROMOTION_REVIEW_VERSION,
            ),
        ).fetchone()
        if cached:
            skipped_cached += 1
            continue
        similar = resolution.candidate_entities(
            conn, str(candidate["name"]), limit=5, threshold=0.35
        )
        evidence_payload = [
            {
                "source_id": int(row["source_id"]),
                "passage_ids": json.loads(str(row["passage_ids"])),
                "source_text": str(row["source_text"]),
                "model_quote": str(row["model_quote"]),
            }
            for row in rows
        ]
        payload = llm.complete_json(
            PROMOTION_SYSTEM,
            """这个名称已在至少 %d 个独立 Passage 中作为 Claim 端点出现。
判断这些原文是否足以确认它是一个稳定、可复指、可独立学习的 Entity。
若与候选 Entity 相同，可返回 same；字符串相似本身不是同一实体证据。
若创建新 Entity，definition 必须能由给出的 source_text 直接支持，并选择 1-3 个真实的 source_id + passage_id 组合。
返回：
{
  "decision": "same | new | uncertain",
  "candidate_id": "仅 same 时填写候选 id，否则 null",
  "canonical_name": "规范名称",
  "definition": "仅依据原文的定义",
  "type_labels": ["1-3 个开放类别词，可为空"],
  "aliases": ["原文支持的别名"],
  "evidence_refs": [{"source_id": 1, "passage_id": "P000001"}],
  "reason": "简短理由"
}

类型标签开放抽取。以下旧标签只是示例，不是白名单：%s

待定名称：%s
候选 Entity：%s
原文证据：%s"""
            % (
                threshold,
                ontology.entity_type_summary(),
                json.dumps(candidate["names"], ensure_ascii=False),
                json.dumps(similar, ensure_ascii=False),
                json.dumps(evidence_payload, ensure_ascii=False),
            ),
        )
        decision, entity_id, reason = _apply_promotion_decision(
            conn,
            candidate,
            similar,
            payload,
            model=model,
        )
        conn.execute(
            """
            INSERT INTO entity_candidate_reviews
            (reference_key,evidence_fingerprint,reviewer_model,reviewer_version,
             decision,entity_id,reason)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                candidate["reference_key"],
                fingerprint,
                model,
                PROMOTION_REVIEW_VERSION,
                decision,
                entity_id,
                reason,
            ),
        )
        if decision == "new" and entity_id is not None:
            created.append(entity_id)
        elif decision == "same" and entity_id is not None:
            linked.append(entity_id)
        else:
            uncertain += 1
        conn.commit()
    return {
        "eligible": len(candidates),
        "created": created,
        "linked": linked,
        "uncertain": uncertain,
        "skipped_cached": skipped_cached,
    }


def _apply_promotion_decision(
    conn: sqlite3.Connection,
    candidate: dict[str, Any],
    similar: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    model: str,
) -> tuple[str, int | None, str]:
    decision = str(payload.get("decision", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if decision == "same":
        try:
            selected = int(payload.get("candidate_id"))
        except (TypeError, ValueError):
            selected = -1
        allowed = {int(item["id"]) for item in similar}
        if selected not in allowed:
            return "uncertain", None, reason or "same 返回了非法 candidate_id"
        for name in candidate["names"]:
            store.add_alias(conn, selected, str(name))
        return "same", selected, reason
    if decision != "new":
        return "uncertain", None, reason or "证据不足"

    canonical = str(payload.get("canonical_name", "")).strip()
    definition = str(payload.get("definition", "")).strip()
    raw_labels = payload.get("type_labels", [])
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    type_labels = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in raw_labels
            if isinstance(item, str) and item.strip()
        )
    )[:3] if isinstance(raw_labels, list) else ()
    legacy_type = str(payload.get("entity_type", "")).strip()
    if legacy_type and legacy_type not in type_labels:
        type_labels = (legacy_type, *type_labels)[:3]
    entity_type = type_labels[0] if type_labels else ""
    aliases_raw = payload.get("aliases", [])
    selected_refs_raw = payload.get("evidence_refs", [])
    if (
        not canonical
        or len(definition) < 4
        or not isinstance(selected_refs_raw, list)
        or not 1 <= len(selected_refs_raw) <= 3
    ):
        return "uncertain", None, reason or "new 缺少可验证的实体字段"
    selected_refs: set[tuple[int, str]] = set()
    for item in selected_refs_raw:
        if not isinstance(item, dict):
            return "uncertain", None, reason or "new 返回了非法证据引用"
        try:
            source_id = int(item.get("source_id"))
        except (TypeError, ValueError):
            return "uncertain", None, reason or "new 返回了非法 source_id"
        passage_id = str(item.get("passage_id", "")).strip()
        if not passage_id:
            return "uncertain", None, reason or "new 返回了空 Passage ID"
        selected_refs.add((source_id, passage_id))
    if not 1 <= len(selected_refs) <= 3:
        return "uncertain", None, reason or "new 返回了重复证据引用"
    available = {
        (int(row["source_id"]), passage_id)
        for row in candidate["rows"]
        for passage_id in json.loads(str(row["passage_ids"]))
    }
    if not selected_refs.issubset(available):
        return "uncertain", None, reason or "new 返回了不存在的 Source/Passage"
    if store.reference_entity_ids(conn, canonical):
        return "uncertain", None, reason or "new 的规范名已经对应现有 Entity"

    aliases = tuple(
        str(item).strip()
        for item in aliases_raw
        if isinstance(item, str) and item.strip()
    ) if isinstance(aliases_raw, list) else ()
    evidence_rows = [
        row
        for row in candidate["rows"]
        if any(
            (int(row["source_id"]), passage_id) in selected_refs
            for passage_id in json.loads(str(row["passage_ids"]))
        )
    ]
    first = evidence_rows[0]
    observation = EntityObservation(
        name=str(candidate["name"]),
        definition=definition,
        entity_type=entity_type,
        type_labels=type_labels,
        model_quote=str(first["model_quote"]),
        source_text=str(first["source_text"]),
        passage_ids=tuple(json.loads(str(first["passage_ids"]))),
        location=str(first["location"]),
        aliases=aliases,
    )
    entity_id = store.create_entity(conn, observation, canonical_name=canonical)
    for row in evidence_rows:
        store.add_evidence(
            conn,
            source_id=int(row["source_id"]),
            source_text=str(row["source_text"]),
            model_quote=str(row["model_quote"]),
            passage_ids=json.loads(str(row["passage_ids"])),
            passage_version=str(row["passage_version"]),
            location=str(row["location"]),
            polarity="support",
            extraction_model=model,
            extraction_prompt_version=PROMOTION_REVIEW_VERSION,
            observed_entity_type=entity_type,
            entity_id=entity_id,
        )
    return "new", entity_id, reason


def _unresolved_reference_groups(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"names": [], "rows": []}
    )
    rows = conn.execute(
        """
        SELECT * FROM claim_observations
        WHERE subject_entity_id IS NULL OR object_entity_id IS NULL
        ORDER BY id
        """
    ).fetchall()
    for row in rows:
        for side in ("subject", "object"):
            if row[f"{side}_entity_id"] is not None:
                continue
            key = str(row[f"{side}_reference_key"])
            name = str(row[f"{side}_name"])
            group = groups[key]
            if name not in group["names"]:
                group["names"].append(name)
            if row not in group["rows"]:
                group["rows"].append(row)
    return groups


def _evidence_units(rows: list[sqlite3.Row]) -> set[tuple[int, str]]:
    return {
        (int(row["source_id"]), passage_id)
        for row in rows
        for passage_id in json.loads(str(row["passage_ids"]))
    }


def _candidate_fingerprint(rows: list[sqlite3.Row]) -> str:
    units = sorted(_evidence_units(rows))
    payload = [units, sorted(int(row["id"]) for row in rows)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def observation_report(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        WITH latest AS (
          SELECT observation_id,verdict FROM (
            SELECT observation_id,verdict,
                   ROW_NUMBER() OVER (
                     PARTITION BY observation_id ORDER BY id DESC
                   ) AS position
            FROM claim_observation_judgments
          ) WHERE position=1
        )
        SELECT COUNT(*) AS observations,
               SUM(o.subject_entity_id IS NULL OR o.object_entity_id IS NULL)
                 AS pending_endpoint,
               SUM(o.claim_id IS NOT NULL) AS materialized,
               SUM(o.materialization_error<>'') AS blocked,
               SUM(o.claim_id IS NULL AND latest.observation_id IS NULL)
                 AS pending_judgment,
               SUM(latest.verdict='supports') AS supports,
               SUM(latest.verdict='insufficient') AS insufficient,
               SUM(latest.verdict='contradicts') AS contradicts,
               SUM(latest.verdict='supports' AND o.claim_id IS NULL)
                 AS supported_unmaterialized
        FROM claim_observations o
        LEFT JOIN latest ON latest.observation_id=o.id
        """
    ).fetchone()
    candidate_count = len(promotion_candidates(conn, threshold=3))
    entity_row = conn.execute(
        """
        SELECT COUNT(*) AS observations,
               SUM(entity_id IS NULL) AS pending,
               SUM(resolution_outcome='same') AS same_count,
               SUM(resolution_outcome='new') AS new_count,
               SUM(resolution_outcome='uncertain') AS uncertain_count
        FROM entity_observations
        """
    ).fetchone()
    return {
        "observations": int(row["observations"] or 0),
        "pending_endpoint": int(row["pending_endpoint"] or 0),
        "materialized": int(row["materialized"] or 0),
        "blocked": int(row["blocked"] or 0),
        "pending_judgment": int(row["pending_judgment"] or 0),
        "supports": int(row["supports"] or 0),
        "insufficient": int(row["insufficient"] or 0),
        "contradicts": int(row["contradicts"] or 0),
        "supported_unmaterialized": int(
            row["supported_unmaterialized"] or 0
        ),
        "promotion_candidates_3plus": candidate_count,
        "entity_observations": int(entity_row["observations"] or 0),
        "entity_resolution_pending": int(entity_row["pending"] or 0),
        "entity_resolution_same": int(entity_row["same_count"] or 0),
        "entity_resolution_new": int(entity_row["new_count"] or 0),
        "entity_resolution_uncertain": int(entity_row["uncertain_count"] or 0),
    }


def observation_audit(
    conn: sqlite3.Connection, *, detail_limit: int = 500
) -> dict[str, Any]:
    """Return inspectable non-materialized observations for the HTML audit.

    Materialized observations are summarized but omitted from ``items`` because
    their provenance is already visible on the corresponding graph edge.
    """
    rows = conn.execute(
        """
        WITH latest AS (
          SELECT * FROM (
            SELECT j.*,
                   ROW_NUMBER() OVER (
                     PARTITION BY observation_id ORDER BY id DESC
                   ) AS position
            FROM claim_observation_judgments j
          ) WHERE position=1
        )
        SELECT o.*,latest.validator_model,latest.validator_prompt_version,
               latest.verdict,latest.reason AS validator_reason,
               s.name AS source_name,s.uri AS source_uri,s.version AS source_version,
               se.canonical_name AS subject_entity_name,
               oe.canonical_name AS object_entity_name
        FROM claim_observations o
        JOIN sources s ON s.id=o.source_id
        LEFT JOIN latest ON latest.observation_id=o.id
        LEFT JOIN entities se ON se.id=o.subject_entity_id
        LEFT JOIN entities oe ON oe.id=o.object_entity_id
        WHERE o.claim_id IS NULL OR o.materialization_error<>''
        ORDER BY
          CASE
            WHEN o.materialization_error<>'' THEN 0
            WHEN latest.observation_id IS NULL THEN 1
            WHEN o.subject_entity_id IS NULL OR o.object_entity_id IS NULL THEN 2
            ELSE 3
          END,
          o.id
        LIMIT ?
        """,
        (max(0, detail_limit),),
    ).fetchall()
    items = [_audit_item(row) for row in rows]
    candidates = []
    for candidate in promotion_candidates(conn, threshold=3):
        evidence = []
        for row in candidate["rows"][:10]:
            evidence.append(
                {
                    "source_id": int(row["source_id"]),
                    "passage_ids": json.loads(str(row["passage_ids"])),
                    "source_text": str(row["source_text"]),
                    "model_quote": str(row["model_quote"]),
                }
            )
        candidates.append(
            {
                "reference_key": str(candidate["reference_key"]),
                "name": str(candidate["name"]),
                "names": list(candidate["names"]),
                "passage_count": int(candidate["passage_count"]),
                "source_count": int(candidate["source_count"]),
                "evidence": evidence,
            }
        )
    return {
        "summary": observation_report(conn),
        "items": items,
        "detail_limit": max(0, detail_limit),
        "promotion_candidates": candidates,
    }


def _audit_item(row: sqlite3.Row) -> dict[str, Any]:
    verdict = str(row["verdict"] or "")
    expected = "supports" if row["polarity"] == "support" else "contradicts"
    statuses = []
    if str(row["materialization_error"]):
        statuses.append("blocked")
    if not verdict:
        statuses.append("pending_judgment")
    elif verdict != expected:
        statuses.append(verdict)
    if row["subject_entity_id"] is None or row["object_entity_id"] is None:
        statuses.append("pending_endpoint")
    if verdict == expected and "pending_endpoint" not in statuses:
        statuses.append("supported_unmaterialized")
    status = statuses[0] if statuses else "supported_unmaterialized"
    return {
        "id": int(row["id"]),
        "status": status,
        "statuses": statuses or [status],
        "source": {
            "id": int(row["source_id"]),
            "name": str(row["source_name"]),
            "uri": str(row["source_uri"]),
            "version": str(row["source_version"]),
        },
        "chunk_index": int(row["chunk_index"]),
        "subject": {
            "name": str(row["subject_name"]),
            "entity_id": (
                int(row["subject_entity_id"])
                if row["subject_entity_id"] is not None
                else None
            ),
            "entity_name": str(row["subject_entity_name"] or ""),
        },
        "relation": str(row["relation"]),
        "object": {
            "name": str(row["object_name"]),
            "entity_id": (
                int(row["object_entity_id"])
                if row["object_entity_id"] is not None
                else None
            ),
            "entity_name": str(row["object_entity_name"] or ""),
        },
        "polarity": str(row["polarity"]),
        "source_text": str(row["source_text"]),
        "model_quote": str(row["model_quote"]),
        "passage_ids": json.loads(str(row["passage_ids"])),
        "passage_version": str(row["passage_version"]),
        "location": str(row["location"]),
        "extraction": {
            "model": str(row["extraction_model"]),
            "prompt_version": str(row["extraction_prompt_version"]),
        },
        "validation": {
            "model": str(row["validator_model"] or ""),
            "prompt_version": str(row["validator_prompt_version"] or ""),
            "verdict": verdict,
            "reason": str(row["validator_reason"] or ""),
        },
        "materialization_error": str(row["materialization_error"]),
    }


def _model_name(llm: JSONLLM) -> str:
    config = getattr(llm, "config", None)
    return str(getattr(config, "model", "") or llm.__class__.__name__)
