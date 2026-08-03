from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB = Path("data/knowledge.db")
SCHEMA = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "8"
CLAIM_OBSERVATION_BACKFILL_VERSION = "1"


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _install_schema(conn)
    return conn


def _install_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
    ).fetchone()
    had_existing_database = existing is not None
    if existing:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(entities)")
        }
        expected = {"id", "canonical_name", "normalized_name", "definition"}
        if not expected.issubset(columns) or "status" in columns:
            raise RuntimeError(
                "目标数据库不是本项目的最小 schema。请改用新的数据库路径；"
                "旧 data/kg.db 不会被自动修改。"
            )
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    _migrate_evidence(conn)
    _migrate_entity_type_to_profile(conn)
    backfill = conn.execute(
        "SELECT value FROM schema_meta WHERE key=?",
        ("claim_observation_backfill_version",),
    ).fetchone()
    backfill_version = str(backfill["value"]) if backfill else ""
    if backfill_version != CLAIM_OBSERVATION_BACKFILL_VERSION:
        # A new database already writes native ClaimObservations before it
        # materializes Claim Evidence, so there is no legacy data to import.
        # Existing databases need this one-time backfill.  The separate marker
        # keeps later read-only commands from re-importing newly written
        # Evidence as duplicate legacy observations.
        if had_existing_database:
            _migrate_claim_observations(conn)
            _remove_duplicate_legacy_claim_observations(conn)
        conn.execute(
            """
            INSERT INTO schema_meta(key,value) VALUES (?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (
                "claim_observation_backfill_version",
                CLAIM_OBSERVATION_BACKFILL_VERSION,
            ),
        )
    # A migration or an Entity/Alias added by another process may make an old
    # endpoint deterministically resolvable.  This step is read-mostly and never
    # calls an LLM or materializes an unjudged Claim.
    from . import observations

    observations.resolve_endpoint_ids(conn)
    conn.execute(
        """
        INSERT INTO schema_meta(key,value) VALUES ('schema_version',?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (SCHEMA_VERSION,),
    )
    conn.commit()


def _migrate_entity_type_to_profile(conn: sqlite3.Connection) -> None:
    """schema 4：Entity 不再有单值 entity_type。

    类型下沉到 evidence.observed_entity_type，Entity 层用 type profile 汇总。
    历史 Evidence 无法回填——我们不知道当时那次观察判的是什么类型，实体上
    存的只是「第一次写入的那个」，拿它回填等于编造观察记录。因此历史行留空。
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(entities)")}
    if "entity_type" in columns:
        # schema 3 在这一列上建过索引，带索引的列 SQLite 不允许删。
        conn.execute("DROP INDEX IF EXISTS idx_entities_type")
        conn.execute("ALTER TABLE entities DROP COLUMN entity_type")


def _migrate_evidence(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(evidence)")
    }
    additions = {
        "model_quote": "TEXT NOT NULL DEFAULT ''",
        "observed_entity_type": "TEXT NOT NULL DEFAULT ''",
        "passage_ids": "TEXT NOT NULL DEFAULT '[]'",
        "passage_version": "TEXT NOT NULL DEFAULT 'source-passages-1'",
        "extraction_model": "TEXT NOT NULL DEFAULT ''",
        "extraction_prompt_version": "TEXT NOT NULL DEFAULT ''",
        "validator_model": "TEXT NOT NULL DEFAULT ''",
        "validator_prompt_version": "TEXT NOT NULL DEFAULT ''",
        "validator_verdict": "TEXT NOT NULL DEFAULT ''",
        "validator_reason": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(
                f"ALTER TABLE evidence ADD COLUMN {name} {definition}"
            )
    if "quote_match" in columns:
        conn.execute("ALTER TABLE evidence DROP COLUMN quote_match")
    # 依赖上面补出来的列，因此不能写在 schema.sql 里。
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evidence_observed_type
        ON evidence(entity_id, observed_entity_type)
        """
    )
    # Existing evidence.excerpt passed the old exact locator, so preserve it as
    # both historical model quote and source text until it is recalibrated.
    conn.execute(
        """
        UPDATE evidence
        SET model_quote=excerpt,passage_version='legacy-exact-quote-1'
        WHERE model_quote='' AND passage_ids='[]'
        """
    )
    conn.execute(
        """
        UPDATE evidence SET passage_version='legacy-exact-quote-1'
        WHERE passage_ids='[]'
        """
    )


def _migrate_claim_observations(conn: sqlite3.Connection) -> None:
    """Backfill accepted Claim Evidence as already-materialized observations.

    Historical rejected strings are intentionally not guessed.  Newer
    structured rejection_details are imported when all grounded fields exist.
    Re-running their source Chunk later is still safe because observation_key
    makes this migration and normal extraction idempotent.
    """
    import hashlib
    import json

    from . import store

    accepted = conn.execute(
        """
        SELECT v.id AS evidence_id,v.*,c.subject_id,c.relation,c.object_id,
               s.canonical_name AS subject_name,o.canonical_name AS object_name
        FROM evidence v
        JOIN claims c ON c.id=v.claim_id
        JOIN entities s ON s.id=c.subject_id
        JOIN entities o ON o.id=c.object_id
        WHERE v.claim_id IS NOT NULL
        ORDER BY v.id
        """
    ).fetchall()
    for row in accepted:
        represented = conn.execute(
            """
            SELECT id FROM claim_observations
            WHERE observation_key NOT LIKE 'legacy-claim-evidence:%'
              AND claim_id=? AND source_id=? AND source_text=?
              AND model_quote=? AND passage_ids=? AND polarity=?
            LIMIT 1
            """,
            (
                int(row["claim_id"]),
                int(row["source_id"]),
                str(row["excerpt"]),
                str(row["model_quote"]),
                str(row["passage_ids"]),
                str(row["polarity"]),
            ),
        ).fetchone()
        if represented:
            continue
        key = f"legacy-claim-evidence:{int(row['evidence_id'])}"
        conn.execute(
            """
            INSERT OR IGNORE INTO claim_observations
            (observation_key,source_id,chunk_index,subject_name,
             subject_reference_key,subject_entity_id,relation,object_name,
             object_reference_key,object_entity_id,polarity,source_text,
             model_quote,passage_ids,passage_version,location,
             extraction_model,extraction_prompt_version,claim_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                key,
                int(row["source_id"]),
                -1,
                str(row["subject_name"]),
                store.reference_key(str(row["subject_name"])),
                int(row["subject_id"]),
                str(row["relation"]),
                str(row["object_name"]),
                store.reference_key(str(row["object_name"])),
                int(row["object_id"]),
                str(row["polarity"]),
                str(row["excerpt"]),
                str(row["model_quote"]),
                str(row["passage_ids"]),
                str(row["passage_version"]),
                str(row["location"]),
                str(row["extraction_model"]),
                str(row["extraction_prompt_version"]),
                int(row["claim_id"]),
            ),
        )
        observation_id = int(
            conn.execute(
                "SELECT id FROM claim_observations WHERE observation_key=?", (key,)
            ).fetchone()[0]
        )
        verdict = str(row["validator_verdict"])
        if verdict in {"supports", "contradicts", "insufficient"}:
            conn.execute(
                """
                INSERT OR IGNORE INTO claim_observation_judgments
                (observation_id,validator_model,validator_prompt_version,verdict,reason)
                VALUES (?,?,?,?,?)
                """,
                (
                    observation_id,
                    str(row["validator_model"]),
                    str(row["validator_prompt_version"]),
                    verdict,
                    str(row["validator_reason"]),
                ),
            )

    progress_rows = conn.execute(
        """
        SELECT source_id,chunk_index,result FROM source_progress
        WHERE status='done' AND json_valid(result)
        """
    ).fetchall()
    for progress in progress_rows:
        try:
            result = json.loads(str(progress["result"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        details = result.get("rejection_details", [])
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            subject = str(detail.get("subject", "")).strip()
            relation = str(detail.get("relation", "")).strip()
            object_name = str(detail.get("object", "")).strip()
            source_text = str(detail.get("source_text", "")).strip()
            passage_ids = detail.get("passage_ids", [])
            if (
                not subject
                or relation not in {"is_a", "part_of", "prerequisite_of"}
                or not object_name
                or not source_text
                or not isinstance(passage_ids, list)
                or not passage_ids
            ):
                continue
            raw_key = json.dumps(
                [
                    int(progress["source_id"]),
                    int(progress["chunk_index"]),
                    subject,
                    relation,
                    object_name,
                    str(detail.get("polarity", "support")),
                    source_text,
                    passage_ids,
                ],
                ensure_ascii=False,
            )
            key = "legacy-rejection:" + hashlib.sha256(
                raw_key.encode("utf-8")
            ).hexdigest()
            conn.execute(
                """
                INSERT OR IGNORE INTO claim_observations
                (observation_key,source_id,chunk_index,subject_name,
                 subject_reference_key,relation,object_name,object_reference_key,
                 polarity,source_text,model_quote,passage_ids,location)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    key,
                    int(progress["source_id"]),
                    int(progress["chunk_index"]),
                    subject,
                    store.reference_key(subject),
                    relation,
                    object_name,
                    store.reference_key(object_name),
                    str(detail.get("polarity", "support")),
                    source_text,
                    str(detail.get("model_quote", "")),
                    json.dumps(passage_ids, ensure_ascii=False),
                    str(detail.get("location", "")),
                ),
            )
            observation_id = int(
                conn.execute(
                    "SELECT id FROM claim_observations WHERE observation_key=?",
                    (key,),
                ).fetchone()[0]
            )
            stage = str(detail.get("stage", ""))
            verdict = str(detail.get("verdict", "")).strip()
            if stage == "claim_write" and not verdict:
                verdict = (
                    "supports"
                    if str(detail.get("polarity", "support")) == "support"
                    else "contradicts"
                )
            if verdict in {"supports", "contradicts", "insufficient"}:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO claim_observation_judgments
                    (observation_id,validator_model,validator_prompt_version,
                     verdict,reason)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        observation_id,
                        "legacy-recorded",
                        "legacy-recorded",
                        verdict,
                        str(detail.get("reason", "")),
                    ),
                )


def _remove_duplicate_legacy_claim_observations(
    conn: sqlite3.Connection,
) -> int:
    """Remove schema-5 backfill rows duplicated by a native observation.

    Early schema-5 installs ran the legacy backfill on every connection.  A
    current pipeline run could therefore gain a second, synthetic Observation
    for the same Claim Evidence when ``status`` or ``check`` reopened the DB.
    Keep genuine legacy rows, and delete only rows whose complete grounded
    Evidence identity is already represented by a non-legacy Observation.
    """
    cursor = conn.execute(
        """
        DELETE FROM claim_observations
        WHERE id IN (
          SELECT legacy.id
          FROM claim_observations legacy
          WHERE legacy.observation_key LIKE 'legacy-claim-evidence:%'
            AND EXISTS (
              SELECT 1 FROM claim_observations native
              WHERE native.id<>legacy.id
                AND native.observation_key NOT LIKE 'legacy-claim-evidence:%'
                AND native.claim_id=legacy.claim_id
                AND native.source_id=legacy.source_id
                AND native.source_text=legacy.source_text
                AND native.model_quote=legacy.model_quote
                AND native.passage_ids=legacy.passage_ids
                AND native.polarity=legacy.polarity
            )
        )
        """
    )
    return cursor.rowcount
