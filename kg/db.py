from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB = Path("data/knowledge.db")
SCHEMA = Path(__file__).with_name("schema.sql")


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
    conn.execute(
        """
        INSERT INTO schema_meta(key,value) VALUES ('schema_version','4')
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """
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
