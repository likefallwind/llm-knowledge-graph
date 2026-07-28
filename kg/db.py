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
        expected = {"id", "canonical_name", "normalized_name", "definition", "entity_type"}
        if not expected.issubset(columns) or "status" in columns:
            raise RuntimeError(
                "目标数据库不是本项目的最小 schema。请改用新的数据库路径；"
                "旧 data/kg.db 不会被自动修改。"
            )
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
