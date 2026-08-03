from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB = Path("data/knowledge-vnext.db")
SCHEMA = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "9"


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
    meta_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    ).fetchone()
    if meta_exists:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        version = str(row["value"]) if row else ""
        if version and version != SCHEMA_VERSION:
            raise RuntimeError(
                f"数据库 schema {version} 属于旧抽取模式；vNext schema "
                f"{SCHEMA_VERSION} 必须使用全新数据库，旧库不会被修改"
            )
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
    ).fetchone()
    if existing and not meta_exists:
        raise RuntimeError(
            "数据库缺少 vNext schema 标记；请使用全新数据库，"
            "旧 data/kg.db 或其他旧库不会被修改"
        )
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute(
        """INSERT INTO schema_meta(key,value) VALUES ('schema_version',?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (SCHEMA_VERSION,),
    )
    for name, kind, description in (
        ("is_a", "is_a", "主语是宾语的一种"),
        ("part_of", "part_of", "主语是宾语的组成部分"),
        ("prerequisite_of", "prerequisite_of", "主语是学习宾语的前置知识"),
    ):
        conn.execute(
            """INSERT OR IGNORE INTO relation_types
               (canonical_name,normalized_name,relation_kind,description)
               VALUES (?,?,?,?)""",
            (name, name, kind, description),
        )
    for name in ("resource", "criterion", "data", "task", "solution", "concept"):
        conn.execute(
            """INSERT OR IGNORE INTO entity_type_vocab
               (canonical_name,normalized_name,description) VALUES (?,?,?)""",
            (name, name, "兼容旧语料的种子类型；开放词表仍可继续增长"),
        )
    conn.commit()
