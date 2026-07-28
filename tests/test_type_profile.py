from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg import db, store
from kg.models import EntityObservation, LoadedSource, SourceSpec


def observation(name: str, entity_type: str) -> EntityObservation:
    return EntityObservation(
        name=name,
        definition=f"{name} 的实质性测试定义",
        entity_type=entity_type,
        model_quote=f"{name} 的实质性测试定义",
        source_text=f"{name} 的实质性测试定义",
        passage_ids=("P000001",),
        location="P000001",
    )


class TypeProfileTest(unittest.TestCase):
    """类型是 mention 级观察，Entity 层用 profile 汇总而不折叠成单值。

    schema 4 之前 `entities.entity_type` 只在建实体时写一次，之后所有观察
    的类型判断都被静默丢弃，合并时还按 `min(id)` 随机保留一边。这组用例
    钉住新行为。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmp.name) / "kg.db")
        self.sources = {}
        for key in ("book-a", "book-b"):
            loaded = LoadedSource(
                spec=SourceSpec(key, key, "textbook"),
                content=f"{key} 正文",
                content_hash=key.ljust(64, "0"),
                version="1",
            )
            self.sources[key], _ = store.add_source(self.conn, loaded)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _observe(self, entity_id: int, entity_type: str, source: str, text: str):
        store.add_evidence(
            self.conn,
            source_id=self.sources[source],
            source_text=text,
            model_quote=text,
            passage_ids=("P000001",),
            location="P000001",
            polarity="support",
            observed_entity_type=entity_type,
            entity_id=entity_id,
        )

    def test_entities_table_has_no_single_type_column(self):
        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(entities)")
        }
        self.assertNotIn("entity_type", columns)

    def test_profile_keeps_every_observed_type_without_collapsing(self):
        entity_id = store.create_entity(self.conn, observation("深度学习", "solution"))
        self._observe(entity_id, "solution", "book-a", "深度学习是一族方法 一")
        self._observe(entity_id, "solution", "book-a", "深度学习是一族方法 二")
        self._observe(entity_id, "concept", "book-b", "深度学习是一个研究方向")

        profile = store.type_profile(self.conn, entity_id)
        self.assertEqual(
            profile,
            [
                {"entity_type": "solution", "observations": 2, "sources": 1},
                {"entity_type": "concept", "observations": 1, "sources": 1},
            ],
        )

    def test_reusing_an_entity_records_the_new_observed_type(self):
        """旧行为下这次判断会被完全丢弃，因为实体已存在、类型只写一次。"""
        entity_id = store.create_entity(self.conn, observation("深度学习", "solution"))
        self._observe(entity_id, "solution", "book-a", "第一次观察")
        self._observe(entity_id, "concept", "book-b", "第二次观察，判成研究方向")

        types = {item["entity_type"] for item in store.type_profile(self.conn, entity_id)}
        self.assertEqual(types, {"solution", "concept"})

    def test_merge_unions_both_profiles_instead_of_picking_a_winner(self):
        """旧行为下 target 的类型胜出、source 的类型随实体一起消失。"""
        target = store.create_entity(self.conn, observation("深度学习", "solution"))
        source = store.create_entity(self.conn, observation("深度学习方法", "concept"))
        self._observe(target, "solution", "book-a", "目标实体的证据")
        self._observe(source, "concept", "book-b", "来源实体的证据")

        store.merge_entities(self.conn, source, target)

        profile = store.type_profile(self.conn, target)
        self.assertEqual(
            {item["entity_type"]: item["observations"] for item in profile},
            {"solution": 1, "concept": 1},
        )
        self.assertIsNone(store.get_entity(self.conn, source))

    def test_sources_and_observations_are_counted_separately(self):
        """同一来源反复出现不等于多个独立来源，两个口径必须分开给。"""
        entity_id = store.create_entity(self.conn, observation("梯度下降", "solution"))
        for index in range(5):
            self._observe(entity_id, "solution", "book-a", f"同一本书第 {index} 次")
        self._observe(entity_id, "solution", "book-b", "另一本书")

        profile = store.type_profile(self.conn, entity_id)
        self.assertEqual(profile[0]["observations"], 6)
        self.assertEqual(profile[0]["sources"], 2)

    def test_unrecorded_legacy_types_are_excluded(self):
        """历史 Evidence 没有类型记录，留空且不参与统计——回填等于编造。"""
        entity_id = store.create_entity(self.conn, observation("反向传播", "solution"))
        self._observe(entity_id, "", "book-a", "历史证据，未记录类型")
        self.assertEqual(store.type_profile(self.conn, entity_id), [])

        self._observe(entity_id, "solution", "book-b", "新证据")
        self.assertEqual(
            store.type_profile(self.conn, entity_id),
            [{"entity_type": "solution", "observations": 1, "sources": 1}],
        )


class MigrationFromSchema3Test(unittest.TestCase):
    """schema 3 → 4 的迁移。其余用例都从空库开始，覆盖不到这条路径。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "legacy.db"

    def tearDown(self):
        self.tmp.cleanup()

    def _build_schema3(self) -> None:
        import sqlite3

        conn = sqlite3.connect(self.path)
        conn.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version','3');
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY, source_key TEXT NOT NULL, name TEXT NOT NULL,
                source_type TEXT NOT NULL, uri TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL, content TEXT NOT NULL,
                content_hash TEXT NOT NULL, language TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_key, content_hash));
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY, canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE, definition TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX idx_entities_type ON entities(entity_type);
            CREATE TABLE entity_aliases (
                id INTEGER PRIMARY KEY, entity_id INTEGER NOT NULL,
                name TEXT NOT NULL, normalized_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_id, normalized_name));
            CREATE TABLE claims (
                id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL,
                relation TEXT NOT NULL, object_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_id, relation, object_id));
            CREATE TABLE evidence (
                id INTEGER PRIMARY KEY, target_key TEXT NOT NULL,
                entity_id INTEGER, claim_id INTEGER, source_id INTEGER NOT NULL,
                excerpt TEXT NOT NULL, excerpt_hash TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '', polarity TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(target_key, source_id, excerpt_hash, polarity));
            CREATE TABLE source_progress (
                source_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(source_id, chunk_index, chunk_hash));
            INSERT INTO entities(canonical_name,normalized_name,definition,entity_type)
            VALUES ('深度学习','深度学习','历史定义','solution');
            INSERT INTO sources(source_key,name,source_type,version,content,content_hash)
            VALUES ('k','n','textbook','1','正文','h');
            INSERT INTO evidence(target_key,entity_id,source_id,excerpt,excerpt_hash,polarity)
            VALUES ('entity:1',1,1,'历史证据','hash1','support');
            """
        )
        conn.commit()
        conn.close()

    def test_drops_the_indexed_type_column_and_keeps_data(self):
        self._build_schema3()
        conn = db.connect(self.path)
        self.addCleanup(conn.close)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(entities)")}
        self.assertNotIn("entity_type", columns)
        self.assertEqual(
            conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'")
            .fetchone()[0],
            "4",
        )
        self.assertEqual(
            int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]), 1
        )
        # 历史 Evidence 不回填，因此 profile 为空而不是编造出一个 solution。
        self.assertEqual(store.type_profile(conn, 1), [])

    def test_migration_is_idempotent(self):
        self._build_schema3()
        db.connect(self.path).close()
        conn = db.connect(self.path)
        self.addCleanup(conn.close)
        self.assertEqual(
            int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]), 1
        )


if __name__ == "__main__":
    unittest.main()
