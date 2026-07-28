from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg import db, store
from kg.models import EntityObservation, LoadedSource, SourceSpec


def observation(name: str, entity_type: str = "concept") -> EntityObservation:
    return EntityObservation(
        name=name,
        definition=f"{name} 的实质性测试定义",
        entity_type=entity_type,
        model_quote=f"{name} 的实质性测试定义",
        source_text=f"{name} 的实质性测试定义",
        passage_ids=("P000001",),
        location="P000001",
    )


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "kg.db"
        self.conn = db.connect(self.path)
        loaded = LoadedSource(
            spec=SourceSpec("test", "Test", "test"),
            content="测试正文",
            content_hash="a" * 64,
            version="1",
        )
        self.source_id, _ = store.add_source(self.conn, loaded)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _entity(self, name: str, entity_type: str = "concept") -> int:
        entity_id = store.create_entity(
            self.conn, observation(name, entity_type)
        )
        store.add_evidence(
            self.conn,
            source_id=self.source_id,
            source_text=f"{name} 的实质性测试定义",
            model_quote=f"{name} 的实质性测试定义",
            passage_ids=("P000001",),
            location="test",
            polarity="support",
            entity_id=entity_id,
        )
        self.conn.commit()
        return entity_id

    def test_source_versions_are_immutable_and_idempotent(self):
        same = LoadedSource(
            spec=SourceSpec("test", "Test", "test"),
            content="测试正文",
            content_hash="a" * 64,
            version="1",
        )
        self.assertEqual(store.add_source(self.conn, same), (self.source_id, False))
        changed = LoadedSource(
            spec=SourceSpec("test", "Test", "test"),
            content="新正文",
            content_hash="b" * 64,
            version="2",
        )
        changed_id, created = store.add_source(self.conn, changed)
        self.assertTrue(created)
        self.assertNotEqual(changed_id, self.source_id)

    def test_claim_deduplicates_and_accumulates_evidence(self):
        left = self._entity("批量梯度下降法", "solution")
        right = self._entity("梯度下降法", "solution")
        claim_id, created, _ = store.upsert_claim(
            self.conn, left, "is_a", right
        )
        self.assertTrue(created)
        again, created, _ = store.upsert_claim(self.conn, left, "is_a", right)
        self.assertEqual(again, claim_id)
        self.assertFalse(created)
        store.add_evidence(
            self.conn,
            source_id=self.source_id,
            source_text="真实原文一",
            model_quote="证据一",
            passage_ids=("P000001",),
            location="1",
            polarity="support",
            claim_id=claim_id,
        )
        other = LoadedSource(
            spec=SourceSpec("other", "Other", "test"),
            content="证据二",
            content_hash="c" * 64,
            version="1",
        )
        other_id, _ = store.add_source(self.conn, other)
        store.add_evidence(
            self.conn,
            source_id=other_id,
            source_text="真实原文二",
            model_quote="证据二",
            passage_ids=("P000002",),
            location="2",
            polarity="support",
            claim_id=claim_id,
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE claim_id=?", (claim_id,)
        ).fetchone()[0]
        self.assertEqual(count, 2)

    def test_self_loops_and_cycles_are_rejected(self):
        a, b, c = (self._entity(name) for name in ("A", "B", "C"))
        self.assertIsNone(store.upsert_claim(self.conn, a, "is_a", a)[0])
        self.assertIsNotNone(store.upsert_claim(self.conn, a, "is_a", b)[0])
        self.assertIsNotNone(store.upsert_claim(self.conn, b, "is_a", c)[0])
        claim_id, _, reason = store.upsert_claim(self.conn, c, "is_a", a)
        self.assertIsNone(claim_id)
        self.assertIn("循环", reason)

    def test_merge_retargets_and_deduplicates_claims(self):
        duplicate = self._entity("梯度下降算法", "solution")
        target = self._entity("梯度下降法", "solution")
        parent = self._entity("优化算法", "solution")
        old_claim, _, _ = store.upsert_claim(
            self.conn, duplicate, "is_a", parent
        )
        new_claim, _, _ = store.upsert_claim(self.conn, target, "is_a", parent)
        for claim_id, excerpt in ((old_claim, "来源一"), (new_claim, "来源二")):
            store.add_evidence(
                self.conn,
                source_id=self.source_id,
                source_text=excerpt,
                model_quote=excerpt,
                passage_ids=("P000001",),
                location="test",
                polarity="support",
                claim_id=claim_id,
            )
        store.merge_entities(self.conn, duplicate, target)
        claims = self.conn.execute(
            "SELECT * FROM claims WHERE subject_id=? AND object_id=?",
            (target, parent),
        ).fetchall()
        self.assertEqual(len(claims), 1)
        claim_evidence = self.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE claim_id=?", (claims[0]["id"],)
        ).fetchone()[0]
        self.assertEqual(claim_evidence, 2)
        self.assertFalse(store.get_entity(self.conn, duplicate))

    def test_integrity_requires_evidence_but_allows_isolated_entity(self):
        self._entity("孤立实体")
        report = store.integrity_report(self.conn)
        self.assertTrue(report["ok"])
        self.assertEqual(store.counts(self.conn)["claims"], 0)

    def test_refuses_old_complex_database(self):
        path = Path(self.tmp.name) / "old.db"
        import sqlite3

        old = sqlite3.connect(path)
        old.execute(
            "CREATE TABLE entities(id INTEGER, canonical_name TEXT, "
            "normalized_name TEXT, definition TEXT, entity_type TEXT, status TEXT)"
        )
        old.commit()
        old.close()
        with self.assertRaisesRegex(RuntimeError, "旧 data/kg.db"):
            db.connect(path)


if __name__ == "__main__":
    unittest.main()
