from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg import db, observations, store
from kg.models import ClaimObservation, EntityObservation, LoadedSource, SourceSpec


class ClaimObservationBackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "kg.db"

    def tearDown(self):
        self.tmp.cleanup()

    def _current_claim(self) -> None:
        conn = db.connect(self.path)
        source = LoadedSource(
            spec=SourceSpec("book", "Book", "textbook"),
            content="卷积层是卷积神经网络的组成部分。",
            content_hash="book".ljust(64, "0"),
            version="1",
        )
        source_id, _ = store.add_source(conn, source)
        for name in ("卷积层", "卷积神经网络"):
            item = EntityObservation(
                name=name,
                definition=f"{name} 的稳定测试定义",
                entity_type="solution",
                model_quote=name,
                source_text=f"{name} 的稳定测试定义。",
                passage_ids=("P000001",),
                location="P000001",
            )
            entity_id = store.create_entity(conn, item)
            store.add_evidence(
                conn,
                source_id=source_id,
                source_text=item.source_text,
                model_quote=item.model_quote,
                passage_ids=item.passage_ids,
                location=item.location,
                polarity="support",
                observed_entity_type="solution",
                entity_id=entity_id,
            )
        claim = ClaimObservation(
            subject="卷积层",
            relation="part_of",
            object="卷积神经网络",
            model_quote="卷积层是卷积神经网络的组成部分",
            source_text="卷积层是卷积神经网络的组成部分。",
            passage_ids=("P000001",),
            location="P000001",
            statement_text="卷积层是卷积神经网络的组成部分",
        )
        observation_id, _ = observations.add_claim_observation(
            conn,
            source_id=source_id,
            chunk_index=0,
            claim=claim,
            extraction_model="FakeLLM",
        )
        observations.save_judgment(
            conn,
            observation_id,
            validator_model="FakeLLM",
            verdict="supports",
            reason="原文明示组成关系",
        )
        observations.resolve_endpoint_ids(conn, [observation_id])
        observations.prepare_assertions(conn, [observation_id])
        observations.save_judgment(
            conn,
            observation_id,
            validator_model="FakeLLM",
            verdict="supports",
            reason="规范化后的完整 Assertion 仍由原文支持",
        )
        result = observations.materialize(
            conn, observation_id, validator_model="FakeLLM"
        )
        self.assertEqual(result["outcome"], "materialized")
        conn.commit()
        conn.close()

    def test_reopening_current_database_does_not_duplicate_observation(self):
        self._current_claim()

        db.connect(self.path).close()
        conn = db.connect(self.path)
        self.addCleanup(conn.close)

        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM claim_observations").fetchone()[0],
            1,
        )
        self.assertEqual(
            conn.execute(
                """
                SELECT COUNT(*) FROM claim_observations
                WHERE observation_key LIKE 'legacy-claim-evidence:%'
                """
            ).fetchone()[0],
            0,
        )

    def test_existing_claim_evidence_is_not_reinterpreted_as_observation(self):
        self._current_claim()
        conn = db.connect(self.path)
        conn.execute("DELETE FROM claim_observations")
        conn.execute(
            "DELETE FROM schema_meta WHERE key='claim_observation_backfill_version'"
        )
        conn.commit()
        conn.close()

        db.connect(self.path).close()
        conn = db.connect(self.path)
        self.addCleanup(conn.close)

        rows = conn.execute(
            "SELECT observation_key,claim_id FROM claim_observations"
        ).fetchall()
        self.assertEqual(rows, [])

    def test_old_schema_marker_is_refused_without_mutation(self):
        self._current_claim()
        conn = db.connect(self.path)
        before = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        conn.execute(
            "UPDATE schema_meta SET value='5' WHERE key='schema_version'"
        )
        conn.commit()
        conn.close()
        with self.assertRaisesRegex(RuntimeError, "旧抽取模式"):
            db.connect(self.path)
        import sqlite3
        raw = sqlite3.connect(self.path)
        self.addCleanup(raw.close)
        self.assertEqual(raw.execute("SELECT COUNT(*) FROM claims").fetchone()[0], before)


if __name__ == "__main__":
    unittest.main()
