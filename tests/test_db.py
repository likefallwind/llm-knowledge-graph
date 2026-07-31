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

    def test_existing_claim_evidence_is_backfilled_once(self):
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
        self.assertEqual(len(rows), 1)
        self.assertTrue(str(rows[0]["observation_key"]).startswith(
            "legacy-claim-evidence:"
        ))
        self.assertIsNotNone(rows[0]["claim_id"])

    def test_schema5_duplicate_is_removed_on_upgrade(self):
        self._current_claim()
        conn = db.connect(self.path)
        conn.execute(
            "DELETE FROM schema_meta WHERE key='claim_observation_backfill_version'"
        )
        db._migrate_claim_observations(conn)
        conn.commit()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM claim_observations").fetchone()[0],
            1,
        )
        # Recreate the exact schema-5 failure shape to exercise the cleanup,
        # rather than relying on the corrected backfill to generate bad data.
        row = conn.execute(
            "SELECT * FROM claim_observations WHERE claim_id IS NOT NULL"
        ).fetchone()
        evidence_id = conn.execute(
            "SELECT id FROM evidence WHERE claim_id=?", (int(row["claim_id"]),)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO claim_observations
            (observation_key,source_id,chunk_index,subject_name,
             subject_reference_key,subject_entity_id,relation,object_name,
             object_reference_key,object_entity_id,polarity,source_text,
             model_quote,passage_ids,passage_version,location,
             extraction_model,extraction_prompt_version,claim_id)
            SELECT ?,source_id,-1,subject_name,subject_reference_key,
                   subject_entity_id,relation,object_name,object_reference_key,
                   object_entity_id,polarity,source_text,model_quote,passage_ids,
                   passage_version,location,extraction_model,
                   extraction_prompt_version,claim_id
            FROM claim_observations WHERE id=?
            """,
            (f"legacy-claim-evidence:{evidence_id}", int(row["id"])),
        )
        conn.commit()
        conn.close()

        conn = db.connect(self.path)
        self.addCleanup(conn.close)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM claim_observations").fetchone()[0],
            1,
        )
        self.assertEqual(
            conn.execute(
                """
                SELECT value FROM schema_meta
                WHERE key='claim_observation_backfill_version'
                """
            ).fetchone()[0],
            db.CLAIM_OBSERVATION_BACKFILL_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
