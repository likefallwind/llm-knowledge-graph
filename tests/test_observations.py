from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg import db, observations, store
from kg.models import ClaimObservation, EntityObservation, LoadedSource, SourceSpec
from tests.helpers import FakeLLM


class ClaimObservationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmp.name) / "kg.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _source(self, key: str, content: str) -> int:
        source = LoadedSource(
            spec=SourceSpec(key, key, "textbook"),
            content=content,
            content_hash=(key * 64)[:64],
            version="1",
        )
        return store.add_source(self.conn, source)[0]

    def _entity(self, source_id: int, name: str) -> int:
        item = EntityObservation(
            name=name,
            definition=f"{name} 是一个具有稳定身份的测试对象",
            entity_type="concept",
            model_quote=name,
            source_text=f"{name} 是一个具有稳定身份的测试对象。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, item)
        store.add_evidence(
            self.conn,
            source_id=source_id,
            source_text=item.source_text,
            model_quote=item.model_quote,
            passage_ids=item.passage_ids,
            location=item.location,
            polarity="support",
            observed_entity_type="concept",
            entity_id=entity_id,
        )
        self.conn.commit()
        return entity_id

    def _claim(self, subject: str, object_name: str, passage: str) -> ClaimObservation:
        return ClaimObservation(
            subject=subject,
            relation="is_a",
            object=object_name,
            model_quote=f"{subject} 是 {object_name} 的一种",
            source_text=f"{subject} 是 {object_name} 的一种。",
            passage_ids=(passage,),
            location=passage,
        )

    def test_pending_observation_materializes_without_rejudging_after_entity_arrives(self):
        source_id = self._source("a", "待定方法是基础方法的一种。")
        object_id = self._entity(source_id, "基础方法")
        claim = self._claim("待定方法", "基础方法", "P000002")
        observation_id, _ = observations.add_claim_observation(
            self.conn,
            source_id=source_id,
            chunk_index=0,
            claim=claim,
            extraction_model="FakeLLM",
        )
        observations.save_judgment(
            self.conn,
            observation_id,
            validator_model="FakeLLM",
            verdict="supports",
            reason="原文明确定义类属关系",
        )
        observations.resolve_endpoint_ids(self.conn, [observation_id])
        self.conn.commit()

        row = observations.get_observation(self.conn, observation_id)
        self.assertIsNone(row["subject_entity_id"])
        self.assertEqual(row["object_entity_id"], object_id)

        self._entity(source_id, "待定方法")
        no_calls = FakeLLM()
        report = observations.replay_pending(
            self.conn, no_calls, promote_threshold=3
        )

        self.assertEqual(report["judged"], 0)
        self.assertEqual(store.counts(self.conn)["claims"], 1)
        self.assertEqual(len(no_calls.calls), 0)
        row = observations.get_observation(self.conn, observation_id)
        self.assertIsNotNone(row["claim_id"])

    def test_ambiguous_alias_keeps_endpoint_pending(self):
        source_id = self._source("b", "X 是基础方法的一种。")
        object_id = self._entity(source_id, "基础方法")
        left = self._entity(source_id, "对象甲")
        right = self._entity(source_id, "对象乙")
        store.add_alias(self.conn, left, "X")
        store.add_alias(self.conn, right, "X")
        claim = self._claim("X", "基础方法", "P000002")
        observation_id, _ = observations.add_claim_observation(
            self.conn,
            source_id=source_id,
            chunk_index=0,
            claim=claim,
            extraction_model="FakeLLM",
        )

        observations.resolve_endpoint_ids(self.conn, [observation_id])
        row = observations.get_observation(self.conn, observation_id)

        self.assertIsNone(row["subject_entity_id"])
        self.assertEqual(row["object_entity_id"], object_id)
        self.assertEqual(store.reference_entity_ids(self.conn, "X"), [left, right])

    def test_promotion_counts_distinct_passages_not_duplicate_observations(self):
        source_id = self._source("c", "候选方法是基础方法的一种。")
        self._entity(source_id, "基础方法")
        for chunk_index in range(3):
            observations.add_claim_observation(
                self.conn,
                source_id=source_id,
                chunk_index=chunk_index,
                claim=self._claim("候选方法", "基础方法", "P000002"),
                extraction_model=f"FakeLLM-{chunk_index}",
            )
        self.conn.commit()

        candidates = observations.promotion_candidates(self.conn, threshold=3)

        self.assertEqual(candidates, [])

    def test_three_passages_promote_entity_and_replay_all_claim_evidence(self):
        source_ids = [
            self._source(str(index), f"候选方法是基础方法的一种。证据 {index}")
            for index in range(3)
        ]
        self._entity(source_ids[0], "基础方法")
        for index, source_id in enumerate(source_ids, start=1):
            # Passage IDs are scoped to a Source, so each source legitimately
            # starts at P000001.
            claim = self._claim("候选方法", "基础方法", "P000001")
            observation_id, _ = observations.add_claim_observation(
                self.conn,
                source_id=source_id,
                chunk_index=0,
                claim=claim,
                extraction_model="FakeLLM",
            )
            observations.save_judgment(
                self.conn,
                observation_id,
                validator_model="FakeLLM",
                verdict="supports",
                reason="原文明确定义类属关系",
            )
        self.conn.commit()
        llm = FakeLLM(
            {
                "decision": "new",
                "candidate_id": None,
                "canonical_name": "候选方法",
                "definition": "一种由三个独立原文片段共同说明的方法",
                "entity_type": "solution",
                "aliases": [],
                "evidence_refs": [
                    {"source_id": source_ids[0], "passage_id": "P000001"},
                    {"source_id": source_ids[1], "passage_id": "P000001"},
                    {"source_id": source_ids[2], "passage_id": "P000001"},
                ],
                "reason": "三个独立片段稳定指向同一对象",
            }
        )

        report = observations.replay_pending(
            self.conn, llm, promote_threshold=3
        )

        self.assertEqual(len(report["promotion"]["created"]), 1)
        self.assertEqual(store.counts(self.conn)["claims"], 1)
        claim_evidence = self.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE claim_id IS NOT NULL"
        ).fetchone()[0]
        self.assertEqual(claim_evidence, 3)
        self.assertTrue(store.integrity_report(self.conn)["ok"])
        llm.assert_finished()

        cached = FakeLLM()
        second = observations.replay_pending(
            self.conn, cached, promote_threshold=3
        )
        self.assertEqual(second["judged"], 0)
        self.assertEqual(len(cached.calls), 0)


if __name__ == "__main__":
    unittest.main()
