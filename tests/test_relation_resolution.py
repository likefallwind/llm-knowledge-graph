from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg import db, observations, store, vocabulary
from tests.helpers import FakeLLM


class RelationResolutionStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmp.name) / "kg.db")
        self.source_id = int(
            self.conn.execute(
                """INSERT INTO sources
                   (source_key,name,source_type,version,content,content_hash)
                   VALUES ('s','S','test','1','甲与乙','hash')"""
            ).lastrowid
        )
        self.subject_id = self._entity("甲")
        self.object_id = self._entity("乙")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _entity(self, name: str) -> int:
        return int(
            self.conn.execute(
                """INSERT INTO entities
                   (canonical_name,normalized_name,definition)
                   VALUES (?,?,?)""",
                (name, store.normalize_name(name), name),
            ).lastrowid
        )

    def _observation(self, relation: str, statement: str = "甲通过关系指向乙") -> int:
        return int(
            self.conn.execute(
                """INSERT INTO claim_observations
                   (observation_key,source_id,chunk_index,subject_name,
                    subject_reference_key,subject_entity_id,raw_relation,
                    relation,object_name,object_reference_key,object_entity_id,
                    polarity,statement_text,source_text)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"o-{relation}", self.source_id, 0, "甲",
                    store.reference_key("甲"), self.subject_id, relation,
                    relation, "乙", store.reference_key("乙"), self.object_id,
                    "support", statement, statement,
                ),
            ).lastrowid
        )

    def _claim(self, observation_id: int):
        return observations.as_claim(
            self.conn, observations.get_observation(self.conn, observation_id)
        )

    def test_exact_name_is_only_a_candidate_and_does_not_bypass_context(self):
        observation_id = self._observation("is_a", "甲并不是乙的一种")
        llm = FakeLLM({
            "decision": "non_projectable",
            "reason": "原文否定且当前端点不能形成该投影",
        })

        result = vocabulary.resolve_relation(
            self.conn, llm, self._claim(observation_id)
        )

        self.assertEqual(result.outcome, "non_projectable")
        self.assertIn(1, result.candidates)
        self.assertEqual(len(llm.calls), 1)

    def test_catalog_has_supported_open_relations_without_core_slots(self):
        custom_id = int(
            self.conn.execute(
                """INSERT INTO relation_types
                   (canonical_name,normalized_name,relation_kind,description)
                   VALUES ('用于','用于','other','主语用于宾语')"""
            ).lastrowid
        )
        self.conn.execute(
            """INSERT INTO claims
               (subject_id,relation_type_id,relation,object_id)
               VALUES (?,?,?,?)""",
            (self.subject_id, custom_id, "用于", self.object_id),
        )

        candidates = vocabulary._relation_candidates(self.conn, "任意新关系")

        self.assertEqual([item["id"] for item in candidates], [custom_id])

    def test_new_relation_is_a_proposal_until_finalized(self):
        observation_id = self._observation("帮助完成", "甲帮助完成乙")
        llm = FakeLLM({
            "decision": "new",
            "canonical_name": "促进完成",
            "relation_kind": "other",
            "description": "主语促进宾语的完成",
            "projection_statement": "甲促进乙完成",
            "register_alias": True,
            "reason": "没有同义已有关系",
        })
        claim = self._claim(observation_id)

        result = vocabulary.resolve_relation(self.conn, llm, claim)
        self.assertIsNone(result.relation_type_id)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM relation_types").fetchone()[0],
            3,
        )
        self.conn.execute(
            """UPDATE claim_observations SET relation=?,relation_kind=?
               WHERE id=?""",
            (result.canonical_name, result.relation_kind, observation_id),
        )
        vocabulary.save_relation_resolution(
            self.conn, observation_id, claim.raw_relation, result, model="test"
        )
        self.assertEqual(observations.prepare_assertions(self.conn, [observation_id]), 1)
        self.assertTrue(
            observations.get_observation(self.conn, observation_id)[
                "assertion_fingerprint"
            ]
        )

        relation_id = vocabulary.finalize_relation_resolution(
            self.conn, observation_id, claim.raw_relation, result, model="test"
        )

        self.assertIsNotNone(relation_id)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM relation_types").fetchone()[0],
            4,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM relation_resolutions").fetchone()[0],
            1,
        )

    def test_non_projectable_is_audited_but_not_prepared_for_judging(self):
        observation_id = self._observation(
            "产出", "甲的参数经过训练后产出乙"
        )
        llm = FakeLLM({
            "decision": "non_projectable",
            "projection_statement": "甲产出乙",
            "reason": "真正产出乙的是甲的参数，不是甲",
        })
        claim = self._claim(observation_id)

        result = vocabulary.resolve_relation(self.conn, llm, claim)
        vocabulary.save_relation_resolution(
            self.conn, observation_id, claim.raw_relation, result, model="test"
        )

        self.assertEqual(observations.prepare_assertions(self.conn, [observation_id]), 0)
        attempt = self.conn.execute(
            "SELECT * FROM relation_resolution_attempts"
        ).fetchone()
        self.assertEqual(attempt["outcome"], "non_projectable")
        self.assertIn("参数", attempt["reason"])

    def test_same_alias_is_registered_only_after_final_support(self):
        part_of = 2
        self.conn.execute(
            """INSERT INTO claims
               (subject_id,relation_type_id,relation,object_id)
               VALUES (?,?,?,?)""",
            (self.subject_id, part_of, "part_of", self.object_id),
        )
        observation_id = self._observation("构成", "甲构成乙的一部分")
        llm = FakeLLM({
            "decision": "same",
            "candidate_id": part_of,
            "projection_statement": "甲是乙的组成部分",
            "register_alias": True,
            "reason": "语义与方向相同",
        })
        claim = self._claim(observation_id)

        result = vocabulary.resolve_relation(self.conn, llm, claim)
        self.assertIsNone(
            self.conn.execute(
                "SELECT 1 FROM relation_aliases WHERE normalized_name='构成'"
            ).fetchone()
        )
        vocabulary.finalize_relation_resolution(
            self.conn, observation_id, claim.raw_relation, result, model="test"
        )

        alias = self.conn.execute(
            "SELECT relation_type_id FROM relation_aliases WHERE normalized_name='构成'"
        ).fetchone()
        self.assertEqual(alias["relation_type_id"], part_of)


if __name__ == "__main__":
    unittest.main()
