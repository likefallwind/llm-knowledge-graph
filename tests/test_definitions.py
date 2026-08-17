from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg import db, definitions, export, observations, store
from kg.models import EntityObservation, LoadedSource, Resolution, SourceSpec
from tests.helpers import FakeLLM


class DefinitionSynthesisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmp.name) / "kg.db")
        loaded = LoadedSource(
            spec=SourceSpec("book", "Book", "textbook"),
            content="测试正文",
            content_hash="source".ljust(64, "0"),
            version="1",
        )
        self.source_id, _ = store.add_source(self.conn, loaded)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _observation(
        self, definition: str, passage_id: str, chunk_index: int
    ) -> EntityObservation:
        return EntityObservation(
            name="卷积神经网络",
            definition=definition,
            entity_type="solution",
            aliases=("CNN",),
            model_quote=definition,
            source_text=definition + "。",
            passage_ids=(passage_id,),
            location=passage_id,
        )

    def _entity_with_two_observations(self) -> tuple[int, list[int]]:
        first = self._observation("处理图像的强大工具", "P000001", 0)
        entity_id = store.create_entity(self.conn, first)
        ids = []
        for index, item in enumerate(
            (
                first,
                self._observation(
                    "包含卷积层的一类特殊神经网络", "P000002", 1
                ),
            )
        ):
            observation_id, _ = observations.add_entity_observation(
                self.conn,
                source_id=self.source_id,
                chunk_index=index,
                observation=item,
                extraction_model="FakeLLM",
            )
            observations.save_entity_resolution(
                self.conn,
                observation_id,
                Resolution(entity_id, "new" if index == 0 else "same"),
                resolver_model="FakeLLM",
            )
            ids.append(observation_id)
        self.conn.commit()
        return entity_id, ids

    def test_prompt_uses_general_knowledge_without_losing_source_boundary(self):
        prompt = definitions.SYSTEM_PROMPT + definitions.USER_PROMPT

        self.assertIn("可靠的通用知识", prompt)
        self.assertIn("通常含义", prompt)
        self.assertIn("source_text", prompt)
        self.assertIn("原文中特有的事实", prompt)
        self.assertIn("不得让一次局部", prompt)
        self.assertNotIn("每个实质性陈述都必须", prompt)
        self.assertNotIn("不得使用模型记忆补充任何事实", prompt)

    def test_default_synthesizes_entity_with_one_observation(self):
        item = self._observation("卷积神经网络可用于处理图像", "P000003", 0)
        entity_id = store.create_entity(self.conn, item)
        observation_id, _ = observations.add_entity_observation(
            self.conn,
            source_id=self.source_id,
            chunk_index=0,
            observation=item,
            extraction_model="FakeLLM",
        )
        observations.save_entity_resolution(
            self.conn,
            observation_id,
            Resolution(entity_id, "new"),
            resolver_model="FakeLLM",
        )
        self.conn.commit()
        llm = FakeLLM(
            {
                "definition": (
                    "卷积神经网络是一类使用卷积运算提取局部特征的神经网络，"
                    "常用于处理图像等网格结构数据。"
                ),
                "supporting_observations": [
                    {
                        "observation_id": observation_id,
                        "passage_ids": ["P000003"],
                        "support": "锚定为用于图像处理的卷积神经网络义项",
                    }
                ],
                "rejected_candidates": ["只写图像用途会遮蔽通常含义"],
                "limitation": "卷积运算和局部特征由可靠通用知识补足。",
            }
        )

        result = definitions.synthesize_pending(
            self.conn, llm, entity_ids=[entity_id]
        )

        self.assertEqual(len(result["processed"]), 1)
        self.assertFalse(result["failures"])
        self.assertIn(
            "使用卷积运算",
            store.get_entity(self.conn, entity_id)["definition"],
        )
        llm.assert_finished()

    def test_regenerates_once_when_first_payload_fails_validation(self):
        entity_id, observation_ids = self._entity_with_two_observations()
        citation = {
            "observation_id": observation_ids[1],
            "passage_ids": ["P000002"],
            "support": "直接给出上位类别和结构特征",
        }
        llm = FakeLLM(
            {
                "definition": "卷积神经网络是包含卷积层的一类特殊神经网络。",
                "supporting_observations": [citation],
                # 真实见过的坏输出：区间写进字符串时把数组结构写漏了。
                "rejected_candidates": ["裁剪到 [0", 1],
                "limitation": "",
            },
            {
                "definition": "卷积神经网络是包含卷积层的一类特殊神经网络。",
                "supporting_observations": [citation],
                "rejected_candidates": ["强大工具只说明作用"],
                "limitation": "",
            },
        )

        result = definitions.synthesize_pending(
            self.conn, llm, entity_ids=[entity_id]
        )

        self.assertFalse(result["failures"])
        self.assertEqual(len(result["processed"]), 1)
        self.assertEqual(len(llm.calls), 2)
        synthesis = self.conn.execute(
            "SELECT * FROM entity_definition_syntheses WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        self.assertEqual(
            json.loads(synthesis["rejected_candidates"]), ["强大工具只说明作用"]
        )
        llm.assert_finished()

    def test_synthesizes_from_all_observations_and_caches_fingerprint(self):
        entity_id, observation_ids = self._entity_with_two_observations()
        llm = FakeLLM(
            {
                "definition": "卷积神经网络是包含卷积层的一类特殊神经网络。",
                "supporting_observations": [
                    {
                        "observation_id": observation_ids[1],
                        "passage_ids": ["P000002"],
                        "support": "直接给出上位类别和结构特征",
                    }
                ],
                "rejected_candidates": ["强大工具只说明作用"],
                "limitation": "",
            }
        )

        result = definitions.synthesize_pending(
            self.conn, llm, entity_ids=[entity_id]
        )

        self.assertFalse(result["failures"])
        self.assertEqual(len(result["processed"]), 1)
        self.assertEqual(
            store.get_entity(self.conn, entity_id)["definition"],
            "卷积神经网络是包含卷积层的一类特殊神经网络。",
        )
        synthesis = self.conn.execute(
            "SELECT * FROM entity_definition_syntheses WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        self.assertEqual(synthesis["synthesizer_model"], "FakeLLM")
        citations = json.loads(synthesis["supporting_observations"])
        self.assertEqual(citations[0]["observation_id"], observation_ids[1])
        exported = export.graph_dict(self.conn)["entities"][0]
        self.assertEqual(
            exported["definition_synthesis"]["supporting_observations"][0][
                "observation_id"
            ],
            observation_ids[1],
        )
        llm.assert_finished()

        no_calls = FakeLLM()
        cached = definitions.synthesize_pending(
            self.conn, no_calls, entity_ids=[entity_id]
        )
        self.assertEqual(len(cached["skipped"]), 1)
        self.assertEqual(len(no_calls.calls), 0)

    def test_invalid_citation_preserves_previous_definition(self):
        entity_id, _ = self._entity_with_two_observations()
        old_definition = str(store.get_entity(self.conn, entity_id)["definition"])
        llm = FakeLLM(
            {
                "definition": "一个无依据的新定义",
                "supporting_observations": [
                    {
                        "observation_id": 999999,
                        "passage_ids": ["P999999"],
                        "support": "不存在的证据",
                    }
                ],
                "rejected_candidates": [],
                "limitation": "",
            }
        )

        result = definitions.synthesize_pending(
            self.conn, llm, entity_ids=[entity_id]
        )

        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(
            store.get_entity(self.conn, entity_id)["definition"], old_definition
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM entity_definition_syntheses"
            ).fetchone()[0],
            0,
        )

    def test_old_prompt_version_does_not_suppress_regeneration(self):
        entity_id, observation_ids = self._entity_with_two_observations()
        items = definitions._observations(self.conn, entity_id)
        fingerprint = definitions.observation_fingerprint(items)
        self.conn.execute(
            """
            INSERT INTO entity_definition_syntheses
            (entity_id,observation_fingerprint,synthesizer_model,prompt_version,
             definition,supporting_observations,rejected_candidates,limitation)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                entity_id,
                fingerprint,
                "FakeLLM",
                "entity-definition-observations-2",
                "旧版过窄定义",
                "[]",
                "[]",
                "",
            ),
        )
        self.conn.commit()
        llm = FakeLLM(
            {
                "definition": "卷积神经网络是一类使用卷积运算处理网格结构数据的神经网络。",
                "supporting_observations": [
                    {
                        "observation_id": observation_ids[1],
                        "passage_ids": ["P000002"],
                        "support": "锚定卷积神经网络义项",
                    }
                ],
                "rejected_candidates": ["旧定义只强调局部用途"],
                "limitation": "通常含义由可靠通用知识补足。",
            }
        )

        result = definitions.synthesize_pending(
            self.conn, llm, entity_ids=[entity_id]
        )

        self.assertEqual(len(result["processed"]), 1)
        versions = self.conn.execute(
            """
            SELECT prompt_version FROM entity_definition_syntheses
            WHERE entity_id=? ORDER BY id
            """,
            (entity_id,),
        ).fetchall()
        self.assertEqual(
            [row["prompt_version"] for row in versions],
            [
                "entity-definition-observations-2",
                definitions.DEFINITION_PROMPT_VERSION,
            ],
        )
        llm.assert_finished()


if __name__ == "__main__":
    unittest.main()
