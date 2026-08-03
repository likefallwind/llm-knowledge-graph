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


if __name__ == "__main__":
    unittest.main()
