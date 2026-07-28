from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg import db, pipeline, resolution, store
from kg.models import EntityObservation
from tests.helpers import FakeLLM


def entity_payload(
    name: str,
    definition: str,
    quote: str,
    *,
    entity_type: str = "solution",
) -> dict:
    return {
        "name": name,
        "definition": definition,
        "entity_type": entity_type,
        "aliases": [],
        "evidence": {
            "passage_ids": ["P000001"],
            "quote": quote,
        },
    }


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.conn = db.connect(self.root / "kg.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _catalog(self, texts: list[str]) -> Path:
        sources = []
        for index, text in enumerate(texts):
            path = self.root / f"source-{index}.txt"
            path.write_text(text, encoding="utf-8")
            sources.append(
                {
                    "key": f"source-{index}",
                    "name": f"来源 {index}",
                    "type": "textbook",
                    "path": path.name,
                    "language": "zh",
                }
            )
        catalog = self.root / "sources.json"
        catalog.write_text(
            json.dumps({"sources": sources}, ensure_ascii=False),
            encoding="utf-8",
        )
        return catalog

    def test_end_to_end_claim_aggregation_and_resume(self):
        first = (
            "梯度下降法是一种迭代优化算法。"
            "批量梯度下降法是梯度下降法的一种。"
        )
        second = (
            "本教材也说明：批量梯度下降法是梯度下降法的一种，"
            "它每次使用全部训练样本。"
        )
        catalog = self._catalog([first, second])
        first_extraction = {
            "entities": [
                entity_payload(
                    "梯度下降法",
                    "一种迭代优化算法",
                    "梯度下降法是一种迭代优化算法",
                ),
                entity_payload(
                    "批量梯度下降法",
                    "每次使用全部样本的梯度下降方法",
                    "批量梯度下降法是梯度下降法的一种",
                ),
            ],
            "claims": [
                {
                    "subject": "批量梯度下降法",
                    "relation": "is_a",
                    "object": "梯度下降法",
                    "stance": "support",
                    "evidence": {
                        "passage_ids": ["P000001"],
                        "quote": "批量梯度下降法是梯度下降法的一种",
                    },
                }
            ],
        }
        second_extraction = {
            "entities": [
                entity_payload(
                    "梯度下降法",
                    "一种迭代优化算法",
                    "梯度下降法的一种",
                ),
                entity_payload(
                    "批量梯度下降法",
                    "每次使用全部训练样本的方法",
                    "批量梯度下降法是梯度下降法的一种",
                ),
            ],
            "claims": [
                {
                    "subject": "批量梯度下降法",
                    "relation": "is_a",
                    "object": "梯度下降法",
                    "stance": "support",
                    "evidence": {
                        "passage_ids": ["P000001"],
                        "quote": "批量梯度下降法是梯度下降法的一种",
                    },
                }
            ],
        }
        llm = FakeLLM(
            first_extraction,
            {
                "decision": "new",
                "canonical_name": "梯度下降法",
                "reason": "没有同一对象候选",
            },
            {
                "decision": "new",
                "canonical_name": "批量梯度下降法",
                "reason": "不同粒度对象",
            },
            {"verdict": "supports", "reason": "明确说是一种"},
            second_extraction,
            {"verdict": "supports", "reason": "独立来源明确支持"},
        )
        result = pipeline.process_catalog(self.conn, llm, catalog)
        self.assertFalse(result["failures"])
        self.assertEqual(store.counts(self.conn)["sources"], 2)
        self.assertEqual(store.counts(self.conn)["entities"], 2)
        self.assertEqual(store.counts(self.conn)["claims"], 1)
        evidence = self.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE claim_id IS NOT NULL"
        ).fetchone()[0]
        self.assertEqual(evidence, 2)
        rows = self.conn.execute(
            """
            SELECT model_quote,excerpt,passage_ids,extraction_model,
                   extraction_prompt_version,validator_prompt_version,
                   validator_verdict,validator_reason
            FROM evidence WHERE claim_id IS NOT NULL ORDER BY id
            """
        ).fetchall()
        self.assertTrue(all(row["model_quote"] for row in rows))
        self.assertTrue(all(row["excerpt"] for row in rows))
        self.assertTrue(all(row["passage_ids"] == '["P000001"]' for row in rows))
        self.assertTrue(all(row["extraction_model"] == "FakeLLM" for row in rows))
        self.assertTrue(all(row["extraction_prompt_version"] for row in rows))
        self.assertTrue(all(row["validator_prompt_version"] for row in rows))
        self.assertTrue(all(row["validator_verdict"] == "supports" for row in rows))
        self.assertTrue(all(row["validator_reason"] for row in rows))
        self.assertTrue(store.integrity_report(self.conn)["ok"])
        llm.assert_finished()

        no_calls = FakeLLM()
        rerun = pipeline.process_catalog(self.conn, no_calls, catalog)
        self.assertFalse(rerun["failures"])
        self.assertEqual(
            sum(item["skipped_chunks"] for item in rerun["completed"]), 2
        )
        self.assertEqual(len(no_calls.calls), 0)

    def test_invalid_passage_cannot_create_knowledge(self):
        catalog = self._catalog(["这里只介绍优化。"])
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload(
                        "Transformer",
                        "一种神经网络架构",
                        "Transformer 是一种神经网络架构",
                    )
                ],
                "claims": [],
            }
        )
        llm.responses[0]["entities"][0]["evidence"]["passage_ids"] = [
            "P999999"
        ]
        result = pipeline.process_catalog(self.conn, llm, catalog)
        self.assertFalse(result["failures"])
        self.assertEqual(store.counts(self.conn)["entities"], 0)
        self.assertTrue(result["completed"][0]["rejected"])

    def test_entity_writes_roll_back_when_chunk_fails(self):
        catalog = self._catalog(["实体甲和实体乙都在当前语料中。"])
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload("实体甲", "第一个测试实体", "实体甲"),
                    entity_payload("实体乙", "第二个测试实体", "实体乙"),
                ],
                "claims": [],
            },
            {
                "decision": "new",
                "canonical_name": "实体甲",
                "reason": "新实体",
            },
            # 第二个实体解析时 FakeLLM 无响应，模拟远端失败。
        )
        result = pipeline.process_catalog(self.conn, llm, catalog)
        self.assertTrue(result["failures"])
        self.assertEqual(store.counts(self.conn)["entities"], 0)
        self.assertEqual(store.counts(self.conn)["evidence"], 0)

    def test_claim_judge_failure_rolls_back_the_whole_chunk(self):
        catalog = self._catalog(["实体甲是实体乙的一种。"])
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload("实体甲", "第一个测试实体", "实体甲"),
                    entity_payload(
                        "实体乙",
                        "第二个测试实体",
                        "实体乙",
                        entity_type="concept",
                    ),
                ],
                "claims": [
                    {
                        "subject": "实体甲",
                        "relation": "is_a",
                        "object": "实体乙",
                        "stance": "support",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "实体甲是实体乙的一种",
                        },
                    }
                ],
            },
            {
                "decision": "new",
                "canonical_name": "实体甲",
                "reason": "新实体",
            },
            {
                "decision": "new",
                "canonical_name": "实体乙",
                "reason": "新实体",
            },
        )
        result = pipeline.process_catalog(self.conn, llm, catalog)
        self.assertTrue(result["failures"])
        self.assertEqual(store.counts(self.conn)["entities"], 0)
        self.assertEqual(store.counts(self.conn)["claims"], 0)
        self.assertEqual(store.counts(self.conn)["evidence"], 0)

    def test_uncertain_keeps_independent_entity_then_reconcile_merges(self):
        source_id = self.conn.execute(
            """
            INSERT INTO sources
            (source_key,name,source_type,version,content,content_hash)
            VALUES ('s','S','test','1','正文','hash')
            """
        ).lastrowid
        first = EntityObservation(
            name="梯度下降法",
            definition="一种迭代优化算法",
            entity_type="solution",
            model_quote="梯度下降法",
            source_text="梯度下降法",
            passage_ids=("P000001",),
            location="P000001",
        )
        first_id = store.create_entity(self.conn, first)
        store.add_evidence(
            self.conn,
            source_id=source_id,
            source_text="梯度下降法",
            model_quote="梯度下降法",
            passage_ids=("P000001",),
            location="1",
            polarity="support",
            entity_id=first_id,
        )
        observed = EntityObservation(
            name="梯度下降算法",
            definition="沿负梯度更新参数的算法",
            entity_type="solution",
            model_quote="梯度下降算法",
            source_text="梯度下降算法",
            passage_ids=("P000002",),
            location="P000002",
        )
        uncertain_llm = FakeLLM(
            {
                "decision": "uncertain",
                "canonical_name": "梯度下降法",
                "reason": "当前语境不足",
            }
        )
        resolved = resolution.resolve_observation(
            self.conn, uncertain_llm, observed
        )
        self.assertEqual(resolved.outcome, "uncertain")
        self.assertNotEqual(resolved.entity_id, first_id)
        store.add_evidence(
            self.conn,
            source_id=source_id,
            source_text="梯度下降算法",
            model_quote="梯度下降算法",
            passage_ids=("P000002",),
            location="2",
            polarity="support",
            entity_id=resolved.entity_id,
        )
        self.conn.commit()
        self.assertEqual(store.counts(self.conn)["entities"], 2)

        same_llm = FakeLLM(
            {
                "decision": "same",
                "canonical_name": "梯度下降法",
                "reason": "新增定义足以确认同一对象",
            }
        )
        report = resolution.reconcile(self.conn, same_llm, limit=10)
        self.assertEqual(len(report["merged"]), 1)
        self.assertEqual(store.counts(self.conn)["entities"], 1)
        self.assertTrue(store.integrity_report(self.conn)["ok"])

    def test_new_entity_uses_llm_canonical_name_and_keeps_source_alias(self):
        observed = EntityObservation(
            name="GD",
            definition="沿负梯度方向更新参数的优化算法",
            entity_type="solution",
            model_quote="GD",
            source_text="GD",
            passage_ids=("P000001",),
            location="P000001",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "梯度下降法",
                "reason": "根据观察语境规范名称",
            }
        )
        resolved = resolution.resolve_observation(self.conn, llm, observed)
        row = store.get_entity(self.conn, resolved.entity_id)
        self.assertEqual(row["canonical_name"], "梯度下降法")
        self.assertIn("GD", store.aliases_for(self.conn, resolved.entity_id))

    def test_insufficient_relationship_does_not_enter_graph(self):
        text = "梯度下降法和机器学习都在本章出现。"
        catalog = self._catalog([text])
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload(
                        "梯度下降法", "一种优化方法", "梯度下降法"
                    ),
                    entity_payload(
                        "机器学习",
                        "一个研究领域",
                        "机器学习",
                        entity_type="concept",
                    ),
                ],
                "claims": [
                    {
                        "subject": "梯度下降法",
                        "relation": "part_of",
                        "object": "机器学习",
                        "stance": "support",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "梯度下降法和机器学习都在本章出现",
                        },
                    }
                ],
            },
            {
                "decision": "new",
                "canonical_name": "梯度下降法",
                "reason": "没有同一对象候选",
            },
            {
                "decision": "new",
                "canonical_name": "机器学习",
                "reason": "不同对象",
            },
            {"verdict": "insufficient", "reason": "只有共现"},
        )
        pipeline.process_catalog(self.conn, llm, catalog)
        self.assertEqual(store.counts(self.conn)["entities"], 2)
        self.assertEqual(store.counts(self.conn)["claims"], 0)


if __name__ == "__main__":
    unittest.main()
