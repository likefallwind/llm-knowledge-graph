from __future__ import annotations

import unittest

from kg.extraction import extract, parse_payload
from kg.sources import segment_text
from tests.helpers import FakeLLM


class ExtractionTest(unittest.TestCase):
    def test_prompt_requires_every_claim_endpoint_to_be_an_entity(self):
        passages = segment_text("批量梯度下降法是梯度下降法的一种。")
        llm = FakeLLM({"entities": [], "claims": []})

        extract(llm, "测试片段", passages=passages)

        prompt = llm.calls[0][1]
        self.assertIn("每个 Claim 的 subject 和 object", prompt)
        self.assertIn("entities 数组", prompt)
        self.assertIn("最多输出 30 个实体", prompt)

    def test_entity_cap_keeps_claim_endpoints_before_unreferenced_entities(self):
        passages = segment_text("A、B、C 都有定义，B 是 C 的一种。")
        payload = {
            "entities": [
                {
                    "name": name,
                    "definition": f"{name} 是一个有实质定义的对象",
                    "entity_type": "concept",
                    "evidence": {
                        "passage_ids": ["P000001"],
                        "quote": f"{name} 有定义",
                    },
                }
                for name in ("A", "B", "C")
            ],
            "claims": [
                {
                    "subject": "B",
                    "relation": "is_a",
                    "object": "C",
                    "stance": "support",
                    "evidence": {
                        "passage_ids": ["P000001"],
                        "quote": "B 是 C 的一种",
                    },
                }
            ],
        }

        batch = parse_payload(payload, passages, max_entities=2)

        self.assertEqual([item.name for item in batch.entities], ["B", "C"])
        self.assertEqual(len(batch.claims), 1)
        self.assertIn("已优先保留 Claim 端点", batch.rejected[-1])

    def test_paraphrased_quote_and_actual_source_are_both_preserved(self):
        passages = segment_text(
            "梯度下降通过沿目标函数的负梯度方向更新参数。"
        )
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "梯度下降法",
                        "definition": "一种迭代优化算法",
                        "entity_type": "solution",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "沿着梯度相反方向逐步调整参数",
                        },
                    }
                ],
                "claims": [],
            },
            passages,
        )
        self.assertEqual(len(batch.entities), 1)
        self.assertEqual(
            batch.entities[0].source_text,
            "梯度下降通过沿目标函数的负梯度方向更新参数。",
        )
        self.assertEqual(
            batch.entities[0].model_quote,
            "沿着梯度相反方向逐步调整参数",
        )

    def test_rejects_invalid_passage_references(self):
        text = "原文只介绍梯度下降法是一种迭代优化算法。"
        passages = segment_text(text)
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "Transformer",
                        "definition": "一种神经网络架构",
                        "entity_type": "solution",
                        "evidence": {
                            "passage_ids": ["P999999"],
                            "quote": "Transformer 是一种神经网络架构",
                        },
                    },
                    {
                        "name": "梯度下降法",
                        "definition": "一种迭代优化算法",
                        "entity_type": "solution",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "梯度下降是一种优化方法",
                        },
                    },
                ],
                "claims": [
                    {
                        "subject": "梯度下降法",
                        "relation": "is_a",
                        "object": "优化算法",
                        "stance": "support",
                        "evidence": {
                            "passage_ids": ["P999999"],
                            "quote": "梯度下降法是一种优化算法",
                        },
                    }
                ],
            },
            passages,
        )
        self.assertEqual([item.name for item in batch.entities], ["梯度下降法"])
        self.assertEqual(batch.claims, ())
        self.assertGreaterEqual(len(batch.rejected), 2)

    def test_rejects_duplicate_passage_references(self):
        passages = segment_text("梯度下降法是一种迭代优化算法。")
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "梯度下降法",
                        "definition": "一种迭代优化算法",
                        "entity_type": "solution",
                        "evidence": {
                            "passage_ids": ["P000001", "P000001"],
                            "quote": "梯度下降法是一种迭代优化算法",
                        },
                    }
                ],
                "claims": [],
            },
            passages,
        )
        self.assertFalse(batch.entities)
        self.assertIn("不能重复", batch.rejected[0])

    def test_only_six_types_and_three_relations_are_accepted(self):
        text = "A 是 B 的一种。"
        passages = segment_text(text)
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "A",
                        "definition": "一个可稳定复指的测试对象",
                        "entity_type": "model",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "A 是 B 的一种",
                        },
                    }
                ],
                "claims": [
                    {
                        "subject": "A",
                        "relation": "related_to",
                        "object": "B",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "A 是 B 的一种",
                        },
                    }
                ],
            },
            passages,
        )
        self.assertFalse(batch.entities)
        self.assertFalse(batch.claims)


if __name__ == "__main__":
    unittest.main()
