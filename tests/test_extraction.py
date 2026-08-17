from __future__ import annotations

import unittest

from kg.extraction import extract, parse_payload
from kg.sources import segment_text
from tests.helpers import FakeLLM


class ExtractionTest(unittest.TestCase):
    def test_entity_prompt_is_open_and_relation_free(self):
        passages = segment_text("批量梯度下降法是梯度下降法的一种。")
        llm = FakeLLM({"entities": [], "claims": []})

        extract(llm, "测试片段", passages=passages)

        prompt = llm.calls[0][1]
        self.assertIn("类型标签是开放的", prompt)
        self.assertIn("这一阶段不要输出关系", prompt)
        self.assertIn("最多输出 50 个实体", prompt)
        self.assertIn("train_ch6", prompt)
        self.assertIn("证据来源测试", prompt)
        self.assertIn("不能单独产生 Entity", prompt)
        self.assertIn("d2l.Timer", prompt)
        self.assertIn("召回保护", prompt)
        self.assertIn("随机梯度下降与梯度下降", prompt)

    def test_negative_statement_is_support_not_opposing_evidence(self):
        passages = segment_text("在非凸问题中，SGD 通常不收敛到全局最优解。")
        batch = parse_payload(
            {
                "entities": [],
                "claims": [
                    {
                        "subject": "SGD",
                        "relation": "通常不收敛到",
                        "object": "全局最优解",
                        "statement": "在非凸问题中，SGD 通常不收敛到全局最优解。",
                        "scope": "在非凸问题中",
                        "scope_is_restrictive": True,
                        "stance": "oppose",
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "SGD 通常不收敛到全局最优解",
                        },
                    }
                ],
            },
            passages,
        )

        self.assertEqual(len(batch.claims), 1)
        self.assertEqual(batch.claims[0].polarity, "support")

    def test_complete_assertion_preserves_restrictive_scope(self):
        passages = segment_text(
            "对于凸问题，当学习率选择适当时，随机梯度下降收敛到全局最优解。"
        )
        batch = parse_payload(
            {
                "entities": [],
                "claims": [
                    {
                        "subject": "随机梯度下降",
                        "relation": "收敛到",
                        "object": "全局最优解",
                        "statement": (
                            "对于凸问题，当学习率选择适当时，"
                            "随机梯度下降收敛到全局最优解"
                        ),
                        "scope": "对于凸问题，当学习率选择适当时",
                        "scope_is_restrictive": True,
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "对于凸问题，当学习率选择适当时",
                        },
                    }
                ],
            },
            passages,
        )

        self.assertEqual(len(batch.claims), 1)
        claim = batch.claims[0]
        self.assertTrue(claim.scope_is_restrictive)
        self.assertIn("凸问题", claim.scope_text)
        self.assertIn("学习率", claim.statement_text)

    def test_relation_without_complete_statement_is_rejected(self):
        passages = segment_text("A 是 B 的一种。")
        batch = parse_payload(
            {
                "entities": [],
                "claims": [
                    {
                        "subject": "A",
                        "relation": "is_a",
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

        self.assertEqual(batch.claims, ())
        self.assertIn("缺少完整 statement", batch.rejected[0])

    def test_endpoint_mismatch_rejection_keeps_inspectable_payload(self):
        passages = segment_text("在非凸情况下，SGD 的保证通常不可用。")
        batch = parse_payload(
            {
                "entities": [],
                "claims": [
                    {
                        "subject": "随机梯度下降",
                        "relation": "保证不可用",
                        "object": "非凸问题",
                        "statement": "在非凸情况下，SGD 的保证通常不可用。",
                        "scope": "非凸情况下",
                        "scope_is_restrictive": True,
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "SGD 的保证通常不可用",
                        },
                    }
                ],
            },
            passages,
        )

        self.assertEqual(batch.claims, ())
        reason = batch.rejected[0]
        self.assertIn("subject='随机梯度下降'", reason)
        self.assertIn("object='非凸问题'", reason)
        self.assertIn("statement='在非凸情况下", reason)

    def test_entity_cap_does_not_discard_claim_observation(self):
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
                    "statement": "B 是 C 的一种",
                    "scope": "",
                    "scope_is_restrictive": False,
                    "stance": "support",
                    "evidence": {
                        "passage_ids": ["P000001"],
                        "quote": "B 是 C 的一种",
                    },
                }
            ],
        }

        batch = parse_payload(payload, passages, max_entities=2)

        self.assertEqual([item.name for item in batch.entities], ["A", "B"])
        self.assertEqual(len(batch.claims), 1)
        self.assertIn("entities 超过上限 2，已截断", batch.rejected[-1])

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
                        "statement": "梯度下降法是一种优化算法",
                        "scope": "",
                        "scope_is_restrictive": False,
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

    def test_open_types_and_relations_are_accepted(self):
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
                        "statement": "A 与 B 相关",
                        "scope": "",
                        "scope_is_restrictive": False,
                        "evidence": {
                            "passage_ids": ["P000001"],
                            "quote": "A 是 B 的一种",
                        },
                    }
                ],
            },
            passages,
        )
        self.assertEqual(batch.entities[0].type_labels, ("model",))
        self.assertEqual(batch.claims[0].raw_relation, "related_to")


if __name__ == "__main__":
    unittest.main()
