from __future__ import annotations

import unittest

from kg.extraction import evidence_in_text, parse_payload


class ExtractionTest(unittest.TestCase):
    def test_evidence_tolerates_whitespace_and_ellipsis(self):
        self.assertTrue(
            evidence_in_text(
                "梯度下降…负梯度方向",
                "梯度 下降通过沿目标函数的负梯度方向更新参数。",
            )
        )

    def test_rejects_ungrounded_model_output(self):
        text = "原文只介绍梯度下降法是一种迭代优化算法。"
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "Transformer",
                        "definition": "一种神经网络架构",
                        "entity_type": "solution",
                        "evidence": "Transformer 是一种神经网络架构",
                    },
                    {
                        "name": "梯度下降法",
                        "definition": "一种迭代优化算法",
                        "entity_type": "solution",
                        "evidence": "梯度下降法是一种迭代优化算法",
                    },
                ],
                "claims": [
                    {
                        "subject": "梯度下降法",
                        "relation": "is_a",
                        "object": "优化算法",
                        "stance": "support",
                        "evidence": "梯度下降法是一种优化算法",
                    }
                ],
            },
            text,
        )
        self.assertEqual([item.name for item in batch.entities], ["梯度下降法"])
        self.assertEqual(batch.claims, ())
        self.assertGreaterEqual(len(batch.rejected), 2)

    def test_only_six_types_and_three_relations_are_accepted(self):
        text = "A 是 B 的一种。"
        batch = parse_payload(
            {
                "entities": [
                    {
                        "name": "A",
                        "definition": "一个可稳定复指的测试对象",
                        "entity_type": "model",
                        "evidence": "A 是 B 的一种",
                    }
                ],
                "claims": [
                    {
                        "subject": "A",
                        "relation": "related_to",
                        "object": "B",
                        "evidence": "A 是 B 的一种",
                    }
                ],
            },
            text,
        )
        self.assertFalse(batch.entities)
        self.assertFalse(batch.claims)


if __name__ == "__main__":
    unittest.main()
