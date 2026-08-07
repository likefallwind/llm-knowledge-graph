from __future__ import annotations


import unittest

from kg import db, extraction, ontology, resolution, validation
from kg.models import ENTITY_TYPES, RELATIONS, ClaimObservation, EntityObservation
from tests.helpers import FakeLLM


class OntologyDefinitionTest(unittest.TestCase):
    def test_definitions_cover_exactly_the_stored_vocabularies(self):
        self.assertEqual({d.name for d in ontology.ENTITY_TYPE_DEFS}, ENTITY_TYPES)
        self.assertEqual({d.name for d in ontology.RELATION_DEFS}, RELATIONS)

    def test_every_definition_has_a_test_and_exclusions(self):
        for item in (*ontology.ENTITY_TYPE_DEFS, *ontology.RELATION_DEFS):
            with self.subTest(name=item.name):
                self.assertTrue(item.test.strip(), "缺少判定测试")
                self.assertTrue(item.excludes, "缺少排除项")
                self.assertTrue(item.positive, "缺少正例")


class DefinitionsReachThePromptsTest(unittest.TestCase):
    """定义必须真的送到 LLM。曾经最强的 part_of 措辞只存在于文档里，
    抽取和裁判看到的都是剥掉排除项的弱版本，因此需要这组回归。"""

    def test_two_pass_prompts_keep_open_vocab_and_evidence_boundary(self):
        self.assertIn("类型标签是开放的", extraction.ENTITY_PROMPT)
        self.assertIn("不受预设关系词表限制", extraction.RELATION_PROMPT)
        self.assertIn("目录和摘要只提供定位上下文", extraction.RELATION_PROMPT)
        self.assertIn("不能单独证明关系", extraction.RELATION_PROMPT)

    def test_judge_prompt_carries_the_relation_exclusions(self):
        for relation in sorted(RELATIONS):
            llm = FakeLLM({"verdict": "insufficient", "reason": "r"})
            validation.judge_claim(
                llm,
                ClaimObservation(
                    subject="甲",
                    relation=relation,
                    object="乙",
                    model_quote="q",
                    source_text="t",
                    passage_ids=("P000001",),
                    location="loc",
                    polarity="support",
                ),
            )
            user_prompt = llm.calls[0][1]
            for text in ontology.RELATION_BY_NAME[relation].excludes:
                with self.subTest(relation=relation, exclude=text[:20]):
                    self.assertIn(text, user_prompt)

    def test_live_failure_boundaries_remain_explicit(self):
        part_of = ontology.RELATION_BY_NAME["part_of"]
        prerequisite = ontology.RELATION_BY_NAME["prerequisite_of"]
        self.assertTrue(
            any("完整 resource 身份" in text for text in part_of.excludes)
        )
        self.assertTrue(
            any("可替换选项" in text for text in part_of.excludes)
        )
        self.assertTrue(
            any("建议先浏览" in text for text in prerequisite.excludes)
        )
        self.assertIn("实质性学习依赖", prerequisite.test)
        self.assertIn("不要求绝对逻辑必需", prerequisite.test)
        self.assertTrue(
            any("具体依赖机制" in text for text in prerequisite.excludes)
        )

    def test_resolution_prompt_explains_entity_types(self):
        prompt_seen: list[str] = []

        class RecordingLLM(FakeLLM):
            def complete_json(self, system: str, user: str, **kwargs):
                prompt_seen.append(user)
                return super().complete_json(system, user, **kwargs)

        conn = db.connect(":memory:")
        self.addCleanup(conn.close)
        # 先建一个实体，否则唯一精确匹配路径会跳过 LLM 调用。
        conn.execute(
            "INSERT INTO entities(canonical_name,normalized_name,definition)"
            " VALUES ('梯度下降法','梯度下降法','一种优化方法')"
        )
        conn.commit()
        llm = RecordingLLM(
            {"decision": "uncertain", "canonical_name": "梯度下降", "reason": "r"}
        )
        resolution.resolve_observation(
            conn,
            llm,
            EntityObservation(
                name="梯度下降",
                definition="沿负梯度方向更新参数的优化方法",
                entity_type="solution",
                model_quote="q",
                source_text="t",
                passage_ids=("P000001",),
                location="loc",
            ),
        )
        self.assertTrue(prompt_seen, "身份裁决应当调用了 LLM")
        for item in ontology.ENTITY_TYPE_DEFS:
            with self.subTest(name=item.name):
                self.assertIn(item.name, prompt_seen[0])


class PromptVersionsAreBumpedTest(unittest.TestCase):
    """三个提示词版本都参与断点续跑指纹，语义改动后必须换值。"""

    def test_versions_are_distinct_and_current(self):
        versions = {
            extraction.EXTRACTION_PROMPT_VERSION,
            resolution.RESOLUTION_PROMPT_VERSION,
            validation.VALIDATION_PROMPT_VERSION,
        }
        self.assertEqual(len(versions), 3)
        self.assertNotIn("grounded-extract-passages-1", versions)
        self.assertNotIn("relation-judge-passages-1", versions)
        self.assertEqual(
            resolution.RESOLUTION_PROMPT_VERSION,
            "entity-identity-ontology-3",
        )
        self.assertEqual(extraction.ENTITY_PROMPT_VERSION, "open-entities-section-1")
        self.assertEqual(extraction.RELATION_PROMPT_VERSION, "open-relations-section-1")
        self.assertEqual(validation.VALIDATION_PROMPT_VERSION, "open-relation-judge-1")


if __name__ == "__main__":
    unittest.main()
