from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from kg import db, pipeline, resolution, store
from kg.models import ClaimObservation, EntityObservation, ExtractionBatch
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

    def test_claim_judges_can_run_in_parallel(self):
        barrier = threading.Barrier(2)
        thread_ids: set[int] = set()

        def judge(_llm, claim):
            thread_ids.add(threading.get_ident())
            barrier.wait(timeout=1)
            return "supports", claim

        with mock.patch(
            "kg.pipeline.validation.judge_claim", side_effect=judge
        ):
            results = pipeline._judge_claims(
                object(), ["first", "second"], workers=2
            )

        self.assertEqual(
            results,
            [("supports", "first"), ("supports", "second")],
        )
        self.assertEqual(len(thread_ids), 2)

    def test_chunk_extraction_parallelism_preserves_serial_write_order(self):
        text = "\n\n".join(
            f"标记{index}：这一段用于验证有序并行抽取。" * 8
            for index in range(4)
        )
        catalog = self._catalog([text])

        def run(target, workers, *, require_overlap):
            conn = db.connect(target)
            barrier = threading.Barrier(2) if require_overlap else None
            thread_ids: set[int] = set()

            def extract(_llm, chunk_text, *, passages, location, **_kwargs):
                thread_ids.add(threading.get_ident())
                if barrier is not None and len(thread_ids) <= 2:
                    barrier.wait(timeout=2)
                marker = next(
                    value for value in range(4) if f"标记{value}" in chunk_text
                )
                passage = passages[0]
                return ExtractionBatch(
                    entities=(),
                    claims=(
                        ClaimObservation(
                            subject=f"主体{marker}",
                            relation="part_of",
                            object=f"整体{marker}",
                            model_quote=passage.text,
                            source_text=passage.text,
                            passage_ids=(passage.passage_id,),
                            location=location,
                        ),
                    ),
                )

            with mock.patch("kg.pipeline.extraction.extract", side_effect=extract), mock.patch(
                "kg.pipeline.validation.judge_claim",
                return_value=("supports", "测试关系"),
            ):
                result = pipeline.process_catalog(
                    conn,
                    object(),
                    catalog,
                    max_chunks=4,
                    chunk_chars=240,
                    overlap_chars=0,
                    chunk_workers=workers,
                )
            rows = conn.execute(
                """
                SELECT id,chunk_index,subject_name,relation,object_name
                FROM claim_observations ORDER BY id
                """
            ).fetchall()
            snapshot = [tuple(row) for row in rows]
            conn.close()
            self.assertFalse(result["failures"])
            return snapshot, thread_ids

        serial, _ = run(self.root / "serial.db", 1, require_overlap=False)
        parallel, parallel_threads = run(
            self.root / "parallel.db", 2, require_overlap=True
        )

        self.assertEqual(parallel, serial)
        self.assertEqual(
            [row[1] for row in parallel],
            sorted(row[1] for row in parallel),
        )
        self.assertEqual(len(parallel_threads), 2)

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

    def test_start_chunk_ignores_earlier_chunks_without_spending_limit(self):
        text = "\n\n".join(
            f"第 {index} 段包含足够长的测试正文。" * 12
            for index in range(4)
        )
        catalog = self._catalog([text])
        llm = FakeLLM({"entities": [], "claims": []})

        result = pipeline.process_catalog(
            self.conn,
            llm,
            catalog,
            start_chunk=1,
            max_chunks=1,
            chunk_chars=240,
            overlap_chars=0,
        )

        self.assertFalse(result["failures"])
        self.assertEqual(result["completed"][0]["before_start_chunks"], 1)
        self.assertEqual(result["completed"][0]["processed_chunks"], 1)
        progress = self.conn.execute(
            "SELECT chunk_index,status FROM source_progress"
        ).fetchall()
        self.assertEqual(
            [(row["chunk_index"], row["status"]) for row in progress],
            [(1, "done")],
        )
        llm.assert_finished()

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

    def test_claim_judge_failure_preserves_grounded_observation(self):
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
        observation = self.conn.execute(
            "SELECT * FROM claim_observations"
        ).fetchone()
        self.assertIsNotNone(observation)
        self.assertEqual(observation["subject_name"], "实体甲")
        self.assertTrue(observation["source_text"])
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM claim_observation_judgments"
            ).fetchone()[0],
            0,
        )

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

    def test_reconcile_limit_prioritizes_highest_similarity_pair(self):
        source_id = self.conn.execute(
            """
            INSERT INTO sources
            (source_key,name,source_type,version,content,content_hash)
            VALUES ('rank','Rank','test','1','正文','rank-hash')
            """
        ).lastrowid
        ids = []
        for name in ("实体甲", "实体乙", "实体丙"):
            item = EntityObservation(
                name=name,
                definition=f"{name} 的独立定义",
                entity_type="concept",
                model_quote=name,
                source_text=name,
                passage_ids=("P000001",),
                location="P000001",
            )
            entity_id = store.create_entity(self.conn, item)
            store.add_evidence(
                self.conn,
                source_id=source_id,
                source_text=name,
                model_quote=name,
                passage_ids=("P000001",),
                location="P000001",
                polarity="support",
                observed_entity_type="concept",
                entity_id=entity_id,
            )
            ids.append(entity_id)
        self.conn.commit()

        def candidates(_conn, name, **_kwargs):
            if name != "实体甲":
                return []
            return [
                {"id": ids[1], "score": 0.6},
                {"id": ids[2], "score": 0.95},
            ]

        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "",
                "reason": "不同对象",
            }
        )
        with mock.patch(
            "kg.resolution.candidate_entities", side_effect=candidates
        ):
            report = resolution.reconcile(self.conn, llm, limit=1)

        self.assertEqual(report["distinct"][0]["ids"], [ids[0], ids[2]])
        self.assertEqual(report["distinct"][0]["score"], 0.95)

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

    def test_exact_name_with_conflicting_type_requires_llm_identity_judgment(self):
        section = EntityObservation(
            name="玻尔兹曼机",
            definition="第 15 章下编号为 15.1 的章节资源",
            entity_type="resource",
            model_quote="15.1 玻尔兹曼机",
            source_text="第 15 章 深度信念网络\n15.1 玻尔兹曼机",
            passage_ids=("P000001",),
            location="P000001",
        )
        section_id = store.create_entity(self.conn, section)
        store.add_evidence(
            self.conn,
            source_id=self.conn.execute(
                """
                INSERT INTO sources
                (source_key,name,source_type,version,content,content_hash)
                VALUES ('toc','目录','textbook','1','目录','toc-hash')
                """
            ).lastrowid,
            source_text=section.source_text,
            model_quote=section.model_quote,
            passage_ids=section.passage_ids,
            location=section.location,
            polarity="support",
            observed_entity_type="resource",
            entity_id=section_id,
        )
        algorithm = EntityObservation(
            name="玻尔兹曼机",
            definition="由能量函数定义的随机神经网络模型",
            entity_type="solution",
            model_quote="玻尔兹曼机是一种随机神经网络",
            source_text="玻尔兹曼机是一种随机神经网络",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "玻尔兹曼机（模型）",
                "reason": "同名章节资源与模型是不同对象",
            }
        )

        resolved = resolution.resolve_observation(self.conn, llm, algorithm)

        self.assertEqual(resolved.outcome, "new")
        self.assertNotEqual(resolved.entity_id, section_id)
        self.assertEqual(
            store.get_entity(self.conn, resolved.entity_id)["canonical_name"],
            "玻尔兹曼机（模型）",
        )
        self.assertIn("同名也不构成 same", llm.calls[0][1])
        llm.assert_finished()

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
        row = self.conn.execute(
            """
            SELECT o.subject_name,o.source_text,j.verdict,j.reason
            FROM claim_observations o
            JOIN claim_observation_judgments j ON j.observation_id=o.id
            """
        ).fetchone()
        self.assertEqual(row["subject_name"], "梯度下降法")
        self.assertEqual(row["verdict"], "insufficient")
        self.assertTrue(row["source_text"])


if __name__ == "__main__":
    unittest.main()
