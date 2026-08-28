from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from kg import (
    db,
    llm as llm_module,
    observations,
    pipeline,
    resolution,
    store,
    vocabulary,
)
from kg.models import (
    ClaimObservation,
    EntityObservation,
    ExtractionBatch,
    Resolution,
)
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

    def test_consecutive_failures_pause_after_three_and_success_resets(self):
        pauser = pipeline._ConsecutiveFailurePauser()

        with mock.patch("kg.pipeline.time.sleep") as sleep:
            pauser.record_failure()
            pauser.record_failure()
            sleep.assert_not_called()

            pauser.record_success()
            pauser.record_failure()
            pauser.record_failure()
            sleep.assert_not_called()

            with self.assertLogs("kg.pipeline", level="WARNING") as logs:
                pauser.record_failure()

        sleep.assert_called_once_with(600)
        self.assertEqual(pauser.count, 0)
        self.assertIn("连续 3 个 Chunk 失败，暂停 600 秒后继续", logs.output[0])

    def test_quota_exhausted_pauses_on_the_first_failure(self):
        pauser = pipeline._ConsecutiveFailurePauser()

        with mock.patch("kg.pipeline.time.sleep") as sleep:
            with self.assertLogs("kg.pipeline", level="WARNING") as logs:
                pauser.record_failure(immediate=True)

        sleep.assert_called_once_with(600)
        self.assertEqual(pauser.count, 0)
        self.assertIn("额度耗尽，暂停 600 秒后继续", logs.output[0])

    def test_quota_exhausted_is_detected_through_a_wrapped_exception(self):
        quota = llm_module.LLMResponseError(
            {"status_code": 2067, "status_msg": "当前已达到 Token Plan 用量上限。"}
        )
        balance = llm_module.LLMResponseError({"status_code": 1008})
        rate_limited = llm_module.LLMResponseError({"status_code": 2062})

        self.assertTrue(llm_module.is_quota_exhausted(quota))
        self.assertTrue(llm_module.is_quota_exhausted(balance))
        # 限流和普通失败必须继续走"连续 3 次"这条路，不能一次就停 10 分钟。
        self.assertFalse(llm_module.is_quota_exhausted(rate_limited))
        self.assertFalse(llm_module.is_quota_exhausted(RuntimeError("boom")))
        self.assertFalse(llm_module.is_quota_exhausted(None))

        # 真实路径：异常在抽取深处抛出，被逐层包裹后才到 pipeline。
        try:
            try:
                raise quota
            except llm_module.LLMResponseError as exc:
                raise RuntimeError("chunk 130 抽取失败") from exc
        except RuntimeError as wrapped:
            self.assertTrue(llm_module.is_quota_exhausted(wrapped))

        # 额度耗尽不该再浪费本层的秒级重试。
        self.assertFalse(quota.retryable)
        self.assertTrue(rate_limited.retryable)

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
                            statement_text=f"主体{marker} 是 整体{marker} 的一部分",
                        ),
                    ),
                )

            with mock.patch("kg.pipeline.extraction.extract", side_effect=extract), mock.patch(
                "kg.pipeline.validation.judge_claim",
                return_value=("supports", "测试关系"),
            ), mock.patch(
                "kg.pipeline.vocabulary.resolve_relation",
                return_value=vocabulary.RelationResolution(
                    2, "part_of", "part_of", "same", "测试关系"
                ),
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
                    "statement": "批量梯度下降法是梯度下降法的一种",
                    "scope": "",
                    "scope_is_restrictive": False,
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
                    "statement": "批量梯度下降法是梯度下降法的一种",
                    "scope": "",
                    "scope_is_restrictive": False,
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
            {"decision": "same", "candidate_id": 1,
             "projection_statement": "批量梯度下降法是梯度下降法的一种",
             "register_alias": False, "reason": "方向和语义一致"},
            {"assertion_verdict": "supports",
             "projection_statement": "卷积神经网络是神经网络的一种",
             "projection_faithful": True, "reason": "明确说是一种"},
            second_extraction,
            {
                "decision": "same",
                "candidate_id": 1,
                "canonical_name": "梯度下降法",
                "accepted_aliases": [],
                "reason": "同一优化算法的重复观察",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "global_name",
                "strongest_identity_conflict": "不存在",
                "reason": "确认是同一算法",
            },
            {
                "decision": "same",
                "candidate_id": 2,
                "canonical_name": "批量梯度下降法",
                "accepted_aliases": [],
                "reason": "同一算法变体的重复观察",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "global_name",
                "strongest_identity_conflict": "不存在",
                "reason": "确认是同一算法变体",
            },
            {"decision": "same", "candidate_id": 1,
             "projection_statement": "批量梯度下降法是梯度下降法的一种",
             "register_alias": False, "reason": "方向和语义一致"},
            {"assertion_verdict": "supports",
             "projection_statement": "卷积神经网络是神经网络的一种",
             "projection_faithful": True, "reason": "独立来源明确支持"},
        )
        result = pipeline.process_catalog(self.conn, llm, catalog)
        self.assertFalse(result["failures"])
        self.assertEqual(store.counts(self.conn)["sources"], 2)
        self.assertEqual(store.counts(self.conn)["entities"], 2)
        self.assertEqual(store.counts(self.conn)["claims"], 1)
        self.assertEqual(store.counts(self.conn)["assertions"], 1)
        entity_observations = self.conn.execute(
            "SELECT * FROM entity_observations ORDER BY id"
        ).fetchall()
        self.assertEqual(len(entity_observations), 4)
        self.assertTrue(all(row["entity_id"] for row in entity_observations))
        self.assertEqual(
            {str(row["resolution_outcome"]) for row in entity_observations},
            {"new", "same"},
        )
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

    def test_catalog_main_flow_synthesizes_changed_entity_definitions(self):
        catalog = self._catalog(
            [
                "卷积神经网络是处理图像的强大工具。",
                "卷积神经网络是包含卷积层的一类特殊神经网络。",
            ]
        )
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload(
                        "卷积神经网络",
                        "处理图像的强大工具",
                        "卷积神经网络是处理图像的强大工具",
                    )
                ],
                "claims": [],
            },
            {
                "decision": "new",
                "canonical_name": "卷积神经网络",
                "reason": "当前语料首次定义该实体",
            },
            {
                "entities": [
                    entity_payload(
                        "卷积神经网络",
                        "包含卷积层的一类特殊神经网络",
                        "卷积神经网络是包含卷积层的一类特殊神经网络",
                    )
                ],
                "claims": [],
            },
            {
                "decision": "same",
                "candidate_id": 1,
                "canonical_name": "卷积神经网络",
                "accepted_aliases": [],
                "reason": "同一概念的另一条定义观察",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "global_name",
                "strongest_identity_conflict": "不存在",
                "reason": "确认是同一概念",
            },
            {
                "definition": "卷积神经网络是包含卷积层的一类特殊神经网络。",
                "supporting_observations": [
                    {
                        "observation_id": 2,
                        "passage_ids": ["P000001"],
                        "support": "直接给出上位类别和结构特征",
                    }
                ],
                "rejected_candidates": ["强大工具只说明作用"],
                "limitation": "",
            },
        )

        result = pipeline.process_catalog(
            self.conn, llm, catalog, synthesize_definitions=True
        )

        self.assertFalse(result["failures"])
        self.assertEqual(len(result["definition_synthesis"]["processed"]), 1)
        row = self.conn.execute(
            "SELECT definition FROM entities WHERE canonical_name='卷积神经网络'"
        ).fetchone()
        self.assertEqual(
            row["definition"],
            "卷积神经网络是包含卷积层的一类特殊神经网络。",
        )
        llm.assert_finished()

        no_calls = FakeLLM()
        rerun = pipeline.process_catalog(
            self.conn, no_calls, catalog, synthesize_definitions=True
        )
        self.assertFalse(rerun["failures"])
        self.assertEqual(len(rerun["definition_synthesis"]["skipped"]), 1)
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
        observations = self.conn.execute(
            "SELECT * FROM entity_observations ORDER BY id"
        ).fetchall()
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(row["entity_id"] is None for row in observations))
        self.assertTrue(
            all(str(row["resolution_outcome"]) == "" for row in observations)
        )

    def test_entity_observation_records_resolution_provenance(self):
        catalog = self._catalog(["GD 是沿负梯度方向更新参数的优化方法。"])
        llm = FakeLLM(
            {
                "entities": [
                    entity_payload(
                        "GD",
                        "沿负梯度方向更新参数的优化方法",
                        "GD 是沿负梯度方向更新参数的优化方法",
                    )
                ],
                "claims": [],
            },
            {
                "decision": "new",
                "canonical_name": "梯度下降法",
                "reason": "当前原文给出了完整定义",
            },
        )

        result = pipeline.process_catalog(
            self.conn, llm, catalog, max_entities=1
        )

        self.assertFalse(result["failures"])
        row = self.conn.execute("SELECT * FROM entity_observations").fetchone()
        self.assertEqual(row["name"], "GD")
        self.assertEqual(row["resolution_outcome"], "new")
        self.assertEqual(row["resolution_reason"], "当前原文给出了完整定义")
        self.assertEqual(row["resolver_model"], "FakeLLM")
        self.assertTrue(row["resolver_prompt_version"])
        self.assertIsNotNone(row["entity_id"])
        self.assertEqual(
            result["completed"][0]["entity_observations"], 1
        )
        self.assertEqual(
            result["completed"][0]["entity_cap_hit_chunks"], [0]
        )

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
                        "statement": "实体甲是实体乙的一种",
                        "scope": "",
                        "scope_is_restrictive": False,
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
            {"decision": "same", "candidate_id": 1,
             "projection_statement": "实体甲是实体乙的一种",
             "register_alias": False, "reason": "方向和语义一致"},
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

    def test_reconcile_uses_saved_cross_language_uncertain_candidates(self):
        source_id = self.conn.execute(
            """
            INSERT INTO sources
            (source_key,name,source_type,version,content,content_hash)
            VALUES ('saved-cross-lang','S','test','1','正文','saved-cross-lang-hash')
            """
        ).lastrowid
        chinese = EntityObservation(
            name="支持向量机",
            definition="最大间隔分类方法",
            entity_type="algorithm",
            model_quote="支持向量机",
            source_text="支持向量机使用最大间隔。",
            passage_ids=("P000001",),
            location="P000001",
        )
        english = replace(
            chinese,
            name="Support Vector Machine",
            model_quote="Support Vector Machine",
            source_text="Support Vector Machine maximizes the margin.",
            passage_ids=("P000002",),
            location="P000002",
        )
        chinese_id = store.create_entity(self.conn, chinese)
        english_id = store.create_entity(self.conn, english)
        observation_id, _ = observations.add_entity_observation(
            self.conn,
            source_id=source_id,
            chunk_index=0,
            observation=english,
            extraction_model="test",
        )
        observations.save_entity_resolution(
            self.conn,
            observation_id,
            Resolution(
                entity_id=english_id,
                outcome="uncertain",
                reason="跨语言候选待复核",
                candidates=(chinese_id,),
            ),
            resolver_model="test",
        )
        self.conn.commit()
        llm = FakeLLM(
            {
                "decision": "same",
                "canonical_name": "支持向量机",
                "reason": "中英文名称是同一算法",
            }
        )

        with mock.patch("kg.resolution.candidate_entities", return_value=[]):
            report = resolution.reconcile(self.conn, llm, limit=1)

        self.assertEqual(len(report["merged"]), 1)
        self.assertEqual(report["merged"][0]["score"], 1.0)
        self.assertEqual(store.counts(self.conn)["entities"], 1)

    def test_reconcile_ignores_invalid_missing_or_self_saved_candidates(self):
        source_id = self.conn.execute(
            """
            INSERT INTO sources
            (source_key,name,source_type,version,content,content_hash)
            VALUES ('invalid-saved','S','test','1','正文','invalid-saved-hash')
            """
        ).lastrowid
        observed = EntityObservation(
            name="独立对象",
            definition="独立定义",
            entity_type="concept",
            model_quote="独立对象",
            source_text="独立对象",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, observed)
        observation_id, _ = observations.add_entity_observation(
            self.conn,
            source_id=source_id,
            chunk_index=0,
            observation=observed,
            extraction_model="test",
        )
        observations.save_entity_resolution(
            self.conn,
            observation_id,
            Resolution(
                entity_id=entity_id,
                outcome="uncertain",
                reason="无效候选测试",
                candidates=(entity_id, 999999),
            ),
            resolver_model="test",
        )
        self.conn.commit()

        with mock.patch("kg.resolution.candidate_entities", return_value=[]):
            report = resolution.reconcile(self.conn, FakeLLM(), limit=10)

        self.assertEqual(report["examined"], 0)
        self.assertEqual(report["merged"], [])
        self.assertEqual(store.counts(self.conn)["entities"], 1)

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

    def test_resolver_promotes_only_explicitly_accepted_alias_suggestions(self):
        observed = EntityObservation(
            name="小批量随机梯度下降",
            definition="使用小批量样本估计梯度的优化算法",
            entity_type="solution",
            model_quote="小批量随机梯度下降",
            source_text="小批量随机梯度下降使用小批量样本估计梯度。",
            passage_ids=("P000001",),
            location="P000001",
            aliases=("mini-batch SGD", "随机梯度下降"),
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "小批量随机梯度下降",
                "accepted_aliases": ["mini-batch SGD", "未被抽取的别名"],
                "knowledge_aliases": ["mini-batch stochastic gradient descent"],
                "identity_basis": "首次建立独立算法对象",
                "distinguishing_test": "可与随机梯度下降比较，不能合并",
                "reason": "新实体",
            }
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)
        aliases = store.aliases_for(self.conn, resolved.entity_id)

        self.assertIn("mini-batch SGD", aliases)
        self.assertNotIn("mini-batch stochastic gradient descent", aliases)
        self.assertIn(
            "mini-batch stochastic gradient descent",
            store.alias_candidates_for(self.conn, resolved.entity_id),
        )
        self.assertNotIn("随机梯度下降", aliases)
        self.assertNotIn("未被抽取的别名", aliases)
        resolver_request = llm.calls[0][1]
        self.assertIn(
            '"aliases": ["mini-batch SGD", "随机梯度下降"]',
            resolver_request,
        )

    def test_knowledge_aliases_are_candidate_only_until_observed(self):
        observed = EntityObservation(
            name="支持向量机",
            definition="通过最大化分类间隔构造决策边界的监督学习方法",
            entity_type="method",
            model_quote="支持向量机使用最大间隔分类器。",
            source_text="支持向量机使用最大间隔分类器。",
            passage_ids=("P000001",),
            location="P000001",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "支持向量机",
                "accepted_aliases": [],
                "knowledge_aliases": ["Support Vector Machine", "SVM"],
                "reason": "标准英文全称与缩写可跨语境互换",
            }
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)
        aliases = store.aliases_for(self.conn, resolved.entity_id)
        candidates = store.alias_candidates_for(self.conn, resolved.entity_id)

        self.assertNotIn("Support Vector Machine", aliases)
        self.assertNotIn("SVM", aliases)
        self.assertIn("Support Vector Machine", candidates)
        self.assertIn("SVM", candidates)
        self.assertEqual(store.exact_entity_ids(self.conn, "SVM"), [])

    def test_knowledge_alias_recall_promotes_real_observation_after_same(self):
        chinese = EntityObservation(
            name="支持向量机",
            definition="通过最大化分类间隔构造决策边界的监督学习方法",
            entity_type="method",
            model_quote="支持向量机",
            source_text="支持向量机使用最大间隔分类器。",
            passage_ids=("P000001",),
            location="P000001",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "支持向量机",
                "knowledge_aliases": ["SVM"],
                "reason": "新实体",
            }
        )
        first = resolution.resolve_observation(self.conn, llm, chinese)
        incoming = replace(
            chinese,
            name="SVM",
            model_quote="SVM",
            source_text="SVM maximizes the classification margin.",
            passage_ids=("P000002",),
            location="P000002",
        )
        recalled = resolution.candidate_entities(
            self.conn, incoming.name, observation=incoming
        )
        self.assertEqual([item["id"] for item in recalled], [first.entity_id])
        self.assertEqual(recalled[0]["aliases"], ["支持向量机"])
        self.assertEqual(recalled[0]["candidate_aliases"], ["SVM"])

        same_llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": first.entity_id,
                "canonical_name": "支持向量机",
                "accepted_aliases": [],
                "knowledge_aliases": [],
                "reason": "SVM 指向支持向量机",
            },
            {
                "decision": "same",
                "identity_scope": "global_name",
                "reason": "标准缩写",
            },
        )
        second = resolution.resolve_observation(self.conn, same_llm, incoming)

        self.assertEqual(second.entity_id, first.entity_id)
        self.assertEqual(second.outcome, "same")
        self.assertIn("SVM", store.aliases_for(self.conn, first.entity_id))
        self.assertNotIn(
            "SVM", store.alias_candidates_for(self.conn, first.entity_id)
        )
        self.assertEqual(
            store.exact_entity_ids(self.conn, "SVM"), [first.entity_id]
        )

    def test_training_set_knowledge_hint_does_not_pollute_dataset_aliases(self):
        dataset = EntityObservation(
            name="数据集",
            definition="由多个数据样本组成的集合",
            entity_type="data",
            model_quote="数据集",
            source_text="每个数据集由多个样本组成。",
            passage_ids=("P000001",),
            location="P000001",
        )
        first = resolution.resolve_observation(
            self.conn,
            FakeLLM(
                {
                    "decision": "new",
                    "canonical_name": "数据集",
                    "knowledge_aliases": ["training set", "训练数据集"],
                    "reason": "新实体",
                }
            ),
            dataset,
        )

        self.assertNotIn("training set", store.aliases_for(self.conn, first.entity_id))
        self.assertIn(
            "training set",
            store.alias_candidates_for(self.conn, first.entity_id),
        )
        self.assertEqual(store.exact_entity_ids(self.conn, "training set"), [])

        training_set = replace(
            dataset,
            name="training set",
            definition="专门用于拟合模型参数的数据子集",
            model_quote="training set",
            source_text="A training set is used to fit model parameters.",
            passage_ids=("P000002",),
            location="P000002",
        )
        recalled = resolution.candidate_entities(
            self.conn, training_set.name, observation=training_set
        )
        self.assertEqual([item["id"] for item in recalled], [first.entity_id])
        second = resolution.resolve_observation(
            self.conn,
            FakeLLM(
                {
                    "decision": "new",
                    "canonical_name": "训练集",
                    "knowledge_aliases": [],
                    "reason": "训练集是数据集的特定子集，不是同一对象",
                }
            ),
            training_set,
        )

        self.assertEqual(second.outcome, "new")
        self.assertNotEqual(second.entity_id, first.entity_id)
        self.assertEqual(store.counts(self.conn)["entities"], 2)
        self.assertNotIn("training set", store.aliases_for(self.conn, first.entity_id))
        self.assertIn("training set", store.aliases_for(self.conn, second.entity_id))

    def test_candidate_recall_uses_incoming_observation_aliases(self):
        existing = EntityObservation(
            name="Gated Recurrent Unit",
            definition="使用门控机制的循环神经网络单元",
            entity_type="component",
            aliases=("GRU",),
            model_quote="Gated Recurrent Unit",
            source_text="Gated Recurrent Unit",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        incoming = EntityObservation(
            name="门控循环单元",
            definition="使用门控机制控制循环状态的信息流",
            entity_type="component",
            aliases=("GRU",),
            model_quote="门控循环单元",
            source_text="门控循环单元使用门控机制。",
            passage_ids=("P000002",),
            location="P000002",
        )

        candidates = resolution.candidate_entities(
            self.conn, incoming.name, observation=incoming
        )

        candidate = next(item for item in candidates if item["id"] == entity_id)
        self.assertEqual(candidate["score"], 1.0)

    def test_candidate_embedding_excludes_quotes_and_source_text(self):
        existing = EntityObservation(
            name="Support Vector Machine",
            definition="A maximum-margin classifier",
            entity_type="method",
            aliases=("SVM",),
            model_quote="EXISTING_QUOTE_MUST_NOT_BE_EMBEDDED",
            source_text="EXISTING_SOURCE_MUST_NOT_BE_EMBEDDED",
            passage_ids=("P000001",),
            location="P000001",
        )
        store.create_entity(self.conn, existing)
        incoming = EntityObservation(
            name="支持向量机",
            definition="最大化分类间隔的分类器",
            entity_type="method",
            aliases=("SVM",),
            model_quote="INCOMING_QUOTE_MUST_NOT_BE_EMBEDDED",
            source_text="INCOMING_SOURCE_MUST_NOT_BE_EMBEDDED",
            passage_ids=("P000002",),
            location="P000002",
        )

        with mock.patch(
            "kg.embeddings.cosine_scores", return_value=[0.9]
        ) as scorer:
            resolution.candidate_entities(
                self.conn, incoming.name, observation=incoming
            )

        query, passages = scorer.call_args.args
        embedded = "\n".join((query, *passages))
        self.assertIn("支持向量机", query)
        self.assertIn("SVM", query)
        self.assertIn("最大化分类间隔的分类器", query)
        self.assertIn("Support Vector Machine", passages[0])
        self.assertNotIn("MUST_NOT_BE_EMBEDDED", embedded)

    def test_candidate_recall_returns_ten_by_default(self):
        for index in range(12):
            item = EntityObservation(
                name=f"候选概念{index}",
                definition=f"候选概念{index}的解释",
                entity_type="concept",
                model_quote=f"候选概念{index}",
                source_text=f"候选概念{index}",
                passage_ids=("P000001",),
                location="P000001",
            )
            store.create_entity(self.conn, item)

        candidates = resolution.candidate_entities(self.conn, "候选概念")

        self.assertEqual(len(candidates), 10)

    def test_same_decision_does_not_promote_rejected_alias_suggestions(self):
        existing = EntityObservation(
            name="随机梯度下降",
            definition="每次使用一个样本估计梯度的优化算法",
            entity_type="solution",
            model_quote="随机梯度下降",
            source_text="随机梯度下降的批量大小为1。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = EntityObservation(
            name="随机梯度下降算法",
            definition="随机梯度下降的名称变体",
            entity_type="method",
            model_quote="随机梯度下降算法",
            source_text="随机梯度下降算法也称SGD。",
            passage_ids=("P000002",),
            location="P000002",
            aliases=("SGD", "mini-batch SGD"),
        )
        llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": entity_id,
                "accepted_aliases": ["SGD"],
                "identity_basis": "名称变体和缩写指向同一算法",
                "distinguishing_test": "名称可互换且对象边界不变",
                "reason": "同一算法的语言和缩写变体",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "global_name",
                "strongest_identity_conflict": "不存在；只是名称后缀和缩写差异",
                "reason": "独立确认是同一算法",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)
        aliases = store.aliases_for(self.conn, entity_id)

        self.assertEqual(resolved.outcome, "same")
        self.assertIn("随机梯度下降算法", aliases)
        self.assertIn("SGD", aliases)
        self.assertNotIn("mini-batch SGD", aliases)

    def test_exact_canonical_name_still_requires_knowledge_identity_judgment(self):
        existing = EntityObservation(
            name="全连接层",
            definition="GoogLeNet 中与 Inception 块串联的网络层",
            entity_type="网络层",
            model_quote="全连接层",
            source_text="GoogLeNet 串联卷积层和全连接层。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = EntityObservation(
            name="全连接层",
            definition="通过权重矩阵和偏置变换输入的神经网络层",
            entity_type="神经网络层",
            model_quote="全连接层",
            source_text="全连接层计算输入的仿射变换。",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": entity_id,
                "canonical_name": "全连接层",
                "accepted_aliases": [],
                "reason": "应用场景不同但仍是同一通用网络层概念",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "global_name",
                "strongest_identity_conflict": "不存在；GoogLeNet 只是应用场景",
                "reason": "通用知识确认是同一概念",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "same")
        self.assertEqual(resolved.entity_id, entity_id)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("可靠的通用知识", llm.calls[0][0])
        self.assertIn("定义不完整", llm.calls[1][1])
        llm.assert_finished()

    def test_tentative_same_requires_independent_identity_confirmation(self):
        existing = EntityObservation(
            name="感官输入",
            definition="认知过程接收的感官信息",
            entity_type="concept",
            model_quote="感官输入",
            source_text="注意力会选择感官输入。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        # Simulate a legacy/polluted alias: it may retrieve a candidate, but it
        # must not bypass either identity judgment.
        store.add_alias(self.conn, entity_id, "值")
        observed = EntityObservation(
            name="值",
            definition="注意力机制中与键配对的技术成分",
            entity_type="component",
            model_quote="值（感官输入）",
            source_text="教学类比把值写作感官输入。",
            passage_ids=("P000002",),
            location="P000002",
            aliases=("感官输入",),
        )
        llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": entity_id,
                "canonical_name": "值（注意力机制）",
                "accepted_aliases": ["感官输入"],
                "reason": "括号中并列出现",
            },
            {
                "verdict": "reject_same",
                "identity_scope": "passage_local",
                "strongest_identity_conflict": "技术角色与教学类比对象可区分",
                "reason": "局部映射不能成为全局同义",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "new")
        self.assertNotEqual(resolved.entity_id, entity_id)
        self.assertNotIn("感官输入", store.aliases_for(self.conn, resolved.entity_id))
        self.assertIn("独立复核身份", llm.calls[1][1])
        self.assertIn("可靠通用知识", llm.calls[1][1])
        self.assertIn("应用场景不同当成冲突", llm.calls[1][1])

    def test_tentative_same_confirmation_can_remain_uncertain(self):
        existing = EntityObservation(
            name="共享名称",
            definition="候选对象的定义",
            entity_type="concept",
            model_quote="共享名称",
            source_text="候选对象",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = replace(
            existing,
            name="上下文名称",
            definition="语境不足，无法判断具体义项",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": entity_id,
                "canonical_name": "上下文名称",
                "reason": "初步认为相同",
            },
            {
                "decision": "uncertain",
                "identity_scope": "uncertain",
                "reason": "当前义项仍无法确定",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "uncertain")
        self.assertNotEqual(resolved.entity_id, entity_id)
        self.assertEqual(store.counts(self.conn)["entities"], 2)

    def test_candidate_recall_includes_entity_named_in_definition(self):
        key = EntityObservation(
            name="键",
            definition="注意力机制中与查询匹配的技术成分",
            entity_type="component",
            model_quote="键与查询匹配",
            source_text="键与查询匹配",
            passage_ids=("P000001",),
            location="P000001",
        )
        key_id = store.create_entity(self.conn, key)
        observed = EntityObservation(
            name="非自主性提示",
            definition="注意力机制中与值配对的键",
            entity_type="component",
            model_quote="每个值都与一个键配对",
            source_text="键可以想象为感官输入的非自主性提示。",
            passage_ids=("P000002",),
            location="P000002",
        )

        candidates = resolution.candidate_entities(
            self.conn, observed.name, observation=observed
        )

        candidate = next(item for item in candidates if item["id"] == key_id)
        self.assertGreater(candidate["score"], 0.0)

    def test_contextual_surface_name_links_without_global_alias(self):
        general = EntityObservation(
            name="梯度下降",
            definition="沿负梯度方向更新参数的一般优化方法",
            entity_type="优化算法",
            model_quote="梯度下降",
            source_text="梯度下降沿负梯度更新参数。",
            passage_ids=("P000001",),
            location="P000001",
        )
        batch = EntityObservation(
            name="批量梯度下降",
            definition="每一步使用全部样本计算梯度",
            entity_type="优化算法",
            model_quote="批量梯度下降",
            source_text="批量梯度下降使用全部样本。",
            passage_ids=("P000002",),
            location="P000002",
        )
        general_id = store.create_entity(self.conn, general)
        batch_id = store.create_entity(self.conn, batch)
        observed = EntityObservation(
            name="梯度下降",
            definition="一次以大批量处理全部数据的优化方法",
            entity_type="优化算法",
            model_quote="大批量一次处理数据的梯度下降",
            source_text="小批量随机梯度下降比梯度下降更快。",
            passage_ids=("P000003",),
            location="P000003",
            aliases=("gradient descent",),
        )
        llm = FakeLLM(
            {
                "decision": "same",
                "candidate_id": batch_id,
                "canonical_name": "批量梯度下降",
                "accepted_aliases": ["gradient descent"],
                "reason": "当前观察实际指向全批量变体",
            },
            {
                "verdict": "confirmed_same",
                "identity_scope": "passage_referent",
                "strongest_identity_conflict": "表面名称较宽，但实际指代明确",
                "reason": "确认当前语料指向批量梯度下降",
            },
        )
        candidates = [
            {
                "id": general_id,
                "canonical_name": "梯度下降",
                "aliases": [],
                "definition": general.definition,
                "type_profile": [],
                "evidence": [],
                "score": 1.0,
            },
            {
                "id": batch_id,
                "canonical_name": "批量梯度下降",
                "aliases": [],
                "definition": batch.definition,
                "type_profile": [],
                "evidence": [],
                "score": 0.72,
            },
        ]

        with mock.patch("kg.resolution.candidate_entities", return_value=candidates):
            resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "same")
        self.assertEqual(resolved.entity_id, batch_id)
        self.assertNotIn("梯度下降", store.aliases_for(self.conn, batch_id))
        self.assertNotIn("gradient descent", store.aliases_for(self.conn, batch_id))
        self.assertIn("passage_referent", llm.calls[1][1])

    def test_distinct_same_surface_requires_disambiguated_canonical_name(self):
        existing = EntityObservation(
            name="卷积",
            definition="卷积层中不翻转卷积核的互相关运算",
            entity_type="数学运算",
            model_quote="卷积",
            source_text="卷积层实际执行互相关。",
            passage_ids=("P000001",),
            location="P000001",
        )
        existing_id = store.create_entity(self.conn, existing)
        observed = EntityObservation(
            name="卷积",
            definition="翻转一个函数后计算重叠的数学运算",
            entity_type="数学运算",
            model_quote="数学中的卷积",
            source_text="数学中的卷积需要翻转函数。",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "数学卷积",
                "reason": "与深度学习中俗称卷积的互相关不同",
            }
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertNotEqual(resolved.entity_id, existing_id)
        self.assertEqual(
            store.get_entity(self.conn, resolved.entity_id)["canonical_name"],
            "数学卷积",
        )

    def test_colliding_new_name_rechecks_identity_then_renames_if_still_new(self):
        existing = EntityObservation(
            name="梯度下降",
            definition="一般优化方法",
            entity_type="优化算法",
            model_quote="梯度下降",
            source_text="梯度下降",
            passage_ids=("P000001",),
            location="P000001",
        )
        store.create_entity(self.conn, existing)
        observed = EntityObservation(
            name="梯度下降",
            definition="语义不同但尚未正确命名的对象",
            entity_type="优化算法",
            model_quote="梯度下降",
            source_text="梯度下降",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "梯度下降",
                "reason": "不同对象但没有完成名称消歧",
            },
            {
                "decision": "new",
                "identity_scope": "not_same",
                "reason": "复判确认是不同算法",
            },
            {
                "canonical_name": "批量梯度下降",
                "naming_basis": "语料明确说明每步使用全部样本",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "new")
        self.assertEqual(store.counts(self.conn)["entities"], 2)
        self.assertEqual(
            store.get_entity(self.conn, resolved.entity_id)["canonical_name"],
            "批量梯度下降",
        )
        self.assertNotIn("梯度下降", store.aliases_for(self.conn, resolved.entity_id))
        self.assertIn("独立复核身份", llm.calls[1][1])
        self.assertIn("只修正语义命名", llm.calls[2][1])

    def test_colliding_new_name_recheck_can_link_existing_entity(self):
        existing = EntityObservation(
            name="支持向量机",
            definition="最大间隔分类方法",
            entity_type="algorithm",
            model_quote="支持向量机",
            source_text="支持向量机使用最大间隔。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = replace(
            existing,
            name="Support Vector Machine",
            model_quote="Support Vector Machine",
            source_text="Support Vector Machine maximizes the margin.",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "支持向量机",
                "reason": "首次漏掉跨语言身份",
            },
            {
                "decision": "same",
                "identity_scope": "global_name",
                "reason": "中英文标准名称指向同一算法",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "same")
        self.assertEqual(resolved.entity_id, entity_id)
        self.assertEqual(store.counts(self.conn)["entities"], 1)
        self.assertIn("Support Vector Machine", store.aliases_for(self.conn, entity_id))

    def test_colliding_new_passage_referent_rejects_all_observation_aliases(self):
        existing = EntityObservation(
            name="优化算法",
            definition="调整模型参数以优化目标函数的方法",
            entity_type="algorithm",
            model_quote="优化算法",
            source_text="优化算法用于更新参数。",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = replace(
            existing,
            name="算法",
            model_quote="算法（algorithm）",
            source_text="调整模型参数以优化目标函数的算法（algorithm）。",
            passage_ids=("P000002",),
            location="P000002",
            aliases=("algorithm",),
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "优化算法",
                "accepted_aliases": ["algorithm"],
                "reason": "首次裁决未识别局部泛称",
            },
            {
                "decision": "same",
                "identity_scope": "passage_referent",
                "reason": "当前段落中的算法指优化算法，但不是全局名称",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)
        aliases = store.aliases_for(self.conn, entity_id)

        self.assertEqual(resolved.outcome, "same")
        self.assertEqual(resolved.entity_id, entity_id)
        self.assertNotIn("算法", aliases)
        self.assertNotIn("algorithm", aliases)

    def test_colliding_new_name_recheck_can_be_uncertain(self):
        existing = EntityObservation(
            name="共享标准名",
            definition="候选对象",
            entity_type="concept",
            model_quote="共享标准名",
            source_text="候选对象",
            passage_ids=("P000001",),
            location="P000001",
        )
        entity_id = store.create_entity(self.conn, existing)
        observed = replace(
            existing,
            name="未消歧表面名",
            definition="语境不足的另一次观察",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "共享标准名",
                "reason": "首次判断为新对象",
            },
            {
                "decision": "uncertain",
                "identity_scope": "uncertain",
                "reason": "复判仍无法确定对象边界",
            },
        )

        resolved = resolution.resolve_observation(self.conn, llm, observed)

        self.assertEqual(resolved.outcome, "uncertain")
        self.assertNotEqual(resolved.entity_id, entity_id)
        self.assertEqual(
            store.get_entity(self.conn, resolved.entity_id)["canonical_name"],
            "未消歧表面名",
        )

    def test_colliding_new_name_is_rejected_if_retry_still_collides(self):
        existing = EntityObservation(
            name="梯度下降",
            definition="一般优化方法",
            entity_type="优化算法",
            model_quote="梯度下降",
            source_text="梯度下降",
            passage_ids=("P000001",),
            location="P000001",
        )
        store.create_entity(self.conn, existing)
        observed = replace(
            existing,
            definition="语义不同但尚未正确命名的对象",
            passage_ids=("P000002",),
            location="P000002",
        )
        llm = FakeLLM(
            {
                "decision": "new",
                "canonical_name": "梯度下降",
                "reason": "不同对象但没有完成名称消歧",
            },
            {
                "decision": "new",
                "identity_scope": "not_same",
                "reason": "复判确认不同",
            },
            {"canonical_name": "梯度下降", "naming_basis": "无区分"},
            {"canonical_name": "梯度下降", "naming_basis": "仍无区分"},
        )

        with self.assertRaisesRegex(ValueError, "语义重命名仍与已有 Entity 冲突"):
            resolution.resolve_observation(self.conn, llm, observed)
        self.assertEqual(store.counts(self.conn)["entities"], 1)

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
                        "statement": "梯度下降法是机器学习的一部分",
                        "scope": "",
                        "scope_is_restrictive": False,
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
            {"decision": "same", "candidate_id": 2,
             "projection_statement": "梯度下降法是机器学习的一部分",
             "register_alias": False, "reason": "待最终证据裁判"},
            {"assertion_verdict": "insufficient",
             "projection_statement": "候选实体与基础实体相关",
             "projection_faithful": True, "reason": "只有共现"},
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
