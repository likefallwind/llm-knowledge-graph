from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kg import db, extraction, pipeline, sources, store, structure, viz
from kg.models import (
    ClaimObservation,
    EntityObservation,
    ExtractionBatch,
    LoadedSource,
    SourceSpec,
)
from tests.helpers import FakeLLM


class VNextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.conn = db.connect(self.root / "vnext.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _source(self, text: str) -> tuple[int, tuple]:
        loaded = LoadedSource(
            SourceSpec("book", "教材", "textbook"), text, "hash", "v1"
        )
        source_id, _ = store.add_source(self.conn, loaded)
        chunks = sources.chunk_text(text, max_chars=8000, overlap_chars=0)
        passages = tuple(
            {p.passage_id: p for chunk in chunks for p in chunk.passages}.values()
        )
        structure.sync_source_structure(self.conn, source_id, passages)
        return source_id, passages

    def test_two_pass_extraction_keeps_open_types_and_predicates(self):
        passages = tuple(sources.segment_text("A 改进自 B。"))
        llm = FakeLLM(
            {
                "entities": [
                    {"name": "A", "definition": "一种改进模型", "type_labels": ["视觉模型"],
                     "evidence": {"passage_ids": ["P000001"], "quote": "A 改进自 B"}},
                    {"name": "B", "definition": "一种基础模型", "type_labels": ["模型架构"],
                     "evidence": {"passage_ids": ["P000001"], "quote": "A 改进自 B"}},
                ]
            },
            {"relations": [{"subject": "A", "predicate": "改进自", "object": "B",
                             "evidence": {"passage_ids": ["P000001"], "quote": "A 改进自 B"}}]},
        )
        batch = extraction.extract(llm, "A 改进自 B。", passages=passages)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(batch.entities[0].type_labels, ("视觉模型",))
        self.assertEqual(batch.claims[0].raw_relation, "改进自")

    def test_section_tree_is_persisted_without_becoming_claims(self):
        source_id, passages = self._source("# 卷积网络\n\n## VGG\n\nVGG 使用小卷积核。")
        rows = self.conn.execute(
            "SELECT title,depth FROM source_sections WHERE source_id=? ORDER BY depth",
            (source_id,),
        ).fetchall()
        self.assertEqual([(row["title"], row["depth"]) for row in rows],
                         [("卷积网络", 1), ("VGG", 2)])
        self.assertTrue(all(p.section_path for p in passages))
        self.assertEqual(store.counts(self.conn)["claims"], 0)

    def test_section_summaries_parallelize_with_depth_barrier(self):
        source_id, _ = self._source(
            "# 根章节\n\n## 子节一\n\n内容一。\n\n## 子节二\n\n内容二。"
        )

        class ConcurrentSummaryLLM:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.calls: list[str] = []

            def complete_json(self, system: str, user: str) -> dict:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    self.calls.append(user)
                time.sleep(0.05)
                with self.lock:
                    self.active -= 1
                return {"summary": "有据摘要", "passage_ids": []}

        llm = ConcurrentSummaryLLM()
        result = structure.summarize_source(
            self.conn, llm, source_id, model="simple", workers=2
        )

        self.assertEqual(result, {"processed": 3, "skipped": 0, "failed": 0})
        self.assertEqual(llm.max_active, 2)
        self.assertEqual(len(llm.calls), 3)
        self.assertIn('"children": [{"title": "子节一", "summary": "有据摘要"}',
                      llm.calls[-1])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM section_summaries").fetchone()[0],
            3,
        )

    def test_section_summary_persists_before_slow_peer_finishes(self):
        source_id, _ = self._source(
            "# 根章节\n\n## 快子节\n\n内容一。\n\n## 慢子节\n\n内容二。"
        )
        release_slow = threading.Event()
        observed_write = threading.Event()

        class OneSlowSummaryLLM:
            def __init__(self):
                self.lock = threading.Lock()
                self.call_count = 0

            def complete_json(self, system: str, user: str) -> dict:
                with self.lock:
                    self.call_count += 1
                    call_number = self.call_count
                if call_number == 2:
                    release_slow.wait(timeout=2)
                return {"summary": "有据摘要", "passage_ids": []}

        def observe_database() -> None:
            observer = sqlite3.connect(self.root / "vnext.db")
            try:
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    count = observer.execute(
                        "SELECT COUNT(*) FROM section_summaries"
                    ).fetchone()[0]
                    if count:
                        observed_write.set()
                        break
                    time.sleep(0.01)
            finally:
                release_slow.set()
                observer.close()

        observer = threading.Thread(target=observe_database)
        observer.start()
        result = structure.summarize_source(
            self.conn, OneSlowSummaryLLM(), source_id, model="simple", workers=2
        )
        observer.join()

        self.assertTrue(observed_write.is_set())
        self.assertEqual(result["processed"], 3)

    def test_open_relation_normalizes_then_materializes_with_passage_evidence(self):
        source_id, passages = self._source("# 模型\n\nA 改进自 B。")
        passage = passages[-1]
        entities = tuple(
            EntityObservation(
                name=name,
                definition=f"{name} 是有稳定含义的模型",
                entity_type="solution",
                type_labels=("solution",),
                model_quote="A 改进自 B",
                source_text=passage.text,
                passage_ids=(passage.passage_id,),
                location=passage.location,
            )
            for name in ("A", "B")
        )
        claim = ClaimObservation(
            "A", "改进自", "B", "A 改进自 B", passage.text,
            (passage.passage_id,), passage.location, raw_relation="改进自"
        )
        simple_llm = FakeLLM(
            {"decision": "new", "canonical_name": "改进自", "relation_kind": "other",
             "description": "主语是在宾语基础上的改进", "reason": "不同于种子关系"},
        )
        llm = FakeLLM(
            {"decision": "new", "canonical_name": "A", "reason": "首次出现"},
            {"decision": "new", "canonical_name": "B", "reason": "首次出现"},
            {"verdict": "supports", "reason": "原文明示"},
        )
        result = pipeline.process_chunk(
            self.conn,
            llm,
            source_id=source_id,
            text=passage.text,
            passages=(passage,),
            location=passage.location,
            batch=ExtractionBatch(entities, (claim,)),
            simple_llm=simple_llm,
        )
        self.assertEqual(len(simple_llm.calls), 1)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(result.claims, 1)
        row = self.conn.execute(
            """SELECT c.relation,r.relation_kind FROM claims c
               JOIN relation_types r ON r.id=c.relation_type_id"""
        ).fetchone()
        self.assertEqual((row["relation"], row["relation_kind"]), ("改进自", "other"))
        self.assertEqual(store.integrity_report(self.conn)["ok"], True)

    def test_all_visualization_views_are_self_contained(self):
        self._source("# 卷积网络\n\n## NiN\n\nNiN 是一种网络架构。")
        for view in ("semantic", "document", "mixed"):
            output = viz.write_html(self.conn, self.root / f"{view}.html", view=view)
            text = output.read_text(encoding="utf-8")
            self.assertIn("graph-data", text)
            self.assertNotIn("__GRAPH_DATA__", text)


if __name__ == "__main__":
    unittest.main()
