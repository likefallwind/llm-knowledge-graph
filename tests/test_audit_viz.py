from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg import audit, db, observations, store, viz
from kg.models import (
    ClaimObservation,
    EntityObservation,
    LoadedSource,
    SourceSpec,
)


def observation(name: str) -> EntityObservation:
    return EntityObservation(
        name=name,
        definition=f"{name} 的测试定义",
        entity_type="concept",
        model_quote=f"{name} 的关键引文",
        source_text=f"{name} 的真实原文",
        passage_ids=("P000001",),
        location="P000001",
    )


class AuditVizTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.conn = db.connect(self.root / "kg.db")
        source = LoadedSource(
            spec=SourceSpec("test", "测试教材", "textbook"),
            content="实体甲是实体乙的一种。",
            content_hash="a" * 64,
            version="1",
        )
        self.source_id, _ = store.add_source(self.conn, source)
        self.left = self._entity("实体甲")
        self.right = self._entity("实体乙")
        claim_id, _, _ = store.upsert_claim(
            self.conn, self.left, "is_a", self.right
        )
        store.add_evidence(
            self.conn,
            source_id=self.source_id,
            source_text="实体甲是实体乙的一种。",
            model_quote="实体甲是实体乙的一种。",
            passage_ids=("P000001",),
            location="P000001",
            polarity="support",
            extraction_model="FakeLLM",
            extraction_prompt_version="test-extract",
            validator_model="FakeLLM",
            validator_prompt_version="test-validator",
            validator_verdict="supports",
            validator_reason="原文明确定义类属关系",
            claim_id=claim_id,
        )
        store.save_progress(
            self.conn,
            self.source_id,
            0,
            "fingerprint",
            status="done",
            result={
                "entities": 2,
                "claims": 1,
                "evidence": 3,
                "rejected": [
                    "Claim 端点无法唯一解析: 缺失实体 -> 实体乙",
                    "Claim 证据裁决为 insufficient: A is_a B; 只有共现",
                ],
            },
        )
        pending_id, _ = observations.add_claim_observation(
            self.conn,
            source_id=self.source_id,
            chunk_index=1,
            claim=ClaimObservation(
                subject="缺失实体",
                relation="is_a",
                object="实体乙",
                model_quote="缺失实体是实体乙的一种",
                source_text="缺失实体是实体乙的一种。",
                passage_ids=("P000001",),
                location="P000001",
            ),
            extraction_model="FakeLLM",
            extraction_prompt_version="test-extract",
        )
        observations.save_judgment(
            self.conn,
            pending_id,
            validator_model="FakeLLM",
            validator_prompt_version="test-validator",
            verdict="supports",
            reason="原文明确定义类属关系",
        )
        observations.resolve_endpoint_ids(self.conn, [pending_id])
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _entity(self, name: str) -> int:
        entity_id = store.create_entity(self.conn, observation(name))
        store.add_evidence(
            self.conn,
            source_id=self.source_id,
            source_text=f"{name} 的真实原文",
            model_quote=f"{name} 的关键引文",
            passage_ids=("P000001",),
            location="P000001",
            polarity="support",
            extraction_model="FakeLLM",
            extraction_prompt_version="test-extract",
            observed_entity_type="concept",
            entity_id=entity_id,
        )
        return entity_id

    def test_rejection_report_separates_algorithmic_and_semantic_rejections(self):
        report = audit.rejection_report(self.conn)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["categories"]["endpoint_unresolved"], 1)
        self.assertEqual(report["categories"]["insufficient"], 1)
        self.assertEqual(report["algorithmic_loss"], 1)
        self.assertEqual(report["semantic_rejection"], 1)

    def test_generated_scripts_have_no_broken_string_literals(self):
        """模板是非 raw 的三引号字符串，JS 里的 \\n 必须写成 \\\\n。

        漏写会让 Python 把它变成真实换行，把 JS 字符串字面量拦腰截断，
        整个 <script> 编译失败——页面只剩空壳，而且不会有任何运行时报错，
        肉眼很难发现。按行检查引号配对，不必引入 JS 引擎就能抓住这类断裂。
        """
        for view in ("semantic", "document", "mixed"):
            out = viz.write_html(
                self.conn, self.root / f"{view}.html", view=view
            )
            html = out.read_text(encoding="utf-8")
            for block in _script_blocks(html):
                for lineno, line in enumerate(block.split("\n"), 1):
                    stripped = line.replace('\\"', "").replace("\\'", "")
                    for quote in ('"', "'"):
                        self.assertEqual(
                            stripped.count(quote) % 2,
                            0,
                            f"{view} 视图第 {lineno} 行 {quote} 未配对，"
                            f"字符串字面量被换行截断: {line[:160]}",
                        )

    def test_visualization_is_self_contained_and_includes_evidence(self):
        output = viz.write_html(self.conn, self.root / "graph.html")

        html = output.read_text(encoding="utf-8")
        self.assertIn("实体甲", html)
        self.assertIn("原文明确定义类属关系", html)
        self.assertIn("endpoint_unresolved", html)
        self.assertIn("Observation 审计", html)
        self.assertIn("定义聚合", html)
        self.assertIn("缺失实体", html)
        self.assertIn("<canvas", html)
        self.assertIn("strokeText(node.canonical_name", html)
        self.assertIn("startSimulation()", html)
        self.assertIn("拖拽节点", html)
        self.assertNotIn('src="http', html)
        self.assertNotIn("unpkg.com", html)

        marker = '<script id="graph-data" type="application/json">'
        payload_text = html.split(marker, 1)[1].split("</script>", 1)[0]
        payload = json.loads(payload_text)
        self.assertEqual(len(payload["entities"]), 2)
        self.assertEqual(len(payload["claims"]), 1)
        observation_audit = payload["observation_audit"]
        self.assertEqual(observation_audit["summary"]["pending_endpoint"], 1)
        self.assertEqual(len(observation_audit["items"]), 1)
        self.assertEqual(
            observation_audit["items"][0]["status"], "pending_endpoint"
        )
        self.assertEqual(
            observation_audit["items"][0]["object"]["entity_name"], "实体乙"
        )


def _script_blocks(html: str) -> list[str]:
    """取出所有可执行 JS 块（跳过 application/json 数据块）。"""
    blocks = []
    rest = html
    while True:
        start = rest.find("<script")
        if start < 0:
            break
        head_end = rest.find(">", start)
        head = rest[start:head_end]
        body_end = rest.find("</script>", head_end)
        if "application/json" not in head:
            blocks.append(rest[head_end + 1:body_end])
        rest = rest[body_end + len("</script>"):]
    return blocks


if __name__ == "__main__":
    unittest.main()
