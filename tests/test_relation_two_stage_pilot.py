import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from experiments.relation_two_stage_pilot import keyed, parse_groups, parse_catalog, materialize, fine_stage, checked_completion


def fixture():
    return dict(relations=[dict(id=i, name=f"r{i}", description="", uses=1) for i in (1, 2)],
                claims=[dict(id=i, subject_id=10, object_id=20, relation_type_id=i) for i in (1, 2)],
                facts=[dict(id=i, claim_id=i, subject_id=10, object_id=20, relation_type_id=i,
                            subject="X", object="Y", statement="fact", scope="condition",
                            scope_is_restrictive=True) for i in (1, 2)],
                evidence=[dict(id=i, assertion_id=i, excerpt=f"source {i}") for i in (1, 2)])


class PilotTests(unittest.TestCase):
    def test_ordinal_ids_are_reasked_never_assigned_positionally(self):
        class FakeLLM:
            calls = 0

            def complete(self, **kwargs):
                self.calls += 1
                return dict(checks=[dict(id=i, valid=True) for i in ([1, 2] if self.calls == 1 else [54, 100])])

        llm = FakeLLM()
        rows = checked_completion(llm, kind="review", prompt="facts", field="checks", ids=[54, 100])
        self.assertEqual(llm.calls, 2)
        self.assertEqual([r["id"] for r in rows], [54, 100])

    def test_reject_duplicate_missing_and_invented_ids(self):
        for ids in ([1, 1], [1], [1, 3], [True, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                keyed([dict(id=i) for i in ids], "id", [1, 2])

    def test_no_transitive_or_overlapping_group_rewrites(self):
        with self.assertRaises(ValueError):
            parse_groups(dict(groups=[dict(canonical_id=1, member_ids=[1, 2]),
                                      dict(canonical_id=2, member_ids=[2, 3])]), {1, 2, 3})

    def test_invalid_catalog(self):
        for rows in ([], [dict(name="x", definition="")], [dict(name="x", definition="d")]*2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_catalog(dict(relations=rows), 35)

    def test_merge_keeps_every_assertion_and_evidence_and_lineage(self):
        sample = fixture()
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"derived.db"
            n = materialize(path, sample, {1: 1, 2: 1},
                            [dict(id=1, coarse_id=5, accepted=True), dict(id=2, coarse_id=None, accepted=False)])
            self.assertEqual(n, 1)
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute("select count(*) from claim_lineage").fetchone()[0], 2)
                self.assertEqual(c.execute("select count(*) from coarse_lineage").fetchone()[0], 2)
                for kind in ("facts", "evidence"):
                    rows = [json.loads(r[0]) for r in c.execute("select payload from source_rows where kind=? order by id", (kind,))]
                    self.assertEqual(rows, sample[kind])
                self.assertEqual(c.execute("pragma foreign_key_check").fetchall(), [])
            with self.assertRaises(ValueError):
                materialize(path, sample, {1: 1, 2: 1}, [])

    def test_mixed_assertions_do_not_trigger_whole_claim_replacement(self):
        sample = fixture()
        sample["facts"].append({**sample["facts"][1], "id": 3})

        class FakeLLM:
            max_concurrency = 1

            def complete(self, **kwargs):
                if "discovery" in kwargs["kind"]:
                    return dict(groups=[dict(canonical_id=1, member_ids=[1, 2])])
                return dict(lexically_equivalent=True, checks=[dict(id=i, equivalent=i != 3) for i in (1, 2, 3)])

        with TemporaryDirectory() as tmp:
            _, mapping, _ = fine_stage(FakeLLM(), sample, Path(tmp))
            self.assertEqual(mapping, {1: 1, 2: 2})


if __name__ == "__main__":
    unittest.main()
