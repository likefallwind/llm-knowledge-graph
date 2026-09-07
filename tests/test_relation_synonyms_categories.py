import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from experiments.relation_synonyms_categories import canonicalize, categories, materialize, validate_catalog
from tests.test_relation_two_stage_pilot import fixture


class SynonymCategoryTests(unittest.TestCase):
    def test_similar_chain_does_not_become_transitive_merge(self):
        sample = fixture()
        for kind in ("relations", "claims", "facts", "evidence"):
            row = {**sample[kind][-1], "id": 3}
            if kind in ("claims", "facts"):
                row["relation_type_id"] = 3
            if kind == "facts":
                row["claim_id"] = 3
            sample[kind].append(row)
        reviews = [dict(pair=[1, 2], checks=[dict(id=i, equivalent=True) for i in (1, 2)]),
                   dict(pair=[2, 3], checks=[dict(id=i, equivalent=True) for i in (2, 3)])]
        _, mapping = canonicalize(sample, reviews)
        self.assertEqual(mapping, {1: 1, 2: 1, 3: 3})

    def test_mixed_claim_stays_fine_and_unsupported_target_blocks_merge(self):
        sample = fixture()
        sample["facts"].append({**sample["facts"][1], "id": 3})
        for rejected in (1, 3):
            with self.subTest(rejected=rejected):
                review = dict(pair=[1, 2], checks=[dict(id=i, equivalent=i != rejected) for i in (1, 2, 3)])
                _, mapping = canonicalize(sample, [review])
                self.assertEqual(mapping, {1: 1, 2: 2})

    def test_coarse_classification_preserves_opposite_fine_relations_and_pending(self):
        sample = fixture()
        sample["relations"][0]["name"] = "提高"
        sample["relations"][1]["name"] = "降低"
        category = dict(id=1, name="影响", definition="对指标产生作用", inclusion_rule="正负作用均纳入",
                        exclusion_rule="仅陈述目的归用途", support_relation_ids=[1, 2])

        class FakeLLM:
            def complete(self, **kwargs):
                kind = kwargs["kind"]
                if "category-induction" in kind:
                    return dict(categories=[dict(category)])
                if kind.endswith("category-assignment"):
                    return dict(classifications=[dict(id=i, category_id=1) for i in (1, 2)])
                return dict(checks=[dict(id=i, valid=i == 1) for i in (1, 2)])

        with TemporaryDirectory() as tmp:
            catalog, mappings = categories(FakeLLM(), sample, Path(tmp))
            path = Path(tmp)/"derived.db"
            materialize(path, sample, sample, {1: 1, 2: 2}, catalog, mappings)
            with sqlite3.connect(path) as c:
                rows = c.execute("SELECT fine_relation,coarse_category,status FROM classified_facts ORDER BY assertion_id").fetchall()
                self.assertEqual(rows, [("提高", "影响", "accepted"), ("降低", None, "pending")])
                self.assertEqual(c.execute("SELECT count(*) FROM fine_claims").fetchone()[0], 2)
                for kind in ("facts", "evidence"):
                    original = [json.loads(r[0]) for r in c.execute("SELECT payload FROM source_rows WHERE kind=? ORDER BY id", (kind,))]
                    self.assertEqual(original, sample[kind])

    def test_category_provenance_rejects_unknown_or_boolean_ids(self):
        for rid in (999, True):
            with self.subTest(rid=rid), self.assertRaises(ValueError):
                validate_catalog(dict(categories=[dict(name="影响", definition="d", inclusion_rule="i",
                    exclusion_rule="e", support_relation_ids=[rid])]), {1, 2})


if __name__ == "__main__":
    unittest.main()
