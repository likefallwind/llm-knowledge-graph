from __future__ import annotations

import json
import threading

from experiments.relation_coarsening_experiment import (
    CachedLLM,
    Candidate,
    RelationRecord,
    add_proposals,
    candidate_gate_passes,
    evenly_spaced,
    full_validation_passes,
    parse_embedding_subsets,
    parse_mapping,
    prune_weak_candidates,
    seed_candidates,
    select_pilot,
    Proposal,
    refine_oversized_clusters,
    synthesize_embedding_proposals,
)


def relation(relation_id: int, uses: int) -> RelationRecord:
    return RelationRecord(
        id=relation_id,
        name=f"关系{relation_id}",
        description=f"关系{relation_id}的定义",
        uses=uses,
        assertions=(f"关系{relation_id}的完整命题",),
        examples=((f"主体{relation_id}", f"客体{relation_id}", f"关系{relation_id}的完整命题"),),
    )


def test_seed_candidates_freezes_usage_at_ten_or_more() -> None:
    seeds = seed_candidates([relation(1, 9), relation(2, 10), relation(3, 20)])
    assert [item.id for item in seeds] == ["seed-3", "seed-2"]


def test_pilot_is_stratified_and_deterministic() -> None:
    records = [
        *[relation(index, 3 + index % 7) for index in range(1, 101)],
        *[relation(index, 2) for index in range(101, 201)],
        *[relation(index, 1) for index in range(201, 401)],
        relation(999, 12),
    ]
    first = select_pilot(records, 200)
    second = select_pilot(records, 200)
    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 200
    assert sum(3 <= item.uses <= 9 for item in first) == 80
    assert sum(item.uses == 2 for item in first) == 60
    assert sum(item.uses == 1 for item in first) == 60
    assert all(item.id != 999 for item in first)


def test_evenly_spaced_includes_both_ends() -> None:
    records = [relation(index, 1) for index in range(10)]
    selected = evenly_spaced(records, 3)
    assert [item.id for item in selected] == [0, 4, 9]


def test_parse_mapping_rejects_unknown_candidate() -> None:
    candidate_ids = {"seed-1"}
    assert parse_mapping({"decision": "map", "candidate_id": "seed-1"}, candidate_ids) == "seed-1"
    assert parse_mapping({"decision": "map", "candidate_id": "made-up"}, candidate_ids) is None
    assert parse_mapping({"decision": "no_map", "candidate_id": "seed-1"}, candidate_ids) is None


def test_candidate_gate_requires_explicit_valid_decision() -> None:
    assert candidate_gate_passes({"decision": "valid_candidate"})
    assert candidate_gate_passes({"decision": " VALID_CANDIDATE "})
    assert not candidate_gate_passes({"decision": "invalid_candidate"})
    assert not candidate_gate_passes({"decision": "valid"})


def test_full_validation_requires_every_numbered_example() -> None:
    valid = {
        "decision": "full_map",
        "checks": [
            {"example_index": 1, "valid": True},
            {"example_index": 2, "valid": True},
            {"example_index": 3, "valid": True},
        ],
    }
    assert full_validation_passes(valid, 3)
    assert not full_validation_passes(
        {"decision": "full_map", "checks": valid["checks"][:2]}, 3
    )
    assert not full_validation_passes(
        {
            "decision": "full_map",
            "checks": [
                {"example_index": 1, "valid": True},
                {"example_index": 2, "valid": False},
                {"example_index": 3, "valid": True},
            ],
        },
        3,
    )
    assert not full_validation_passes(
        {
            "decision": "full_map",
            "checks": [
                {"example_index": 1, "valid": True},
                {"example_index": 1, "valid": True},
                {"example_index": 3, "valid": True},
            ],
        },
        3,
    )


def test_embedding_subset_parser_keeps_valid_partial_groups_disjoint() -> None:
    cluster = [relation(index, 1) for index in range(1, 7)]
    result = {
        "proposals": [
            {
                "canonical_name": "共同关系A",
                "definition": "X 对 Y 执行共同动作A",
                "source_relation_ids": [1, 2, 3],
            },
            {
                "canonical_name": "重叠关系",
                "definition": "不应保留",
                "source_relation_ids": [3, 4, 5],
            },
            {
                "canonical_name": "成员不足",
                "definition": "不应保留",
                "source_relation_ids": [5, 6],
            },
        ]
    }
    proposals = parse_embedding_subsets(result, cluster, round_index=2)
    assert len(proposals) == 1
    assert proposals[0].name == "共同关系A"
    assert proposals[0].source_relation_ids == (1, 2, 3)
    assert proposals[0].source == "embedding-r2"


def test_embedding_subset_parser_can_admit_pairs_when_configured() -> None:
    cluster = [relation(index, 1) for index in range(1, 4)]
    result = {
        "proposals": [
            {
                "canonical_name": "同义关系",
                "definition": "X 对 Y 执行相同动作",
                "source_relation_ids": [1, 2],
            }
        ]
    }
    assert parse_embedding_subsets(result, cluster, round_index=1) == []
    proposals = parse_embedding_subsets(
        result, cluster, round_index=1, min_member_types=2
    )
    assert len(proposals) == 1
    assert proposals[0].source_relation_ids == (1, 2)


def test_weak_new_candidate_is_removed_and_members_released() -> None:
    records = {index: relation(index, 1) for index in range(1, 5)}
    candidates = [
        Candidate("seed-9", "已有", "已有定义", "seed", [9]),
        Candidate("embedding-r1-1", "新候选", "新定义", "embedding-r1", [1, 2]),
    ]
    mappings = {1: "embedding-r1-1", 2: "embedding-r1-1", 3: "seed-9"}
    removed, released = prune_weak_candidates(
        candidates=candidates,
        mappings=mappings,
        records=records,
        candidate_ids={"embedding-r1-1"},
        min_member_types=3,
        min_member_uses=3,
    )
    assert removed == ["embedding-r1-1"]
    assert released == [1, 2]
    assert [item.id for item in candidates] == ["seed-9"]
    assert mappings == {3: "seed-9"}


def test_add_proposals_respects_name_dedup_and_budget() -> None:
    candidates = [Candidate("seed-1", "用于", "用途关系", "seed", [1])]
    proposals = [
        Proposal("用于", "重复", (2, 3, 4), "llm-r1"),
        Proposal("属性比较", "比较关系", (5, 6, 7), "llm-r1"),
        Proposal("训练目标", "训练关系", (8, 9, 10), "llm-r1"),
    ]
    added = add_proposals(
        candidates=candidates,
        proposals=proposals,
        arm="llm",
        round_index=1,
        max_candidates=2,
    )
    assert added == ["llm-r1-1"]
    assert [item.name for item in candidates] == ["用于", "属性比较"]


def test_oversized_cluster_refinement_defers_tighter_outliers(monkeypatch) -> None:
    source = [relation(index, 1) for index in range(1, 9)]

    def fake_clusters(items, *, distance_threshold, min_cluster_size):
        assert distance_threshold == 0.05
        assert min_cluster_size == 3
        return [list(items[:4])]

    monkeypatch.setattr(
        "experiments.relation_coarsening_experiment.embedding_clusters",
        fake_clusters,
    )
    refined, details = refine_oversized_clusters(
        [source],
        max_cluster_size=5,
        split_distance_threshold=0.05,
        min_cluster_size=3,
    )
    assert [[item.id for item in group] for group in refined] == [[1, 2, 3, 4]]
    assert details[0]["deferred_member_ids"] == [5, 6, 7, 8]


def test_exhaustive_embedding_extraction_peels_multiple_subsets(monkeypatch) -> None:
    source = [relation(index, 1) for index in range(1, 7)]
    monkeypatch.setattr(
        "experiments.relation_coarsening_experiment.embedding_clusters",
        lambda *args, **kwargs: [source],
    )

    class FakeLLM:
        max_concurrency = 1

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            ids = [1, 2] if self.calls == 1 else [3, 4]
            return {
                "proposals": [
                    {
                        "canonical_name": f"候选{self.calls}",
                        "definition": "同一谓词",
                        "source_relation_ids": ids,
                    }
                ]
            }

    llm = FakeLLM()
    proposals, generation = synthesize_embedding_proposals(
        llm=llm,
        residual=source,
        round_index=1,
        max_proposals=10,
        distance_threshold=0.1,
        min_cluster_size=3,
        min_member_types=2,
        exhaustive_cluster_extraction=True,
        max_cluster_size=10,
        max_cluster_extraction_passes=10,
    )
    assert [item.source_relation_ids for item in proposals] == [(1, 2), (3, 4)]
    assert llm.calls == 2
    assert len(generation["clusters"][0]["attempts"]) == 2


def test_cached_llm_retries_malformed_json(tmp_path) -> None:
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, system, user):
            self.calls += 1
            if self.calls < 3:
                raise json.JSONDecodeError("bad", "{", 1)
            return {"decision": "ok"}

    llm = object.__new__(CachedLLM)
    llm.client = FakeClient()
    llm.max_concurrency = 1
    llm.cache_path = tmp_path / "cache.jsonl"
    llm._lock = threading.Lock()
    llm._cache = {}
    llm.metrics = {
        "cache_hits": 0,
        "api_calls": 0,
        "prompt_chars": 0,
        "elapsed_seconds": 0.0,
    }
    assert llm.complete(kind="test", system="s", user="u") == {"decision": "ok"}
    assert llm.client.calls == 3
    assert llm.metrics["api_calls"] == 1
