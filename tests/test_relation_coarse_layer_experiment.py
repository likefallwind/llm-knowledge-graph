from __future__ import annotations

from experiments.relation_coarse_layer_experiment import (
    build_candidates,
    build_mapping_rows,
    parse_single_proposal,
    retrieve_top_candidates,
)
from experiments.relation_coarsening_experiment import Candidate, Proposal, RelationRecord


def relation(relation_id: int, name: str, uses: int = 1) -> RelationRecord:
    return RelationRecord(
        id=relation_id,
        name=name,
        description=f"{name}定义",
        uses=uses,
        assertions=(f"{name}断言",),
        examples=((f"主体{relation_id}", f"客体{relation_id}", f"{name}断言"),),
    )


def test_parse_single_proposal_keeps_only_cluster_member_ids() -> None:
    cluster = [relation(1, "用于"), relation(2, "应用于"), relation(3, "包含")]
    proposal = parse_single_proposal(
        {
            "proposal": {
                "canonical_name": "用于",
                "definition": "主体被用于客体所代表的目的",
                "source_relation_ids": [1, 2, 99],
            }
        },
        cluster,
    )
    assert proposal is not None
    assert proposal.source_relation_ids == (1, 2)


def test_build_candidates_preserves_source_mapping() -> None:
    frozen = [Candidate("seed-9", "使用", "主体使用客体", "high-frequency", [9])]
    proposals = [Proposal("用于", "主体用于客体", (1, 2), "simple-consolidated")]
    candidates = build_candidates(frozen, proposals)
    assert [item.id for item in candidates] == ["seed-9", "coarse-1"]
    assert candidates[1].source_relation_ids == [1, 2]


def test_retrieval_forces_source_candidate_into_top_k(monkeypatch) -> None:
    records = [relation(1, "训练目标")]
    candidates = [
        Candidate("seed-9", "训练", "训练关系", "high-frequency", [9]),
        Candidate("coarse-1", "完全不相似的名称", "目标关系", "simple", [1]),
    ]

    def fake_encode(texts):
        vectors = {
            "训练目标": [1.0, 0.0],
            "训练": [1.0, 0.0],
            "完全不相似的名称": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    monkeypatch.setattr(
        "experiments.relation_coarse_layer_experiment._encode", fake_encode
    )
    retrieved = retrieve_top_candidates(records, candidates, top_k=1)
    assert [item.id for item in retrieved[1]] == ["coarse-1"]


def test_mapping_rows_include_identity_mapped_and_unmapped_relations() -> None:
    records = [
        relation(1, "低频已映射"),
        relation(2, "低频未映射"),
        relation(9, "高频", uses=10),
    ]
    candidates = [
        Candidate("seed-9", "高频", "高频定义", "high-frequency", [9]),
        Candidate("coarse-1", "粗关系", "粗关系定义", "simple", [1]),
    ]
    rows = build_mapping_rows(
        relations=records,
        candidates=candidates,
        high_frequency_threshold=10,
        mappings={1: "coarse-1"},
        mapping_judgments=[],
        validation_judgments=[],
    )
    by_id = {item["source_relation_id"]: item for item in rows}
    assert by_id[1]["mapping_status"] == "mapped"
    assert by_id[1]["target_relation_id"] == "coarse-1"
    assert by_id[2]["mapping_status"] == "unmapped"
    assert by_id[2]["target_relation_id"] is None
    assert by_id[9]["mapping_status"] == "identity_high_frequency"
    assert by_id[9]["target_relation_id"] == "seed-9"
