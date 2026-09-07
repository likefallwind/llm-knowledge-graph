from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from kg.embeddings import _encode

from experiments.relation_coarsening_experiment import (
    SYSTEM,
    CachedLLM,
    Candidate,
    Proposal,
    RelationRecord,
    _compact_text,
    evenly_spaced,
    load_relations,
    log,
    parse_mapping,
    relation_payload,
    save_json,
    validate_all_triples,
)


def fixed_count_name_clusters(
    relations: Sequence[RelationRecord], *, cluster_count: int
) -> list[list[RelationRecord]]:
    """Cluster relation names only; evidence is reserved for LLM synthesis."""
    if not relations or cluster_count <= 0:
        return []
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError(
            "粗关系实验需要安装项目的 sentence-transformers 依赖"
        ) from exc

    actual_count = min(cluster_count, len(relations))
    matrix = np.asarray(_encode([item.name for item in relations]), dtype="float32")
    if actual_count == len(relations):
        labels = list(range(len(relations)))
    else:
        labels = AgglomerativeClustering(
            n_clusters=actual_count,
            metric="cosine",
            linkage="average",
        ).fit_predict(matrix)

    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)

    clusters: list[list[RelationRecord]] = []
    for indices in grouped.values():
        submatrix = matrix[indices]
        centroid = submatrix.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        ordered = sorted(
            indices,
            key=lambda index: float(1.0 - np.dot(matrix[index], centroid)),
        )
        clusters.append([relations[index] for index in ordered])
    clusters.sort(key=lambda group: (-sum(item.uses for item in group), -len(group)))
    return clusters


def parse_single_proposal(
    result: dict[str, Any],
    cluster: Sequence[RelationRecord],
) -> Proposal | None:
    item = result.get("proposal")
    if not isinstance(item, dict):
        return None
    name = str(item.get("canonical_name", "")).strip()
    definition = str(item.get("definition", "")).strip()
    valid_ids = {relation.id for relation in cluster}
    source_ids = tuple(
        sorted(
            {
                int(value)
                for value in item.get("source_relation_ids", [])
                if isinstance(value, int) and value in valid_ids
            }
        )
    )
    if not name or not definition or len(source_ids) < 2:
        return None
    return Proposal(
        name=name,
        definition=definition,
        source_relation_ids=source_ids,
        source="simple-name-cluster",
    )


def synthesize_cluster_proposals(
    *,
    llm: CachedLLM,
    clusters: Sequence[Sequence[RelationRecord]],
) -> tuple[list[Proposal], list[dict[str, Any]]]:
    """Ask for at most one reusable coarse predicate from each name cluster."""

    def synthesize(index: int, cluster: Sequence[RelationRecord]) -> tuple[int, dict[str, Any]]:
        members = [relation_payload(item) for item in cluster]
        prompt = """下面是一组仅按关系名称 Embedding 得到的低频关系。请结合定义、
Assertion 和代表三元组，判断其中是否存在一个能覆盖至少两个成员的、明确且可复用的
粗粒度二元关系。

本实验每个簇最多生成一个粗关系。可以只选择簇内一部分成员；其他成员之后仍可映射到
别的主关系。粗关系允许省略程度、条件和机制细节，因为这些信息由 Assertion 保留，但
使用原 subject、原 object 组成的新三元组必须仍然成立。必须保持方向和端点角色，不得
生成“相关”“涉及”“影响”等空泛兜底类型，也不得用“或”“/”连接多个谓词。
不得升级命题强度：目标/意图不等于实际效果，可能不等于已经发生，有助于执行不等于
直接执行，相关性不等于因果性。

若存在，返回 {"proposal":{"canonical_name":"","definition":"说明主体、客体
角色及方向","source_relation_ids":[1,2]}}；否则返回 {"proposal":null}。
簇成员=%s""" % json.dumps(members, ensure_ascii=False)
        return index, llm.complete(
            kind=f"simple-coarse-synthesis:cluster-{index}",
            system=SYSTEM,
            user=prompt,
        )

    raw: list[dict[str, Any]] = []
    proposals: list[Proposal] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {
            executor.submit(synthesize, index, cluster): index
            for index, cluster in enumerate(clusters, start=1)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, result = future.result()
            cluster = clusters[index - 1]
            proposal = parse_single_proposal(result, cluster)
            raw.append(
                {
                    "cluster_index": index,
                    "member_ids": [item.id for item in cluster],
                    "result": result,
                    "accepted": asdict(proposal) if proposal else None,
                }
            )
            if proposal is not None:
                proposals.append(proposal)
            if completed % 10 == 0 or completed == len(futures):
                log(f"coarse synthesis completed {completed}/{len(futures)}")
    raw.sort(key=lambda item: int(item["cluster_index"]))
    proposals.sort(key=lambda item: (-len(item.source_relation_ids), item.name))
    return proposals, raw


def consolidate_proposals(
    *,
    llm: CachedLLM,
    proposals: Sequence[Proposal],
    frozen: Sequence[Candidate],
    max_new_types: int,
) -> tuple[list[Proposal], dict[str, Any] | None]:
    if max_new_types <= 0 or not proposals:
        return [], None
    payload = [asdict(item) for item in proposals]
    prompt = """将下面按不同 Embedding 簇生成的粗关系整理为最终主关系表。

只合并含义相同或明显属于同一个稳定二元谓词的提案；删除与冻结高频关系重复的提案。
不要进一步发明层次结构，不要生成空泛兜底关系。最终最多保留 %d 个新主关系。合并
提案时，source_relation_ids 必须取被合并提案原 ID 的并集，不能添加输入中不存在的 ID。

返回 {"proposals":[{"canonical_name":"","definition":"明确主体、客体角色和
方向","source_relation_ids":[]}]}。
冻结高频关系=%s
待整理提案=%s""" % (
        max_new_types,
        json.dumps(
            [{"name": item.name, "definition": item.definition} for item in frozen],
            ensure_ascii=False,
        ),
        json.dumps(payload, ensure_ascii=False),
    )
    result = llm.complete(
        kind="simple-coarse-consolidation",
        system=SYSTEM,
        user=prompt,
    )
    valid_ids = {value for item in proposals for value in item.source_relation_ids}
    frozen_names = {item.name.strip().casefold() for item in frozen}
    consolidated: list[Proposal] = []
    seen_names: set[str] = set()
    items = result.get("proposals", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("canonical_name", "")).strip()
            definition = str(item.get("definition", "")).strip()
            normalized_name = name.casefold()
            source_ids = tuple(
                sorted(
                    {
                        int(value)
                        for value in item.get("source_relation_ids", [])
                        if isinstance(value, int) and value in valid_ids
                    }
                )
            )
            if (
                not name
                or not definition
                or len(source_ids) < 2
                or normalized_name in frozen_names
                or normalized_name in seen_names
            ):
                continue
            seen_names.add(normalized_name)
            consolidated.append(
                Proposal(
                    name=name,
                    definition=definition,
                    source_relation_ids=source_ids,
                    source="simple-consolidated",
                )
            )
            if len(consolidated) >= max_new_types:
                break
    return consolidated, result


def build_candidates(
    frozen: Sequence[Candidate], proposals: Sequence[Proposal]
) -> list[Candidate]:
    candidates = [Candidate(**asdict(item)) for item in frozen]
    candidates.extend(
        Candidate(
            id=f"coarse-{index}",
            name=proposal.name,
            definition=proposal.definition,
            source=proposal.source,
            source_relation_ids=list(proposal.source_relation_ids),
        )
        for index, proposal in enumerate(proposals, start=1)
    )
    return candidates


def retrieve_top_candidates(
    relations: Sequence[RelationRecord],
    candidates: Sequence[Candidate],
    *,
    top_k: int,
) -> dict[int, list[Candidate]]:
    if not candidates:
        return {item.id: [] for item in relations}
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("粗关系实验需要 numpy") from exc
    candidate_matrix = np.asarray(_encode([item.name for item in candidates]), dtype="float32")
    relation_matrix = np.asarray(_encode([item.name for item in relations]), dtype="float32")
    candidate_matrix /= np.maximum(
        np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12
    )
    relation_matrix /= np.maximum(
        np.linalg.norm(relation_matrix, axis=1, keepdims=True), 1e-12
    )
    source_candidates: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        for relation_id in candidate.source_relation_ids:
            source_candidates.setdefault(relation_id, []).append(index)

    result: dict[int, list[Candidate]] = {}
    limit = min(max(top_k, 1), len(candidates))
    for relation, vector in zip(relations, relation_matrix, strict=True):
        similarities = candidate_matrix @ vector
        ranked = sorted(range(len(candidates)), key=lambda index: -float(similarities[index]))
        forced = sorted(
            source_candidates.get(relation.id, []),
            key=lambda index: -float(similarities[index]),
        )
        selected: list[int] = []
        for index in [*forced, *ranked]:
            if index not in selected:
                selected.append(index)
            if len(selected) >= limit:
                break
        result[relation.id] = [candidates[index] for index in selected]
    return result


def map_relations_top_k(
    *,
    llm: CachedLLM,
    relations: Sequence[RelationRecord],
    retrieved: dict[int, list[Candidate]],
    checkpoint_path: Path,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    accepted: dict[int, str] = {}
    judgments_by_id: dict[int, dict[str, Any]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        accepted = {int(key): str(value) for key, value in checkpoint["mappings"].items()}
        judgments_by_id = {
            int(item["relation_id"]): item for item in checkpoint["judgments"]
        }
    pending = [item for item in relations if item.id not in judgments_by_id]
    log(
        f"mapping {len(pending)} pending low-frequency relations "
        f"({len(judgments_by_id)} restored)"
    )

    def judge(item: RelationRecord) -> tuple[int, str | None, dict[str, Any], list[str]]:
        candidates = retrieved[item.id]
        candidate_ids = {candidate.id for candidate in candidates}
        prompt = """判断当前低频关系能否映射到一个候选主关系。Embedding 只负责召回。
Assertion 保存程度、条件和机制等细节，因此候选关系更粗不是拒绝理由；但必须把原
subject、原 object 原封不动代入候选关系，并保证每个代表三元组仍为真。不得交换方向，
不得把端点重新解释为其属性、输出、类别或其他隐藏对象。没有准确候选就 no_map。
还必须保持命题强度：目标/意图不蕴含实际效果，可能/可以不蕴含实际发生，有助于某动作
不蕴含主体直接完成该动作，相关性不蕴含因果性；这些变化不是可省略的普通细节。
若 Assertion 只说“X 的某个组成部分、损失函数、步骤或属性基于/作用于 Y”，不能省略
中间路径后声称“X 整体基于/作用于 Y”。只能依据给出的 Assertion，不能用常识补桥。

返回 {"decision":"map|no_map","candidate_id":null,
"projections":["逐条写出 subject→候选→object"],"reason":""}。
当前关系=%s
候选主关系=%s""" % (
            json.dumps(relation_payload(item), ensure_ascii=False),
            json.dumps(
                [
                    {
                        "id": candidate.id,
                        "name": candidate.name,
                        "definition": candidate.definition,
                    }
                    for candidate in candidates
                ],
                ensure_ascii=False,
            ),
        )
        result = llm.complete(
            kind=f"simple-coarse-map:relation-{item.id}",
            system=SYSTEM,
            user=prompt,
        )
        return item.id, parse_mapping(result, candidate_ids), result, sorted(candidate_ids)

    def checkpoint() -> None:
        save_json(
            checkpoint_path,
            {
                "mappings": accepted,
                "judgments": [judgments_by_id[key] for key in sorted(judgments_by_id)],
            },
        )

    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {executor.submit(judge, item): item.id for item in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            relation_id, candidate_id, result, candidate_ids = future.result()
            if candidate_id is not None:
                accepted[relation_id] = candidate_id
            judgments_by_id[relation_id] = {
                "relation_id": relation_id,
                "candidate_ids": candidate_ids,
                "selected_candidate_id": candidate_id,
                "result": result,
            }
            if completed % 10 == 0 or completed == len(futures):
                checkpoint()
                log(f"mapping completed {completed}/{len(futures)}")
    return accepted, [judgments_by_id[key] for key in sorted(judgments_by_id)]


def build_mapping_rows(
    *,
    relations: Sequence[RelationRecord],
    candidates: Sequence[Candidate],
    high_frequency_threshold: int,
    mappings: dict[int, str],
    mapping_judgments: Sequence[dict[str, Any]],
    validation_judgments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {item.id: item for item in candidates}
    map_judgment = {int(item["relation_id"]): item for item in mapping_judgments}
    validation = {int(item["relation_id"]): item for item in validation_judgments}
    rows: list[dict[str, Any]] = []
    for relation in sorted(relations, key=lambda item: item.id):
        if relation.uses >= high_frequency_threshold:
            target_id = f"seed-{relation.id}"
            status = "identity_high_frequency"
            validated = True
        else:
            target_id = mappings.get(relation.id)
            status = "mapped" if target_id else "unmapped"
            validated = bool(target_id)
        target = candidate_by_id.get(target_id) if target_id else None
        initial = map_judgment.get(relation.id, {})
        final = validation.get(relation.id, {})
        rows.append(
            {
                "source_relation_id": relation.id,
                "source_relation_name": relation.name,
                "uses": relation.uses,
                "mapping_status": status,
                "target_relation_id": target_id,
                "target_relation_name": target.name if target else None,
                "target_relation_source": target.source if target else None,
                "validated": validated,
                "mapping_reason": initial.get("result", {}).get("reason"),
                "validation_reason": final.get("result", {}).get("reason"),
            }
        )
    return rows


def save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--high-frequency-threshold", type=int, default=10)
    parser.add_argument(
        "--low-sample-size",
        type=int,
        help="仅用于冒烟测试；从低频关系中等距抽取指定数量",
    )
    parser.add_argument("--max-coarse-types", type=int, default=100)
    parser.add_argument("--raw-cluster-multiplier", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="https://api.minimaxi.com/v1")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers 必须在 1 到 8 之间")
    if args.high_frequency_threshold < 2:
        parser.error("--high-frequency-threshold 必须至少为 2")
    if args.low_sample_size is not None and args.low_sample_size < 1:
        parser.error("--low-sample-size 必须为正数")
    if args.max_coarse_types < 1:
        parser.error("--max-coarse-types 必须为正数")
    if args.raw_cluster_multiplier < 1:
        parser.error("--raw-cluster-multiplier 不得小于 1")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"环境变量 {args.api_key_env} 未设置")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_relations = load_relations(args.db)
    high = [
        item for item in all_relations if item.uses >= args.high_frequency_threshold
    ]
    all_low = [
        item for item in all_relations if item.uses < args.high_frequency_threshold
    ]
    low = (
        evenly_spaced(all_low, args.low_sample_size)
        if args.low_sample_size is not None
        else all_low
    )
    relations = [*high, *low]
    records = {item.id: item for item in relations}
    frozen = [
        Candidate(
            id=f"seed-{item.id}",
            name=item.name,
            definition=item.description,
            source="high-frequency",
            source_relation_ids=[item.id],
        )
        for item in sorted(high, key=lambda item: (-item.uses, item.id))
    ]
    new_type_budget = max(args.max_coarse_types - len(frozen), 0)
    raw_cluster_count = min(
        len(low), max(new_type_budget, round(new_type_budget * args.raw_cluster_multiplier))
    )
    manifest = {
        "db": str(args.db.resolve()),
        "db_relation_types": len(all_relations),
        "db_claims": sum(item.uses for item in all_relations),
        "input_relation_types": len(relations),
        "input_claims": sum(item.uses for item in relations),
        "high_frequency_threshold": args.high_frequency_threshold,
        "high_frequency_types": len(high),
        "low_frequency_types": len(low),
        "low_sample_size": args.low_sample_size,
        "max_coarse_types": args.max_coarse_types,
        "new_type_budget": new_type_budget,
        "raw_cluster_multiplier": args.raw_cluster_multiplier,
        "raw_cluster_count": raw_cluster_count,
        "top_k": args.top_k,
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
    }
    save_json(args.output_dir / "manifest.json", manifest)
    llm = CachedLLM(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_concurrency=args.workers,
        cache_path=args.output_dir / "llm_cache.jsonl",
    )

    candidates_path = args.output_dir / "candidate_generation.json"
    if candidates_path.exists():
        generation = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates = [Candidate(**item) for item in generation["candidates"]]
    else:
        log(
            f"clustering {len(low)} low-frequency relation names into "
            f"{raw_cluster_count} recall groups"
        )
        clusters = fixed_count_name_clusters(low, cluster_count=raw_cluster_count)
        raw_proposals, synthesis = synthesize_cluster_proposals(llm=llm, clusters=clusters)
        proposals, consolidation = consolidate_proposals(
            llm=llm,
            proposals=raw_proposals,
            frozen=frozen,
            max_new_types=new_type_budget,
        )
        candidates = build_candidates(frozen, proposals)
        generation = {
            "clusters": [[item.id for item in cluster] for cluster in clusters],
            "synthesis": synthesis,
            "raw_proposals": [asdict(item) for item in raw_proposals],
            "consolidation": consolidation,
            "final_proposals": [asdict(item) for item in proposals],
            "candidates": [asdict(item) for item in candidates],
        }
        save_json(candidates_path, generation)
    log(
        f"candidate catalog ready: {len(frozen)} frozen + "
        f"{len(candidates) - len(frozen)} generated = {len(candidates)}"
    )

    retrieved = retrieve_top_candidates(low, candidates, top_k=args.top_k)
    preliminary, mapping_judgments = map_relations_top_k(
        llm=llm,
        relations=low,
        retrieved=retrieved,
        checkpoint_path=args.output_dir / "mapping_checkpoint.json",
    )
    save_json(args.output_dir / "mapping_judgments.json", mapping_judgments)
    validated, rejected, validation_judgments = validate_all_triples(
        llm=llm,
        accepted=preliminary,
        candidates=candidates,
        records=records,
        label="simple-coarse-layer",
    )
    save_json(args.output_dir / "validation_judgments.json", validation_judgments)

    rows = build_mapping_rows(
        relations=relations,
        candidates=candidates,
        high_frequency_threshold=args.high_frequency_threshold,
        mappings=validated,
        mapping_judgments=mapping_judgments,
        validation_judgments=validation_judgments,
    )
    member_ids: dict[str, list[int]] = {item.id: [] for item in candidates}
    for row in rows:
        target_id = row["target_relation_id"]
        if target_id:
            member_ids[target_id].append(int(row["source_relation_id"]))
    coarse_rows = []
    for candidate in candidates:
        members = member_ids[candidate.id]
        coarse_rows.append(
            {
                **asdict(candidate),
                "mapped_relation_ids": members,
                "mapped_relation_type_count": len(members),
                "mapped_claim_count": sum(records[item].uses for item in members),
            }
        )
    save_json(args.output_dir / "coarse_relations.json", coarse_rows)
    save_json(args.output_dir / "relation_mappings.json", rows)
    save_jsonl(args.output_dir / "relation_mappings.jsonl", rows)

    low_claims = sum(item.uses for item in low)
    mapped_low_claims = sum(records[item].uses for item in validated)
    used_candidate_ids = {value for value in validated.values()}.union(
        candidate.id for candidate in frozen
    )
    summary = {
        "status": "complete",
        "input_relation_types": len(relations),
        "input_claims": sum(item.uses for item in relations),
        "high_frequency_relation_types": len(high),
        "low_frequency_relation_types": len(low),
        "coarse_catalog_types": len(candidates),
        "used_coarse_catalog_types": len(used_candidate_ids),
        "generated_coarse_types": len(candidates) - len(frozen),
        "preliminary_mapped_low_types": len(preliminary),
        "validation_rejected_low_types": len(rejected),
        "mapped_low_relation_types": len(validated),
        "unmapped_low_relation_types": len(low) - len(validated),
        "low_type_coverage": len(validated) / max(len(low), 1),
        "mapped_low_claims": mapped_low_claims,
        "low_claims": low_claims,
        "low_claim_coverage": mapped_low_claims / max(low_claims, 1),
        "all_types_with_coarse_mapping": len(high) + len(validated),
        "all_type_coverage": (len(high) + len(validated)) / max(len(relations), 1),
        "all_claims_with_coarse_mapping": sum(item.uses for item in high)
        + mapped_low_claims,
        "all_claim_coverage": (
            sum(item.uses for item in high) + mapped_low_claims
        )
        / max(sum(item.uses for item in relations), 1),
        "effective_type_count_if_unmapped_retained": len(used_candidate_ids)
        + len(low)
        - len(validated),
        "runtime": llm.metrics,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
