from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from kg.embeddings import _encode
from kg.llm import ChatCompletionsJSONLLM, LLMConcurrencyLimiter, LLMConfig


SYSTEM = """你是知识图谱粗粒度关系归并专家。Assertion 保存完整事实，粗粒度关系只表达
稳定、可复用的核心关系。不能改变主客体角色或方向，不能把没有共同核心的关系强行归入
“相关于”“涉及”“作用于”等空泛类别。候选发现阶段根据提供的代表性三元组判断；成员
映射最终必须检查该 RelationType 在图谱中的全部三元组。使用候选粗关系连接每个原
subject 和 object 时都必须仍然成立，不得改变、扩展或重新解释端点。只输出符合要求的
JSON 对象。"""

TRIPLE_VALIDATION_SYSTEM = """你是知识图谱粗三元组的事实校验员。本阶段只判断原
Assertion 是否支持使用候选粗关系连接原 subject 和 object。粗关系比原关系更宽、信息量
更少本身不是拒绝理由，省略的细节由 Assertion 保留；不要评价候选是否过粗。只有投影
三元组不受原 Assertion 直接支持、方向错误、端点改变，或端点被替换成隐藏属性/输出/
类别时才拒绝。注意：端点字符串没变不等于端点语义没变。如果 Assertion 实际关系目标
是“Y 的属性/组成部分/输出/尺寸”等，不能省略该路径后声称关系目标就是 Y。只输出符合
要求的 JSON 对象。

不得把较弱命题升级成较强命题：目标/意图不代表效果已经发生，“可能/可以”不代表实际
发生，“有助于/促进”不代表主体直接完成该动作，相关性不代表因果性。上述限定会改变
投影命题的真值条件，不属于可由 Assertion 单独保留的普通细节。"""


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


@dataclass(frozen=True)
class RelationRecord:
    id: int
    name: str
    description: str
    uses: int
    assertions: tuple[str, ...]
    examples: tuple[tuple[str, str, str], ...]
    all_examples: tuple[tuple[str, str, str], ...] = ()


@dataclass
class Candidate:
    id: str
    name: str
    definition: str
    source: str
    source_relation_ids: list[int]


@dataclass(frozen=True)
class Proposal:
    name: str
    definition: str
    source_relation_ids: tuple[int, ...]
    source: str


class CachedLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrency: int,
        cache_path: Path,
    ) -> None:
        if not 1 <= max_concurrency <= 8:
            raise ValueError("实验全局并发必须在 1 到 8 之间")
        self.client = ChatCompletionsJSONLLM(
            LLMConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=600.0,
                retries=3,
            ),
            limiter=LLMConcurrencyLimiter(max_concurrency),
        )
        self.max_concurrency = max_concurrency
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self.metrics = {
            "cache_hits": 0,
            "api_calls": 0,
            "prompt_chars": 0,
            "elapsed_seconds": 0.0,
        }
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                self._cache[str(item["key"])] = item

    def complete(self, *, kind: str, system: str, user: str) -> dict[str, Any]:
        key = hashlib.sha256(
            json.dumps(
                {"kind": kind, "system": system, "user": user},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.metrics["cache_hits"] += 1
                return dict(cached["result"])
        started = time.monotonic()
        for parse_attempt in range(1, 4):
            try:
                result = self.client.complete_json(system, user)
                break
            except json.JSONDecodeError:
                if parse_attempt == 3:
                    raise
                log(
                    f"{kind}: MiniMax returned malformed JSON; "
                    f"retrying ({parse_attempt}/2)"
                )
        elapsed = time.monotonic() - started
        record = {
            "key": key,
            "kind": kind,
            "prompt_chars": len(system) + len(user),
            "elapsed_seconds": elapsed,
            "result": result,
        }
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self.metrics["cache_hits"] += 1
                return dict(existing["result"])
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._cache[key] = record
            self.metrics["api_calls"] += 1
            self.metrics["prompt_chars"] += int(record["prompt_chars"])
            self.metrics["elapsed_seconds"] += elapsed
        return result


def _compact_text(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _sample_three(values: Sequence[str]) -> tuple[str, ...]:
    unique = list(dict.fromkeys(value for value in values if value.strip()))
    if len(unique) <= 3:
        return tuple(unique)
    return (unique[0], unique[len(unique) // 2], unique[-1])


def load_relations(db_path: Path) -> list[RelationRecord]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT rt.id,rt.canonical_name,rt.description,COUNT(c.id) AS uses
        FROM relation_types rt
        JOIN claims c ON c.relation_type_id=rt.id
        GROUP BY rt.id
        ORDER BY rt.id
        """
    ).fetchall()
    assertions: dict[int, list[str]] = {}
    examples: dict[int, list[tuple[str, str, str]]] = {}
    for row in conn.execute(
        """
        SELECT c.relation_type_id,a.normalized_text
        FROM claims c JOIN assertions a ON a.claim_id=c.id
        ORDER BY c.relation_type_id,c.id,a.id
        """
    ):
        assertions.setdefault(int(row[0]), []).append(str(row[1]))
    for row in conn.execute(
        """
        SELECT c.relation_type_id,s.canonical_name,o.canonical_name,
               COALESCE((SELECT group_concat(a.normalized_text, '; ')
                         FROM assertions a WHERE a.claim_id=c.id),'')
        FROM claims c
        JOIN entities s ON s.id=c.subject_id
        JOIN entities o ON o.id=c.object_id
        ORDER BY c.relation_type_id,c.id
        """
    ):
        examples.setdefault(int(row[0]), []).append(
            (str(row[1]), str(row[2]), str(row[3]))
        )
    conn.close()
    records: list[RelationRecord] = []
    for row in rows:
        relation_id = int(row["id"])
        all_examples = tuple(dict.fromkeys(examples.get(relation_id, [])))
        records.append(
            RelationRecord(
                id=relation_id,
                name=str(row["canonical_name"]),
                description=str(row["description"]),
                uses=int(row["uses"]),
                assertions=_sample_three(assertions.get(relation_id, [])),
                examples=tuple(_sample_three_examples(all_examples)),
                all_examples=all_examples,
            )
        )
    return records


def _sample_three_examples(
    values: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    unique = list(dict.fromkeys(values))
    if len(unique) <= 3:
        return unique
    return [unique[0], unique[len(unique) // 2], unique[-1]]


def evenly_spaced(values: Sequence[RelationRecord], count: int) -> list[RelationRecord]:
    if count >= len(values):
        return list(values)
    if count <= 0:
        return []
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in dict.fromkeys(indices)]


def select_pilot(
    relations: Sequence[RelationRecord], sample_size: int | None
) -> list[RelationRecord]:
    residual = [item for item in relations if item.uses < 10]
    residual.sort(key=lambda item: (-item.uses, item.id))
    if sample_size is None or sample_size >= len(residual):
        return residual
    if sample_size == 200:
        groups = [
            ([item for item in residual if 3 <= item.uses <= 9], 80),
            ([item for item in residual if item.uses == 2], 60),
            ([item for item in residual if item.uses == 1], 60),
        ]
        selected = [item for values, count in groups for item in evenly_spaced(values, count)]
        return sorted(selected, key=lambda item: (-item.uses, item.id))
    return evenly_spaced(residual, sample_size)


def seed_candidates(relations: Sequence[RelationRecord]) -> list[Candidate]:
    return [
        Candidate(
            id=f"seed-{item.id}",
            name=item.name,
            definition=item.description,
            source="seed",
            source_relation_ids=[item.id],
        )
        for item in sorted(relations, key=lambda item: (-item.uses, item.id))
        if item.uses >= 10
    ]


def candidate_payload(
    candidates: Sequence[Candidate], mappings: dict[int, str], records: dict[int, RelationRecord]
) -> list[dict[str, Any]]:
    totals: dict[str, int] = {
        candidate.id: sum(
            records[relation_id].uses
            for relation_id in candidate.source_relation_ids
            if candidate.source == "seed" and relation_id in records
        )
        for candidate in candidates
    }
    for relation_id, candidate_id in mappings.items():
        if candidate_id in totals and relation_id in records:
            totals[candidate_id] += records[relation_id].uses
    return [
        {
            "id": item.id,
            "name": item.name,
            "definition": _compact_text(item.definition, 220),
            "mapped_uses": totals[item.id],
        }
        for item in sorted(candidates, key=lambda item: (-totals[item.id], item.id))
    ]


def relation_payload(item: RelationRecord) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "description": _compact_text(item.description, 360),
        "uses": item.uses,
        "assertions": [_compact_text(value, 420) for value in item.assertions],
        "examples": [
            {
                "subject": subject,
                "object": object_,
                "assertion": _compact_text(assertion, 420),
            }
            for subject, object_, assertion in item.examples
        ],
    }


def parse_mapping(result: dict[str, Any], candidate_ids: set[str]) -> str | None:
    if str(result.get("decision", "")).strip().lower() != "map":
        return None
    selected = str(result.get("candidate_id", "")).strip()
    return selected if selected in candidate_ids else None


def full_validation_passes(result: dict[str, Any], expected_examples: int) -> bool:
    if str(result.get("decision", "")).strip().lower() != "full_map":
        return False
    checks = result.get("checks")
    if not isinstance(checks, list) or len(checks) != expected_examples:
        return False
    seen: set[int] = set()
    for check in checks:
        if not isinstance(check, dict) or check.get("valid") is not True:
            return False
        index = check.get("example_index")
        if not isinstance(index, int):
            return False
        seen.add(index)
    return seen == set(range(1, expected_examples + 1))


def validate_all_triples(
    *,
    llm: CachedLLM,
    accepted: dict[int, str],
    candidates: Sequence[Candidate],
    records: dict[int, RelationRecord],
    label: str,
) -> tuple[dict[int, str], list[int], list[dict[str, Any]]]:
    """Validate every stored claim triple before accepting a type mapping."""

    if not accepted:
        return {}, [], []
    candidate_by_id = {item.id: item for item in candidates}
    log(f"{label}: validating all triples for {len(accepted)} mapped relations")

    def judge(
        relation_id: int, candidate_id: str
    ) -> tuple[int, str, bool, dict[str, Any]]:
        relation = records[relation_id]
        candidate = candidate_by_id[candidate_id]
        examples = relation.all_examples or relation.examples
        payload = relation_payload(relation)
        payload["examples"] = [
            {
                "example_index": index,
                "subject": subject,
                "object": object_,
                "assertion": _compact_text(assertion, 600),
            }
            for index, (subject, object_, assertion) in enumerate(examples, start=1)
        ]
        prompt = """这是成员映射的最终全量三元组事实校验，不是抽样。下面 examples
包含当前 RelationType 在图谱中的全部 subject/object 三元组及其 Assertion。必须逐条把
原 subject、原 object 原封不动代入候选粗关系；不得把 object 改成它的属性、输出、类别
或其他隐藏对象，不得交换端点。

本阶段只检查投影三元组是否由原 Assertion 支持。候选关系更宽、信息量更少、未保留原
关系的机制/程度/条件等细节，不构成拒绝理由，因为这些细节仍保留在 Assertion 中。不要
在本阶段判断候选是否过于宽泛，也不要因为信息损失而判 no_map。

“可省略的关系细节”和“不可省略的端点路径”必须严格区分：
- 可以：`X 是 Y 的机制` 投影为 `X 存在于 Y`，只要 Assertion 直接支持 X 位于 Y 中；
- 不可以：Assertion 为 `X 表示 Y 的尺寸/属性` 时投影为 `X 表示 Y`。真实关系目标是
  Y 的隐藏属性而不是 Y，哪怕输入中的 object 字符串仍写作 Y，也属于端点语义扩展。
- 不可以：Assertion 只说 `X 的损失函数/步骤/组成部分基于 Y` 时投影为 `X 基于 Y`。
  这省略了 subject 到其局部机制的路径，不能凭常识把局部依据提升为整体依据。

对每条先做反例检查：在原 Assertion 完全为真的情况下，投影三元组是否仍可能为假？
如果可能，就必须 valid=false 并 no_map；不得靠补入“Y 所代表的属性/信息”等未出现在
投影中的词来使句子成立，也不得使用“说明、体现、可视为、参与”等推断桥接词把未直接
陈述的关系补出来。

只有每一条“subject→候选关系→object”都仍然为真时才能 full_map；任一条不成立就
no_map。checks 必须与 examples 一一对应，不能遗漏或合并。
返回 {"decision":"full_map|no_map","checks":[{"example_index":1,
"projection":"严格使用原 subject 和 object 的句子","valid":true,"reason":""}],
"reason":""}。
当前关系=%s
唯一候选=%s""" % (
            json.dumps(payload, ensure_ascii=False),
            json.dumps(
                {
                    "id": candidate.id,
                    "name": candidate.name,
                    "definition": candidate.definition,
                },
                ensure_ascii=False,
            ),
        )
        result = llm.complete(
            kind=f"full-triple-validation-truth-only:{label}",
            system=TRIPLE_VALIDATION_SYSTEM,
            user=prompt,
        )
        passed = full_validation_passes(result, len(examples))
        return relation_id, candidate_id, passed, result

    validated: dict[int, str] = {}
    rejected: list[int] = []
    judgments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {
            executor.submit(judge, relation_id, candidate_id): relation_id
            for relation_id, candidate_id in accepted.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            relation_id, candidate_id, passed, result = future.result()
            judgments.append(
                {
                    "relation_id": relation_id,
                    "candidate_id": candidate_id,
                    "example_count": len(
                        records[relation_id].all_examples
                        or records[relation_id].examples
                    ),
                    "passed": passed,
                    "result": result,
                }
            )
            if passed:
                validated[relation_id] = candidate_id
            else:
                rejected.append(relation_id)
            if completed % 10 == 0 or completed == len(futures):
                log(f"{label}: completed {completed}/{len(futures)}")
    judgments.sort(key=lambda item: int(item["relation_id"]))
    return validated, sorted(rejected), judgments


def candidate_gate_passes(result: dict[str, Any]) -> bool:
    return str(result.get("decision", "")).strip().lower() == "valid_candidate"


def validate_proposals(
    *,
    llm: CachedLLM,
    proposals: Sequence[Proposal],
    records: dict[int, RelationRecord],
    label: str,
) -> tuple[list[Proposal], list[dict[str, Any]]]:
    """Reject union-like or non-entailing coarse predicates before mapping."""

    if not proposals:
        return [], []
    log(f"{label}: validating atomicity of {len(proposals)} candidates")

    def judge(index: int, proposal: Proposal) -> tuple[int, dict[str, Any]]:
        members = [
            relation_payload(records[relation_id])
            for relation_id in proposal.source_relation_ids
            if relation_id in records
        ]
        prompt = """严格审核一个候选粗关系是否是合格的二元原子谓词。粗关系可以省略
原关系中的程度、数值、机制等细节，因为 Assertion 会保留这些信息；但候选本身必须满足：
1. 能写成一个单一的“X 对 Y 做什么/具有什么关系”，不能用“或”“以及”“/”拼接多个
核心动作、多个语义分支或多个主客体角色；
2. 每一个来源关系在保持原 subject、object 和方向不变时，都必然蕴含这个候选关系；
不能把“可能”提升为确定，不能把“能力”改成已发生，不能改变否定范围；
3. 不能用“相关”“评价”“影响”等几乎任何关系都能落入的空泛兜底词，除非所有来源
确实共享同一种明确的核心作用；
4. 任一来源不满足就必须 invalid_candidate，不能只删掉坏成员，也不能为了覆盖率放行。
返回 {"decision":"valid_candidate|invalid_candidate",
"atomic_paraphrase":"X→谓词→Y 的单句改写", "checks":[{"relation_id":1,
"entailed":true,"reason":""}], "reason":""}。
候选=%s
来源关系=%s""" % (
            json.dumps(asdict(proposal), ensure_ascii=False),
            json.dumps(members, ensure_ascii=False),
        )
        return index, llm.complete(
            kind=f"candidate-gate:{label}", system=SYSTEM, user=prompt
        )

    accepted: list[tuple[int, Proposal]] = []
    judgments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {
            executor.submit(judge, index, proposal): (index, proposal)
            for index, proposal in enumerate(proposals)
        }
        for future in as_completed(futures):
            index, result = future.result()
            proposal = proposals[index]
            passed = candidate_gate_passes(result)
            judgments.append(
                {
                    "proposal_index": index,
                    "proposal": asdict(proposal),
                    "passed": passed,
                    "result": result,
                }
            )
            if passed:
                accepted.append((index, proposal))
    accepted.sort(key=lambda item: item[0])
    judgments.sort(key=lambda item: int(item["proposal_index"]))
    log(f"{label}: candidate gate retained {len(accepted)}/{len(proposals)}")
    return [proposal for _, proposal in accepted], judgments


def map_relations(
    *,
    llm: CachedLLM,
    relations: Sequence[RelationRecord],
    candidates: Sequence[Candidate],
    mappings: dict[int, str],
    records: dict[int, RelationRecord],
    label: str,
) -> tuple[dict[int, str], list[int], list[dict[str, Any]]]:
    shown = candidate_payload(candidates, mappings, records)
    candidate_ids = {item.id for item in candidates}
    log(f"{label}: mapping {len(relations)} relations against {len(candidates)} candidates")

    def judge(item: RelationRecord) -> tuple[int, str | None, dict[str, Any]]:
        prompt = """判断当前细粒度关系能否归入一个候选粗关系。Assertion 保留具体属性、
程度、条件、机制和否定信息；候选只保留稳定核心。但主客体角色、关系方向和核心动作必须
兼容。必须逐条检查 examples：把原 subject、原 object 原封不动代入候选关系后，得到的
“subject→候选关系→object”必须仍然为真。不得把 object 重新解释成它的属性、输出、类别
或其他隐藏对象，也不得交换端点。任一代表例子不成立就 no_map。若没有准确候选必须
no_map，不得为了提高覆盖率强行选择。
返回 {"decision":"map|no_map","candidate_id":null,
"projections":["逐条口头化 subject→候选→object"],"reason":"..."}。
当前关系=%s
候选关系=%s""" % (
            json.dumps(relation_payload(item), ensure_ascii=False),
            json.dumps(shown, ensure_ascii=False),
        )
        result = llm.complete(kind=f"map:{label}", system=SYSTEM, user=prompt)
        return item.id, parse_mapping(result, candidate_ids), result

    accepted: dict[int, str] = {}
    residual: list[int] = []
    judgments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {executor.submit(judge, item): item.id for item in relations}
        for completed, future in enumerate(as_completed(futures), start=1):
            relation_id, candidate_id, result = future.result()
            judgments.append({"relation_id": relation_id, "result": result})
            if candidate_id is None:
                residual.append(relation_id)
            else:
                accepted[relation_id] = candidate_id
            if completed % 10 == 0 or completed == len(futures):
                log(f"{label}: completed {completed}/{len(futures)}")
    judgments.sort(key=lambda item: int(item["relation_id"]))
    residual.sort()
    return accepted, residual, judgments


def embedding_clusters(
    relations: Sequence[RelationRecord],
    *,
    distance_threshold: float,
    min_cluster_size: int = 3,
) -> list[list[RelationRecord]]:
    if min_cluster_size < 2:
        raise ValueError("Embedding 聚类最少成员数必须至少为 2")
    if len(relations) < min_cluster_size:
        return []
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError(
            "Embedding 聚类实验需要安装项目的 sentence-transformers 依赖"
        ) from exc
    # Embedding is deliberately only a high-recall grouping signal.  Predicate
    # definitions and evidence are left to the LLM subset-selection step below.
    texts = [item.name for item in relations]
    matrix = np.asarray(_encode(texts), dtype="float32")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(matrix)
    grouped: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(labels):
        grouped.setdefault(int(cluster_id), []).append(index)
    clusters: list[list[RelationRecord]] = []
    for indices in grouped.values():
        if len(indices) < min_cluster_size:
            continue
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


def refine_oversized_clusters(
    clusters: Sequence[Sequence[RelationRecord]],
    *,
    max_cluster_size: int,
    split_distance_threshold: float,
    min_cluster_size: int,
) -> tuple[list[list[RelationRecord]], list[dict[str, Any]]]:
    """Split broad recall clusters before asking the LLM for a predicate.

    Members that do not form a sufficiently large tighter subcluster are left
    for the next global round. They are deliberately not bundled into an
    arbitrary overflow group, because that would recreate the noisy large
    cluster that this refinement is intended to avoid.
    """
    if max_cluster_size < min_cluster_size:
        raise ValueError("Embedding 簇最大成员数不得小于最少成员数")
    refined: list[list[RelationRecord]] = []
    records: list[dict[str, Any]] = []
    for parent_index, original in enumerate(clusters):
        cluster = list(original)
        if len(cluster) <= max_cluster_size:
            refined.append(cluster)
            continue
        children = embedding_clusters(
            cluster,
            distance_threshold=split_distance_threshold,
            min_cluster_size=min_cluster_size,
        )
        if (
            split_distance_threshold <= 0.0100001
            and len(children) == 1
            and {item.id for item in children[0]} == {item.id for item in cluster}
        ):
            children = []
        covered_ids = {item.id for child in children for item in child}
        if children:
            child_refined, child_records = refine_oversized_clusters(
                children,
                max_cluster_size=max_cluster_size,
                split_distance_threshold=max(split_distance_threshold * 0.8, 0.01),
                min_cluster_size=min_cluster_size,
            )
            refined.extend(child_refined)
        else:
            child_records = []
        records.append(
            {
                "parent_cluster_index": parent_index,
                "parent_size": len(cluster),
                "split_distance_threshold": split_distance_threshold,
                "child_sizes": [len(child) for child in children],
                "deferred_member_ids": [
                    item.id for item in cluster if item.id not in covered_ids
                ],
            }
        )
        records.extend(child_records)
    refined.sort(key=lambda group: (-sum(item.uses for item in group), -len(group)))
    return refined, records


def parse_embedding_subsets(
    result: dict[str, Any],
    cluster: Sequence[RelationRecord],
    *,
    round_index: int,
    min_member_types: int = 3,
) -> list[Proposal]:
    cluster_ids = {item.id for item in cluster}
    used_ids: set[int] = set()
    proposals: list[Proposal] = []
    items = result.get("proposals", [])
    if not isinstance(items, list):
        return proposals
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("canonical_name", "")).strip()
        definition = str(item.get("definition", "")).strip()
        source_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in item.get("source_relation_ids", [])
                    if isinstance(value, int)
                    and int(value) in cluster_ids
                    and int(value) not in used_ids
                }
            )
        )
        if not name or not definition or len(source_ids) < min_member_types:
            continue
        used_ids.update(source_ids)
        proposals.append(
            Proposal(
                name=name,
                definition=definition,
                source_relation_ids=source_ids,
                source=f"embedding-r{round_index}",
            )
        )
    return proposals


def synthesize_embedding_proposals(
    *,
    llm: CachedLLM,
    residual: Sequence[RelationRecord],
    round_index: int,
    max_proposals: int,
    distance_threshold: float,
    min_cluster_size: int,
    min_member_types: int,
    exhaustive_cluster_extraction: bool = False,
    max_cluster_size: int = 20,
    split_distance_threshold: float = 0.075,
    max_cluster_extraction_passes: int = 10,
) -> tuple[list[Proposal], dict[str, Any]]:
    original_clusters = embedding_clusters(
        residual,
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size,
    )
    if exhaustive_cluster_extraction:
        clusters, refinements = refine_oversized_clusters(
            original_clusters,
            max_cluster_size=max_cluster_size,
            split_distance_threshold=split_distance_threshold,
            min_cluster_size=min_cluster_size,
        )
    else:
        clusters = original_clusters
        refinements = []
    selected = clusters[:max_proposals]
    log(
        f"embedding:r{round_index}: {len(original_clusters)} eligible clusters, "
        f"{len(clusters)} after refinement; "
        f"synthesizing {len(selected)}"
    )

    def synthesize(
        index: int, cluster: list[RelationRecord]
    ) -> tuple[int, list[Proposal], list[dict[str, Any]]]:
        remaining = list(cluster)
        cluster_proposals: list[Proposal] = []
        attempts: list[dict[str, Any]] = []
        pass_limit = max_cluster_extraction_passes if exhaustive_cluster_extraction else 1
        for pass_index in range(1, pass_limit + 1):
            if len(remaining) < min_cluster_size:
                break
            members = []
            for position, item in enumerate(remaining):
                payload = relation_payload(item)
                payload["center_rank"] = position + 1
                members.append(payload)
            prompt = """下面是一组仅根据关系名称 Embedding 得到的相似关系，已按距离中心
从近到远排列。Embedding 只负责召回，不代表整个簇必须合并。请结合每条关系的定义、
提供的代表性 Assertion 和 subject/object 例子，从中挑出真正共享同一个粗粒度二元谓词
的子集。候选成员稍后还会使用图谱中的全部三元组做最终校验。

可以只选择部分成员：例如 5 条中只有 3 条兼容，就只用这 3 条生成候选，其余不合并。
一个簇中若存在两个互不相同但各自有效的子集，可以输出两个候选。每个候选至少包含
%d 个成员，成员不能在多个候选中重复。候选必须保持主客体角色、方向、否定和模态；
不能用“或”“/”拼接多个核心动作，也不能生成“相关”“影响”一类空泛兜底关系。
如果找不到至少 %d 条真正兼容的成员，返回空 proposals。

返回 {"proposals":[{"canonical_name":"","definition":"必须说明单一核心谓词、
主客体角色和方向","source_relation_ids":[1,2,3],"reason":""}],
"unmerged_relation_ids":[4,5]}。
聚类成员=%s""" % (
                min_member_types,
                min_member_types,
                json.dumps(members, ensure_ascii=False),
            )
            result = llm.complete(
                kind=(
                    f"embedding-synthesis:r{round_index}:"
                    f"cluster{index}:pass{pass_index}"
                ),
                system=SYSTEM,
                user=prompt,
            )
            parsed = parse_embedding_subsets(
                result,
                remaining,
                round_index=round_index,
                min_member_types=min_member_types,
            )
            selected_ids = {
                relation_id
                for proposal in parsed
                for relation_id in proposal.source_relation_ids
            }
            attempts.append(
                {
                    "pass": pass_index,
                    "member_ids": [item.id for item in remaining],
                    "result": result,
                    "accepted_source_ids": sorted(selected_ids),
                }
            )
            cluster_proposals.extend(parsed)
            if not selected_ids:
                break
            remaining = [item for item in remaining if item.id not in selected_ids]
        return index, cluster_proposals, attempts

    raw: list[dict[str, Any]] = []
    proposals: list[Proposal] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {
            executor.submit(synthesize, index, cluster): (index, cluster)
            for index, cluster in enumerate(selected)
        }
        for future in as_completed(futures):
            index, parsed, attempts = future.result()
            cluster = selected[index]
            raw.append(
                {
                    "cluster_index": index,
                    "member_ids": [item.id for item in cluster],
                    "attempts": attempts,
                }
            )
            proposals.extend(parsed)
    raw.sort(key=lambda item: int(item["cluster_index"]))
    proposals.sort(
        key=lambda item: (
            min(item.source_relation_ids),
            item.name,
            item.source_relation_ids,
        )
    )
    return proposals[:max_proposals], {
        "cluster_count": len(original_clusters),
        "refined_cluster_count": len(clusters),
        "selected_cluster_count": len(selected),
        "cluster_refinements": refinements,
        "exhaustive_cluster_extraction": exhaustive_cluster_extraction,
        "clusters": raw,
    }


def _batch(values: Sequence[RelationRecord], size: int) -> Iterable[list[RelationRecord]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def synthesize_llm_proposals(
    *,
    llm: CachedLLM,
    residual: Sequence[RelationRecord],
    existing: Sequence[Candidate],
    round_index: int,
    max_proposals: int,
    batch_size: int,
) -> tuple[list[Proposal], dict[str, Any]]:
    batches = list(_batch(residual, batch_size))
    per_batch = max(2, math.ceil(max_proposals / max(len(batches), 1)) + 1)
    log(
        f"llm:r{round_index}: inducing candidates from {len(residual)} relations "
        f"in {len(batches)} batches"
    )

    def induce(index: int, items: list[RelationRecord]) -> tuple[int, dict[str, Any]]:
        payload = [
            {
                "id": item.id,
                "name": item.name,
                "description": _compact_text(item.description, 260),
                "uses": item.uses,
                "assertion": _compact_text(item.assertions[0], 260)
                if item.assertions
                else "",
                "example": {
                    "subject": item.examples[0][0],
                    "object": item.examples[0][1],
                    "assertion": _compact_text(item.examples[0][2], 260),
                }
                if item.examples
                else None,
            }
            for item in items
        ]
        prompt = """这些关系尚不能归入现有粗关系。直接从整体语义出发，提出尽量少、
但有清晰边界且能覆盖多个成员的新粗粒度关系。不要逐条改名，不要制造只能覆盖一条关系
的类型，也不要使用空泛兜底类型。每个提案列出它应覆盖的原关系 ID。
最多提出 %d 个。返回 {"proposals":[{"canonical_name":"","definition":"包含主客体
角色和方向","source_relation_ids":[1,2,3]}]}。
现有候选名称=%s
未映射关系=%s""" % (
            per_batch,
            json.dumps([item.name for item in existing], ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False),
        )
        return index, llm.complete(
            kind=f"llm-induction:r{round_index}", system=SYSTEM, user=prompt
        )

    batch_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as executor:
        futures = {
            executor.submit(induce, index, items): index
            for index, items in enumerate(batches)
        }
        for future in as_completed(futures):
            index, result = future.result()
            batch_results.append({"batch_index": index, "result": result})
    batch_results.sort(key=lambda item: int(item["batch_index"]))

    valid_ids = {item.id for item in residual}
    raw_proposals: list[dict[str, Any]] = []
    for batch_result in batch_results:
        proposals = batch_result["result"].get("proposals", [])
        if not isinstance(proposals, list):
            continue
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            name = str(proposal.get("canonical_name", "")).strip()
            definition = str(proposal.get("definition", "")).strip()
            ids = proposal.get("source_relation_ids", [])
            source_ids = sorted(
                {
                    int(value)
                    for value in ids
                    if isinstance(value, int) and int(value) in valid_ids
                }
            )
            if name and definition and len(source_ids) >= 2:
                raw_proposals.append(
                    {
                        "canonical_name": name,
                        "definition": definition,
                        "source_relation_ids": source_ids,
                    }
                )
    if not raw_proposals:
        return [], {"batch_results": batch_results, "consolidation": None}

    consolidation_prompt = """合并下面各批次提出的粗粒度关系，删除与现有候选重复的
提案，并把含义相同的提案合并。直接决定最终需要增加几个类型，最多 %d 个。每个最终
类型必须覆盖至少三个给出的原关系 ID，并保留清晰的主客体角色和方向。不要为了覆盖率
生成空泛类别。
返回 {"proposals":[{"canonical_name":"","definition":"","source_relation_ids":[]}]}。
现有候选=%s
批次提案=%s""" % (
        max_proposals,
        json.dumps(
            [{"name": item.name, "definition": item.definition} for item in existing],
            ensure_ascii=False,
        ),
        json.dumps(raw_proposals, ensure_ascii=False),
    )
    consolidated = llm.complete(
        kind=f"llm-consolidation:r{round_index}",
        system=SYSTEM,
        user=consolidation_prompt,
    )
    proposals: list[Proposal] = []
    items = consolidated.get("proposals", [])
    if isinstance(items, list):
        for item in items[:max_proposals]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("canonical_name", "")).strip()
            definition = str(item.get("definition", "")).strip()
            ids = item.get("source_relation_ids", [])
            source_ids = tuple(
                sorted(
                    {
                        int(value)
                        for value in ids
                        if isinstance(value, int) and int(value) in valid_ids
                    }
                )
            )
            if name and definition and len(source_ids) >= 3:
                proposals.append(
                    Proposal(
                        name=name,
                        definition=definition,
                        source_relation_ids=source_ids,
                        source=f"llm-r{round_index}",
                    )
                )
    return proposals, {
        "batch_results": batch_results,
        "raw_proposals": raw_proposals,
        "consolidation": consolidated,
    }


def add_proposals(
    *,
    candidates: list[Candidate],
    proposals: Sequence[Proposal],
    arm: str,
    round_index: int,
    max_candidates: int,
) -> list[str]:
    existing_names = {item.name.strip().casefold() for item in candidates}
    added: list[str] = []
    for proposal in proposals:
        if len(candidates) >= max_candidates:
            break
        normalized = proposal.name.strip().casefold()
        if not normalized or normalized in existing_names:
            continue
        candidate_id = f"{arm}-r{round_index}-{len(added) + 1}"
        candidates.append(
            Candidate(
                id=candidate_id,
                name=proposal.name,
                definition=proposal.definition,
                source=proposal.source,
                source_relation_ids=list(proposal.source_relation_ids),
            )
        )
        existing_names.add(normalized)
        added.append(candidate_id)
    return added


def prune_weak_candidates(
    *,
    candidates: list[Candidate],
    mappings: dict[int, str],
    records: dict[int, RelationRecord],
    candidate_ids: set[str],
    min_member_types: int,
    min_member_uses: int,
) -> tuple[list[str], list[int]]:
    members: dict[str, list[int]] = {candidate_id: [] for candidate_id in candidate_ids}
    for relation_id, candidate_id in mappings.items():
        if candidate_id in members:
            members[candidate_id].append(relation_id)
    removed: list[str] = []
    released: list[int] = []
    for candidate_id, relation_ids in members.items():
        uses = sum(records[relation_id].uses for relation_id in relation_ids)
        if len(relation_ids) < min_member_types or uses < min_member_uses:
            removed.append(candidate_id)
            released.extend(relation_ids)
    if removed:
        removed_set = set(removed)
        candidates[:] = [item for item in candidates if item.id not in removed_set]
        for relation_id in released:
            mappings.pop(relation_id, None)
    return sorted(removed), sorted(set(released))


def state_metrics(
    *,
    candidates: Sequence[Candidate],
    mappings: dict[int, str],
    sample_ids: set[int],
    records: dict[int, RelationRecord],
) -> dict[str, Any]:
    mapped_ids = sample_ids.intersection(mappings)
    mapped_claims = sum(records[item].uses for item in mapped_ids)
    total_claims = sum(records[item].uses for item in sample_ids)
    counts: dict[str, int] = {item.id: 0 for item in candidates}
    for relation_id in mapped_ids:
        candidate_id = mappings[relation_id]
        if candidate_id in counts:
            counts[candidate_id] += 1
    return {
        "candidate_count": len(candidates),
        "new_candidate_count": sum(item.source != "seed" for item in candidates),
        "mapped_relation_types": len(mapped_ids),
        "residual_relation_types": len(sample_ids - mapped_ids),
        "type_coverage": len(mapped_ids) / max(len(sample_ids), 1),
        "mapped_claims": mapped_claims,
        "total_claims": total_claims,
        "claim_coverage": mapped_claims / max(total_claims, 1),
        "candidate_member_counts": counts,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_arm(
    *,
    arm: str,
    llm: CachedLLM,
    records: dict[int, RelationRecord],
    sample: Sequence[RelationRecord],
    initial_candidates: Sequence[Candidate],
    initial_mappings: dict[int, str],
    initial_residual: Sequence[int],
    output_dir: Path,
    rounds: int,
    max_candidates: int,
    max_new_per_round: int,
    distance_threshold: float,
    min_cluster_size: int,
    llm_batch_size: int,
    min_member_types: int,
    min_member_uses: int,
    stop_when_round_reduction_below: int | None,
    exhaustive_cluster_extraction: bool,
    max_cluster_size: int,
    split_distance_threshold: float,
    max_cluster_extraction_passes: int,
) -> dict[str, Any]:
    state_path = output_dir / arm / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved_next_round = int(state.get("next_round", 1))
        if state.get("status") == "complete" and saved_next_round > rounds:
            return state
        candidates = [Candidate(**item) for item in state["candidates"]]
        mappings = {int(key): str(value) for key, value in state["mappings"].items()}
        residual_ids = [int(value) for value in state["residual_ids"]]
        round_records = list(state.get("rounds", []))
        start_round = saved_next_round
    else:
        candidates = [Candidate(**asdict(item)) for item in initial_candidates]
        mappings = dict(initial_mappings)
        residual_ids = list(initial_residual)
        round_records = []
        start_round = 1

    sample_ids = {item.id for item in sample}
    next_round = start_round
    stop_reason: str | None = None
    for round_index in range(start_round, rounds + 1):
        residual = [records[item] for item in residual_ids]
        remaining_budget = max_candidates - len(candidates)
        proposal_limit = min(max_new_per_round, max(remaining_budget, 0))
        log(
            f"{arm}: round {round_index} starts with {len(residual)} residuals, "
            f"{len(candidates)} candidates, proposal budget {proposal_limit}"
        )
        if not residual or proposal_limit <= 0:
            stop_reason = "no_residuals" if not residual else "candidate_budget_exhausted"
            break
        if arm == "embedding":
            proposals, generation = synthesize_embedding_proposals(
                llm=llm,
                residual=residual,
                round_index=round_index,
                max_proposals=proposal_limit,
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                min_member_types=min_member_types,
                exhaustive_cluster_extraction=exhaustive_cluster_extraction,
                max_cluster_size=max_cluster_size,
                split_distance_threshold=split_distance_threshold,
                max_cluster_extraction_passes=max_cluster_extraction_passes,
            )
        elif arm == "llm":
            proposals, generation = synthesize_llm_proposals(
                llm=llm,
                residual=residual,
                existing=candidates,
                round_index=round_index,
                max_proposals=proposal_limit,
                batch_size=llm_batch_size,
            )
        else:
            raise ValueError(f"未知实验组: {arm}")
        if arm == "llm":
            proposals, candidate_gate = validate_proposals(
                llm=llm,
                proposals=proposals,
                records=records,
                label=f"{arm}:r{round_index}",
            )
        else:
            candidate_gate = []
        generation["candidate_gate"] = candidate_gate
        added = add_proposals(
            candidates=candidates,
            proposals=proposals,
            arm=arm,
            round_index=round_index,
            max_candidates=max_candidates,
        )
        if not added:
            log(f"{arm}: round {round_index} produced no admissible candidate")
            round_records.append(
                {
                    "round": round_index,
                    "residual_before": len(residual_ids),
                    "generation": generation,
                    "added_candidate_ids": [],
                    "stopped": "no_valid_proposals",
                }
            )
            next_round = round_index + 1
            stop_reason = "no_valid_proposals"
            break
        accepted, still_residual, judgments = map_relations(
            llm=llm,
            relations=residual,
            # The shared seed catalog was judged exactly once before the arms
            # diverged.  Each discovery round may only earn coverage through
            # candidates created in that round; otherwise repeated MiniMax
            # sampling against unchanged seeds confounds the A/B comparison.
            candidates=[item for item in candidates if item.id in set(added)],
            mappings=mappings,
            records=records,
            label=f"{arm}:r{round_index}",
        )
        accepted, full_rejected, full_validation = validate_all_triples(
            llm=llm,
            accepted=accepted,
            candidates=[item for item in candidates if item.id in set(added)],
            records=records,
            label=f"{arm}:r{round_index}",
        )
        still_residual = sorted(set(still_residual).union(full_rejected))
        mappings.update(accepted)
        removed, released = prune_weak_candidates(
            candidates=candidates,
            mappings=mappings,
            records=records,
            candidate_ids=set(added),
            min_member_types=min_member_types,
            min_member_uses=min_member_uses,
        )
        residual_ids = sorted(set(still_residual).union(released))
        round_reduction = len(residual) - len(residual_ids)
        log(
            f"{arm}: round {round_index} added {len(added) - len(removed)} candidates; "
            f"{len(residual_ids)} residuals remain (reduction={round_reduction})"
        )
        round_records.append(
            {
                "round": round_index,
                "residual_before": len(residual),
                "generation": generation,
                "proposals": [asdict(item) for item in proposals],
                "added_candidate_ids": added,
                "removed_candidate_ids": removed,
                "residual_after": len(residual_ids),
                "residual_reduction": round_reduction,
                "judgments": judgments,
                "full_validation_rejected_ids": full_rejected,
                "full_validation_judgments": full_validation,
            }
        )
        state = {
            "status": "running",
            "arm": arm,
            "next_round": round_index + 1,
            "candidates": [asdict(item) for item in candidates],
            "mappings": mappings,
            "residual_ids": residual_ids,
            "rounds": round_records,
            "metrics": state_metrics(
                candidates=candidates,
                mappings=mappings,
                sample_ids=sample_ids,
                records=records,
            ),
        }
        save_json(state_path, state)
        next_round = round_index + 1
        if (
            stop_when_round_reduction_below is not None
            and round_reduction < stop_when_round_reduction_below
        ):
            stop_reason = (
                f"round_reduction_{round_reduction}_below_"
                f"{stop_when_round_reduction_below}"
            )
            log(f"{arm}: stopping because {stop_reason}")
            break
    else:
        stop_reason = "round_limit_reached"

    # Accepted mappings stay fixed. Rejudging the same relation against an
    # unchanged catalog introduced model-sampling noise in the smoke test and
    # would confound the comparison between candidate-generation strategies.
    state = {
        "status": "complete",
        "arm": arm,
        "next_round": next_round,
        "candidates": [asdict(item) for item in candidates],
        "mappings": mappings,
        "residual_ids": residual_ids,
        "rounds": round_records,
        "final_removed_candidate_ids": [],
        "final_judgments": [],
        "stop_reason": stop_reason,
        "metrics": state_metrics(
            candidates=candidates,
            mappings=mappings,
            sample_ids=sample_ids,
            records=records,
        ),
    }
    save_json(state_path, state)
    return state


def build_report(
    *,
    output_dir: Path,
    common: dict[str, Any],
    embedding_state: dict[str, Any],
    llm_state: dict[str, Any],
    llm_metrics: dict[str, Any],
) -> None:
    report = {
        "common": common["metrics"],
        "embedding": embedding_state["metrics"],
        "llm": llm_state["metrics"],
        "runtime": llm_metrics,
    }
    save_json(output_dir / "metrics.json", report)
    lines = [
        "# Relation coarsening experiment",
        "",
        "| metric | common seeds | embedding clusters | LLM induction |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "candidate_count",
        "new_candidate_count",
        "mapped_relation_types",
        "residual_relation_types",
        "type_coverage",
        "mapped_claims",
        "claim_coverage",
    ):
        values = [common["metrics"].get(key, 0), embedding_state["metrics"].get(key, 0), llm_state["metrics"].get(key, 0)]
        rendered = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        lines.append(f"| {key} | {rendered[0]} | {rendered[1]} | {rendered[2]} |")
    lines.extend(
        [
            "",
            "This report contains structural metrics only. Semantic winner selection requires the blinded disagreement audit.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-new-per-round", type=int, default=15)
    parser.add_argument("--distance-threshold", type=float, default=0.075)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="进入 LLM 子集选择的 Embedding 簇最少成员数",
    )
    parser.add_argument(
        "--exhaustive-cluster-extraction",
        action="store_true",
        help="对大簇细分，并在每个簇内反复提取互不重叠的兼容子集",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=20,
        help="穷尽提取模式下，送入单次 LLM 判断的最大簇成员数",
    )
    parser.add_argument(
        "--split-distance-threshold",
        type=float,
        default=0.075,
        help="穷尽提取模式下拆分大簇的更严格 Embedding 距离阈值",
    )
    parser.add_argument(
        "--max-cluster-extraction-passes",
        type=int,
        default=10,
        help="每个簇反复剥离兼容子集的最多 LLM 轮数",
    )
    parser.add_argument("--llm-batch-size", type=int, default=60)
    parser.add_argument("--min-member-types", type=int, default=3)
    parser.add_argument("--min-member-uses", type=int, default=5)
    parser.add_argument(
        "--stop-when-round-reduction-below",
        type=int,
        help="某轮 residual 减少量低于此值时停止后续轮次",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:8111/v1")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--api-key-env", default="API_GATEWAY")
    parser.add_argument(
        "--arms",
        default="embedding,llm",
        help="逗号分隔的实验组：embedding,llm",
    )
    parser.add_argument(
        "--common-from",
        type=Path,
        help="复用另一个实验目录的 common.json，避免重新判断共同种子",
    )
    args = parser.parse_args()
    if args.workers > 8:
        parser.error("--workers 不得超过 8")
    if args.min_cluster_size < 2:
        parser.error("--min-cluster-size 不得小于 2")
    if args.max_cluster_size < args.min_cluster_size:
        parser.error("--max-cluster-size 不得小于 --min-cluster-size")
    if args.max_cluster_extraction_passes < 1:
        parser.error("--max-cluster-extraction-passes 不得小于 1")
    if (
        args.stop_when_round_reduction_below is not None
        and args.stop_when_round_reduction_below < 0
    ):
        parser.error("--stop-when-round-reduction-below 不得小于 0")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"环境变量 {args.api_key_env} 未设置")
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not arms or any(item not in {"embedding", "llm"} for item in arms):
        parser.error("--arms 只能包含 embedding 和 llm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    relations = load_relations(args.db)
    records = {item.id: item for item in relations}
    sample = select_pilot(relations, args.sample_size)
    seeds = seed_candidates(relations)
    llm = CachedLLM(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_concurrency=args.workers,
        cache_path=args.output_dir / "llm_cache.jsonl",
    )
    manifest = {
        "db": str(args.db.resolve()),
        "db_size": args.db.stat().st_size,
        "sample_size": len(sample),
        "sample_relation_ids": [item.id for item in sample],
        "seed_count": len(seeds),
        "seed_ids": [item.id for item in relations if item.uses >= 10],
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "rounds": args.rounds,
        "max_candidates": args.max_candidates,
        "max_new_per_round": args.max_new_per_round,
        "distance_threshold": args.distance_threshold,
        "min_cluster_size": args.min_cluster_size,
        "exhaustive_cluster_extraction": args.exhaustive_cluster_extraction,
        "max_cluster_size": args.max_cluster_size,
        "split_distance_threshold": args.split_distance_threshold,
        "max_cluster_extraction_passes": args.max_cluster_extraction_passes,
        "llm_batch_size": args.llm_batch_size,
        "min_member_types": args.min_member_types,
        "min_member_uses": args.min_member_uses,
        "stop_when_round_reduction_below": args.stop_when_round_reduction_below,
        "full_triple_validation": True,
        "arms": arms,
        "common_from": str(args.common_from.resolve()) if args.common_from else None,
    }
    save_json(args.output_dir / "manifest.json", manifest)
    log(
        f"starting experiment: sample={len(sample)}, seeds={len(seeds)}, "
        f"model={args.model}, workers={args.workers}"
    )

    common_path = args.output_dir / "common.json"
    if common_path.exists():
        common = json.loads(common_path.read_text(encoding="utf-8"))
    elif args.common_from is not None:
        common = json.loads(args.common_from.read_text(encoding="utf-8"))
        expected_ids = {item.id for item in sample}
        actual_ids = set(int(value) for value in common["mappings"]).union(
            int(value) for value in common["residual_ids"]
        )
        if actual_ids != expected_ids:
            parser.error("--common-from 的样本关系集合与当前实验不一致")
        save_json(common_path, common)
    else:
        common_mappings, common_residual, judgments = map_relations(
            llm=llm,
            relations=sample,
            candidates=seeds,
            mappings={},
            records=records,
            label="common-seeds",
        )
        common = {
            "candidates": [asdict(item) for item in seeds],
            "mappings": common_mappings,
            "residual_ids": common_residual,
            "judgments": judgments,
            "metrics": state_metrics(
                candidates=seeds,
                mappings=common_mappings,
                sample_ids={item.id for item in sample},
                records=records,
            ),
        }
        save_json(common_path, common)

    if "full_validation_judgments" not in common:
        common_candidates = [Candidate(**item) for item in common["candidates"]]
        common_mappings = {
            int(key): str(value) for key, value in common["mappings"].items()
        }
        common_mappings, rejected, full_validation = validate_all_triples(
            llm=llm,
            accepted=common_mappings,
            candidates=common_candidates,
            records=records,
            label="common-seeds",
        )
        common["mappings"] = common_mappings
        common["residual_ids"] = sorted(
            set(int(value) for value in common["residual_ids"]).union(rejected)
        )
        common["full_validation_rejected_ids"] = rejected
        common["full_validation_judgments"] = full_validation
        common["metrics"] = state_metrics(
            candidates=common_candidates,
            mappings=common_mappings,
            sample_ids={item.id for item in sample},
            records=records,
        )
        save_json(common_path, common)

    initial_candidates = [Candidate(**item) for item in common["candidates"]]
    initial_mappings = {int(key): str(value) for key, value in common["mappings"].items()}
    initial_residual = [int(value) for value in common["residual_ids"]]
    states: dict[str, dict[str, Any]] = {}
    for arm in arms:
        states[arm] = run_arm(
            arm=arm,
            llm=llm,
            records=records,
            sample=sample,
            initial_candidates=initial_candidates,
            initial_mappings=initial_mappings,
            initial_residual=initial_residual,
            output_dir=args.output_dir,
            rounds=args.rounds,
            max_candidates=args.max_candidates,
            max_new_per_round=args.max_new_per_round,
            distance_threshold=args.distance_threshold,
            min_cluster_size=args.min_cluster_size,
            llm_batch_size=args.llm_batch_size,
            min_member_types=args.min_member_types,
            min_member_uses=args.min_member_uses,
            stop_when_round_reduction_below=args.stop_when_round_reduction_below,
            exhaustive_cluster_extraction=args.exhaustive_cluster_extraction,
            max_cluster_size=args.max_cluster_size,
            split_distance_threshold=args.split_distance_threshold,
            max_cluster_extraction_passes=args.max_cluster_extraction_passes,
        )
    if set(states) == {"embedding", "llm"}:
        build_report(
            output_dir=args.output_dir,
            common=common,
            embedding_state=states["embedding"],
            llm_state=states["llm"],
            llm_metrics=llm.metrics,
        )
    print(
        json.dumps(
            {**{arm: state["metrics"] for arm, state in states.items()}, "runtime": llm.metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
