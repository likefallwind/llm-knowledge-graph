from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from . import ontology, store
from .llm import JSONLLM
from .models import EntityObservation, Resolution


RESOLUTION_PROMPT_VERSION = "entity-identity-ontology-4-strict-identity"

RESOLUTION_SYSTEM = """你是实体身份裁判，不是知识来源。
只能根据给出的语料观察与候选实体判断身份，禁止补充外部知识。
宁可 uncertain，也不要错误合并。只输出 JSON 对象。"""


def candidate_entities(
    conn: sqlite3.Connection,
    name: str,
    *,
    limit: int = 5,
    threshold: float = 0.35,
    exclude_id: int | None = None,
) -> list[dict[str, Any]]:
    query = store.normalize_name(name)
    candidates: list[dict[str, Any]] = []
    for row in store.list_entities(conn):
        entity_id = int(row["id"])
        if entity_id == exclude_id:
            continue
        names = [str(row["canonical_name"]), *store.aliases_for(conn, entity_id)]
        score = max(
            SequenceMatcher(None, query, store.normalize_name(value)).ratio()
            for value in names
        )
        compact_query = query.replace(" ", "")
        compact_names = [store.normalize_name(value).replace(" ", "") for value in names]
        if any(
            compact_query in value or value in compact_query for value in compact_names
        ):
            score = max(score, 0.55)
        if score >= threshold:
            candidates.append(
                {
                    "id": entity_id,
                    "canonical_name": str(row["canonical_name"]),
                    "aliases": names[1:],
                    "definition": str(row["definition"]),
                    "type_profile": store.type_profile(conn, entity_id),
                    "evidence": store.evidence_for_entity(conn, entity_id),
                    "score": round(score, 4),
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["id"])))
    return candidates[:limit]


def resolve_observation(
    conn: sqlite3.Connection, llm: JSONLLM, observation: EntityObservation
) -> Resolution:
    exact = store.exact_entity_ids(conn, observation.name)
    if len(exact) == 1:
        entity_id = exact[0]
        entity = store.get_entity(conn, entity_id)
        type_profile = store.type_profile(conn, entity_id)
        # Only an exact canonical name is safe enough to skip identity review.
        # Alias rows in an older graph may predate strict review, so matching an
        # alias must still go through the LLM rather than circularly proving it.
        canonical_matches = bool(
            entity
            and store.normalize_name(str(entity["canonical_name"]))
            == store.normalize_name(observation.name)
        )
        if canonical_matches and any(
            item["entity_type"] == observation.entity_type
            for item in type_profile
        ):
            return Resolution(
                entity_id=entity_id,
                outcome="same",
                reason=(
                    "exact canonical name with compatible observed type; "
                    "unreviewed alias suggestions were not promoted"
                ),
            )

    candidates = candidate_entities(conn, observation.name)
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """严格判断新观察与候选是否指向完全相同的知识对象。

这是 identity 判断，不是相关性、相似性或归类判断。same 的门槛很高：只有在不同
语境中互换两个名称仍不会改变对象边界，才能判为 same。用途相近、定义相关、共同出现、
一个实现另一个、一个是另一个的变体/子类/实例/配置/角色/数据集版本，或者当前段落把
二者对应起来，都不足以构成 same。括号解释、教学类比、角色映射或“在这里称为”只证明
当前 passage 的局部对应；除非语料同时证明名称在其他语境也指向同一对象，否则不能提升为
全局 alias。以上只是原则示例，不是要枚举所有情况。

判定 same 前必须做两个检查：
1. 区分检查：语料是否可能同时谈论二者，并比较、区分或在二者之间建立有方向的关系？
   如果可以，它们就不是同一个对象。
2. 合并反事实：合并后，下面任一名称、定义或 source_text 是否会变假、丢失限定条件，
   或把局部语境中的对应关系扩大为全局同义？如果会，就不能判 same。

decision 必须与上述分析一致：只要 identity_basis 或 distinguishing_test 承认二者在严格语义下
可以区分、只是“当前语境中”把一个名称用于另一个对象，decision 就只能是 new 或 uncertain，
绝不能是 same。基础概念/算法族名称与带有限定词的变体名称应保留为不同实体；即使当前
passage 用基础名称描述的操作恰好采用该变体，也不能因此注册全局 alias。

候选 aliases 以及新观察 aliases 都只是上游模型提供的待判断线索，不是已经证明的事实，
不得以“候选已有此 alias”为理由循环证明 same。只有缩写/全称、翻译、拼写格式变体、
正式名/简称等确实指向同一对象的名称，才能放入 accepted_aliases。实现名、类名、实例名、
角色映射和只在当前语境成立的称呼不得作为全局 alias。

候选的 type_profile 是历次观察类型的汇总而非单一白名单；类型不一致也许可以解释，
但它是身份边界证据，不能忽略。若无法用现有名称、定义和原文排除身份差异，应返回
uncertain；宁可暂时保留重复实体，也不要错误合并。

例如「随机梯度下降」与「小批量随机梯度下降」可以在同一段中被比较，是不同算法；即使
某段把读取小批量样本的过程简称为“随机梯度下降”，也只是该段的宽泛用词，不构成身份；
「Vocab 类」实现「词表」而不是词表概念本身；作品与由作品构成的数据集也不是同一对象；
教学类比中写作“值（感官输入）”或“查询（自主性提示）”，只是用熟悉事物解释注意力角色，
不能据此把值与感官输入、查询与自主性提示注册成可跨语境互换的全局同义词。
相反，SGD 与 stochastic gradient descent 在同一对象边界下只是缩写与全称。

同名也不构成 same：教材章节、目录条目等 resource 与其讲述的同名算法、模型或
概念是不同知识对象。比如「15.1 玻尔兹曼机」这一节不能与「玻尔兹曼机」算法合并。
canonical_name 必须保留区分知识对象身份所必需的信息。若观察是章节、节、附录等
resource，不能删去章节编号或载体限定后变成同名知识内容；例如应保留
「15.1 玻尔兹曼机」，不能规范成「玻尔兹曼机」。
返回：
{
  "decision": "same | new | uncertain",
  "candidate_id": 仅 same 时填写候选 id，否则为 null,
  "canonical_name": new/uncertain 时给出规范正式名称；只能规范化观察名，不能创造新知识,
  "accepted_aliases": ["仅从新观察 aliases 中选择确认是全局同一名称的字符串"],
  "identity_basis": "same 时说明为何是同一对象；new/uncertain 时说明身份边界或缺失证据",
  "distinguishing_test": "说明二者能否共现并被比较/建立关系，以及合并反事实是否安全",
  "reason": "简短理由"
}

type_labels 是开放类别词。以下旧标签仅用于解释历史观察，不是白名单：%s

新观察：
%s

候选：
%s"""
        % (
            ontology.entity_type_summary(),
            json.dumps(
                {
                    "name": observation.name,
                    "definition": observation.definition,
                    "type_labels": observation.type_labels
                    or ((observation.entity_type,) if observation.entity_type else ()),
                    "model_quote": observation.model_quote,
                    "source_text": observation.source_text,
                    "passage_ids": observation.passage_ids,
                },
                ensure_ascii=False,
            ),
            json.dumps(candidates, ensure_ascii=False),
        ),
    )
    decision = str(payload.get("decision", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    accepted_aliases = _accepted_aliases(payload, observation)
    candidate_ids = tuple(int(item["id"]) for item in candidates)
    if decision == "same":
        try:
            selected = int(payload.get("candidate_id"))
        except (TypeError, ValueError):
            selected = -1
        if selected in candidate_ids:
            candidate = next(item for item in candidates if int(item["id"]) == selected)
            confirmation = _confirm_same(
                llm,
                observation=observation,
                candidate=candidate,
                proposed_reason=reason,
            )
            if confirmation[0]:
                for alias in (observation.name, *accepted_aliases):
                    store.add_alias(conn, selected, alias)
                return Resolution(
                    entity_id=selected,
                    outcome="same",
                    reason=confirmation[1] or reason,
                    candidates=candidate_ids,
                )
            decision = "uncertain"
            accepted_aliases = ()
            reason = confirmation[1] or "same 未通过独立身份否证确认"
        else:
            decision = "uncertain"
            accepted_aliases = ()
            reason = reason or "same 返回了非法 candidate_id"

    canonical = str(payload.get("canonical_name", "")).strip()
    if decision not in {"new", "uncertain"}:
        decision = "uncertain"
        reason = reason or "resolver 返回了非法 decision"
    # A `new` or `uncertain` answer must never collapse into an existing row
    # merely because the model suggested an already-used canonical spelling.
    if canonical and store.exact_entity_ids(conn, canonical):
        canonical = observation.name
    reviewed_observation = replace(observation, aliases=accepted_aliases)
    entity_id = store.create_entity(
        conn, reviewed_observation, canonical_name=canonical or observation.name
    )
    return Resolution(
        entity_id=entity_id,
        outcome=decision,
        reason=reason,
        candidates=candidate_ids,
    )


def _confirm_same(
    llm: JSONLLM,
    *,
    observation: EntityObservation,
    candidate: dict[str, Any],
    proposed_reason: str,
) -> tuple[bool, str]:
    """Adversarially verify a tentative merge before mutating graph identity."""
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """第一次裁决拟将下面两个实体合并。现在进行独立的身份否证检查。

你的任务不是支持第一次答案，而是主动寻找二者仍可作为两个知识对象存在的证据。
候选 aliases 不是身份证据；当前段落中的括号解释、教学类比、角色对应、宽泛用词，
也不能证明跨语境全局同义。只要存在合理的对象边界差异，或现有语料不足以排除差异，
就必须 reject_same。只有缩写/全称、翻译、拼写变体等在对象边界内可安全互换，且合并
不会把局部关系提升成全局 alias 时，才能 confirmed_same。

必须判断“两个名称之间的身份关系本身”是否独立于正在审查的 passage 成立。不能因为
当前定义写成“A 被称为 B”或“A（B）”，就用这句话循环证明 A/B 是全局同义词；如果
离开该教学段落后，一个名称仍是技术角色、另一个仍是被类比或被表示的对象，就应拒绝。
例如“值（感官输入）”和“查询（自主性提示）”必须 reject_same，因为这是注意力机制
角色与认知类比对象的局部映射，不是名称层面的全局同义。该例用于说明一般原则，不能只
匹配字面词语作答。

特别检查第一次理由是否自相矛盾：如果理由承认二者理论上可区分、一个是另一个的限定
变体/实现/角色，或只是“在此处”这样称呼，就必须 reject_same。

返回：
{
  "verdict": "confirmed_same | reject_same",
  "identity_scope": "global_name | passage_local | uncertain",
  "strongest_identity_conflict": "最强的身份边界冲突；若确认相同则说明为何不存在冲突",
  "reason": "简短结论"
}

新观察：
%s

拟合并候选：
%s

第一次裁决理由：
%s"""
        % (
            json.dumps(
                {
                    "name": observation.name,
                    "definition": observation.definition,
                    "type_labels": observation.type_labels
                    or ((observation.entity_type,) if observation.entity_type else ()),
                    "model_quote": observation.model_quote,
                    "source_text": observation.source_text,
                    "passage_ids": observation.passage_ids,
                },
                ensure_ascii=False,
            ),
            json.dumps(candidate, ensure_ascii=False),
            proposed_reason,
        ),
    )
    verdict = str(payload.get("verdict", "")).strip().lower()
    identity_scope = str(payload.get("identity_scope", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    return verdict == "confirmed_same" and identity_scope == "global_name", reason


def _accepted_aliases(
    payload: dict[str, Any], observation: EntityObservation
) -> tuple[str, ...]:
    """Keep only resolver-reviewed aliases that were proposed by extraction."""
    raw = payload.get("accepted_aliases", [])
    if not isinstance(raw, list):
        return ()
    proposed = {
        store.normalize_name(alias): alias
        for alias in observation.aliases
        if alias.strip()
    }
    accepted: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            continue
        normalized = store.normalize_name(value)
        if normalized in proposed and normalized not in seen:
            accepted.append(proposed[normalized])
            seen.add(normalized)
    return tuple(accepted)


def reconcile(
    conn: sqlite3.Connection, llm: JSONLLM, *, limit: int = 20
) -> dict[str, Any]:
    """Revisit similar existing entities; merge only explicit `same` decisions."""
    pairs: dict[
        tuple[int, int], tuple[dict[str, Any], dict[str, Any], float]
    ] = {}
    entities = [_entity_context(conn, int(row["id"])) for row in store.list_entities(conn)]
    by_id = {int(item["id"]): item for item in entities}
    for entity in entities:
        source_id = int(entity["id"])
        for candidate in candidate_entities(
            conn,
            str(entity["canonical_name"]),
            exclude_id=source_id,
            threshold=0.55,
        ):
            pair = tuple(sorted((source_id, int(candidate["id"]))))
            score = float(candidate["score"])
            if pair not in pairs or score > pairs[pair][2]:
                pairs[pair] = (by_id[pair[0]], by_id[pair[1]], score)
    examined = 0
    merged: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    distinct: list[dict[str, Any]] = []
    ranked = sorted(
        pairs.items(),
        key=lambda item: (-item[1][2], item[0][0], item[0][1]),
    )
    for (left_id, right_id), (left, right, score) in ranked[:limit]:
        if not store.get_entity(conn, left_id) or not store.get_entity(conn, right_id):
            continue
        examined += 1
        payload = llm.complete_json(
            RESOLUTION_SYSTEM,
            """两个已有实体是否指向同一个知识对象？
返回 {"decision":"same|new|uncertain","canonical_name":"若 same 给出更规范名称","reason":"..."}。
实体 A：%s
实体 B：%s"""
            % (
                json.dumps(left, ensure_ascii=False),
                json.dumps(right, ensure_ascii=False),
            ),
        )
        decision = str(payload.get("decision", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if decision == "same":
            target_id, source_id = min(left_id, right_id), max(left_id, right_id)
            store.merge_entities(conn, source_id, target_id)
            store.set_canonical_name(
                conn, target_id, str(payload.get("canonical_name", ""))
            )
            merged.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "score": score,
                    "reason": reason,
                }
            )
        elif decision == "uncertain":
            uncertain.append(
                {
                    "ids": [left_id, right_id],
                    "score": score,
                    "reason": reason,
                }
            )
        else:
            distinct.append(
                {
                    "ids": [left_id, right_id],
                    "score": score,
                    "reason": reason,
                }
            )
    from . import observations

    replayed = observations.resolve_and_materialize_cached(conn)
    return {
        "examined": examined,
        "merged": merged,
        "uncertain": uncertain,
        "distinct": distinct,
        "replayed": replayed,
    }


def _entity_context(
    conn: sqlite3.Connection, entity_id: int
) -> dict[str, Any]:
    row = store.get_entity(conn, entity_id)
    if not row:
        raise ValueError(f"实体不存在: {entity_id}")
    return {
        "id": entity_id,
        "canonical_name": str(row["canonical_name"]),
        "aliases": store.aliases_for(conn, entity_id),
        "definition": str(row["definition"]),
        "type_profile": store.type_profile(conn, entity_id),
        "evidence": store.evidence_for_entity(conn, entity_id),
    }
