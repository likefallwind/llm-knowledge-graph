from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from . import ontology, store
from .llm import JSONLLM
from .models import EntityObservation, Resolution


RESOLUTION_PROMPT_VERSION = "entity-identity-ontology-5-semantic-naming"

RESOLUTION_SYSTEM = """你是实体身份裁判，不是知识来源。
只能根据给出的语料观察与候选实体判断身份，禁止补充外部知识。
宁可 uncertain，也不要错误合并。只输出 JSON 对象。"""


def candidate_entities(
    conn: sqlite3.Connection,
    name: str,
    *,
    observation: EntityObservation | None = None,
    limit: int = 5,
    threshold: float = 0.35,
    exclude_id: int | None = None,
) -> list[dict[str, Any]]:
    query = store.normalize_name(name)
    semantic_text = ""
    if observation is not None:
        semantic_text = store.normalize_name(
            " ".join((observation.definition, observation.model_quote))
        ).replace(" ", "")
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
        mentioned_names = [
            value
            for value in compact_names
            if value and semantic_text and value in semantic_text
        ]
        if mentioned_names:
            # Definitions and quotes often reveal the actual referent even when
            # the passage uses a broad or analogical surface name.  This is
            # deterministic candidate recall, not identity evidence by itself.
            score = max(score, 0.72)
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
                    "mentioned_in_observation": bool(mentioned_names),
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

    candidates = candidate_entities(
        conn, observation.name, observation=observation
    )
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """严格判断新观察实际描述的知识对象，并将它与候选 Entity 对齐。

先根据 name、definition、model_quote 和 source_text 识别本观察的实际指代，再判断
它是否与某个候选 Entity 是同一知识对象。原文使用的表面名称可能比实际指代宽泛，
也可能是教学类比中的局部称呼；表面名称不是全局 alias，不妨碍当前 observation
指向语料已经明确表达的候选 Entity。此时可以判 same，但不得把该表面名称放入
accepted_aliases。比如某段把全批量优化简称为「梯度下降」，而定义和比较明确指向
候选「批量梯度下降」，应判 same 并选择该候选，但不能把「梯度下降」注册为它的
全局 alias；把注意力机制的「键」类比为「非自主性提示」时，若观察定义实际描述键，
应指向候选「键」，同样不能把类比名称注册为全局 alias。

这是 identity 判断，不是相关性、相似性或归类判断。same 的门槛很高：只有在不同
语境中实际指代同一对象，才能判为 same。用途相近、定义相关、共同出现、
一个实现另一个、一个是另一个的变体/子类/实例/配置/角色/数据集版本，或者当前段落把
二者对应起来，都不足以构成 same。括号解释、教学类比、角色映射或“在这里称为”只证明
当前 passage 的局部对应；除非语料同时证明名称在其他语境也指向同一对象，否则不能提升为
全局 alias。以上只是原则示例，不是要枚举所有情况。

判定 same 前必须做两个检查：
1. 区分检查：语料是否可能同时谈论二者，并比较、区分或在二者之间建立有方向的关系？
   如果可以，它们就不是同一个对象。
2. 合并反事实：合并后，下面任一名称、定义或 source_text 是否会变假、丢失限定条件，
   或把局部语境中的对应关系扩大为全局同义？如果会，就不能判 same。

decision 必须与上述分析一致。基础概念/算法族与带有限定词的变体若实际对象边界不同，
应保留为不同实体；但若当前 passage 的定义、性质和比较已经明确说明宽泛表面名称实际
指向某个已有的具体候选，则 observation 可以对齐该候选，同时不注册表面名称为 alias。

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

若判 new，canonical_name 必须准确表达 definition 中的实际语义。观察名过宽、同名但
异义或遗漏了造成身份差异的限定时，必须使用 source_text、model_quote 或 definition 已
明确支持的最小限定来消歧，例如「数学卷积」「一元语法模型」「机器学习注意力机制」。
不得引入语料没有提供的新知识。若无法给出不与已有名称冲突的有据名称，应返回 uncertain，
不能创建另一个无法区分的同名 Entity。

同名也不构成 same：教材章节、目录条目等 resource 与其讲述的同名算法、模型或
概念是不同知识对象。比如「15.1 玻尔兹曼机」这一节不能与「玻尔兹曼机」算法合并。
canonical_name 必须保留区分知识对象身份所必需的信息。若观察是章节、节、附录等
resource，不能删去章节编号或载体限定后变成同名知识内容；例如应保留
「15.1 玻尔兹曼机」，不能规范成「玻尔兹曼机」。
返回：
{
  "decision": "same | new | uncertain",
  "candidate_id": 仅 same 时填写候选 id，否则为 null,
  "canonical_name": "实际语义的规范名称；所有 decision 都填写，new/uncertain 必须有据且可区分",
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
                aliases = accepted_aliases
                if confirmation[1] == "global_name":
                    aliases = (observation.name, *aliases)
                for alias in aliases:
                    store.add_alias(conn, selected, alias)
                return Resolution(
                    entity_id=selected,
                    outcome="same",
                    reason=confirmation[2] or reason,
                    candidates=candidate_ids,
                )
            decision = "uncertain"
            accepted_aliases = ()
            reason = confirmation[2] or "same 未通过独立身份否证确认"
        else:
            decision = "uncertain"
            accepted_aliases = ()
            reason = reason or "same 返回了非法 candidate_id"

    canonical = _canonical_name(payload.get("canonical_name"))
    if decision not in {"new", "uncertain"}:
        decision = "uncertain"
        reason = reason or "resolver 返回了非法 decision"
    # A distinct judgment with a colliding name is an invalid semantic naming
    # result. Never silently turn it into the existing Entity while recording
    # `new`/`uncertain`; only a `new` result gets one bounded naming-only retry.
    if not canonical:
        raise ValueError(f"{decision} entity 缺少有效 canonical_name")
    collisions = store.exact_entity_ids(conn, canonical)
    if collisions:
        if decision == "new":
            canonical = _retry_colliding_new_name(
                conn,
                llm,
                observation=observation,
                proposed_name=canonical,
                collision_ids=collisions,
                candidates=candidates,
                identity_reason=reason,
            )
        else:
            raw_name = _canonical_name(observation.name)
            if raw_name and not store.exact_entity_ids(conn, raw_name):
                canonical = raw_name
            else:
                raise ValueError(
                    f"{decision} entity canonical_name 与已有 Entity 冲突: "
                    f"{canonical!r} -> {collisions}"
                )
    reviewed_observation = replace(observation, aliases=accepted_aliases)
    entity_id = store.create_entity(
        conn,
        reviewed_observation,
        canonical_name=canonical,
        # A surface name that already identifies another Entity is contextual
        # evidence for this observation, not a safe global alias for the new
        # semantically disambiguated Entity.
        include_observation_name_alias=not bool(
            store.exact_entity_ids(conn, observation.name)
        ),
    )
    return Resolution(
        entity_id=entity_id,
        outcome=decision,
        reason=reason,
        candidates=candidate_ids,
    )


def _retry_colliding_new_name(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    *,
    observation: EntityObservation,
    proposed_name: str,
    collision_ids: list[int],
    candidates: list[dict[str, Any]],
    identity_reason: str,
) -> str:
    """Retry semantic naming only after a grounded `new` decision collides."""
    conflicts = [_entity_context(conn, entity_id) for entity_id in collision_ids]

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        canonical = _canonical_name(payload.get("canonical_name"))
        if not canonical:
            raise ValueError("语义重命名缺少有效 canonical_name")
        collisions = store.exact_entity_ids(conn, canonical)
        if collisions:
            raise ValueError(
                "语义重命名仍与已有 Entity 冲突: "
                f"{canonical!r} -> {collisions}"
            )
        return payload

    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """第一次身份裁决已经确定新观察是独立知识对象（decision=new），但给出的
canonical_name 与已有 Entity 冲突。现在只修正语义命名，不得重新判断身份，不得合并
实体，也不得添加或修改 alias。

名称必须准确表达 definition 中的实际对象，并使用 source_text、model_quote 或
definition 已明确支持的最小限定来说明它与冲突实体的身份边界。不要只重复观察原名或
冲突名称；不要引入语料没有提供的知识。若对象是某种结构、模型、语言单元、实现函数、
算法变体或其他带限定的对象，应把造成身份差异的限定保留在 canonical_name 中。

只返回：
{
  "canonical_name": "不与已有 Entity 冲突、且有语料依据的语义规范名",
  "naming_basis": "该限定由哪段输入支持"
}

新观察：
%s

第一次裁决理由：
%s

第一次冲突名称：
%s

直接冲突实体：
%s

第一次裁决看到的全部候选：
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
            identity_reason,
            proposed_name,
            json.dumps(conflicts, ensure_ascii=False),
            json.dumps(candidates, ensure_ascii=False),
        ),
        validate=validate,
    )
    return _canonical_name(payload.get("canonical_name"))


def _confirm_same(
    llm: JSONLLM,
    *,
    observation: EntityObservation,
    candidate: dict[str, Any],
    proposed_reason: str,
) -> tuple[bool, str, str]:
    """Adversarially verify a tentative merge before mutating graph identity."""
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """第一次裁决拟将下面两个实体合并。现在进行独立的身份否证检查。

你的任务不是支持第一次答案，而是主动寻找当前 observation 的实际指代与候选仍是
两个知识对象的证据。
候选 aliases 不是身份证据；当前段落中的括号解释、教学类比、角色对应、宽泛用词，
也不能证明跨语境全局同义。只要存在合理的对象边界差异，或现有语料不足以排除实际
指代相同，就必须 reject_same。只有缩写/全称、翻译、拼写变体等名称可跨语境安全互换，
且合并不会把局部关系提升成全局 alias 时，identity_scope 才能是 global_name。

必须分别判断实体指代与名称范围。不能因为
当前定义写成“A 被称为 B”或“A（B）”，就用这句话循环证明 A/B 是全局同义词；如果
离开该教学段落后，一个名称仍是技术角色、另一个仍是被类比或被表示的对象，就应拒绝。
例如“值（感官输入）”和“查询（自主性提示）”必须 reject_same，因为这是注意力机制
角色与认知类比对象的局部映射，不是名称层面的全局同义。该例用于说明一般原则，不能只
匹配字面词语作答。

如果表面名称只在当前 passage 指向候选，但 definition、model_quote 和 source_text 已经
明确当前 observation 的实际对象就是候选，可以 confirmed_same，并把 identity_scope
设为 passage_referent；这不会批准表面名称成为全局 alias。只有名称本身可跨语境安全
互换时才设为 global_name。若实际对象仍不同或证据不足，必须 reject_same。

审查对象是 observation 的实际指代，不是 observation.name 这个字符串本身。例如观察
名为「非自主性提示」，但 definition 的主语和知识内容实际描述注意力机制的「键」，原文
只是把键类比为非自主性提示，而拟合并候选正是「键」，则应 confirmed_same 且设为
passage_referent；「非自主性提示」与「键」不是全局同义，只决定不能设为 global_name，
不能反过来否定 observation 实际指向候选「键」。相反，若拟合并候选是心理学概念
「非自主性提示」，才应因技术角色与类比对象不同而 reject_same。

返回：
{
  "verdict": "confirmed_same | reject_same",
  "identity_scope": "global_name | passage_referent | uncertain",
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
    confirmed = verdict == "confirmed_same" and identity_scope in {
        "global_name",
        "passage_referent",
    }
    return confirmed, identity_scope, reason


def _canonical_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    canonical = value.strip()
    if store.normalize_name(canonical) in {"", "none", "null", "nil", "n/a"}:
        return ""
    return canonical


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
