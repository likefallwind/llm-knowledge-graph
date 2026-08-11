from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from . import store
from .llm import JSONLLM
from .models import EntityObservation, Resolution


RESOLUTION_PROMPT_VERSION = "entity-identity-ontology-10-knowledge-aliases-top10"

IDENTITY_KNOWLEDGE_POLICY = """你可以使用可靠的通用知识判断术语的通常含义、同义关系、
翻译、缩写，以及概念、实现、子类、实例之间的身份边界。原文和语境用于确定当前名称
实际指向哪个义项，辅助你判断，但不得仅因两个片段描述了不同应用场景，就判断为不同
Entity。要基于知识做最终判断。

输入中的 definition 可能只是某次观察或若干语料的局部概括，可能不完整或带有应用场景；
它是义项线索，不是预先成立的身份边界。原文没有重新定义一个通用术语，也不构成 new 或
uncertain 的理由。"""

RESOLUTION_SYSTEM = f"""你是实体身份裁判，不是知识内容抽取器。
{IDENTITY_KNOWLEDGE_POLICY}
只有概念义项或对象边界确实无法判断时才返回 uncertain；不要因为原文证据不完整而机械
拆分实体，也不要因为名称相似而错误合并。只输出 JSON 对象。"""


def candidate_entities(
    conn: sqlite3.Connection,
    name: str,
    *,
    observation: EntityObservation | None = None,
    limit: int = 10,
    threshold: float = 0.35,
    exclude_id: int | None = None,
) -> list[dict[str, Any]]:
    query_names = [name]
    if observation is not None:
        query_names.extend(observation.aliases)
    queries = [
        store.normalize_name(value)
        for value in dict.fromkeys(query_names)
        if value.strip()
    ]
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
            for query in queries
            for value in names
        )
        compact_queries = [query.replace(" ", "") for query in queries]
        compact_names = [store.normalize_name(value).replace(" ", "") for value in names]
        if any(
            query in value or value in query
            for query in compact_queries
            for value in compact_names
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
    candidates = candidate_entities(
        conn, observation.name, observation=observation
    )
    payload = llm.complete_json(
        RESOLUTION_SYSTEM,
        """严格判断新观察实际描述的知识对象，并将它与候选 Entity 对齐。

第一步结合通用知识与 name、definition、model_quote、source_text 判断本观察实际指向的
通常概念或具体对象；原文主要用于义项消歧。第二步再判断它与候选是否具有同一身份。

这是 identity 判断，不是相关性、相似性或归类判断：
- same：跨语境仍是同一概念或同一对象。定义详略、讲解角度、属性、公式、用途、所属模型
  或应用场景不同，都不会自动产生新 Entity。通用概念先在 GoogLeNet、LSTM、CNN、RNN、
  BERT 等具体场景中出现，后来又以一般形式出现，通常仍是同一个概念。
- new：可靠通用知识或当前义项表明确实是不同对象，例如概念与其实现、类与概念、基础概念
  与子类/变体/实例/配置、作品与数据集、技术角色与教学类比对象。
- uncertain：综合通用知识和语境后仍无法确定当前义项或身份边界。不得仅因原文没有给出
  完整定义而返回 uncertain。

判定前检查：
1. 两组描述能否作为同一标准对象在不同场景中的性质同时成立？能则场景差异不支持 new。
2. 二者是否可以在同一知识体系中同时出现并被比较、实现、包含或建立其他有方向关系？
   若是，通常是两个对象而不是 same。
3. 合并是否会抹掉真正属于身份的限定，或把局部角色映射扩大成全局同义？若会则不能 same。

候选 aliases 以及新观察 aliases 都只是上游模型提供的待判断线索，不是已经证明的事实，
不得以“候选已有此 alias”为理由循环证明 same。只有缩写/全称、翻译、拼写格式变体、
正式名/简称等确实指向同一对象的名称，才能放入 accepted_aliases。实现名、类名、实例名、
角色映射和只在当前语境成立的称呼不得作为全局 alias。

你还可以基于可靠通用知识在 knowledge_aliases 中补充当前语料没有直接列出的标准翻译、
英文全称、通行缩写、正式名/简称或纯拼写格式变体，最多 5 个。这里只能放跨语境仍唯一
指向同一对象的标准名称；普通近义词、上位/下位概念、相关对象、实现/API/实例、局部角色、
教学类比和不确定名称一律不得加入。没有高度可靠的补充名称时返回空数组。

候选的 type_profile 是历次局部类型观察，只是辅助线索，不是身份白名单；类型不同不能
覆盖对术语通常含义和实际对象边界的判断。

例如「随机梯度下降」与「小批量随机梯度下降」可以在同一段中被比较，是不同算法；即使
某段把读取小批量样本的过程简称为“随机梯度下降”，也只是该段的宽泛用词，不构成身份；
「Vocab 类」实现「词表」而不是词表概念本身；作品与由作品构成的数据集也不是同一对象；
教学类比中写作“值（感官输入）”或“查询（自主性提示）”，只是用熟悉事物解释注意力角色，
不能据此把值与感官输入、查询与自主性提示注册成可跨语境互换的全局同义词。
相反，SGD 与 stochastic gradient descent 在同一对象边界下只是缩写与全称。

若判 new，canonical_name 应使用可靠通用知识中的标准名称，并结合当前语境保留真正造成
身份差异的最小限定，例如「数学卷积」「一元语法模型」「机器学习注意力机制」。不得为了
避开名称冲突而虚构并不存在的子类、版本或限定。若无法给出可区分的规范名称，应返回
uncertain，不能创建另一个无法区分的同名 Entity。

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
  "knowledge_aliases": ["可靠通用知识确认的标准翻译、全称、缩写或名称变体，最多5个"],
  "identity_basis": "说明基于通用知识与当前义项得出的身份结论",
  "distinguishing_test": "说明是否存在真正的对象边界，而不是场景或定义详略差异",
  "reason": "简短理由"
}

新观察：
%s

候选：
%s"""
        % (
            json.dumps(
                {
                    "name": observation.name,
                    "aliases": observation.aliases,
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
    """Retry semantic naming only after a knowledge-based `new` decision collides."""
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

使用可靠通用知识中的标准名称，并结合当前语境保留真正造成身份差异的最小限定。不要只
重复观察原名或冲突名称，也不要为了避开冲突而虚构不存在的子类、版本或限定。若对象是
某种结构、模型、语言单元、实现函数、算法变体或其他带限定的对象，应把身份限定保留在
canonical_name 中；应用场景本身不是限定。

只返回：
{
  "canonical_name": "不与已有 Entity 冲突的标准规范名",
  "naming_basis": "该名称对应的知识对象边界"
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
                    "aliases": observation.aliases,
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
        """第一次裁决拟将下面的 observation 关联到候选 Entity。现在独立复核身份。

使用可靠通用知识判断二者是否真是同一概念或对象，原文只用于确认当前 observation 的
具体义项。主动寻找最强的真实身份冲突，但不能把定义不完整、原文没有重新定义、讲解角度
不同或应用场景不同当成冲突。只有通用知识或语境表明二者是概念/实现、上下位、变体、
实例、配置、资源/内容、技术角色/类比对象等不同对象时，才 reject_same。

必须分别判断“实际指代”和“名称能否成为全局 alias”：
- 实际指代相同，即使表面名称只是当前 passage 的宽泛用词或角色称呼，也可
  confirmed_same，并把 identity_scope 设为 passage_referent；此时不批准全局 alias。
- 名称本身是可跨语境安全互换的缩写、全称、翻译、拼写或正式名变体时，设为 global_name。
- 实际对象不同才 reject_same。候选 aliases 以及“A 被称为 B”“A（B）”等局部文字不能
  循环证明身份。

例如，注意力讲解把技术角色「值」类比为「感官输入」时，值与感官输入是不同对象；但若
观察名是一个局部称呼、其实际指代明确为候选「键」，则 observation 可以关联到键而不把
局部称呼注册成键的全局 alias。

返回：
{
  "verdict": "confirmed_same | reject_same",
  "identity_scope": "global_name | passage_referent | uncertain",
  "strongest_identity_conflict": "最强的真实身份边界冲突；若确认相同则说明为何场景差异不构成冲突",
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
                    "aliases": observation.aliases,
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
    """Return source-proposed and resolver-generated standard name variants."""
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
    knowledge = payload.get("knowledge_aliases", [])
    if isinstance(knowledge, list):
        for value in knowledge[:5]:
            if not isinstance(value, str) or not value.strip():
                continue
            alias = value.strip()
            normalized = store.normalize_name(alias)
            if normalized not in seen:
                accepted.append(alias)
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
            """使用可靠通用知识判断两个已有 Entity 是否指向同一个知识对象。原文、
definition、type profile 和 Evidence 只用于确认各自义项；不得仅因定义详略、属性、用途
或应用场景不同而拆分。概念与实现、上下位、变体、实例、配置、资源与内容仍是不同对象。
只有义项或身份边界确实无法判断时才 uncertain。

返回 {"decision":"same|new|uncertain","canonical_name":"若 same 给出标准规范名称","reason":"..."}。
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
