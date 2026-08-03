from __future__ import annotations

import re
from typing import Any

from . import ontology
from .llm import JSONLLM
from .models import (
    POLARITIES,
    ClaimObservation,
    EntityObservation,
    ExtractionBatch,
    SourcePassage,
)


ENTITY_PROMPT_VERSION = "open-entities-section-1"
RELATION_PROMPT_VERSION = "open-relations-section-1"
EXTRACTION_PROMPT_VERSION = (
    f"{ENTITY_PROMPT_VERSION}+{RELATION_PROMPT_VERSION}"
)
PASSAGE_VERSION = "source-passages-2"

SYSTEM_PROMPT = """你是语料约束的知识抽取器。
你只能理解用户给出的原文，禁止用模型参数记忆补充原文没有表达的知识。
没有有效原文段落依据的对象或关系必须省略。只输出 JSON 对象。"""

ENTITY_PROMPT = """从下面的语料片段抽取 EntityObservation。

Entity 必须是在本片段中有稳定名称、可复指，并有实质性定义或知识含义的对象。
类型标签是开放的：使用原文语境中简洁、可复用的类别词，可为空，不得为了满足
预设词表而扭曲实体。每个实体最多给出 3 个 type_labels。

规则：
1. evidence.passage_ids 必须选择片段中真实存在的段落 ID，最多 3 个。
2. evidence.quote 是你认为最关键的引文。应尽量忠实引用，但允许轻微省略或改写。
3. aliases 只列出本片段表达过的别名。
4. 这一阶段不要输出关系。
5. 最多输出 {max_entities} 个实体。

输出：
{{
  "entities": [
    {{
      "name": "原文中的名称",
      "definition": "仅依据原文的定义或知识含义",
      "type_labels": ["原文语境中的开放类别词"],
      "aliases": ["原文中实际出现的别名"],
      "evidence": {{
        "passage_ids": ["P000001"],
        "quote": "最关键的原文引文，可轻微省略"
      }}
    }}
  ]
}}

语料片段位置：{location}
---
{text}
---"""

RELATION_PROMPT = """从下面的原始语料中抽取开放式 subject-predicate-object 关系。

实体已经由上一阶段识别。subject 和 object 必须使用实体清单中的完整名称；不要重新
抽实体。predicate 使用原文关系的简洁、可复用表达，不受预设关系词表限制。

规则：
1. 只能依据原始 Passage，目录和摘要只提供定位上下文，不能单独证明关系。
2. evidence.passage_ids 必须来自下面真实存在的 Passage，最多 3 个。
3. 共现、章节相邻、主题相似或模型常识不能构成关系。
4. 不确定时省略；最多输出 {max_claims} 条。

输出：
{{"relations":[{{
  "subject":"实体清单中的完整名称",
  "predicate":"开放关系谓词",
  "object":"实体清单中的完整名称",
  "stance":"support|oppose",
  "evidence":{{"passage_ids":["P000001"],"quote":"关键引文"}}
}}]}}

目录上下文：{section_context}
实体清单：{entities}
原始语料：
---
{text}
---"""


def extract(
    llm: JSONLLM,
    text: str,
    *,
    passages: tuple[SourcePassage, ...],
    location: str = "",
    max_entities: int = 50,
    max_claims: int = 30,
) -> ExtractionBatch:
    # The vNext contract is two-pass.  Accepting relations accidentally returned
    # by an older one-pass prompt keeps interrupted/test runs replayable without
    # changing the normal production path.
    payload = llm.complete_json(
        SYSTEM_PROMPT,
        ENTITY_PROMPT.format(
            text=text, location=location, max_entities=max_entities
        ),
    )
    first = parse_payload(
        payload,
        passages,
        max_entities=max_entities,
        max_claims=max_claims,
    )
    entities, entity_rejected = first.entities, first.rejected
    if "claims" in payload or "relations" in payload:
        allowed = {_compact(item.name) for item in entities}
        claims = tuple(
            claim
            for claim in first.claims
            if _compact(claim.subject) in allowed
            and _compact(claim.object) in allowed
        )
        return ExtractionBatch(
            entities=entities,
            claims=claims,
            rejected=entity_rejected,
        )
    claims, claim_rejected = extract_relations(
        llm,
        text,
        passages=passages,
        entities=entities,
        section_context=location,
        max_claims=max_claims,
    )
    return ExtractionBatch(
        entities=entities,
        claims=claims,
        rejected=entity_rejected + claim_rejected,
    )


def extract_entities(
    llm: JSONLLM,
    text: str,
    *,
    passages: tuple[SourcePassage, ...],
    location: str = "",
    max_entities: int = 50,
) -> tuple[tuple[EntityObservation, ...], tuple[str, ...]]:
    payload = llm.complete_json(
        SYSTEM_PROMPT,
        ENTITY_PROMPT.format(
            text=text,
            location=location,
            max_entities=max_entities,
        ),
    )
    batch = parse_payload(payload, passages, max_entities=max_entities, max_claims=0)
    return batch.entities, batch.rejected


def extract_relations(
    llm: JSONLLM,
    text: str,
    *,
    passages: tuple[SourcePassage, ...],
    entities: tuple[EntityObservation, ...],
    section_context: str = "",
    max_claims: int = 30,
) -> tuple[tuple[ClaimObservation, ...], tuple[str, ...]]:
    entity_names = list(dict.fromkeys(item.name for item in entities))
    if not entity_names:
        return (), ()
    payload = llm.complete_json(
        SYSTEM_PROMPT,
        RELATION_PROMPT.format(
            text=text,
            entities=entity_names,
            section_context=section_context,
            max_claims=max_claims,
        ),
    )
    if "claims" not in payload and "relations" in payload:
        payload = {"entities": [], "claims": payload.get("relations", [])}
    batch = parse_payload(payload, passages, max_entities=0, max_claims=max_claims)
    allowed = {_compact(name) for name in entity_names}
    claims: list[ClaimObservation] = []
    rejected = list(batch.rejected)
    for index, claim in enumerate(batch.claims):
        if _compact(claim.subject) not in allowed or _compact(claim.object) not in allowed:
            rejected.append(f"claim[{index}] 端点不在实体清单")
            continue
        claims.append(claim)
    return tuple(claims), tuple(rejected)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def parse_payload(
    payload: dict[str, Any],
    passages: tuple[SourcePassage, ...] | list[SourcePassage],
    *,
    max_entities: int = 50,
    max_claims: int = 30,
) -> ExtractionBatch:
    entities: list[EntityObservation] = []
    claims: list[ClaimObservation] = []
    rejected: list[str] = []
    seen_entities: set[str] = set()
    passage_by_id = {item.passage_id: item for item in passages}

    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []
        rejected.append("entities 不是数组")
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
        rejected.append("claims 不是数组")
    for index, raw in enumerate(raw_entities[:max_entities]):
        if not isinstance(raw, dict):
            rejected.append(f"entity[{index}] 不是对象")
            continue
        name = _string(raw, "name")
        definition = _string(raw, "definition")
        raw_labels = raw.get("type_labels", raw.get("entity_types", []))
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        if not isinstance(raw_labels, list):
            raw_labels = []
        labels = tuple(
            dict.fromkeys(
                value.strip()
                for value in raw_labels
                if isinstance(value, str) and value.strip()
            )
        )[:3]
        legacy_type = _string(raw, "entity_type")
        if legacy_type and legacy_type not in labels:
            labels = (legacy_type, *labels)[:3]
        entity_type = labels[0] if labels else ""
        if not name or len(definition) < 4:
            rejected.append(f"entity[{index}] 缺少名称或实质性 definition")
            continue
        grounded = _resolve_evidence(raw.get("evidence"), passage_by_id)
        if isinstance(grounded, str):
            rejected.append(f"entity[{index}] {grounded}")
            continue
        key = _compact(name)
        if key in seen_entities:
            rejected.append(f"entity[{index}] 片段内重复: {name!r}")
            continue
        seen_entities.add(key)
        aliases_raw = raw.get("aliases", [])
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            for alias in aliases_raw:
                if isinstance(alias, str) and alias.strip():
                    aliases.append(alias.strip())
        quote, source_text, passage_ids, source_location = grounded
        entities.append(
            EntityObservation(
                name=name,
                definition=definition,
                entity_type=entity_type,
                model_quote=quote,
                source_text=source_text,
                passage_ids=passage_ids,
                location=source_location,
                aliases=tuple(dict.fromkeys(aliases)),
                type_labels=labels,
            )
        )

    for index, raw in enumerate(raw_claims[:max_claims]):
        if not isinstance(raw, dict):
            rejected.append(f"claim[{index}] 不是对象")
            continue
        subject = _string(raw, "subject")
        relation = _string(raw, "predicate") or _string(raw, "relation")
        object_ = _string(raw, "object")
        polarity = _string(raw, "stance") or "support"
        if not relation or len(relation) > 120:
            rejected.append(f"claim[{index}] 缺少或过长 relation")
            continue
        if polarity not in POLARITIES:
            rejected.append(f"claim[{index}] 非法 stance: {polarity!r}")
            continue
        if not subject or not object_ or _compact(subject) == _compact(object_):
            rejected.append(f"claim[{index}] 端点为空或自环")
            continue
        grounded = _resolve_evidence(raw.get("evidence"), passage_by_id)
        if isinstance(grounded, str):
            rejected.append(f"claim[{index}] {grounded}")
            continue
        quote, source_text, passage_ids, source_location = grounded
        claims.append(
            ClaimObservation(
                subject=subject,
                relation=relation,
                object=object_,
                model_quote=quote,
                source_text=source_text,
                passage_ids=passage_ids,
                location=source_location,
                polarity=polarity,
                raw_relation=relation,
            )
        )
    if len(raw_entities) > max_entities:
        rejected.append(f"entities 超过上限 {max_entities}，已截断")
    if len(raw_claims) > max_claims:
        rejected.append(f"claims 超过上限 {max_claims}，已截断")
    return ExtractionBatch(
        entities=tuple(entities),
        claims=tuple(claims),
        rejected=tuple(rejected),
    )


def _resolve_evidence(
    raw: Any, passage_by_id: dict[str, SourcePassage]
) -> tuple[str, str, tuple[str, ...], str] | str:
    if not isinstance(raw, dict):
        return "evidence 必须包含 passage_ids 和 quote"
    quote = _string(raw, "quote")
    raw_ids = raw.get("passage_ids")
    if not quote:
        return "evidence.quote 为空"
    if not isinstance(raw_ids, list):
        return "evidence.passage_ids 不是数组"
    normalized_ids = [
        str(item).strip() for item in raw_ids if str(item).strip()
    ]
    if len(normalized_ids) != len(set(normalized_ids)):
        return "evidence.passage_ids 不能重复"
    passage_ids = tuple(normalized_ids)
    if not passage_ids or len(passage_ids) > 3:
        return "evidence.passage_ids 必须包含当前片段中的 1–3 个段落"
    missing = [item for item in passage_ids if item not in passage_by_id]
    if missing:
        return f"evidence 引用了当前片段不存在的段落: {missing}"
    selected = sorted(
        (passage_by_id[item] for item in passage_ids),
        key=lambda item: item.start,
    )
    passage_ids = tuple(item.passage_id for item in selected)
    source_text = "\n\n".join(item.text for item in selected)
    location = "; ".join(
        f"{item.passage_id} {item.location}" for item in selected
    )
    return quote, source_text, passage_ids, location
