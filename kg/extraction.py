from __future__ import annotations

import re
from typing import Any

from . import ontology
from .llm import JSONLLM
from .models import (
    ENTITY_TYPES,
    POLARITIES,
    RELATIONS,
    ClaimObservation,
    EntityObservation,
    ExtractionBatch,
    SourcePassage,
)


EXTRACTION_PROMPT_VERSION = "grounded-extract-ontology-2"
PASSAGE_VERSION = "source-passages-1"

SYSTEM_PROMPT = """你是语料约束的知识抽取器。
你只能理解用户给出的原文，禁止用模型参数记忆补充原文没有表达的知识。
没有有效原文段落依据的对象或关系必须省略。只输出 JSON 对象。"""

EXTRACTION_PROMPT = """从下面的单个语料片段抽取 Entity、Claim 和 Evidence。

Entity 必须是在本片段中有稳定名称、可复指，并有实质性定义或知识含义的对象。

六种 entity_type。根据原文给出的 definition 判断，不按名称字面猜测，concept 兜底：
{entity_types}

只允许三种 relation。先做判定测试，再逐条核对排除项；两者冲突时以排除项为准：
{relations}

规则：
1. evidence.passage_ids 必须选择片段中真实存在的段落 ID，最多 3 个。
2. evidence.quote 是你认为最关键的引文。应尽量忠实引用，但允许轻微省略或改写。
3. aliases 只列出本片段表达过的别名。
4. Claim 的两个端点都必须使用本片段表达的完整名称，不得截短限定词。
5. 判不准关系属于哪一种，或命题只是常识为真而原文没有陈述时，不要输出该 Claim。
   宁可漏，不要猜。
6. stance 是 support 或 oppose；不确定的关系不要输出。
7. 最多输出 {max_entities} 个实体和 {max_claims} 个 Claim。

输出：
{{
  "entities": [
    {{
      "name": "原文中的名称",
      "definition": "仅依据原文的定义或知识含义",
      "entity_type": "六种类型之一",
      "aliases": ["原文中实际出现的别名"],
      "evidence": {{
        "passage_ids": ["P000001"],
        "quote": "最关键的原文引文，可轻微省略"
      }}
    }}
  ],
  "claims": [
    {{
      "subject": "完整名称",
      "relation": "三种关系之一",
      "object": "完整名称",
      "stance": "support",
      "evidence": {{
        "passage_ids": ["P000002"],
        "quote": "最关键的关系引文，可轻微省略"
      }}
    }}
  ]
}}

语料片段位置：{location}
---
{text}
---"""


def extract(
    llm: JSONLLM,
    text: str,
    *,
    passages: tuple[SourcePassage, ...],
    location: str = "",
    max_entities: int = 20,
    max_claims: int = 30,
) -> ExtractionBatch:
    payload = llm.complete_json(
        SYSTEM_PROMPT,
        EXTRACTION_PROMPT.format(
            text=text,
            location=location,
            max_entities=max_entities,
            max_claims=max_claims,
            entity_types=ontology.entity_type_block(),
            relations=ontology.relation_block(),
        ),
    )
    return parse_payload(
        payload, passages, max_entities=max_entities, max_claims=max_claims
    )


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def parse_payload(
    payload: dict[str, Any],
    passages: tuple[SourcePassage, ...] | list[SourcePassage],
    *,
    max_entities: int = 20,
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
    for index, raw in enumerate(raw_entities[:max_entities]):
        if not isinstance(raw, dict):
            rejected.append(f"entity[{index}] 不是对象")
            continue
        name = _string(raw, "name")
        definition = _string(raw, "definition")
        entity_type = _string(raw, "entity_type")
        if not name or len(definition) < 4:
            rejected.append(f"entity[{index}] 缺少名称或实质性 definition")
            continue
        if entity_type not in ENTITY_TYPES:
            rejected.append(f"entity[{index}] 非法 entity_type: {entity_type!r}")
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
            )
        )

    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
        rejected.append("claims 不是数组")
    for index, raw in enumerate(raw_claims[:max_claims]):
        if not isinstance(raw, dict):
            rejected.append(f"claim[{index}] 不是对象")
            continue
        subject = _string(raw, "subject")
        relation = _string(raw, "relation")
        object_ = _string(raw, "object")
        polarity = _string(raw, "stance") or "support"
        if relation not in RELATIONS:
            rejected.append(f"claim[{index}] 非法 relation: {relation!r}")
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
