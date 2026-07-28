from __future__ import annotations

import re
from typing import Any

from .llm import JSONLLM
from .models import (
    ENTITY_TYPES,
    POLARITIES,
    RELATIONS,
    ClaimObservation,
    EntityObservation,
    ExtractionBatch,
)


SYSTEM_PROMPT = """你是语料约束的知识抽取器。
你只能理解用户给出的原文，禁止用模型参数记忆补充原文没有表达的知识。
没有可逐字定位证据的对象或关系必须省略。只输出 JSON 对象。"""

EXTRACTION_PROMPT = """从下面的单个语料片段抽取 Entity、Claim 和 Evidence。

Entity 必须是在本片段中有稳定名称、可复指，并有实质性定义或知识含义的对象。
不要抽取“定义、公式、方法、过程、属性”等离开上下文没有明确身份的通用词。

六种 entity_type（根据 definition 判断，concept 兜底）：
- resource：论文、教材、课程、文档等知识载体
- criterion：损失函数、指标、优化目标、评测协议
- data：数据集、训练数据、样本集合
- task：有目标、输入输出或成功条件的问题
- solution：算法、过程、模型、架构、系统或工具
- concept：数学对象、性质、规律、现象、研究领域等其他对象

只允许三种 relation：
- is_a：subject 是 object 的一种
- part_of：subject 是 object 的真实组成部分或明确阶段
- prerequisite_of：理解 subject 是学习 object 的实质性前提

规则：
1. evidence 必须逐字摘自片段；definition 可以忠实概括 evidence。
2. aliases 只列出本片段实际出现的别名。
3. Claim 的两个端点都必须使用本片段实际出现的完整名称，不得截短限定词。
4. 共现、超链接和章节顺序都不能证明关系。
5. stance 是 support 或 oppose；不确定的关系不要输出。
6. 最多输出 {max_entities} 个实体和 {max_claims} 个 Claim。

输出：
{{
  "entities": [
    {{
      "name": "原文中的名称",
      "definition": "仅依据原文的定义或知识含义",
      "entity_type": "六种类型之一",
      "aliases": ["原文中实际出现的别名"],
      "evidence": "逐字摘录",
      "location": "可选的段落说明"
    }}
  ],
  "claims": [
    {{
      "subject": "完整名称",
      "relation": "三种关系之一",
      "object": "完整名称",
      "stance": "support",
      "evidence": "逐字摘录",
      "location": "可选的段落说明"
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
        ),
    )
    return parse_payload(
        payload, text, max_entities=max_entities, max_claims=max_claims
    )


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def evidence_in_text(excerpt: str, source_text: str) -> bool:
    if not excerpt.strip() or not source_text:
        return False
    haystack = _compact(source_text)
    parts = [
        _compact(part)
        for part in re.split(r"(?:\.{3}|…+)", excerpt)
        if _compact(part)
    ]
    return bool(parts) and all(part in haystack for part in parts)


def _name_in_text(name: str, source_text: str) -> bool:
    return bool(name.strip()) and _compact(name) in _compact(source_text)


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def parse_payload(
    payload: dict[str, Any],
    source_text: str,
    *,
    max_entities: int = 20,
    max_claims: int = 30,
) -> ExtractionBatch:
    entities: list[EntityObservation] = []
    claims: list[ClaimObservation] = []
    rejected: list[str] = []
    seen_entities: set[str] = set()

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
        evidence = _string(raw, "evidence")
        if not name or len(definition) < 4:
            rejected.append(f"entity[{index}] 缺少名称或实质性 definition")
            continue
        if entity_type not in ENTITY_TYPES:
            rejected.append(f"entity[{index}] 非法 entity_type: {entity_type!r}")
            continue
        if not _name_in_text(name, source_text):
            rejected.append(f"entity[{index}] 名称未在语料中出现: {name!r}")
            continue
        if not evidence_in_text(evidence, source_text):
            rejected.append(f"entity[{index}] evidence 无法在语料中定位")
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
                if (
                    isinstance(alias, str)
                    and alias.strip()
                    and _name_in_text(alias, source_text)
                ):
                    aliases.append(alias.strip())
        entities.append(
            EntityObservation(
                name=name,
                definition=definition,
                entity_type=entity_type,
                evidence=evidence,
                location=_string(raw, "location"),
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
        evidence = _string(raw, "evidence")
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
        if not _name_in_text(subject, source_text) or not _name_in_text(
            object_, source_text
        ):
            rejected.append(f"claim[{index}] 端点未完整出现在语料中")
            continue
        if not evidence_in_text(evidence, source_text):
            rejected.append(f"claim[{index}] evidence 无法在语料中定位")
            continue
        claims.append(
            ClaimObservation(
                subject=subject,
                relation=relation,
                object=object_,
                evidence=evidence,
                location=_string(raw, "location"),
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
