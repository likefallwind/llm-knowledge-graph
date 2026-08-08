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


ENTITY_PROMPT_VERSION = "open-entities-section-4-recall-protection"
RELATION_PROMPT_VERSION = "open-relations-assertion-4-recall-protection"
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

准入边界：
- 可以抽取正文明确介绍或解释、脱离当前示例后仍有独立教学意义的概念、方法、模型、
  数据集。库、框架或 API 只有在正文把它本身作为教学对象解释时才可准入，不能因为
  它在代码中被导入、实例化或调用就准入。
- 不要抽取只在当前代码示例中存在的局部变量、临时函数、演示类、占位符、文件名、
  图片名、图表编号、公式片段、辅助计时/绘图/累计工具、界面按钮、操作菜单或命令输出。
- 仅仅“在代码中出现”不是 Entity；代码符号必须指向一个脱离该示例仍可独立理解、
  稳定复用的对象。教材临时定义的 train_ch6、fancy_func 一类名字通常不准入。
- 不要把练习题中的假设对象、提问本身或某次演示操作当作 Entity。
- definition 必须描述对象本身，不能只描述它在当前示例中的一次操作或某个固定数值。
- 证据来源测试：假设删掉代码块、练习题和界面操作步骤，读者是否仍能仅根据叙述正文
  识别并解释该对象？如果不能，必须省略。代码只能作为正文已介绍概念的补充证据，
  不能单独产生 Entity。
- 例如 LeNet-5、随机梯度下降可在正文有实质介绍时准入；train_ch6、d2l.Timer、
  d2l.Animator、d2l.Accumulator、Stopping 按钮、Image→Create 操作不得仅凭示例准入。
- 召回保护：叙述正文明确陈述定义、性质、比较、因果、组成、适用条件或限制时，构成
  这些知识陈述所需的全部具名领域对象都应抽取。某对象在当前段落没有被重新完整定义，
  但正文明确陈述了它的性质或它与其他对象的比较，也已经具有实质性知识含义，不得
  因此省略。例如正文比较随机梯度下降与梯度下降时，两者都应准入。
- 上一条召回保护只适用于陈述事实的叙述正文，不适用于代码块、练习题、问句和界面
  操作步骤；不得借召回保护重新引入实现辅助对象。

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

RELATION_PROMPT = """从下面的原始语料中抽取开放式关系及其完整 Assertion。

实体已经由上一阶段识别。subject 和 object 必须使用实体清单中的完整名称；不要重新
抽实体。predicate 使用原文关系的简洁、可复用表达，不受预设关系词表限制。

规则：
1. 只能依据原始 Passage，目录和摘要只提供定位上下文，不能单独证明关系。
2. evidence.passage_ids 必须来自下面真实存在的 Passage，最多 3 个。
3. 共现、章节相邻、主题相似或模型常识不能构成关系。
4. statement 必须是一条可独立判断真假的完整陈述，必须出现 subject 和 object 的完整
   名称，并保留原文中的条件、适用范围、时间、否定、可能性和数量限制。不能把
   “在某条件下成立”改写成无条件成立。
5. scope 摘出会限制关系成立范围的前提或语境；没有则为空字符串。去掉 scope 会使
   statement 变假或明显扩大适用范围时，scope_is_restrictive 必须为 true。
6. 只抽取可复用的教学知识。仅描述当前代码调用了哪个辅助函数、计时器、绘图器、
   累计器、损失类或配置项的实现事实，以及界面点击顺序、云资源操作步骤，不形成关系。
7. 证据来源测试：假设删掉代码块、练习题和界面操作步骤，叙述正文是否仍明确表达该
   关系？如果不能，必须省略。不得仅根据代码反推“模型使用某工具/API”。
8. 正文明确讲授的模型、算法、机制、适用条件之间的关系可以保留；代码只能作为补充
   证据，不能成为关系的唯一来源。
9. 召回检查：逐句检查叙述正文中的定义、性质、比较、因果、组成、适用条件和限制，
   只要两个端点都在实体清单中，就不要因端点未在本段重新定义而漏掉关系。
10. subject 和 object 必须逐字复制实体清单中的完整名称；statement 中也必须逐字包含
   这两个完整名称。即使原文使用简称、代词或“非凸情况下”等语法变形，也要在忠于
   原意的前提下把实体清单中的完整名称写入 statement。
11. stance 表示原文是否支持这条完整 statement，不表示 statement 内部是否含否定。
   本任务抽取的是原文实际陈述的命题，因此始终输出 support；否定事实应写进 statement
   和 predicate，不得因“并非如此”“不收敛”等否定词输出 oppose。
12. predicate 的方向必须与 statement 完全一致，并在输出前核对主客体方向。
13. 不确定时省略；最多输出 {max_claims} 条。

输出：
{{"relations":[{{
  "subject":"实体清单中的完整名称",
  "predicate":"开放关系谓词",
  "object":"实体清单中的完整名称",
  "statement":"包含全部必要条件的完整关系表述",
  "scope":"限制成立范围的条件或语境；没有则为空字符串",
  "scope_is_restrictive":true,
  "stance":"support",
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
        statement_text = _string(raw, "statement")
        scope_text = _string(raw, "scope")
        scope_is_restrictive = raw.get("scope_is_restrictive", False)
        polarity = _string(raw, "stance") or "support"
        if not relation or len(relation) > 120:
            rejected.append(f"claim[{index}] 缺少或过长 relation")
            continue
        if polarity not in POLARITIES:
            rejected.append(f"claim[{index}] 非法 stance: {polarity!r}")
            continue
        # A freshly extracted complete statement is, by contract, the
        # proposition asserted by its cited source.  Logical negation belongs
        # in the statement/predicate; it is not opposing evidence for itself.
        # Keep ``oppose`` in the storage model for later/manual evidence, while
        # normalizing model confusion at the extraction boundary.
        polarity = "support"
        if not subject or not object_ or _compact(subject) == _compact(object_):
            rejected.append(f"claim[{index}] 端点为空或自环")
            continue
        if not statement_text:
            rejected.append(f"claim[{index}] 缺少完整 statement")
            continue
        compact_statement = _compact(statement_text)
        if (
            _compact(subject) not in compact_statement
            or _compact(object_) not in compact_statement
        ):
            rejected.append(
                f"claim[{index}] statement 未完整包含两个端点: "
                f"subject={subject!r}, object={object_!r}, "
                f"statement={statement_text[:240]!r}"
            )
            continue
        if not isinstance(scope_is_restrictive, bool):
            rejected.append(f"claim[{index}] scope_is_restrictive 不是布尔值")
            continue
        if scope_is_restrictive and not scope_text:
            rejected.append(f"claim[{index}] 限制性 scope 不能为空")
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
                statement_text=statement_text,
                scope_text=scope_text,
                scope_is_restrictive=scope_is_restrictive,
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
