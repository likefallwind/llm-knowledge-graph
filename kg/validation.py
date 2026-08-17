from __future__ import annotations

import json

from . import ontology
from .llm import JSONLLM
from .models import ClaimObservation


VALIDATION_PROMPT_VERSION = "canonical-assertion-judge-5-scoped-projection"

VALIDATION_SYSTEM = """你是关系证据裁判，不是知识来源。
只根据程序从 Source 取得的 source_text 判断关系；禁止使用外部知识补足省略信息。
model_quote 只是上一步模型指出的关注重点，可能有轻微改写，不能取代 source_text。
命题在常识上成立、但 source_text 没有把它陈述出来时，一律返回 insufficient。
裁判对象是实体和关系规范化后的完整 Assertion；若规范化扩大了端点含义、漏掉必要
条件，或者把练习题、问句、局部代码行为、界面操作写成一般知识，也返回 insufficient。
即使 source_text 能证明某段代码确实调用了某个辅助函数、工具类、损失类或配置项，或者
能证明某个界面操作步骤，只要该命题主要记录当前实现/操作而不是正文讲授的可复用知识，
仍返回 insufficient。代码只能补充证明正文已经明确介绍的概念关系，不能单独建立关系。
使用删除测试：去掉代码块、练习题和界面操作步骤后，叙述正文不能独立表达完整 Assertion，
则返回 insufficient。
不清楚就返回 insufficient。只输出 JSON 对象。"""


def judge_claim(llm: JSONLLM, claim: ClaimObservation) -> tuple[str, str]:
    relation_kind = claim.relation_kind
    if relation_kind == "other" and claim.relation in {
        "is_a", "part_of", "prerequisite_of"
    }:
        relation_kind = claim.relation
    if relation_kind in {"is_a", "part_of", "prerequisite_of"}:
        relation_definition = ontology.relation_detail(relation_kind)
    else:
        relation_definition = claim.relation_description or (
            "这是开放关系。source_text 必须明确表达 subject 通过该谓词指向 "
            "object；仅共现、主题相近、目录相邻、模型常识或可能的推断均不成立。"
        )
    payload = llm.complete_json(
        VALIDATION_SYSTEM,
        """分别完成两个判断，不得因为完整 Assertion 有原文支持，就默认三元组投影正确。

1. assertion_verdict：只判断 source_text 是否明确支持或反对完整 Assertion。supports 要求
完整保留条件、范围、否定、可能性和数量限制；否则为 insufficient。
2. projection_faithful：先严格按照关系定义，把三元组口头化成
“subject 通过 canonical relation 指向 object”的 projection_statement，再判断它是否与
完整 Assertion 的核心关系参与者、关系含义和方向一致。Claim 是便于导航的紧凑投影，
Assertion 才负责保存条件、范围、数量、时间和其他限定；projection_statement 不需要重复
这些已由 Assertion 保存的限制，不能仅因紧凑边省略限定就判 false。例如 Assertion 说
“标量由只有一个元素的张量表示”时，紧凑投影“标量 represented_by 张量”可以忠实。
但仅仅在 Assertion 中出现两个端点仍然不够；若真正参与关系的主语或宾语是端点的参数、
输出、组成部分等第三个对象，或者关系含义/方向改变，必须为 false。例如“卷积层的权重
被称为卷积核”不能投影成“卷积层是卷积核的别称”。

只有 assertion_verdict=supports 且 projection_faithful=true，最终关系才能作为支持证据；
投影不忠实一律不准入。先做判定测试，再逐条核对排除项；冲突时以排除项为准。
返回 {"assertion_verdict":"supports|contradicts|insufficient",
"projection_statement":"按关系定义口头化后的三元组命题",
"projection_faithful":true,"reason":"同时解释两个判断"}。

三元组投影：%s
完整 Assertion：%s
限制语境：%s
scope_is_restrictive：%s
关系定义：
%s
model_quote：%s
source_text（唯一权威证据）：%s"""
        % (
            json.dumps(
                {
                    "subject": claim.subject,
                    "relation": claim.relation,
                    "object": claim.object,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                claim.normalized_statement or claim.statement_text,
                ensure_ascii=False,
            ),
            json.dumps(claim.scope_text, ensure_ascii=False),
            json.dumps(claim.scope_is_restrictive),
            relation_definition,
            json.dumps(claim.model_quote, ensure_ascii=False),
            json.dumps(claim.source_text, ensure_ascii=False),
        ),
        validate=_validate_judgment_payload,
    )
    verdict = str(payload.get("assertion_verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if payload.get("projection_faithful") is not True:
        return "insufficient", reason or "最终三元组不能忠实投影完整 Assertion"
    return verdict, reason


def _validate_judgment_payload(payload: dict) -> dict:
    verdict = str(payload.get("assertion_verdict", "")).strip().lower()
    if verdict not in {"supports", "contradicts", "insufficient"}:
        raise ValueError("validator 返回非法 assertion_verdict")
    if not isinstance(payload.get("projection_faithful"), bool):
        raise ValueError("validator 缺少 projection_faithful")
    projection = str(payload.get("projection_statement", "")).strip()
    if not projection:
        raise ValueError("validator 缺少 projection_statement")
    return payload
