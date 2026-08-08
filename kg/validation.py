from __future__ import annotations

import json

from . import ontology
from .llm import JSONLLM
from .models import ClaimObservation


VALIDATION_PROMPT_VERSION = "canonical-assertion-judge-3-knowledge-admission"

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
        relation_definition = (
            "这是开放关系。source_text 必须明确表达 subject 通过该谓词指向 "
            "object；仅共现、主题相近、目录相邻、模型常识或可能的推断均不成立。"
        )
    payload = llm.complete_json(
        VALIDATION_SYSTEM,
        """判断 source_text 相对于最终 Assertion 的含义。
supports：source_text 明确表达完整 Assertion，规范化后的端点没有语义扩大，所有限制性
条件、范围、否定、可能性和数量限制均被保留，且关系方向正确。
contradicts：source_text 明确反对完整 Assertion。
insufficient：只是共现、相关、强调、顺序，或触发了排除项、语义不完整、无法判断。
先做判定测试，再逐条核对排除项；两者冲突时以排除项为准。
返回 {"verdict":"supports|contradicts|insufficient","reason":"简短理由"}。

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
    )
    verdict = str(payload.get("verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if verdict not in {"supports", "contradicts", "insufficient"}:
        return "insufficient", reason or "validator 返回非法 verdict"
    return verdict, reason
