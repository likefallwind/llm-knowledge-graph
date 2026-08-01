from __future__ import annotations

import json

from . import ontology
from .llm import JSONLLM
from .models import ClaimObservation


VALIDATION_PROMPT_VERSION = "relation-judge-ontology-4"

VALIDATION_SYSTEM = """你是关系证据裁判，不是知识来源。
只根据程序从 Source 取得的 source_text 判断关系；禁止使用外部知识补足省略信息。
model_quote 只是上一步模型指出的关注重点，可能有轻微改写，不能取代 source_text。
命题在常识上成立、但 source_text 没有把它陈述出来时，一律返回 insufficient。
不清楚就返回 insufficient。只输出 JSON 对象。"""


def judge_claim(llm: JSONLLM, claim: ClaimObservation) -> tuple[str, str]:
    payload = llm.complete_json(
        VALIDATION_SYSTEM,
        """判断 source_text 相对于 Claim 的含义。
supports：source_text 明确表达 Claim，且通过下面关系定义的判定测试、不触发任何排除项。
contradicts：source_text 明确反对 Claim。
insufficient：只是共现、相关、强调、顺序，或触发了排除项、语义不完整、无法判断。
先做判定测试，再逐条核对排除项；两者冲突时以排除项为准。
返回 {"verdict":"supports|contradicts|insufficient","reason":"简短理由"}。

Claim：%s
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
            ontology.relation_detail(claim.relation),
            json.dumps(claim.model_quote, ensure_ascii=False),
            json.dumps(claim.source_text, ensure_ascii=False),
        ),
    )
    verdict = str(payload.get("verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if verdict not in {"supports", "contradicts", "insufficient"}:
        return "insufficient", reason or "validator 返回非法 verdict"
    return verdict, reason
