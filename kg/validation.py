from __future__ import annotations

import json

from .llm import JSONLLM
from .models import ClaimObservation


VALIDATION_PROMPT_VERSION = "relation-judge-passages-1"

VALIDATION_SYSTEM = """你是关系证据裁判，不是知识来源。
只根据程序从 Source 取得的 source_text 判断关系；禁止使用外部知识补足省略信息。
model_quote 只是上一步模型指出的关注重点，可能有轻微改写，不能取代 source_text。
不清楚就返回 insufficient。只输出 JSON 对象。"""

RELATION_MEANINGS = {
    "is_a": "subject 是 object 的一种，而非仅仅相关、使用或属于同一领域",
    "part_of": "subject 是 object 的真实组成部分或正文明确列出的阶段",
    "prerequisite_of": "理解 subject 是学习 object 的实质性前提，而非仅有先后顺序",
}


def judge_claim(llm: JSONLLM, claim: ClaimObservation) -> tuple[str, str]:
    payload = llm.complete_json(
        VALIDATION_SYSTEM,
        """判断 source_text 相对于 Claim 的含义。
supports：证据明确表达 Claim。
contradicts：证据明确反对 Claim。
insufficient：只是共现、相关、顺序、语义不完整或无法判断。
返回 {"verdict":"supports|contradicts|insufficient","reason":"简短理由"}。

Claim：%s
关系含义：%s
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
            RELATION_MEANINGS[claim.relation],
            json.dumps(claim.model_quote, ensure_ascii=False),
            json.dumps(claim.source_text, ensure_ascii=False),
        ),
    )
    verdict = str(payload.get("verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if verdict not in {"supports", "contradicts", "insufficient"}:
        return "insufficient", reason or "validator 返回非法 verdict"
    return verdict, reason
