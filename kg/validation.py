from __future__ import annotations

import json

from .llm import JSONLLM
from .models import ClaimObservation


VALIDATION_SYSTEM = """你是关系证据裁判，不是知识来源。
只判断给出的逐字证据是否真的表达指定关系；禁止使用外部知识补足省略信息。
不清楚就返回 insufficient。只输出 JSON 对象。"""

RELATION_MEANINGS = {
    "is_a": "subject 是 object 的一种，而非仅仅相关、使用或属于同一领域",
    "part_of": "subject 是 object 的真实组成部分或正文明确列出的阶段",
    "prerequisite_of": "理解 subject 是学习 object 的实质性前提，而非仅有先后顺序",
}


def judge_claim(llm: JSONLLM, claim: ClaimObservation) -> tuple[str, str]:
    payload = llm.complete_json(
        VALIDATION_SYSTEM,
        """判断 evidence 相对于 Claim 的含义。
supports：证据明确表达 Claim。
contradicts：证据明确反对 Claim。
insufficient：只是共现、相关、顺序、语义不完整或无法判断。
返回 {"verdict":"supports|contradicts|insufficient","reason":"简短理由"}。

Claim：%s
关系含义：%s
evidence：%s"""
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
            json.dumps(claim.evidence, ensure_ascii=False),
        ),
    )
    verdict = str(payload.get("verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if verdict not in {"supports", "contradicts", "insufficient"}:
        return "insufficient", reason or "validator 返回非法 verdict"
    return verdict, reason
