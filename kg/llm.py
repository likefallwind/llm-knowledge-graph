from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

import json_repair


DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"


class JSONLLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return one JSON object."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 600.0
    retries: int = 3

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/text/chatcompletion_v2") or base.endswith(
            "/chat/completions"
        ):
            return base
        if "api.minimaxi.com" in base:
            return base + "/text/chatcompletion_v2"
        return base + "/chat/completions"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        base_url = (
            os.environ.get("KG_LLM_BASE_URL")
            or DEFAULT_MINIMAX_BASE_URL
        )
        api_key = _normalize_api_key(
            os.environ.get("MINIMAX_API_KEY")
            or os.environ.get("MINIMAX_API")
            or os.environ.get("minimax_api")
            or os.environ.get("KG_LLM_API_KEY")
            or ""
        )
        model = (
            os.environ.get("KG_LLM_MODEL")
            or DEFAULT_MINIMAX_MODEL
        )
        if not api_key:
            raise RuntimeError(
                "缺少 MINIMAX_API_KEY"
            )
        return cls(base_url=base_url, api_key=api_key, model=model)


class ChatCompletionsJSONLLM:
    """Small configurable JSON client for MiniMax and compatible endpoints."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for generation_attempt in range(2):
            data = self._send(request)
            try:
                return parse_json_object(_message_text(data))
            except ValueError:
                if generation_attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                _validate_response(data)
                return data
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= self.config.retries:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(
                        f"LLM HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.config.retries:
                    reason = getattr(exc, "reason", str(exc))
                    raise RuntimeError(f"LLM 连接失败: {reason}") from exc
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


# Compatibility aliases for existing callers.
MiniMaxM3LLM = ChatCompletionsJSONLLM
OpenAICompatibleLLM = ChatCompletionsJSONLLM


def _normalize_api_key(value: str) -> str:
    cleaned = value.strip()
    if cleaned.lower().startswith("bearer "):
        return cleaned[7:].strip()
    return cleaned


def _validate_response(payload: dict[str, Any]) -> None:
    base_resp = payload.get("base_resp")
    if not isinstance(base_resp, dict):
        return
    status_code = base_resp.get("status_code", 0)
    if status_code not in (0, "0", None):
        raise RuntimeError(f"MiniMax API 错误: {base_resp}")


def _message_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "".join(parts)
    raise ValueError("LLM message.content 不是文本")


def parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    if start < 0:
        raise ValueError("LLM 输出中没有 JSON 对象")
    candidate = candidate[start:]
    try:
        value, _ = decoder.raw_decode(candidate)
    except json.JSONDecodeError as original:
        try:
            value = json_repair.loads(candidate, strict=True)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise original
    if not isinstance(value, dict):
        raise ValueError("LLM 输出的顶层 JSON 必须是对象")
    return value
