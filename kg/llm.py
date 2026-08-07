from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

import json_repair


DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_COMPLEX_MODEL = "MiniMax-M3"
DEFAULT_SIMPLE_MODEL = "MiniMax-M2.7"
DEFAULT_MAX_CONCURRENCY = 6
# Compatibility: the primary/complex pipeline remains the default client.
DEFAULT_MINIMAX_MODEL = DEFAULT_COMPLEX_MODEL

# MiniMax 在 HTTP 200 的响应体里用 base_resp.status_code 表达业务错误，
# 其中限流（2062）和内部错误与 HTTP 429/5xx 是同一类问题，必须同样重试。
# 只有鉴权、余额和参数错误重试也不会变，直接抛出让调用方尽快看到。
TERMINAL_RESPONSE_STATUS = frozenset({1004, 1008, 2013, 2049})


class LLMResponseError(RuntimeError):
    """HTTP 200 响应体里的业务错误，按 status_code 决定是否重试。"""

    def __init__(self, base_resp: dict[str, Any]):
        super().__init__(f"MiniMax API 错误: {base_resp}")
        try:
            self.status_code = int(base_resp.get("status_code", 0))
        except (TypeError, ValueError):
            self.status_code = -1

    @property
    def retryable(self) -> bool:
        return self.status_code not in TERMINAL_RESPONSE_STATUS


class JSONLLM(Protocol):
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return one JSON object, optionally normalized by ``validate``."""


class LLMConcurrencyLimiter:
    """Share one request limit across every LLM client in this process."""

    def __init__(self, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
        if max_concurrency < 1:
            raise ValueError("LLM 最大总并发必须至少为 1")
        self.max_concurrency = max_concurrency
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


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
    def from_env(cls, *, role: str = "complex") -> "LLMConfig":
        if role not in {"complex", "simple"}:
            raise ValueError(f"未知 LLM role: {role}")
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
        role_model = (
            os.environ.get("KG_COMPLEX_LLM_MODEL")
            if role == "complex"
            else os.environ.get("KG_SIMPLE_LLM_MODEL")
        )
        model = role_model or os.environ.get("KG_LLM_MODEL") or (
            DEFAULT_COMPLEX_MODEL if role == "complex" else DEFAULT_SIMPLE_MODEL
        )
        if not api_key:
            raise RuntimeError(
                "缺少 MINIMAX_API_KEY"
            )
        return cls(base_url=base_url, api_key=api_key, model=model)


class ChatCompletionsJSONLLM:
    """Small configurable JSON client for MiniMax and compatible endpoints."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        limiter: LLMConcurrencyLimiter | None = None,
    ):
        self.config = config
        self.limiter = limiter or LLMConcurrencyLimiter()

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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
        # 解析失败和载荷校验失败都是同一类问题：这一次采样的输出不能用。
        # 两者共享同一次重新生成机会，最坏开销仍是两次生成。
        for generation_attempt in range(2):
            data = self._send(request)
            try:
                parsed = parse_json_object(_message_text(data))
                return validate(parsed) if validate else parsed
            except ValueError:
                if generation_attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        for attempt in range(self.config.retries + 1):
            try:
                with self.limiter.slot():
                    with urllib.request.urlopen(
                        request, timeout=self.config.timeout
                    ) as response:
                        data = json.loads(response.read().decode("utf-8"))
                _validate_response(data)
                return data
            except LLMResponseError as exc:
                if not exc.retryable or attempt >= self.config.retries:
                    raise
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
        raise LLMResponseError(base_resp)


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
