from __future__ import annotations

from collections import deque
from typing import Any, Callable


class FakeLLM:
    def __init__(self, *responses: dict[str, Any]):
        self.responses = deque(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # 与真实客户端一致：校验失败会消耗一次重新生成机会。
        for attempt in range(2):
            self.calls.append((system, user))
            if not self.responses:
                raise AssertionError("FakeLLM 没有剩余响应")
            payload = self.responses.popleft()
            if validate is None:
                return payload
            try:
                return validate(payload)
            except ValueError:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def assert_finished(self) -> None:
        if self.responses:
            raise AssertionError(f"FakeLLM 还有 {len(self.responses)} 个未使用响应")
