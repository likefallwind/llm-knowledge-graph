from __future__ import annotations

from collections import deque
from typing import Any


class FakeLLM:
    def __init__(self, *responses: dict[str, Any]):
        self.responses = deque(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("FakeLLM 没有剩余响应")
        return self.responses.popleft()

    def assert_finished(self) -> None:
        if self.responses:
            raise AssertionError(f"FakeLLM 还有 {len(self.responses)} 个未使用响应")
