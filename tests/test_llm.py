from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import json
import threading
import time
import unittest
from unittest import mock

from kg.llm import (
    DEFAULT_COMPLEX_MODEL,
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_SIMPLE_MODEL,
    LLMConcurrencyLimiter,
    LLMConfig,
    MiniMaxM3LLM,
    parse_json_object,
)


class LLMTest(unittest.TestCase):
    @staticmethod
    def _response(content: str):
        item = mock.MagicMock()
        item.read.return_value = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        item.__enter__.return_value = item
        return item

    def test_parse_fenced_or_surrounded_json(self):
        self.assertEqual(
            parse_json_object("思考完成\n```json\n{\"ok\": true}\n```"),
            {"ok": True},
        )
        self.assertEqual(
            parse_json_object('prefix {"value": 3} trailing'),
            {"value": 3},
        )

    def test_parse_repairs_unescaped_quotes_in_nested_string_values(self):
        payload = parse_json_object(
            '{"entities":[{"name":"扩散模型",'
            '"definition":"一种称为"生成模型"的模型",'
            '"aliases":["扩散"模型""]}]}'
        )

        self.assertEqual(
            payload["entities"][0],
            {
                "name": "扩散模型",
                "definition": '一种称为"生成模型"的模型',
                "aliases": ['扩散"模型"'],
            },
        )

    def test_top_level_must_be_object(self):
        with self.assertRaises(ValueError):
            parse_json_object("[1, 2]")

    def test_minimax_m3_is_the_default(self):
        with mock.patch.dict(
            os.environ, {"MINIMAX_API_KEY": " Bearer secret "}, clear=True
        ):
            config = LLMConfig.from_env()
        self.assertEqual(config.base_url, DEFAULT_MINIMAX_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(
            config.endpoint,
            "https://api.minimaxi.com/v1/text/chatcompletion_v2",
        )

    def test_model_roles_have_separate_defaults_and_overrides(self):
        with mock.patch.dict(
            os.environ, {"MINIMAX_API_KEY": "secret"}, clear=True
        ):
            complex_config = LLMConfig.from_env(role="complex")
            simple_config = LLMConfig.from_env(role="simple")
        self.assertEqual(complex_config.model, DEFAULT_COMPLEX_MODEL)
        self.assertEqual(simple_config.model, DEFAULT_SIMPLE_MODEL)
        with mock.patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "secret",
                "KG_COMPLEX_LLM_MODEL": "complex-x",
                "KG_SIMPLE_LLM_MODEL": "simple-y",
            },
            clear=True,
        ):
            self.assertEqual(LLMConfig.from_env(role="complex").model, "complex-x")
            self.assertEqual(LLMConfig.from_env(role="simple").model, "simple-y")

    def test_explicit_compatible_gateway_is_still_supported(self):
        with mock.patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "secret",
                "KG_LLM_BASE_URL": "https://gateway.example/v1",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()
        self.assertEqual(
            config.endpoint, "https://gateway.example/v1/chat/completions"
        )

    def test_concurrency_limiter_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "至少为 1"):
            LLMConcurrencyLimiter(0)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_two_clients_share_one_total_concurrency_limit(self, urlopen):
        limiter = LLMConcurrencyLimiter(2)
        config = LLMConfig(
            base_url="https://gateway.example/v1",
            api_key="secret",
            model="MiniMax-M3",
        )
        clients = [
            MiniMaxM3LLM(config, limiter=limiter),
            MiniMaxM3LLM(config, limiter=limiter),
        ]
        lock = threading.Lock()
        active = 0
        peak = 0

        def respond(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return self._response('{"ok": true}')
            finally:
                with lock:
                    active -= 1

        urlopen.side_effect = respond
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda index: clients[index % 2].complete_json(
                        "system", "user"
                    ),
                    range(12),
                )
            )

        self.assertEqual(results, [{"ok": True}] * 12)
        self.assertEqual(peak, 2)

    @mock.patch("kg.llm.time.sleep")
    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_retries_read_timeout(self, urlopen, sleep):
        item = self._response('{"ok": true}')
        urlopen.side_effect = [TimeoutError("read timed out"), item]
        client = MiniMaxM3LLM(
            LLMConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
                model="MiniMax-M3",
            )
        )
        self.assertEqual(client.complete_json("system", "user"), {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    @staticmethod
    def _body_error(status_code: int, status_msg: str = "err"):
        item = mock.MagicMock()
        item.read.return_value = json.dumps(
            {"base_resp": {"status_code": status_code, "status_msg": status_msg}}
        ).encode("utf-8")
        item.__enter__.return_value = item
        return item

    @staticmethod
    def _client():
        return MiniMaxM3LLM(
            LLMConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
                model="MiniMax-M3",
            )
        )

    @mock.patch("kg.llm.time.sleep")
    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_retries_rate_limit_reported_in_response_body(self, urlopen, sleep):
        # 2062 是 HTTP 200 响应体里的限流，与 HTTP 429 属于同一类问题。
        urlopen.side_effect = [
            self._body_error(2062, "已达到 Token Plan 速率限制"),
            self._response('{"ok": true}'),
        ]

        self.assertEqual(self._client().complete_json("system", "user"), {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    @mock.patch("kg.llm.time.sleep")
    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_does_not_retry_terminal_response_status(self, urlopen, sleep):
        urlopen.side_effect = [self._body_error(1004, "鉴权失败")]

        with self.assertRaisesRegex(RuntimeError, "MiniMax API 错误"):
            self._client().complete_json("system", "user")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    @mock.patch("kg.llm.time.sleep")
    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_gives_up_after_exhausting_rate_limit_retries(self, urlopen, sleep):
        urlopen.side_effect = [self._body_error(2062) for _ in range(4)]

        with self.assertRaisesRegex(RuntimeError, "2062"):
            self._client().complete_json("system", "user")
        self.assertEqual(urlopen.call_count, 4)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_regenerates_once_when_validation_fails(self, urlopen):
        # 第一次是真实见过的失败：区间 [0, 1] 把字符串截断成了三个数组元素。
        urlopen.side_effect = [
            self._response('{"rejected_candidates": ["裁剪到 [0", 1]}'),
            self._response('{"rejected_candidates": ["完整理由"]}'),
        ]

        def validate(payload):
            items = payload["rejected_candidates"]
            if not all(isinstance(item, str) for item in items):
                raise ValueError("rejected_candidates 必须是字符串数组")
            return {"normalized": items}

        self.assertEqual(
            self._client().complete_json("system", "user", validate=validate),
            {"normalized": ["完整理由"]},
        )
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_raises_after_validation_fails_twice(self, urlopen):
        urlopen.side_effect = [
            self._response('{"value": 1}'),
            self._response('{"value": 2}'),
        ]

        def validate(payload):
            raise ValueError("载荷不合格")

        with self.assertRaisesRegex(ValueError, "载荷不合格"):
            self._client().complete_json("system", "user", validate=validate)
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_regenerates_once_after_invalid_json(self, urlopen):
        urlopen.side_effect = [
            self._response('{"ok" true}'),
            self._response('{"ok": true}'),
        ]
        client = MiniMaxM3LLM(
            LLMConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
                model="MiniMax-M3",
            )
        )

        self.assertEqual(client.complete_json("system", "user"), {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_repairs_real_unescaped_quotes_without_regeneration(self, urlopen):
        urlopen.return_value = self._response(
            '```json\n{"verdict":"supports","reason":"source_text 明确说'
            '"第二部分 \'基础模型\' 介绍 Transformer"，因此支持。"}\n```'
        )
        client = MiniMaxM3LLM(
            LLMConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
                model="MiniMax-M3",
            )
        )

        self.assertEqual(
            client.complete_json("system", "user"),
            {
                "verdict": "supports",
                "reason": 'source_text 明确说"第二部分 \'基础模型\' 介绍 Transformer"，'
                "因此支持。",
            },
        )
        self.assertEqual(urlopen.call_count, 1)

    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_stops_after_second_invalid_json(self, urlopen):
        urlopen.side_effect = [
            self._response('{"first" true}'),
            self._response('{"second" true}'),
        ]
        client = MiniMaxM3LLM(
            LLMConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
                model="MiniMax-M3",
            )
        )

        with self.assertRaises(json.JSONDecodeError):
            client.complete_json("system", "user")
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
