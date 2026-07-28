from __future__ import annotations

import os
import json
import unittest
from unittest import mock

from kg.llm import (
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    LLMConfig,
    MiniMaxM3LLM,
    parse_json_object,
)


class LLMTest(unittest.TestCase):
    def test_parse_fenced_or_surrounded_json(self):
        self.assertEqual(
            parse_json_object("思考完成\n```json\n{\"ok\": true}\n```"),
            {"ok": True},
        )
        self.assertEqual(
            parse_json_object('prefix {"value": 3} trailing'),
            {"value": 3},
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

    @mock.patch("kg.llm.time.sleep")
    @mock.patch("kg.llm.urllib.request.urlopen")
    def test_client_retries_read_timeout(self, urlopen, sleep):
        item = mock.MagicMock()
        item.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        ).encode("utf-8")
        item.__enter__.return_value = item
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


if __name__ == "__main__":
    unittest.main()
