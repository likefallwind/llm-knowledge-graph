from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

from kg.models import SourceSpec
from kg.sources import chunk_text, load_catalog, load_source, segment_text


class SourcesTest(unittest.TestCase):
    def test_catalog_resolves_relative_paths_and_versions_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "doc.md").write_text("# 标题\n\n正文", encoding="utf-8")
            (root / "sources.json").write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "key": "doc",
                                "name": "文档",
                                "type": "official-doc",
                                "path": "doc.md",
                                "language": "zh",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            spec = load_catalog(root / "sources.json")[0]
            loaded = load_source(spec)
            self.assertEqual(spec.path, (root / "doc.md").resolve())
            self.assertEqual(spec.uri, (root / "doc.md").resolve().as_uri())
            self.assertEqual(loaded.version, loaded.content_hash[:12])
            self.assertIn("正文", loaded.content)

    def test_html_and_jsonl_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html_path = root / "page.html"
            html_path.write_text(
                "<style>x</style><h1>标题</h1><p>正文&amp;证据</p>",
                encoding="utf-8",
            )
            loaded = load_source(
                SourceSpec("html", "HTML", "docs", path=html_path)
            )
            self.assertEqual(loaded.content, "# 标题\n\n正文&证据")

            jsonl_path = root / "items.jsonl"
            jsonl_path.write_text(
                '{"text":"第一段"}\n{"content":"第二段"}\n', encoding="utf-8"
            )
            loaded = load_source(
                SourceSpec("jsonl", "JSONL", "dataset", path=jsonl_path)
            )
            self.assertEqual(loaded.content, "第一段\n\n第二段")

    @mock.patch("kg.sources.subprocess.run")
    def test_pdf_uses_pdftotext(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="第一页\f第二页", stderr=""
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.pdf"
            path.write_bytes(b"%PDF")
            loaded = load_source(
                SourceSpec("book", "Book", "textbook", path=path)
            )
        self.assertIn("\f", loaded.content)
        run.assert_called_once()

    @mock.patch("kg.sources.subprocess.run")
    @mock.patch("kg.sources.urllib.request.urlopen")
    def test_remote_pdf_is_downloaded_as_binary(self, urlopen, run):
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        response = mock.MagicMock()
        response.read.return_value = b"%PDF remote"
        response.headers = headers
        response.__enter__.return_value = response
        urlopen.return_value = response
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="远程 PDF 正文", stderr=""
        )
        loaded = load_source(
            SourceSpec(
                "remote-book",
                "Remote Book",
                "paper",
                uri="https://example.test/paper.pdf",
            )
        )
        self.assertEqual(loaded.content, "远程 PDF 正文")
        self.assertIsInstance(run.call_args.args[0][2], str)

    def test_chunking_has_overlap_and_page_locations(self):
        text = ("第一段。" * 80) + "\f" + ("第二段。" * 80)
        chunks = chunk_text(
            text,
            max_chars=220,
            overlap_chars=20,
            max_passage_chars=200,
        )
        self.assertGreater(len(chunks), 2)
        self.assertEqual([item.index for item in chunks], list(range(len(chunks))))
        self.assertTrue(any("page 2" in item.location for item in chunks))
        passage_ids = [
            passage.passage_id
            for chunk in chunks
            for passage in chunk.passages
        ]
        self.assertIn("P000001", passage_ids)
        self.assertTrue(all("[P" in chunk.text for chunk in chunks))

    def test_passages_preserve_actual_source_text_and_offsets(self):
        text = "第一段真实原文。\n\n第二段真实原文。"
        passages = segment_text(text)
        self.assertEqual(
            [(item.passage_id, item.text) for item in passages],
            [
                ("P000001", "第一段真实原文。"),
                ("P000002", "第二段真实原文。"),
            ],
        )
        for passage in passages:
            self.assertEqual(text[passage.start:passage.end], passage.text)

    def test_chunking_respects_markdown_section_boundaries(self):
        text = (
            "# 第 1 章\n\n章导言。\n\n"
            "## 1.1 线性回归\n\n第一节正文。\n\n"
            "## 1.2 Softmax 回归\n\n第二节正文。"
        )

        chunks = chunk_text(text, max_chars=8000, overlap_chars=500)

        self.assertEqual(
            [item.section_path for item in chunks],
            [
                ("第 1 章",),
                ("第 1 章", "1.1 线性回归"),
                ("第 1 章", "1.2 Softmax 回归"),
            ],
        )
        self.assertNotIn("第二节正文", chunks[1].text)
        self.assertIn("第 1 章 > 1.1 线性回归", chunks[1].location)

    def test_large_section_splits_only_inside_that_section(self):
        text = "# 章节甲\n\n" + ("甲节正文。" * 120) + "\n\n# 章节乙\n\n乙节正文。"

        chunks = chunk_text(
            text,
            max_chars=240,
            overlap_chars=20,
            max_passage_chars=200,
        )

        first = [item for item in chunks if item.section_path == ("章节甲",)]
        second = [item for item in chunks if item.section_path == ("章节乙",)]
        self.assertGreater(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertTrue(all("乙节正文" not in item.text for item in first))


if __name__ == "__main__":
    unittest.main()
