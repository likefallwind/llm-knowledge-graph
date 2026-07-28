from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from .models import LoadedSource, SourceSpec, TextChunk


def load_catalog(path: str | Path) -> list[SourceSpec]:
    catalog_path = Path(path)
    payload = _load_mapping(catalog_path)
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("语料目录的 sources 必须是数组")
    specs: list[SourceSpec] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ValueError(f"sources[{index}] 必须是对象")
        source_path = raw.get("path")
        resolved_path = None
        if source_path:
            candidate = Path(str(source_path))
            resolved_path = (
                candidate
                if candidate.is_absolute()
                else (catalog_path.parent / candidate).resolve()
            )
        uri = str(raw.get("uri", "")).strip()
        if not uri and resolved_path is not None:
            uri = resolved_path.as_uri()
        name = str(raw.get("name", "")).strip()
        source_type = str(raw.get("type", "")).strip()
        key = str(raw.get("key", "")).strip() or uri or str(resolved_path or name)
        if not name or not source_type or (resolved_path is None and not uri):
            raise ValueError(
                f"sources[{index}] 需要 name、type，以及 path 或 uri"
            )
        specs.append(
            SourceSpec(
                key=key,
                name=name,
                source_type=source_type,
                uri=uri,
                path=resolved_path,
                version=str(raw.get("version", "")).strip(),
                language=str(raw.get("language", "")).strip(),
            )
        )
    return specs


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "读取 YAML 目录需要 PyYAML；也可以改用 JSON 目录"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError("语料目录只支持 .json、.yaml 或 .yml")
    if not isinstance(payload, dict):
        raise ValueError("语料目录顶层必须是对象")
    return payload


def load_source(spec: SourceSpec, *, timeout: float = 60.0) -> LoadedSource:
    if spec.path is not None:
        content = _read_path(spec.path)
    else:
        request = urllib.request.Request(
            spec.uri, headers={"User-Agent": "llm-knowledge-graph/0.1"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
        if content_type == "application/pdf" or spec.uri.lower().endswith(".pdf"):
            content = _read_pdf_bytes(raw)
        else:
            text = raw.decode(charset, errors="replace")
            content = _html_to_text(text) if content_type == "text/html" else text
    content = content.replace("\x00", "").strip()
    if not content:
        raise ValueError(f"语料为空: {spec.name}")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return LoadedSource(
        spec=spec,
        content=content,
        content_hash=digest,
        version=spec.version or digest[:12],
    )


def _read_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"语料文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    text = path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        return _html_to_text(text)
    if suffix == ".jsonl":
        lines: list[str] = []
        for index, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if not isinstance(value, str):
                    raise ValueError(f"{path}:{index} 缺少 text/content 字符串")
                lines.append(value)
            else:
                raise ValueError(f"{path}:{index} 必须是字符串或对象")
        return "\n\n".join(lines)
    return text


def _read_pdf(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            return proc.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF 读取需要系统命令 pdftotext，或安装可选依赖 pypdf"
        ) from exc
    pages = [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
    return "\f".join(pages)


def _read_pdf_bytes(raw: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(raw)
            handle.flush()
            proc = subprocess.run(
                ["pdftotext", "-layout", handle.name, "-"],
                check=True,
                capture_output=True,
                text=True,
            )
        if proc.stdout.strip():
            return proc.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF 读取需要系统命令 pdftotext，或安装可选依赖 pypdf"
        ) from exc
    pages = [(page.extract_text() or "") for page in PdfReader(BytesIO(raw)).pages]
    return "\f".join(pages)


class _HTMLTextParser:
    """Small dependency-free HTML cleaner for manually curated sources."""

    BLOCKS = re.compile(
        r"</?(?:p|div|section|article|h[1-6]|li|br|tr|blockquote)[^>]*>",
        re.IGNORECASE,
    )
    TAGS = re.compile(r"<[^>]+>")
    HIDDEN = re.compile(
        r"<(?:script|style|noscript)[^>]*>.*?</(?:script|style|noscript)>",
        re.IGNORECASE | re.DOTALL,
    )


def _html_to_text(value: str) -> str:
    visible = _HTMLTextParser.HIDDEN.sub("", value)
    visible = _HTMLTextParser.BLOCKS.sub("\n", visible)
    visible = _HTMLTextParser.TAGS.sub("", visible)
    visible = html.unescape(visible)
    lines = [re.sub(r"\s+", " ", line).strip() for line in visible.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk_text(
    text: str, *, max_chars: int = 8000, overlap_chars: int = 500
) -> list[TextChunk]:
    if max_chars < 200:
        raise ValueError("max_chars 至少为 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须满足 0 <= overlap < max_chars")
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + max_chars)
        end = hard_end
        if hard_end < length:
            candidates = [
                text.rfind("\n\n", start + max_chars // 2, hard_end),
                text.rfind("\n", start + max_chars // 2, hard_end),
                text.rfind("。", start + max_chars // 2, hard_end),
                text.rfind(". ", start + max_chars // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if text[boundary] != "\n" else 0)
        chunk = text[start:end].strip()
        if chunk:
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            page = text.count("\f", 0, start) + 1
            page_end = text.count("\f", 0, end) + 1
            page_label = (
                f"page {page}" if page == page_end else f"pages {page}-{page_end}"
            )
            chunks.append(
                TextChunk(
                    index=index,
                    text=chunk,
                    location=f"{page_label}, chars {start}-{end}",
                    content_hash=digest,
                )
            )
            index += 1
        if end >= length:
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return chunks
