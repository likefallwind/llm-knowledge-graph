from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .llm import JSONLLM
from .models import SourcePassage


SUMMARY_PROMPT_VERSION = "section-bottom-up-1"
SUMMARY_SYSTEM = """你是教材结构摘要器，不是知识来源。
只能压缩给出的原始 Passage 和子节摘要，不得补充外部知识。只输出 JSON 对象。"""


def _section_key(path: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(path, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def sync_source_structure(
    conn: sqlite3.Connection,
    source_id: int,
    passages: Iterable[SourcePassage],
) -> dict[tuple[str, ...], int]:
    """Persist deterministic TOC nodes and Passage-to-section placement."""
    passage_list = list(passages)
    ordered_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for passage in passage_list:
        for depth in range(1, len(passage.section_path) + 1):
            prefix = passage.section_path[:depth]
            if prefix not in seen:
                seen.add(prefix)
                ordered_paths.append(prefix)

    ids: dict[tuple[str, ...], int] = {}
    children_seen: defaultdict[tuple[str, ...], int] = defaultdict(int)
    for path in ordered_paths:
        parent_path = path[:-1]
        ordinal = children_seen[parent_path]
        children_seen[parent_path] += 1
        conn.execute(
            """
            INSERT OR IGNORE INTO source_sections
            (source_id,section_key,parent_id,title,depth,ordinal,path_json)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                source_id,
                _section_key(path),
                ids.get(parent_path),
                path[-1],
                len(path),
                ordinal,
                json.dumps(path, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT id FROM source_sections WHERE source_id=? AND section_key=?",
            (source_id, _section_key(path)),
        ).fetchone()
        ids[path] = int(row["id"])

    for passage in passage_list:
        conn.execute(
            """
            INSERT INTO source_passages
            (source_id,passage_id,section_id,content_hash,start_offset,end_offset,location)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(source_id,passage_id) DO UPDATE SET
              section_id=excluded.section_id,
              content_hash=excluded.content_hash,
              start_offset=excluded.start_offset,
              end_offset=excluded.end_offset,
              location=excluded.location
            """,
            (
                source_id,
                passage.passage_id,
                ids.get(passage.section_path),
                passage.content_hash,
                passage.start,
                passage.end,
                passage.location,
            ),
        )
    conn.commit()
    return ids


def section_id_for_passages(
    conn: sqlite3.Connection, source_id: int, passage_ids: Iterable[str]
) -> int | None:
    ids = list(passage_ids)
    if not ids:
        return None
    row = conn.execute(
        "SELECT section_id FROM source_passages WHERE source_id=? AND passage_id=?",
        (source_id, ids[0]),
    ).fetchone()
    return int(row["section_id"]) if row and row["section_id"] is not None else None


def latest_summary(conn: sqlite3.Connection, section_id: int) -> str:
    row = conn.execute(
        "SELECT summary FROM section_summaries WHERE section_id=? ORDER BY id DESC LIMIT 1",
        (section_id,),
    ).fetchone()
    return str(row["summary"]) if row else ""


def context_for_section(conn: sqlite3.Connection, section_id: int | None) -> str:
    if section_id is None:
        return ""
    row = conn.execute(
        "SELECT parent_id,path_json FROM source_sections WHERE id=?", (section_id,)
    ).fetchone()
    if not row:
        return ""
    parts = ["目录路径：" + " > ".join(json.loads(str(row["path_json"])))]
    own = latest_summary(conn, section_id)
    if own:
        parts.append("本节摘要：" + own)
    if row["parent_id"] is not None:
        parent = latest_summary(conn, int(row["parent_id"]))
        if parent:
            parts.append("父节摘要：" + parent)
    return "\n".join(parts)


def summarize_source(
    conn: sqlite3.Connection,
    llm: JSONLLM,
    source_id: int,
    *,
    model: str,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Bottom-up summaries used only as extraction context.

    Sections at the same depth are independent and may call the LLM in
    parallel.  Each depth is fully persisted before its parents are prepared,
    and all SQLite access remains in the caller thread.
    """
    if workers < 1:
        raise ValueError("summary_workers 必须至少为 1")
    rows = conn.execute(
        "SELECT * FROM source_sections WHERE source_id=? ORDER BY depth DESC,id",
        (source_id,),
    ).fetchall()
    processed = skipped = failed = 0
    source = conn.execute("SELECT content FROM sources WHERE id=?", (source_id,)).fetchone()
    content = str(source["content"])
    rows_by_depth: defaultdict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        rows_by_depth[int(row["depth"])].append(row)

    def persist(
        task: dict[str, object], result: tuple[str, list[str]] | Exception
    ) -> None:
        nonlocal processed, failed
        if isinstance(result, Exception):
            failed += 1
            return
        summary, cited = result
        try:
            conn.execute(
                """INSERT INTO section_summaries
                   (section_id,input_fingerprint,summarizer_model,
                    prompt_version,summary,supporting_passage_ids)
                   VALUES (?,?,?,?,?,?)""",
                (
                    task["section_id"],
                    task["fingerprint"],
                    model,
                    SUMMARY_PROMPT_VERSION,
                    summary,
                    json.dumps(cited, ensure_ascii=False),
                ),
            )
            conn.commit()
            processed += 1
        except Exception:
            conn.rollback()
            failed += 1

    for depth in sorted(rows_by_depth, reverse=True):
        if limit is not None and processed >= limit:
            break
        tasks: list[dict[str, object]] = []
        for row in rows_by_depth[depth]:
            section_id = int(row["id"])
            passage_rows = conn.execute(
                "SELECT * FROM source_passages WHERE section_id=? ORDER BY passage_id",
                (section_id,),
            ).fetchall()
            child_rows = conn.execute(
                "SELECT id,title FROM source_sections WHERE parent_id=? ORDER BY ordinal,id",
                (section_id,),
            ).fetchall()
            raw = [
                {
                    "passage_id": str(item["passage_id"]),
                    "text": content[
                        int(item["start_offset"]):int(item["end_offset"])
                    ],
                }
                for item in passage_rows
            ]
            children = []
            for item in child_rows:
                child_summary = latest_summary(conn, int(item["id"]))
                if child_summary:
                    children.append(
                        {"title": str(item["title"]), "summary": child_summary}
                    )
            fingerprint = hashlib.sha256(
                json.dumps(
                    [raw, children], ensure_ascii=False, sort_keys=True
                ).encode()
            ).hexdigest()
            exists = conn.execute(
                """SELECT 1 FROM section_summaries
                   WHERE section_id=? AND input_fingerprint=?
                     AND summarizer_model=? AND prompt_version=?""",
                (section_id, fingerprint, model, SUMMARY_PROMPT_VERSION),
            ).fetchone()
            if exists or (not raw and not children):
                skipped += 1
                continue
            tasks.append(
                {
                    "section_id": section_id,
                    "fingerprint": fingerprint,
                    "raw": raw,
                    "children": children,
                }
            )

        task_offset = 0
        while task_offset < len(tasks):
            if limit is not None and processed >= limit:
                break
            batch_size = len(tasks) - task_offset
            if limit is not None:
                batch_size = min(batch_size, limit - processed)
            batch = tasks[task_offset:task_offset + batch_size]
            task_offset += batch_size
            if workers == 1:
                for task in batch:
                    persist(task, _generate_summary(llm, task))
            else:
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(batch))
                ) as executor:
                    future_tasks = {
                        executor.submit(_generate_summary, llm, task): task
                        for task in batch
                    }
                    for future in as_completed(future_tasks):
                        persist(future_tasks[future], future.result())
    return {"processed": processed, "skipped": skipped, "failed": failed}


def _generate_summary(
    llm: JSONLLM, task: dict[str, object]
) -> tuple[str, list[str]] | Exception:
    """Call the LLM without touching SQLite so this is safe in a worker."""
    raw = task["raw"]
    children = task["children"]
    try:
        payload = llm.complete_json(
            SUMMARY_SYSTEM,
            """生成简洁摘要，说明本节实际讲了什么以及出现的关键术语。
摘要只能来自 inputs。返回 {"summary":"...","passage_ids":["P..."]}；
passage_ids 只能引用 inputs.passages，父级摘要可以不引用新的 Passage。
inputs=%s"""
            % json.dumps(
                {"passages": raw, "children": children}, ensure_ascii=False
            ),
        )
        summary = str(payload.get("summary", "")).strip()
        cited = payload.get("passage_ids", [])
        allowed = {str(item["passage_id"]) for item in raw}  # type: ignore[index]
        if not isinstance(cited, list):
            cited = []
        valid_citations = [str(item) for item in cited if str(item) in allowed]
        if not summary:
            raise ValueError("摘要为空")
        return summary, valid_citations
    except Exception as exc:
        return exc
