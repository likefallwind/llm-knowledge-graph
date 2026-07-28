from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, export, pipeline, resolution, store
from .llm import LLMConfig, MiniMaxM3LLM


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg",
        description="语料驱动的最小 AI 知识图谱流水线",
    )
    parser.add_argument(
        "--db",
        default=str(db.DEFAULT_DB),
        help=f"SQLite 路径（默认 {db.DEFAULT_DB}；不会修改旧 data/kg.db）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化最小数据库")
    sub.add_parser("status", help="查看核心对象与处理进度")
    sub.add_parser("check", help="检查外键、证据与关系循环")

    run = sub.add_parser("run", help="批量读取目录并运行核心闭环")
    run.add_argument("catalog", help="JSON/YAML 人工维护语料目录")
    run.add_argument("--source-limit", type=int)
    run.add_argument("--max-chunks", type=int)
    run.add_argument("--chunk-chars", type=int, default=8000)
    run.add_argument("--overlap-chars", type=int, default=500)
    run.add_argument("--max-entities", type=int, default=20)
    run.add_argument("--max-claims", type=int, default=30)
    run.add_argument("--stop-on-error", action="store_true")

    reconcile = sub.add_parser(
        "reconcile", help="重新判断疑似重复实体并安全合并"
    )
    reconcile.add_argument("--limit", type=int, default=20)

    dump = sub.add_parser("export", help="导出不含 Source 正文的 JSON 图谱")
    dump.add_argument("--out", default="out/graph.json")
    return parser


def _status(conn) -> dict:
    progress = {
        str(row["status"]): int(row["count"])
        for row in conn.execute(
            "SELECT status,COUNT(*) AS count FROM source_progress GROUP BY status"
        )
    }
    # 类型是 mention 级观察的汇总，不是实体的单值属性，因此下面按类型统计的
    # 实体数之和会大于实体总数——一个实体可以同时出现在多个类型下。
    entity_types = {
        str(row["entity_type"]): {
            "entities": int(row["entities"]),
            "observations": int(row["observations"]),
            "sources": int(row["sources"]),
        }
        for row in conn.execute(
            """
            SELECT observed_entity_type AS entity_type,
                   COUNT(DISTINCT entity_id) AS entities,
                   COUNT(*) AS observations,
                   COUNT(DISTINCT source_id) AS sources
            FROM evidence
            WHERE entity_id IS NOT NULL AND polarity='support'
              AND observed_entity_type<>''
            GROUP BY observed_entity_type ORDER BY observed_entity_type
            """
        )
    }
    multi_typed = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT entity_id FROM evidence
              WHERE entity_id IS NOT NULL AND polarity='support'
                AND observed_entity_type<>''
              GROUP BY entity_id HAVING COUNT(DISTINCT observed_entity_type) > 1
            )
            """
        ).fetchone()[0]
    )
    relations = {
        str(row["relation"]): int(row["count"])
        for row in conn.execute(
            """
            SELECT relation,COUNT(*) AS count
            FROM claims GROUP BY relation ORDER BY relation
            """
        )
    }
    return {
        "database": str(
            conn.execute("PRAGMA database_list").fetchone()["file"]
        ),
        "counts": store.counts(conn),
        "entity_types": entity_types,
        "multi_typed_entities": multi_typed,
        "relations": relations,
        "progress": progress,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        conn = db.connect(args.db)
        if args.command == "init":
            _json({"database": str(Path(args.db)), "counts": store.counts(conn)})
            return 0
        if args.command == "status":
            _json(_status(conn))
            return 0
        if args.command == "check":
            report = store.integrity_report(conn)
            _json(report)
            return 0 if report["ok"] else 1
        if args.command == "export":
            output = export.write_json(conn, args.out)
            _json({"output": str(output), "counts": store.counts(conn)})
            return 0

        llm = MiniMaxM3LLM(LLMConfig.from_env())
        if args.command == "run":
            result = pipeline.process_catalog(
                conn,
                llm,
                args.catalog,
                source_limit=args.source_limit,
                max_chunks=args.max_chunks,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                max_entities=args.max_entities,
                max_claims=args.max_claims,
                stop_on_error=args.stop_on_error,
            )
            _json(result)
            return 1 if result["failures"] else 0
        if args.command == "reconcile":
            _json(resolution.reconcile(conn, llm, limit=args.limit))
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
