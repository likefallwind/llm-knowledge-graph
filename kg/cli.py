from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    audit,
    db,
    definitions,
    expansion,
    export,
    observations,
    pipeline,
    resolution,
    store,
    viz,
)
from .llm import (
    DEFAULT_MAX_CONCURRENCY,
    LLMConcurrencyLimiter,
    LLMConfig,
    MiniMaxM3LLM,
)


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
    sub.add_parser("audit", help="查看图结构和拒绝原因统计")

    run = sub.add_parser("run", help="批量读取目录并运行核心闭环")
    run.add_argument("catalog", help="JSON/YAML 人工维护语料目录")
    run.add_argument("--source-limit", type=int)
    run.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="从该 Chunk index 开始处理（默认 0）",
    )
    run.add_argument(
        "--skip-section-summaries",
        action="store_true",
        help="跳过目录树自底向上的语料摘要（默认运行）",
    )
    run.add_argument(
        "--summary-limit",
        type=int,
        help="本次最多生成多少个 Section 摘要；默认不限",
    )
    run.add_argument(
        "--summary-workers",
        type=int,
        default=1,
        help="同一目录深度的摘要并发数；深度之间保持自底向上屏障（默认 1）",
    )
    run.add_argument("--max-chunks", type=int)
    run.add_argument("--chunk-chars", type=int, default=8000)
    run.add_argument("--overlap-chars", type=int, default=500)
    run.add_argument("--max-entities", type=int, default=50)
    run.add_argument("--max-claims", type=int, default=30)
    run.add_argument(
        "--chunk-workers",
        type=int,
        default=1,
        help="并行预取 Chunk 抽取数；消歧与数据库写入仍按原顺序串行（默认 1）",
    )
    run.add_argument(
        "--judge-workers",
        type=int,
        default=1,
        help="并行关系裁判数；数据库写入仍保持串行（默认 1）",
    )
    run.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=(
            "复杂/简单模型和所有处理阶段共享的请求总并发上限；"
            f"超过后等待（默认 {DEFAULT_MAX_CONCURRENCY}）"
        ),
    )
    run.add_argument("--stop-on-error", action="store_true")
    run.add_argument(
        "--skip-definition-synthesis",
        action="store_true",
        help="跳过基于全部 EntityObservation 的定义聚合（默认运行）",
    )
    run.add_argument(
        "--definition-limit",
        type=int,
        help="本次最多聚合多少个待更新 Entity；默认不限",
    )

    synthesize = sub.add_parser(
        "synthesize-definitions",
        help="基于每个 Entity 的全部 Observation 生成可追溯定义",
    )
    synthesize.add_argument(
        "--entity-id", type=int, action="append", dest="entity_ids"
    )
    synthesize.add_argument("--limit", type=int)

    reconcile = sub.add_parser(
        "reconcile", help="重新判断疑似重复实体并安全合并"
    )
    reconcile.add_argument("--limit", type=int, default=20)

    replay = sub.add_parser(
        "replay-pending",
        help="用已保存证据重放待定端点、实体晋升与 Claim 落实",
    )
    replay.add_argument("--limit", type=int)
    replay.add_argument("--promote-threshold", type=int, default=3)

    expand = sub.add_parser(
        "expand-relations",
        help="以同节/兄弟节为候选，补抽有 Passage 明证的开放关系",
    )
    expand.add_argument("--limit", type=int, default=50)

    dump = sub.add_parser("export", help="导出不含 Source 正文的 JSON 图谱")
    dump.add_argument("--out", default="out/graph.json")
    visualize = sub.add_parser("viz", help="生成可搜索、可追溯证据的交互式 HTML")
    visualize.add_argument("--out", default="out/graph.html")
    visualize.add_argument(
        "--view",
        choices=("semantic", "document", "mixed"),
        default="mixed",
        help="语义图、教材目录或混合视图（默认 mixed）",
    )
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
            SELECT t.canonical_name AS entity_type,
                   COUNT(DISTINCT o.entity_id) AS entities,
                   COUNT(*) AS observations,
                   COUNT(DISTINCT o.source_id) AS sources
            FROM entity_observation_types ot
            JOIN entity_observations o ON o.id=ot.observation_id
            JOIN entity_type_vocab t ON t.id=ot.type_id
            WHERE o.entity_id IS NOT NULL
            GROUP BY t.id,t.canonical_name ORDER BY t.canonical_name
            """
        )
    }
    multi_typed = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT o.entity_id FROM entity_observation_types ot
              JOIN entity_observations o ON o.id=ot.observation_id
              WHERE o.entity_id IS NOT NULL
              GROUP BY o.entity_id HAVING COUNT(DISTINCT ot.type_id) > 1
            )
            """
        ).fetchone()[0]
    )
    relations = {
        str(row["relation"]): {
            "claims": int(row["count"]),
            "kind": str(row["relation_kind"]),
        }
        for row in conn.execute(
            """
            SELECT c.relation,r.relation_kind,COUNT(*) AS count
            FROM claims c JOIN relation_types r ON r.id=c.relation_type_id
            GROUP BY c.relation,r.relation_kind ORDER BY c.relation
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
        "observations": observations.observation_report(conn),
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
        if args.command == "audit":
            _json(
                {
                    "graph": audit.graph_report(conn),
                    "rejections": audit.rejection_report(conn),
                    "observations": observations.observation_report(conn),
                }
            )
            return 0
        if args.command == "export":
            output = export.write_json(conn, args.out)
            _json({"output": str(output), "counts": store.counts(conn)})
            return 0
        if args.command == "viz":
            output = viz.write_html(conn, args.out, view=args.view)
            _json({"output": str(output), "view": args.view, "counts": store.counts(conn)})
            return 0

        limiter = LLMConcurrencyLimiter(
            getattr(args, "llm_max_concurrency", DEFAULT_MAX_CONCURRENCY)
        )
        llm = MiniMaxM3LLM(
            LLMConfig.from_env(role="complex"), limiter=limiter
        )
        simple_llm = MiniMaxM3LLM(
            LLMConfig.from_env(role="simple"), limiter=limiter
        )
        if args.command == "run":
            result = pipeline.process_catalog(
                conn,
                llm,
                args.catalog,
                source_limit=args.source_limit,
                start_chunk=args.start_chunk,
                max_chunks=args.max_chunks,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                max_entities=args.max_entities,
                max_claims=args.max_claims,
                chunk_workers=args.chunk_workers,
                judge_workers=args.judge_workers,
                stop_on_error=args.stop_on_error,
                synthesize_definitions=not args.skip_definition_synthesis,
                definition_limit=args.definition_limit,
                summarize_sections=not args.skip_section_summaries,
                summary_limit=args.summary_limit,
                summary_workers=args.summary_workers,
                simple_llm=simple_llm,
            )
            _json(result)
            return 1 if result["failures"] else 0
        if args.command == "synthesize-definitions":
            result = definitions.synthesize_pending(
                conn,
                llm,
                entity_ids=args.entity_ids,
                limit=args.limit,
            )
            _json(result)
            return 1 if result["failures"] else 0
        if args.command == "reconcile":
            _json(resolution.reconcile(conn, llm, limit=args.limit))
            return 0
        if args.command == "replay-pending":
            _json(
                observations.replay_pending(
                    conn,
                    llm,
                    limit=args.limit,
                    promote_threshold=args.promote_threshold,
                )
            )
            return 0
        if args.command == "expand-relations":
            _json(
                expansion.expand_relations(
                    conn, llm, limit=args.limit, simple_llm=simple_llm
                )
            )
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
