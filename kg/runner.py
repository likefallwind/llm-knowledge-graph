"""Book-independent orchestration; checkpoints remain in the target database."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import time
from uuid import uuid4

from . import db, pipeline, sources, store
from .llm import (
    LLMConfig,
    LLMConcurrencyLimiter,
    MiniMaxM3LLM,
    UsageLog,
    usage_totals,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def database_lock(path: Path):
    """Lock before opening SQLite; all unified runners share this adjacent lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'数据库已有统一入口任务正在运行: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args):
    """Return (report, exit code): 0 complete, 1 failed, 3 partial."""
    for name in ('max_passes', 'summary_workers', 'chunk_workers', 'judge_workers', 'relation_workers',
                 'llm_max_concurrency', 'max_entities', 'max_claims'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} 必须至少为 1')
    for name in ('source_limit', 'max_chunks', 'summary_limit', 'definition_limit',
                 'start_chunk', 'retry_delay', 'api_retry_delay', 'request_retries',
                 'failure_pause_seconds', 'max_api_retries'):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f'{name} 不能为负数')
    if args.request_timeout <= 0 or args.chunk_chars < 200 or not 0 <= args.overlap_chars < args.chunk_chars:
        raise ValueError('请求超时、chunk_chars 或 overlap_chars 无效')
    catalog = Path(args.catalog).resolve()
    specs = sources.load_catalog(catalog)
    if args.source_key:
        missing = set(args.source_key) - {spec.key for spec in specs}
        if missing:
            raise ValueError(f'未知 source key: {sorted(missing)}')
    configs = []
    for role in ('complex', 'simple'):
        config = LLMConfig.from_env(role=role)
        configs.append(replace(
            config, model=getattr(args, role + '_model') or config.model,
            base_url=args.base_url or config.base_url,
            timeout=args.request_timeout, retries=args.request_retries,
            api_retry_delay=args.api_retry_delay,
            max_api_retries=args.max_api_retries,
        ))
    database = Path(args.db).resolve()
    run_root = Path(args.run_dir).resolve() if args.run_dir else database.parent / (database.name + '.runs')
    # Each invocation has its own history. Repeating a command resumes via SQLite.
    run_dir = run_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8])
    with database_lock(database):
        if args.fresh and database.exists():
            raise ValueError('--fresh 要求数据库路径不存在；不会删除或覆盖已有库')
        run_dir.mkdir(parents=True)
        usage_log = UsageLog(run_dir / 'usage.jsonl')

        def usage():
            # Resumed invocations write separate run directories; the graph total spans all of them.
            return dict(this_run=usage_log.totals(),
                        all_runs=usage_totals(sorted(run_root.glob('*/usage.jsonl'))))

        handler = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
        logger = logging.getLogger('kg')
        previous_level = logger.level
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        conn = None
        try:
            manifest = dict(
                started_at=_now(), database=str(database), catalog=str(catalog),
                catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                options=vars(args),
            models=[dict(model=c.model, base_url=c.base_url, timeout=c.timeout,
                         retries=c.retries, api_retry_delay=c.api_retry_delay,
                         max_api_retries=c.max_api_retries) for c in configs],
                code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob('*.py'))},
            )
            _save(run_dir / 'manifest.json', manifest)
            _save(run_dir / 'status.json', dict(stage='running', updated_at=_now()))
            conn = db.connect(database)
            limiter = LLMConcurrencyLimiter(args.llm_max_concurrency)
            llm, simple_llm = [MiniMaxM3LLM(c, limiter=limiter, usage_log=usage_log) for c in configs]
            options = {name: getattr(args, name) for name in (
                'source_limit', 'start_chunk', 'max_chunks', 'chunk_chars', 'overlap_chars',
                'max_entities', 'max_claims', 'chunk_workers', 'judge_workers',
                'relation_workers',
                'stop_on_error', 'definition_limit', 'summary_limit', 'summary_workers',
                'failure_pause_seconds',
            )}
            options.update(
                source_keys=args.source_key,
                summarize_sections=not args.skip_section_summaries,
                synthesize_definitions=not args.skip_definition_synthesis,
                simple_llm=simple_llm,
                work_selection=set(),
                source_cache={},
            )
            for attempt in range(1, args.max_passes + 1):
                logger.info('Pipeline pass %s/%s', attempt, args.max_passes)
                _save(run_dir / 'status.json', dict(stage='running', attempt=attempt, updated_at=_now()))

                def progress(source_id, chunk_index, status):
                    _save(run_dir / 'status.json', dict(
                        stage='chunks', attempt=attempt, source_id=source_id,
                        chunk_index=chunk_index, chunk_status=status, updated_at=_now(),
                    ))

                result = pipeline.process_catalog(conn, llm, catalog, on_progress=progress, **options)
                _save(run_dir / f'pass-{attempt}.json', result)
                if not result['failures'] or args.stop_on_error:
                    break
                if attempt < args.max_passes:
                    time.sleep(args.retry_delay)
                    continue
                break
            integrity = store.integrity_report(conn)
            quick_check = [row[0] for row in conn.execute('PRAGMA quick_check')]
            _save(run_dir / 'final-check.json', dict(integrity=integrity, quick_check=quick_check))
            failed = bool(result['failures']) or not integrity['ok'] or quick_check != ['ok']
            # Limits/skips describe an intentionally bounded invocation, never full completion.
            partial = (
                any(getattr(args, n) is not None for n in
                    ('source_limit', 'max_chunks', 'summary_limit', 'definition_limit'))
                or args.start_chunk > 0 or args.skip_section_summaries or args.skip_definition_synthesis
                or not result['completed']
                or any(s['done_chunks'] != s['expected_chunks'] for s in result['completed'])
                or result['definition_synthesis']['remaining'] > 0
            )
            stage, code = ('failed', 1) if failed else ('partial', 3) if partial else ('complete', 0)
            report = dict(result, stage=stage, exit_code=code, run_dir=str(run_dir),
                          database=str(database), finished_at=_now(), integrity_ok=integrity['ok'],
                          usage=usage())
            _save(run_dir / 'summary.json', report)
            _save(run_dir / 'status.json', dict(stage=stage, updated_at=_now(), exit_code=code))
            (run_dir / '.exit').write_text(str(code) + '\n')
            if stage == 'complete':
                (run_dir / '.finished').write_text(_now() + '\n')
            return report, code
        except BaseException as exc:
            # Credentials are deliberately absent from manifests and exception summaries.
            message = str(exc)
            for config in configs:
                message = message.replace(config.api_key, '[REDACTED]')
            _save(run_dir / 'status.json', dict(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                              updated_at=_now(), error=message, usage=usage()))
            (run_dir / '.exit').write_text('130\n' if isinstance(exc, KeyboardInterrupt) else '2\n')
            raise
        finally:
            if conn is not None:
                conn.close()
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(previous_level)
