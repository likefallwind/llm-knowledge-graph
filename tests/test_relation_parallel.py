from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from kg import db, extraction, llm, pipeline, resolution, store, validation, vocabulary
from tests.helpers import FakeLLM


RAW_RELATIONS = ['is_a', 'help1', 'help2', 'uncertain', 'bad_projection', 'unfaithful', 'invalid_same']


def extraction_payload():
    return {
        'entities': [dict(name=name, definition=name + '是一种概念', entity_type='concept',
                          aliases=[], evidence=dict(passage_ids=['P000001'], quote=name))
                     for name in ['甲', '乙']],
        'claims': [dict(subject='甲', object='乙', relation=raw,
                        statement='甲通过' + raw + '指向乙', scope='', scope_is_restrictive=False,
                        evidence=dict(passage_ids=['P000001'], quote='甲与乙'))
                   for raw in RAW_RELATIONS],
    }


class RoutedLLM:
    """Responses depend on request content, never on thread completion order."""
    def __init__(self, errors=None, cap=6):
        self.errors = errors or {}
        self.calls = []
        self.finished = []
        self.lock = threading.Lock()
        self.limiter = llm.LLMConcurrencyLimiter(cap)
        self.active = self.peak = 0
        self.thread_ids = set()

    def complete_json(self, system, user, *, validate=None):
        with self.lock:
            self.calls.append((system, user))
        if system == extraction.SYSTEM_PROMPT:
            payload = extraction_payload()
        elif system == resolution.RESOLUTION_SYSTEM:
            data = json.loads(user.split('新观察：\n', 1)[1].split('\n\n候选：', 1)[0])
            payload = dict(decision='new', canonical_name=data['name'], reason='new object')
        elif system == vocabulary.SYSTEM:
            observation = json.loads(user.split('观察=', 1)[1].split('\n候选=', 1)[0])
            raw = observation['raw_relation']
            with self.limiter.slot():
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    self.thread_ids.add(threading.get_ident())
                try:
                    # Later requests return first; serial application must still match.
                    time.sleep(0.008 * (len(RAW_RELATIONS) - RAW_RELATIONS.index(raw)))
                    if raw in self.errors:
                        raise self.errors[raw]
                    decision = ('same' if raw in ('is_a', 'invalid_same') else
                                'uncertain' if raw == 'uncertain' else
                                'non_projectable' if raw == 'bad_projection' else 'new')
                    payload = dict(decision=decision, candidate_id=1 if raw == 'is_a' else 999,
                                   canonical_name='unfaithful' if raw == 'unfaithful' else 'helps',
                                   relation_kind='other', description='甲帮助乙',
                                   projection_statement='甲帮助乙', register_alias=True, reason=raw)
                    with self.lock:
                        self.finished.append(raw)
                finally:
                    with self.lock:
                        self.active -= 1
        elif system == validation.VALIDATION_SYSTEM:
            payload = dict(assertion_verdict='supports', projection_statement='甲指向乙',
                           projection_faithful='"relation": "unfaithful"' not in user,
                           reason='source and projection checked')
        else:
            raise AssertionError('Unexpected model stage: ' + system[:50])
        # Exercise exactly the same schema validator / regeneration contract.
        return FakeLLM(payload, payload).complete_json(system, user, validate=validate)


def snapshot(conn):
    """All persistent state, including IDs, audit records, and progress, sans clocks."""
    result = {}
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for table in tables:
        columns = [r[1] for r in conn.execute('PRAGMA table_info(' + table + ')')
                   if r[1] not in ('created_at', 'updated_at')]
        result[table] = conn.execute('SELECT ' + ','.join(columns) + ' FROM ' + table + ' ORDER BY rowid').fetchall()
        result[table] = [tuple(row) for row in result[table]]
    return result


class ParallelRelationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'source.txt').write_text('甲与乙是概念。甲通过不同关系指向乙。')
        self.catalog = self.root / 'catalog.json'
        self.catalog.write_text(json.dumps({'sources': [dict(key='test', name='test', type='test', path='source.txt')]}))
        self.embedding = mock.patch('kg.embeddings.cosine_scores', side_effect=lambda q, items: [0.0] * len(items))
        self.embedding.start()
        self.addCleanup(self.embedding.stop)

    def execute(self, name, workers, client):
        conn = db.connect(self.root / (name + '.db'))
        self.addCleanup(conn.close)
        sql_threads = set()
        conn.set_trace_callback(lambda sql: sql_threads.add(threading.get_ident()))
        result = pipeline.process_catalog(conn, client, self.catalog, summarize_sections=False,
                                          relation_workers=workers, failure_pause_seconds=0)
        self.assertEqual(sql_threads, {threading.get_ident()})
        return conn, result

    def test_parallel_matches_serial_inputs_graph_audit_and_order(self):
        serial, parallel = RoutedLLM(), RoutedLLM(cap=2)
        a, ra = self.execute('serial', 1, serial)
        b, rb = self.execute('parallel', 6, parallel)
        self.assertFalse(ra['failures'])
        self.assertEqual(ra, rb)
        self.assertEqual(snapshot(a), snapshot(b))
        self.assertEqual(Counter(serial.calls), Counter(parallel.calls))
        self.assertEqual(parallel.peak, 2)
        self.assertGreater(len(parallel.thread_ids), 1)
        self.assertNotEqual(parallel.finished, serial.finished)
        self.assertTrue(store.integrity_report(b)['ok'])
        # Both new proposals collide at finalization, preserving original aliases and IDs.
        self.assertEqual(b.execute("SELECT COUNT(*) FROM relation_types WHERE canonical_name='helps'").fetchone()[0], 1)
        self.assertEqual(b.execute("SELECT COUNT(*) FROM relation_aliases WHERE name IN ('help1','help2')").fetchone()[0], 2)
        self.assertEqual(b.execute('SELECT COUNT(*) FROM claims').fetchone()[0], 2)

    def test_failures_roll_back_identically_then_resume_without_duplicates(self):
        for raw, error in [('is_a', TimeoutError('timeout')), ('help2', ValueError('invalid JSON')),
                           ('invalid_same', RuntimeError('service unavailable'))]:
            with self.subTest(raw=raw):
                a, ra = self.execute(raw + '-s', 1, RoutedLLM({raw: error}))
                b, rb = self.execute(raw + '-p', 6, RoutedLLM({raw: error}))
                self.assertTrue(ra['failures'])
                self.assertEqual(ra, rb)
                self.assertEqual(snapshot(a), snapshot(b))
                self.assertEqual(b.execute('SELECT COUNT(*) FROM claims').fetchone()[0], 0)
                self.assertEqual(b.execute('SELECT COUNT(*) FROM claim_observations').fetchone()[0], len(RAW_RELATIONS))
                for conn, workers in [(a, 1), (b, 6)]:
                    fixed = pipeline.process_catalog(conn, RoutedLLM(), self.catalog,
                                                     summarize_sections=False, relation_workers=workers)
                    self.assertFalse(fixed['failures'])
                self.assertEqual(snapshot(a), snapshot(b))
                self.assertTrue(store.integrity_report(b)['ok'])

    def test_errors_are_reported_in_input_order(self):
        errors = {'is_a': ValueError('first input failed'), 'help2': TimeoutError('later input failed')}
        _, result = self.execute('two-errors', 6, RoutedLLM(errors))
        self.assertEqual(result['failures'][0]['error'], 'first input failed')

    def test_worker_change_reuses_done_fingerprint_without_calls(self):
        conn, _ = self.execute('resume', 1, RoutedLLM())
        client = RoutedLLM()
        result = pipeline.process_catalog(conn, client, self.catalog, summarize_sections=False, relation_workers=6)
        self.assertFalse(result['failures'])
        self.assertEqual(result['completed'][0]['skipped_chunks'], 1)
        self.assertEqual(client.calls, [])

    def test_interrupt_leaves_original_transaction_boundary(self):
        states = []
        for workers in (1, 6):
            conn = db.connect(self.root / f'interrupt-{workers}.db')
            try:
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.process_catalog(conn, RoutedLLM({'help2': KeyboardInterrupt()}), self.catalog,
                                             summarize_sections=False, relation_workers=workers)
                conn.rollback()
                states.append(snapshot(conn))
            finally:
                conn.close()
        self.assertEqual(states[0], states[1])
