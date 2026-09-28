"""Exercise the unified entry point with real SQLite and a deterministic LLM."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kg import cli, db, runner, sources
from kg.llm import LLMConfig
from tests.helpers import FakeLLM


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.database = self.root / 'graph.db'
        self.catalog = self.root / 'catalog.json'
        self.write_catalog(1)
        self.config = mock.patch('kg.runner.LLMConfig.from_env', return_value=LLMConfig(
            base_url='https://example.invalid', api_key='test-secret', model='fake',
        ))
        self.config.start()
        self.addCleanup(self.config.stop)

    def write_catalog(self, count):
        rows = []
        for i in range(count):
            (self.root / f'{i}.txt').write_text(f'Plain source number {i}.')
            rows.append(dict(key=str(i), name=f'Source {i}', type='document', path=f'{i}.txt'))
        self.catalog.write_text(json.dumps(dict(sources=rows)))

    def args(self, *extra):
        return cli._parser().parse_args([
            '--db', str(self.database), 'run', str(self.catalog), '--retry-delay', '0', *extra,
        ])

    def execute(self, client, *extra):
        with mock.patch('kg.runner.MiniMaxM3LLM', return_value=client):
            return runner.run(self.args(*extra))

    def test_fresh_resume_and_artifacts_without_seed(self):
        client = FakeLLM({'entities': [], 'claims': []})
        report, code = self.execute(client, '--fresh')
        self.assertEqual(code, 0)
        self.assertEqual(report['completed'][0]['done_chunks'], 1)
        self.assertTrue((Path(report['run_dir']) / '.finished').exists())
        self.assertNotIn('test-secret', (Path(report['run_dir']) / 'manifest.json').read_text())
        rerun, code = self.execute(FakeLLM())
        self.assertEqual(code, 0)
        self.assertEqual(rerun['completed'][0]['skipped_chunks'], 1)
        self.assertNotEqual(report['run_dir'], rerun['run_dir'])
        with self.assertRaisesRegex(ValueError, '--fresh'):
            self.execute(FakeLLM(), '--fresh')
        with db.connect(self.database) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM sources').fetchone()[0], 1)

    def test_retry_preserves_success_and_does_not_expand_chunk_budget(self):
        self.write_catalog(3)

        class FailsOnce(FakeLLM):
            def complete_json(self, *args, **kwargs):
                if len(self.calls) == 1:
                    self.calls.append(('failed', ''))
                    raise RuntimeError('temporary outage')
                return super().complete_json(*args, **kwargs)

        client = FailsOnce({'entities': [], 'claims': []}, {'entities': [], 'claims': []})
        report, code = self.execute(client, '--max-chunks', '2')
        self.assertEqual(code, 3)
        self.assertEqual(sum(s['done_chunks'] for s in report['completed']), 2)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue((Path(report['run_dir']) / 'pass-2.json').exists())
        self.assertFalse((Path(report['run_dir']) / '.finished').exists())

    def test_summary_failure_blocks_extraction_until_retry_succeeds(self):
        client = FakeLLM({'entities': [], 'claims': []})
        with mock.patch('kg.pipeline.structure.summarize_source', side_effect=[
            dict(processed=0, skipped=0, failed=1), dict(processed=1, skipped=0, failed=0),
        ]):
            report, code = self.execute(client)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.calls), 1)
        first = json.loads((Path(report['run_dir']) / 'pass-1.json').read_text())
        self.assertTrue(first['failures'])

    def test_exhausted_retries_cannot_write_complete_marker(self):
        report, code = self.execute(FakeLLM(), '--max-passes', '2')
        self.assertEqual(code, 1)
        self.assertEqual(report['stage'], 'failed')
        self.assertFalse((Path(report['run_dir']) / '.finished').exists())

    def test_lock_rejects_second_writer_before_database_creation(self):
        with runner.database_lock(self.database):
            with self.assertRaisesRegex(RuntimeError, '正在运行'):
                self.execute(FakeLLM())
        self.assertFalse(self.database.exists())

    def test_unknown_source_rejected_without_database(self):
        with self.assertRaisesRegex(ValueError, '未知 source'):
            self.execute(FakeLLM(), '--source-key', 'missing')
        self.assertFalse(self.database.exists())

    def test_chunk_budget_does_not_summarize_unselected_sources(self):
        self.write_catalog(3)
        with mock.patch('kg.pipeline.structure.summarize_source', return_value=dict(
            processed=0, skipped=0, failed=0,
        )) as summarize:
            self.execute(FakeLLM({'entities': [], 'claims': []}), '--max-chunks', '1')
        self.assertEqual(summarize.call_count, 1)

    def test_source_selection_and_definition_failure_retry(self):
        self.write_catalog(2)
        empty = dict(processed=[], skipped=[], failures=[], remaining=0)
        with mock.patch('kg.pipeline.definitions.synthesize_pending', side_effect=[
            dict(empty, failures=[dict(entity_id=1, error='temporary')]), empty,
        ]):
            report, code = self.execute(FakeLLM({'entities': [], 'claims': []}), '--source-key', '1')
        self.assertEqual(code, 0)
        self.assertEqual([s['source_key'] for s in report['completed']], ['1'])

    def test_integrity_failure_is_not_complete(self):
        with mock.patch('kg.runner.store.integrity_report', return_value={'ok': False}):
            report, code = self.execute(FakeLLM({'entities': [], 'claims': []}))
        self.assertEqual(code, 1)
        self.assertFalse((Path(report['run_dir']) / '.finished').exists())

    def test_prepared_headings_preserve_physical_pages_and_validate_hash(self):
        text = '\f\fSpecial heading\n\nBody text.'
        (self.root / '0.txt').write_text(text)
        (self.root / 'headings.json').write_text(json.dumps([
            dict(marker='Special heading', level=1, title='Verified chapter'),
        ]))
        payload = json.loads(self.catalog.read_text())
        payload['sources'][0].update(headings_path='headings.json', expected_chunks=1,
                                    content_sha256=hashlib.sha256(text.encode()).hexdigest())
        self.catalog.write_text(json.dumps(payload))
        spec = sources.load_catalog(self.catalog)[0]
        loaded = sources.load_source(spec)
        chunks = sources.chunk_text(loaded.content, headings=spec.headings)
        self.assertEqual(chunks[0].section_path, ('Verified chapter',))
        self.assertIn('page 3,', chunks[0].location)
        (self.root / '0.txt').write_text('Changed source')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            sources.load_source(spec)

    def test_context_change_reprocesses_done_chunk(self):
        client = FakeLLM({'entities': [], 'claims': []}, {'entities': [], 'claims': []})
        self.execute(client)
        with mock.patch('kg.pipeline.structure.context_for_section', return_value='Updated summary'):
            report, code = self.execute(client)
        self.assertEqual(code, 0)
        self.assertEqual(report['completed'][0]['processed_chunks'], 1)
        self.assertEqual(len(client.calls), 2)

    def test_interruption_records_status_and_allows_resume(self):
        client = mock.Mock()
        client.complete_json.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.execute(client)
        histories = list(self.root.glob('graph.db.runs/*'))
        self.assertEqual(len(histories), 1)
        status = json.loads((histories[0] / 'status.json').read_text())
        self.assertEqual(status['stage'], 'interrupted')
        self.assertFalse((histories[0] / '.finished').exists())
        report, code = self.execute(FakeLLM({'entities': [], 'claims': []}))
        self.assertEqual(code, 0)

    def test_expected_chunk_mismatch_never_calls_llm(self):
        payload = json.loads(self.catalog.read_text())
        payload['sources'][0]['expected_chunks'] = 2
        self.catalog.write_text(json.dumps(payload))
        client = FakeLLM()
        report, code = self.execute(client, '--max-passes', '1')
        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 0)
        self.assertIn('expected_chunks', report['failures'][0]['error'])

    def test_malformed_headings_fail_before_database_creation(self):
        (self.root / 'headings.json').write_text('[{"marker": "title"}]')
        payload = json.loads(self.catalog.read_text())
        payload['sources'][0]['headings_path'] = 'headings.json'
        self.catalog.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            self.execute(FakeLLM())
        self.assertFalse(self.database.exists())
