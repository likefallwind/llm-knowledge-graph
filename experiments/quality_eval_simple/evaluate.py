"""Two questions per record: correctness and submitted-evidence sufficiency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / 'tmp/d2l-quality-v3-pilot-r3-20260912'
DOCUMENTS = Path('/home/likefallwind/code/llm-graph-benchmark/outputs/d2l-full1105-vnext-20260826/documents.jsonl')
PROMPT = '''核对教材抽取结果，只回答两件事。
1. correctness：结合 reference_context，内容正确用 correct；内容与原文矛盾、对象/方向错或明显虚构用 incorrect；依据不足或确有歧义用 uncertain。允许合理同义改写、领域术语的正常理解和原文直接推论，不要求逐字相同；不能补写原文未表达的实质事实。
2. evidence：只看 submitted_sources，引用支持 target 为真吗？支持用 supported；不支持或反驳 target 用 not_supported；确有歧义用 uncertain。能凭引用发现错误，不等于引用支持这条错误内容。reference_context 的额外段落不能替系统补引用。
括号可能是限定或解释，不自动表示同义词；合理的含义限定不因没有逐字出现就判错或不确定。“A 之前 B”按 A 先于 B 理解。区分缺少依据与内容错误，不因引用无关就自动把内容判错。
按本题 focus 核验，不评价类型标签、别名数量、关系粒度或图谱规模。引用只出现名字不等于支持完整内容。
只返回以下三行标签，reason 用简短中文并注明来源段落编号；不要 JSON，不要额外段落：
<correctness>correct或incorrect或uncertain</correctness>
<evidence>supported或not_supported或uncertain</evidence>
<reason>具体理由</reason>'''
FOCUS = {
    'entity': '检查实体名称所指和边界，以及已经提供的定义。没有定义只检查名称所指，不将缺失定义当成错误。代码对象、事件和示例也可以是实体。',
    'relation': '只检查主语与宾语间的核心关系及方向、否定和必要条件。description 帮助解释关系含义，但本题不评价其中额外细节。related_to 可以正确，描述中已保存的具体关系应正常考虑；明确错误的谓词不能被正确描述修复。',
    'assertion': '检查整条断言：核心关系和 description 中的全部实质陈述，保留否定和影响成立的条件；不补写输出未表达的内容。',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration():
    source = {'id': 'C1', 'text': '序列搜索策略包括贪心搜索、穷举搜索和束搜索。'}
    target = dict(subject='序列搜索策略', predicate='包含', object='贪心搜索（序列解码策略）', description='序列搜索策略包括贪心搜索（序列解码策略）。', scope='', polarity='positive')
    base = dict(kind='assertion', focus=FOCUS['assertion'], target=target, submitted_sources=[source], reference_context=[source])
    cases = []
    def add(name, p, correctness, evidence):
        cases.append(dict(id='cal-' + name, payload=p, expected=dict(correctness=correctness, evidence=evidence)))
    add('parenthetical-qualifier', deepcopy(base), 'correct', 'supported')
    p = deepcopy(base); p['target'].update(subject='生成文件', predicate='之前', object='提交文件', description='生成文件 之前 提交文件')
    p['submitted_sources'] = p['reference_context'] = [{'id': 'C2', 'text': '先生成文件，然后提交文件。'}]
    add('before', deepcopy(p), 'correct', 'supported')
    p['target'].update(subject='提交文件', object='生成文件', description='提交文件 之前 生成文件')
    add('reversed-before', p, 'incorrect', 'not_supported')
    p = deepcopy(base); p['submitted_sources'] = [{'id': 'C3', 'text': '本节说明如何安装软件。'}]
    add('wrong-citation', p, 'correct', 'not_supported')
    p = deepcopy(base); p['submitted_sources'].append({'id': 'C3', 'text': '本节说明如何安装软件。'})
    add('extra-citation', p, 'correct', 'supported')
    p = deepcopy(base); p['target'].update(predicate='related_to', description='')
    add('generic-relation', p, 'correct', 'supported')
    p = deepcopy(base); p['target'].update(predicate='related_to')
    add('generic-with-description', p, 'correct', 'supported')
    p = deepcopy(base); p['target'].update(subject='算法A', predicate='保证收敛到', object='最优解', description='算法A无条件保证收敛到最优解。')
    p['submitted_sources'] = p['reference_context'] = [{'id': 'C4', 'text': '算法A只有在条件H成立时才保证收敛到最优解。'}]
    add('missing-condition', p, 'incorrect', 'not_supported')
    p = dict(kind='entity', focus=FOCUS['entity'], target=dict(name='常量初始化', definition='将参数初始化为给定常数的方法。'), submitted_sources=[{'id': 'C5', 'text': '常量初始化将所有参数初始化为给定的常数。'}])
    p['reference_context'] = p['submitted_sources']
    add('entity-definition', deepcopy(p), 'correct', 'supported')
    p['target']['definition'] = None
    add('entity-no-definition', deepcopy(p), 'correct', 'supported')
    p['target']['definition'] = '将参数初始化为服从正态分布的随机数的方法。'
    add('entity-wrong-definition', p, 'incorrect', 'not_supported')
    p = deepcopy(base); p['submitted_sources'] = []; p['reference_context'] = []
    add('no-information', p, 'uncertain', 'not_supported')
    return cases


def prepare(run, previous):
    if run.exists():
        raise ValueError('Use a new run directory')
    run.mkdir(parents=True)
    original = read(previous / 'tasks.json'); original_key = read(previous / 'private-key.json')
    units = read(DOCUMENTS)['units']; positions = {u['unit_id']: i for i, u in enumerate(units)}
    tasks, key = [], {}
    for old in original:
        p = old['payload']; indices = set()
        for source in p['supplied_sources']:
            pos = positions[source['id']]
            indices.update(range(max(0, pos - 1), min(len(units), pos + 2)))
        context = [{'id': units[i]['unit_id'], 'text': units[i]['text']} for i in sorted(indices)]
        for kind in (['entity'] if p['kind'] == 'entity' else ['relation', 'assertion']):
            uid = old['id'] + '-' + kind
            if kind == 'entity':
                target = {k: p['output'][k] for k in ['name', 'definition']}
            else:
                target = {k: p['output'].get(k, '') for k in ['subject', 'predicate', 'object', 'scope', 'polarity']}
                target['description'] = p['output']['text']
            payload = dict(kind=kind, focus=FOCUS[kind], target=target, submitted_sources=p['supplied_sources'], reference_context=context)
            tasks.append(dict(id=uid, payload=payload, oversized=sum(len(s['text']) for s in context) > 120000))
            key[uid] = {**original_key[old['id']], 'kind': kind, 'original_task_id': old['id']}
    for name, value in [('tasks.json', tasks), ('private-key.json', key), ('calibration.json', calibration())]:
        write(run / name, value)
    (run / 'prompt.txt').write_text(PROMPT)
    shutil.copy2(__file__, run / 'evaluate.py')
    shutil.copy2(Path(__file__).with_name('README.md'), run / 'PROTOCOL.md')
    transport = (previous / 'transport.py').read_text().replace('REQUEST_CONCURRENCY = 4', 'REQUEST_CONCURRENCY = 2')
    (run / 'transport.py').write_text(transport)
    manifest = dict(created_at=time.time(), original_records=len(original), tasks=len(tasks), workers=2, http_slots=2,
                    previous_run=str(previous), source_sampling='exact same 160 records; no replacements',
                    source_files={str(previous / 'tasks.json'): sha(previous / 'tasks.json'), str(previous / 'private-key.json'): sha(previous / 'private-key.json'), str(DOCUMENTS): sha(DOCUMENTS)},
                    reference_context='submitted parent units plus one adjacent unit on each side; same deterministic rule for all methods; not exhaustive book search',
                    frozen_files={name: sha(run / name) for name in ['tasks.json', 'private-key.json', 'calibration.json', 'prompt.txt', 'evaluate.py', 'transport.py', 'PROTOCOL.md']})
    write(run / 'manifest.json', manifest)
    print(json.dumps(dict(records=len(original), tasks=len(tasks), calibration=len(calibration()), oversized=sum(t['oversized'] for t in tasks)), ensure_ascii=False))


def parse(response):
    content = response['choices'][0]['message']['content'].strip()
    match = re.fullmatch(r'<correctness>(correct|incorrect|uncertain)</correctness>\s*<evidence>(supported|not_supported|uncertain)</evidence>\s*<reason>(.+)</reason>', content, re.DOTALL)
    if not match or not match[3].strip():
        raise ValueError('Expected two labels and a reason')
    return dict(correctness=match[1], evidence=match[2], reason=match[3].strip())


def summarize(run):
    key = read(run / 'private-key.json'); tasks = read(run / 'tasks.json')
    groups, review = {}, []
    for t in tasks:
        k = key[t['id']]; path = run / 'results' / (t['id'] + '.json')
        r = read(path) if path.exists() else dict(status='pending')
        groups.setdefault((k['system'], k['kind']), []).append((t, k, r))
        if r['status'] == 'done':
            review.append(dict(id=t['id'], system=k['system'], kind=k['kind'], target=t['payload']['target'], judgment=r['value'], audit=k['audit']))
    summary = {}
    for (system, kind), rows in groups.items():
        done = [(t, k, r['value']) for t, k, r in rows if r['status'] == 'done']
        n = len(rows)
        m = dict(selected=n, judged=len(done), statuses={s: sum(r['status'] == s for _, _, r in rows) for s in sorted({r['status'] for _, _, r in rows})})
        for field, labels in [('correctness', ['correct', 'incorrect', 'uncertain']), ('evidence', ['supported', 'not_supported', 'uncertain'])]:
            m[field] = {label: sum(v[field] == label for _, _, v in done) for label in labels}
        m['confirmed_correct_rate_all_selected'] = m['correctness']['correct'] / n
        m['sufficient_evidence_rate_all_selected'] = m['evidence']['supported'] / n
        supported = [k['audit'] for _, k, v in done if v['evidence'] == 'supported']
        m['median_cited_characters_on_sufficient'] = statistics.median(a['source_characters'] for a in supported) if supported else None
        m['median_cited_units_on_sufficient'] = statistics.median(a['reference_units'] for a in supported) if supported else None
        if kind == 'entity':
            m['definitions_available'] = sum(bool(t['payload']['target']['definition']) for t, _, _ in rows)
        summary.setdefault(system, {})[kind] = m
    write(run / 'summary.json', summary); write(run / 'review.json', review)
    lines = ['# 简化质量试评', '', '同一批 160 个记录，80 个实体 + 80 条关系分别核验核心关系与完整断言，共 240 个判断。每题只判内容正确性、提交引用是否充分。', '', '| 方法 | 已判/任务 | 实体正确 | 核心关系正确 | 完整断言正确 | 实体引用充分 | 断言引用充分 |', '|---|---:|---:|---:|---:|---:|---:|']
    for system in ['ours', 'graphrag', 'autoschemakg', 'kggen']:
        d = summary[system]
        def fraction(kind, field, label):
            return f"{d[kind][field][label]}/{d[kind]['selected']}"
        values = [system, f"{sum(v['judged'] for v in d.values())}/{sum(v['selected'] for v in d.values())}", fraction('entity','correctness','correct'), fraction('relation','correctness','correct'), fraction('assertion','correctness','correct'), fraction('entity','evidence','supported'), fraction('assertion','evidence','supported')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '实体正确性包含已提供的定义；无定义只评名称，并单列可用率。错误、不确定、技术缺失分开计数，见 summary.json。未完成时表中分子仅为已确认数，不得当最终排名。', '', '正确性依赖局部原文参考上下文，不是全书穷尽核验；引用充分性仅使用原先提交的引用。两项有关联，但额外参考上下文不为引用加分。本轮不测召回或全局归并，不报告加权总分、原子拆分得分、关系粒度得分或估计的逐段引用有效率。', '', '同模型小样本初评，需对各方法统一复核；旧分数保留。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def execute(run):
    manifest = read(run / 'manifest.json')
    for name, expected in manifest['frozen_files'].items():
        if sha(run / name) != expected:
            raise ValueError('Frozen file changed: ' + name)
    spec = importlib.util.spec_from_file_location('simple_transport', run / 'transport.py')
    transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    assert transport.REQUEST_CONCURRENCY == 2
    client = transport.Client(run / 'api', run / 'request-slots')
    prompt = (run / 'prompt.txt').read_text()
    (run / '.started').write_text(str(time.time()))

    def one(task):
        path = run / 'results' / (task['id'] + '.json')
        if path.exists():
            return read(path)
        start = time.time()
        if task.get('oversized'):
            result = dict(status='unassessed_size')
        else:
            try:
                response = client.complete([dict(role='system', content=prompt), dict(role='user', content=json.dumps(task['payload'], ensure_ascii=False))], max_tokens=8192, cache=True, validator=parse)
                result = dict(status='done', value=parse(response))
            except transport.ContentRejected:
                result = dict(status='skipped_input_moderation')
            except transport.TerminalProviderError:
                result = dict(status='terminal_provider_error')
            except Exception as exc:
                result = dict(status='failed', error_type=type(exc).__name__)
        result.update(task_id=task['id'], started_at=start, finished_at=time.time())
        write(path, result)
        return result

    for phase, name in [('calibration', 'calibration.json'), ('pilot', 'tasks.json')]:
        tasks = read(run / name)
        write(run / 'progress.json', dict(phase=phase, processed=0, total=len(tasks), updated_at=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(one, t) for t in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                write(run / 'progress.json', dict(phase=phase, processed=count, total=len(tasks), last_status=result['status'], updated_at=time.time()))
                print(f'{phase} {count}/{len(tasks)} {result["status"]}', flush=True)
                if phase == 'pilot':
                    summarize(run)
        if phase == 'calibration':
            cases = []
            for t in tasks:
                r = read(run / 'results' / (t['id'] + '.json'))
                cases.append(dict(id=t['id'], expected=t['expected'], observed=r.get('value'), passed=r['status'] == 'done' and all(r['value'][k] == v for k,v in t['expected'].items())))
            passed = all(c['passed'] for c in cases)
            write(run / 'calibration-report.json', dict(passed=passed, cases=cases))
            if not passed:
                write(run / 'progress.json', dict(phase='calibration_needs_review', processed=len(tasks), total=len(tasks), updated_at=time.time()))
                return 2
    summarize(run)
    statuses = [read(run / 'results' / (t['id'] + '.json'))['status'] for t in read(run / 'tasks.json')]
    complete = all(s == 'done' for s in statuses)
    write(run / 'progress.json', dict(phase='complete' if complete else 'complete_with_unassessed', processed=len(statuses), total=len(statuses), judged=statuses.count('done'), updated_at=time.time()))
    (run / '.finished').write_text(str(time.time()))
    return 0 if complete else 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','report'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == 'prepare':
        prepare(run, args.previous.resolve())
    elif args.action == 'report':
        summarize(run)
    else:
        sys.exit(execute(run))
