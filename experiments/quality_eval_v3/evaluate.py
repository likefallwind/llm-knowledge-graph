"""Frozen, source-grounded output audit. This is not a gold-recall benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import time
import unicodedata

BENCH = Path('/home/likefallwind/code/llm-graph-benchmark/outputs')
BASELINES = BENCH / 'd2l-baseline-correction-m3-c6-20260909-172849'
OURS = BENCH / 'd2l-full1105-vnext-20260826'
SEED = 20260912
LIMIT = 120_000  # Full submitted source text; oversized records are not truncated.
LABELS = {'pass', 'fail', 'uncertain'}
PROMPT = '''你是教材知识图谱的盲态证据审计员。只使用 supplied_sources 的原文核验 output；不得用常识、检索或 reference_fact 补充来源证据。方法名不可见，格式可能暴露身份。
这是来源支持审计，不是无证据的世界知识真假判断。pass=引用足以支持，fail=明确不受引用支持或与引用矛盾，uncertain=语义/公式/指代有实质歧义；缺少足够证据通常 fail，而非默认 pass。说明 fail 属于矛盾还是证据缺失。
entity: core_support 核验 name 所指的概念/对象及局部含义、边界；合理缩写、译名和同义表达允许，代码对象、示例句、事件也可以是实体，不能仅因类型不是概念就扣分。detail_support 核验真实 definition 的全部陈述；definition=null 必须 not_provided。type_support 核验 types 的语义兼容性，不要求固定本体；空 types 必须 not_provided。本轮不评全局归并或别名准确率。
assertion: core_support 核验 subject、predicate、object 的身份、角色、方向、polarity 以及 scope 限制下的关系。detail_support 核验 text/scope 中实际陈述的全部事实；两者均空才 not_provided。宽泛 related_to 可以正确。完整表示中 text 的具体含义必须获得信用，但不能修复一条与它矛盾的明确谓词边。必要条件被删去而扩大断言范围应判 fail；与命题无关的上下文不要求全部复述。type_support 必须 not_provided。
specific_relation 对 assertion 表示完整 output 是否明确表达了超出“有关联/共同出现”的具体属性、机制、类别、动作或角色；仅判表达内容，不判断是否正确，具体但错误仍 true；entity 必须 null。不得从来源脑补输出缺失的具体关系，不按谓词长度或类型数量打分。此标记是信息粒度诊断，并不是说原文本来宽泛的关系低质。
atoms 将 output 的核心命题与真实描述/定义（不含 types）分成少量不重复的可独立核验陈述。保持所有实质承诺；不可为方便而删除错误部分，也不可把原文中额外知识添加到 output。每项 label=pass/fail/uncertain，unit_ids 为真正支持该陈述的 supplied_sources 的 id（无支持则 []）。长记录按记录内原子支持比例汇总，不能以更啰嗦增加权重。
useful_unit_ids 尽量列出所提交来源中实际贡献正面支持的全部段落 id，包括重复的有效佐证；仅提及词面但不支持所评身份/命题不算。需要多段联合支持时允许它们一起计入。不能引用 supplied_sources 以外的 id。irrelevant 段落不应因同属一章就算有效。有效率只是裁判估计，长引用还需人工复核。
reference_fact 仅出现在受控校准题。reference_recovered 只判断完整 output 本身是否表达该事实的全部必要角色、含义和条件，不依赖 supplied_sources；缺省时必须 null。这个判断与引用支持独立：正确输出挂错引用仍可 recovered=true、support=fail。只有关联但未表达具体事实不能 recovered=true。
严格返回一个 JSON 对象，不要 Markdown：
{"core_support":"pass|fail|uncertain","detail_support":"pass|fail|uncertain|not_provided","type_support":"pass|fail|uncertain|not_provided","specific_relation":true或false或null,"reference_recovered":true或false或null,"atoms":[{"claim":"原子陈述","label":"pass|fail|uncertain","unit_ids":["来源id"]}],"useful_unit_ids":["来源id"],"reason":"简短中文解释：注明错误的具体内容、来源位置；区分不支持与矛盾"}
'''


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def source_audit(evidence, units):
    ids = sorted({e['unit_id'] for e in evidence})
    sources = [{'id': uid, 'text': units[uid]['text']} for uid in ids if uid in units]
    quotes = [e for e in evidence if e.get('quote')]
    aligned = sum(bool(normalized(e['quote'])) and e['unit_id'] in units
                  and normalized(e['quote']) in normalized(units[e['unit_id']]['text']) for e in quotes)
    # Parent paragraphs are always read in this pilot, even for valid short quotes.
    # Never claim a quoted-span budget result from a full-paragraph judgment.
    return sources, dict(reference_units=len(ids), resolved_units=len(sources),
                         source_characters=sum(len(s['text']) for s in sources),
                         quotes=len(quotes), aligned_quotes=aligned)


def calibration():
    fact = '汇聚层降低卷积层对位置的敏感性。'
    good_source = {'id': 'C1', 'text': '本节将介绍汇聚（pooling）层，它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。'}
    base = dict(kind='assertion', output=dict(subject='汇聚层', predicate='降低', object='卷积层对位置的敏感性', text=fact, scope='', polarity='positive'), supplied_sources=[good_source], reference_fact=fact)
    cases = []

    def add(name, payload, expected):
        cases.append(dict(id='cal-' + name, payload=payload, expected=expected))

    add('correct', deepcopy(base), dict(core_support='pass', detail_support='pass', specific_relation=True, reference_recovered=True))
    p = deepcopy(base); p['output']['predicate'] = 'related_to'
    add('generic-with-description', p, dict(core_support='pass', specific_relation=True, reference_recovered=True))
    p = deepcopy(base); p['output'].update(predicate='related_to', text='')
    add('generic-only', p, dict(core_support='pass', detail_support='not_provided', specific_relation=False, reference_recovered=False))
    p = deepcopy(base); p['output'].update(predicate='增加', text='汇聚层增加卷积层对位置的敏感性。')
    add('opposite', p, dict(core_support='fail', reference_recovered=False))
    p = deepcopy(base); p['output'].update(subject='卷积层对位置的敏感性', object='汇聚层', text='卷积层对位置的敏感性降低汇聚层。')
    add('roles', p, dict(core_support='fail', reference_recovered=False))
    p = deepcopy(base); p['output'].update(predicate='减小', text='通过汇聚，可减小卷积层对于位置的敏感程度。')
    add('paraphrase', p, dict(core_support='pass', detail_support='pass', reference_recovered=True))
    p = deepcopy(base); p['supplied_sources'] = [{'id': 'C2', 'text': '本节介绍如何创建 GitHub 账户。'}]
    add('wrong-citation', p, dict(core_support='fail', reference_recovered=True, useful_unit_ids=[]))
    p = deepcopy(base); p['supplied_sources'].append({'id': 'C2', 'text': '本节介绍如何创建 GitHub 账户。'})
    add('expanded-citation', p, dict(core_support='pass', reference_recovered=True, useful_unit_ids=['C1']))
    p = deepcopy(base); p['output']['text'] = fact + fact
    add('duplicate', p, dict(core_support='pass', reference_recovered=True))
    # Synthetic conditional claim: a logic check, not a textbook gold annotation.
    p = deepcopy(base); p['output'] = dict(subject='算法A', predicate='收敛', object='最优解', text='算法A总能收敛到最优解。', scope='', polarity='positive')
    p['supplied_sources'] = [{'id': 'C3', 'text': '只有在假设H成立时，算法A才保证收敛到最优解。'}]
    p['reference_fact'] = '算法A在假设H成立时才保证收敛到最优解。'
    add('missing-condition', p, dict(core_support='fail', reference_recovered=False))
    p = dict(kind='entity', output=dict(name='汇聚层', definition='降低卷积层对位置敏感性的神经网络层。', types=['神经网络层']), supplied_sources=[good_source])
    add('entity-valid', deepcopy(p), dict(core_support='pass', detail_support='pass'))
    p['output']['definition'] = '一种用于创建 GitHub 账户的软件工具。'
    add('entity-wrong-definition', deepcopy(p), dict(detail_support='fail'))
    p['output']['definition'] = None; p['output']['types'] = []
    add('entity-absent-definition', p, dict(core_support='pass', detail_support='not_provided', type_support='not_provided'))
    return cases


def prepare(run, count):
    if run.exists():
        raise ValueError('Use a new directory: prepared inputs must remain frozen')
    run.mkdir(parents=True)
    doc_path = OURS / 'documents.jsonl'
    doc = read(doc_path)
    units = {u['unit_id']: u for u in doc['units']}
    tasks, key, inventories, inputs = [], {}, {}, {str(doc_path): sha(doc_path)}
    paths = {'ours': OURS / 'submission.json', **{s: BASELINES / s / 'submission.json' for s in ['graphrag', 'autoschemakg', 'kggen']}}
    for system, path in paths.items():
        inputs[str(path)] = sha(path)
        graph = read(path)['documents'][0]
        entities = {e['id']: e for e in graph['entities']}
        eligible = [a for a in graph['assertions'] if not (system == 'autoschemakg' and a['predicate'].strip().lower() == 'is participated by')]
        inventories[system] = dict(entities=len(entities), assertions=len(graph['assertions']), sampled_assertion_population=len(eligible), excluded_event_participation=len(graph['assertions']) - len(eligible))
        for kind, population in [('entity', graph['entities']), ('assertion', eligible)]:
            rng = random.Random(f'{SEED}:{system}:{kind}')
            for item in rng.sample(sorted(population, key=lambda x: x['id']), count):
                uid = 'q-' + hashlib.sha256(f'{SEED}:{system}:{kind}:{item["id"]}'.encode()).hexdigest()[:16]
                sources, audit = source_audit(item.get('evidence', []), units)
                if kind == 'entity':
                    definition = item.get('definition') if item.get('metadata', {}).get('definition_available', True) else None
                    output = dict(name=item['name'], definition=definition or None, types=item.get('types', []))
                else:
                    output = {f: item.get(f, '') for f in ['predicate', 'text', 'scope', 'polarity']}
                    output.update(subject=entities[item['subject_id']]['name'], object=entities[item['object_id']]['name'])
                payload = dict(kind=kind, output=output, supplied_sources=sources)
                tasks.append(dict(id=uid, payload=payload, oversized=audit['source_characters'] > LIMIT))
                key[uid] = dict(system=system, kind=kind, item_id=item['id'], audit=audit, evidence=item.get('evidence', []))
    random.Random(SEED).shuffle(tasks)
    write(run / 'tasks.json', tasks)
    write(run / 'private-key.json', key)
    write(run / 'calibration.json', calibration())
    (run / 'prompt.txt').write_text(PROMPT)
    shutil.copy2(__file__, run / 'evaluate.py')
    shutil.copy2(BASELINES / 'source-v4/common.py', run / 'transport.py')
    shutil.copy2(Path(__file__).with_name('README.md'), run / 'PROTOCOL.md')
    manifest = dict(version='3-output-pilot-1', created_at=time.time(), seed=SEED, per_system_per_kind=count,
                    total=len(tasks), source_character_limit=LIMIT, model='MiniMax-M3', workers=4,
                    endpoint='https://api.minimaxi.com/v1/text/chatcompletion_v2', inputs=inputs,
                    inventories=inventories, calibration_cases=len(calibration()),
                    provenance='existing submission references; no new retrieval',
                    limitations=['output sample, not recall', 'same-model machine judge', 'no independent human labels', 'no confidence intervals or global identity score'])
    manifest['frozen_files'] = {name: sha(run / name) for name in ['tasks.json', 'private-key.json', 'calibration.json', 'prompt.txt', 'evaluate.py', 'transport.py', 'PROTOCOL.md']}
    write(run / 'manifest.json', manifest)
    print(json.dumps(dict(run=str(run), tasks=len(tasks), calibration_cases=len(calibration()), oversized=sum(t['oversized'] for t in tasks)), ensure_ascii=False), flush=True)


def validate(value, payload):
    required = {'core_support', 'detail_support', 'type_support', 'specific_relation', 'reference_recovered', 'atoms', 'useful_unit_ids', 'reason'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Invalid result fields')
    if value['core_support'] not in LABELS or any(value[f] not in LABELS | {'not_provided'} for f in ['detail_support', 'type_support']):
        raise ValueError('Invalid labels')
    out = payload['output']
    entity = payload['kind'] == 'entity'
    detail_present = bool(out.get('definition')) if entity else bool(out.get('text') or out.get('scope'))
    type_present = entity and bool(out.get('types'))
    for field, present in [('detail_support', detail_present), ('type_support', type_present)]:
        if (value[field] == 'not_provided') != (not present):
            raise ValueError('Availability and judgment disagree')
    if entity and value['specific_relation'] is not None or not entity and type(value['specific_relation']) is not bool:
        raise ValueError('Invalid specificity')
    if 'reference_fact' in payload:
        if type(value['reference_recovered']) is not bool:
            raise ValueError('Missing reference judgment')
    elif value['reference_recovered'] is not None:
        raise ValueError('Cannot infer recall without reference')
    allowed = {s['id'] for s in payload['supplied_sources']}
    def check_ids(ids):
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError('Invalid or invented source IDs')
    check_ids(value['useful_unit_ids'])
    if not isinstance(value['atoms'], list) or not value['atoms']:
        raise ValueError('No evaluated atoms')
    for atom in value['atoms']:
        if not isinstance(atom, dict) or set(atom) != {'claim', 'label', 'unit_ids'} or not isinstance(atom['claim'], str) or not atom['claim'].strip() or atom['label'] not in LABELS:
            raise ValueError('Invalid atom')
        check_ids(atom['unit_ids'])
        if atom['label'] == 'pass' and not atom['unit_ids']:
            raise ValueError('Supported atom requires evidence')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Missing reason')
    return value


def parse(response, payload):
    text = response['choices'][0]['message']['content'].strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    try:
        return validate(json.loads(text), payload)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed judgment') from exc


def judge_input(payload):
    """Task-local contract prevents the shared entity/edge rubric being confused."""
    out = payload['output']
    if payload['kind'] == 'entity':
        contract = '本题是 entity。core_support 评价实体所指；specific_relation=null。'
        detail = bool(out.get('definition'))
        typed = bool(out.get('types'))
    else:
        contract = ('本题是 assertion，不是 entity。core_support 必须评价完整有向关系成立与否，'
                    '绝不只是检查两个实体存在。方向反转、角色交换、错误否定、删除必要条件而泛化，都必须使 core_support=fail。'
                    '理解核心关系时必须结合 text/scope 的限制或明确的无条件声明，不能把“总能”擅自还原为“在某条件下能”。'
                    'atoms 必须忠实复述提交输出，包括其中错误的方向，不能偷偷改写成原文的正确关系。')
        detail = bool(out.get('text') or out.get('scope'))
        typed = False
    contract += (' output 已经提供描述/定义，detail_support 只能是 pass/fail/uncertain，绝不能为 not_provided。即使所有引用无关，也应给 fail，不是 not_provided。' if detail else
                 ' 本题没有描述/定义，detail_support 必须为 not_provided，不能给 pass。')
    contract += (' output 已经提供类型，type_support 只能是 pass/fail/uncertain，不能为 not_provided。' if typed else
                 ' 本题没有需要评价的实体类型，type_support 必须为 not_provided。')
    if payload['kind'] == 'assertion':
        contract += ' specific_relation 必须是 true 或 false，不能是 null；输出具体但引用无关时仍为 true。'
    contract += ' 返回前核对理由与标签是否一致：如果解释说核心关系方向错或条件被删除，不能同时给 core_support=pass。'
    return json.dumps(payload, ensure_ascii=False) + '\n\n本题字段约束：' + contract


def fully_supported(value):
    return value['core_support'] == 'pass' and value['detail_support'] in {'pass', 'not_provided'} and all(a['label'] == 'pass' for a in value['atoms'])


def ratio(a, b):
    return a / b if b else None


def summarize(run):
    tasks, key = read(run / 'tasks.json'), read(run / 'private-key.json')
    groups = {}
    for t in tasks:
        k = key[t['id']]
        rpath = run / 'results' / (t['id'] + '.json')
        result = read(rpath) if rpath.exists() else dict(status='pending')
        groups.setdefault((k['system'], k['kind']), []).append((t, k, result))
    report = {}
    examples = []
    for (system, kind), rows in groups.items():
        completed = [(t, k, r['value']) for t, k, r in rows if r['status'] == 'done']
        n = len(rows); done = len(completed)
        metrics = dict(selected=n, judged=done, statuses={s: sum(r['status'] == s for _, _, r in rows) for s in sorted({r['status'] for _, _, r in rows})},
                       core_supported_count=sum(v['core_support'] == 'pass' for _, _, v in completed),
                       fully_supported_count=sum(fully_supported(v) for _, _, v in completed))
        metrics['core_support_rate_all_selected'] = ratio(metrics['core_supported_count'], n)
        metrics['full_support_rate_all_selected'] = ratio(metrics['fully_supported_count'], n)
        metrics['core_uncertain_count'] = sum(v['core_support'] == 'uncertain' for _, _, v in completed)
        metrics['atomic_support_macro_on_judged'] = statistics.mean(ratio(sum(a['label'] == 'pass' for a in v['atoms']), len(v['atoms'])) for _, _, v in completed) if done else None
        refs = sum(k['audit']['reference_units'] for _, k, _ in completed)
        metrics['citation_useful_micro_on_judged'] = ratio(sum(len(v['useful_unit_ids']) for _, _, v in completed), refs)
        metrics['citation_useful_macro_on_judged'] = statistics.mean(ratio(len(v['useful_unit_ids']), k['audit']['reference_units']) if k['audit']['reference_units'] else 0 for _, k, v in completed) if done else None
        supported = [k['audit'] for _, k, v in completed if fully_supported(v)]
        metrics['median_source_characters_on_supported'] = statistics.median(a['source_characters'] for a in supported) if supported else None
        metrics['median_source_units_on_supported'] = statistics.median(a['reference_units'] for a in supported) if supported else None
        metrics['reference_resolution_rate'] = ratio(sum(k['audit']['resolved_units'] for _, k, _ in rows), sum(k['audit']['reference_units'] for _, k, _ in rows))
        metrics['quote_alignment_rate'] = ratio(sum(k['audit']['aligned_quotes'] for _, k, _ in rows), sum(k['audit']['quotes'] for _, k, _ in rows))
        if kind == 'entity':
            metrics['definitions_present'] = sum(bool(t['payload']['output']['definition']) for t, _, _ in rows)
            metrics['definitions_supported'] = sum(v['detail_support'] == 'pass' for _, _, v in completed)
            metrics['definition_support_rate_on_available'] = ratio(metrics['definitions_supported'], metrics['definitions_present'])
            metrics['definition_available_and_supported_rate'] = ratio(metrics['definitions_supported'], n)
            metrics['types_present'] = sum(bool(t['payload']['output']['types']) for t, _, _ in rows)
            metrics['types_supported'] = sum(v['type_support'] == 'pass' for _, _, v in completed)
            metrics['type_support_rate_on_available'] = ratio(metrics['types_supported'], metrics['types_present'])
        else:
            metrics['specific_count'] = sum(v['specific_relation'] for _, _, v in completed)
            metrics['specific_and_supported_count'] = sum(v['specific_relation'] and fully_supported(v) for _, _, v in completed)
            metrics['specific_and_supported_rate_all_selected'] = ratio(metrics['specific_and_supported_count'], n)
        report.setdefault(system, {})[kind] = metrics
        for t, k, v in completed:
            examples.append(dict(task_id=t['id'], system=system, kind=kind, item_id=k['item_id'], output=t['payload']['output'], source_ids=[s['id'] for s in t['payload']['supplied_sources']], judgment=v, audit=k['audit']))
    write(run / 'summary.json', report)
    write(run / 'review.json', examples)
    def fmt(value):
        return 'N/A' if value is None else f'{100 * value:.1f}%'
    lines = ['# 同书质量试评 v3', '', '160 为默认样本量；实际分母见下表。结果是输出样本的同模型来源支持审计，不是金标召回、全局归并准确率或论文最终排名。未完成/未判定不计通过，进行中的比率仅是已确认支持占固定分母的下界。', '', '| 方法 | 实体已判/抽样 | 实体身份有据 | 定义有且有据 | 关系已判/抽样 | 完整断言有据 | 具体且有据（诊断） | 有据关系引用字符中位数 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in ['ours', 'graphrag', 'autoschemakg', 'kggen']:
        e, a = report[s]['entity'], report[s]['assertion']
        lines.append(f"| {s} | {e['judged']}/{e['selected']} | {fmt(e['core_support_rate_all_selected'])} | {fmt(e['definition_available_and_supported_rate'])} | {a['judged']}/{a['selected']} | {fmt(a['full_support_rate_all_selected'])} | {fmt(a['specific_and_supported_rate_all_selected'])} | {a['median_source_characters_on_supported']} |")
    lines += ['', '具体且有据只表示输出所包含的信息，不衡量遗漏；源文本本来宽泛的关联不因此判错。定义缺失为未提供，不等于幻觉。引用字符统计始终按去重父段落，无短引文预算成绩。各方法样本并非共同事实，长度对比带有样本组成影响。', '', '详细分母、未判定、类型支持、逐字引文对齐、引用有效率在 summary.json；全部已评分例子在 review.json；盲态输入在 tasks.json。历史评测未覆盖。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


def run_evaluation(run):
    manifest = read(run / 'manifest.json')
    for name, expected in manifest['frozen_files'].items():
        if sha(run / name) != expected:
            raise ValueError(f'Frozen file changed: {name}')
    spec = importlib.util.spec_from_file_location('quality_transport', run / 'transport.py')
    transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    client = transport.Client(run / 'api', run / 'request-slots')
    (run / '.started').write_text(str(time.time()))
    prompt = (run / 'prompt.txt').read_text()

    def one(task):
        path = run / 'results' / (task['id'] + '.json')
        if path.exists():
            return read(path)  # Frozen audit: failed/skipped tasks require a new run, not silent retries.
        started = time.time()
        if task.get('oversized'):
            result = dict(status='unassessed_size_limit')
        else:
            try:
                response = client.complete([{'role': 'system', 'content': prompt}, {'role': 'user', 'content': judge_input(task['payload'])}], max_tokens=8192, cache=True, validator=lambda r: parse(r, task['payload']))
                result = dict(status='done', value=parse(response, task['payload']))
            except transport.TerminalProviderError:
                result = dict(status='terminal_provider_error')
            except transport.ContentRejected:
                result = dict(status='skipped_input_moderation')
            except Exception as exc:
                result = dict(status='failed', error_type=type(exc).__name__)
        result.update(task_id=task['id'], started_at=started, finished_at=time.time())
        write(path, result)
        return result

    for phase, filename in [('calibration', 'calibration.json'), ('pilot', 'tasks.json')]:
        tasks = read(run / filename)
        done = 0
        write(run / 'progress.json', dict(phase=phase, completed=0, total=len(tasks), updated_at=time.time()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(one, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result(); done += 1
                write(run / 'progress.json', dict(phase=phase, completed=done, total=len(tasks), last_status=result['status'], updated_at=time.time()))
                print(f'{phase} {done}/{len(tasks)} {result["task_id"]} {result["status"]}', flush=True)
                if phase == 'pilot':
                    summarize(run)
        if phase == 'calibration':
            checks = []
            for task in tasks:
                result = read(run / 'results' / (task['id'] + '.json'))
                mismatches = {field: dict(expected=expected, observed=result.get('value', {}).get(field)) for field, expected in task['expected'].items() if result.get('value', {}).get(field) != expected}
                checks.append(dict(id=task['id'], passed=result['status'] == 'done' and not mismatches, mismatches=mismatches, status=result['status']))
            passed = all(c['passed'] for c in checks)
            write(run / 'calibration-report.json', dict(passed=passed, cases=checks))
            if not passed:
                write(run / 'progress.json', dict(phase='calibration_needs_review', completed=len(tasks), total=len(tasks), updated_at=time.time()))
                summarize(run)
                return 2
    summarize(run)
    statuses = [read(run / 'results' / (t['id'] + '.json'))['status'] for t in read(run / 'tasks.json')]
    success = all(s == 'done' for s in statuses)
    write(run / 'progress.json', dict(phase='complete' if success else 'complete_with_unassessed', completed=len(statuses), total=len(statuses), judged=statuses.count('done'), updated_at=time.time()))
    (run / '.finished').write_text(str(time.time()))
    return 0 if success else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == 'prepare':
        if args.count < 1:
            parser.error('count must be positive')
        prepare(run, args.count)
    elif args.action == 'run':
        return run_evaluation(run)
    else:
        summarize(run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
