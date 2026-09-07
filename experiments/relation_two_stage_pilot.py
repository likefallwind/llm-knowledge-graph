"""Bounded synonym replacement, followed by AI-induced coarse relation mapping.

Source SQLite is read-only. The derived SQLite contains all sampled source rows,
materialized fine edges, and assertion-level coarse lineage, not production schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from experiments.relation_coarsening_experiment import CachedLLM, log, save_json
from kg.llm import LLMConfig

VERSION = "synonym-then-coarse-pilot-1"
SYSTEM = "你是语料约束的关系分析员。只输出 JSON；只根据提供的事实判断，不用常识补充事实。"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(items, key, expected):
    """Reject missing, duplicate, invented IDs and bool-as-int IDs."""
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError("expected a list of records")
    ids = [x.get(key) for x in items]
    if any(type(x) is not int for x in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"invalid {key} coverage")
    return {x[key]: x for x in items}


def checked_completion(llm, *, kind, prompt, field, ids):
    """Retry structural failures without assigning positional IDs to judgments."""
    for attempt in range(3):
        suffix = "" if attempt == 0 else (
            "\n格式重试：id必须逐字复制事实中的真实ID，不能使用批次序号1,2,3。"
            "本批必须且只能包含这些ID：" + json.dumps(ids) + f"。这是第{attempt}次格式重试。"
        )
        answer = llm.complete(kind=kind, system=SYSTEM, user=prompt+suffix)
        try:
            return list(keyed(answer.get(field), "id", ids).values())
        except ValueError:
            if attempt == 2:
                raise
            log(f"{kind}: invalid IDs; requesting structural retry")


def load_sample(db, size=80, *, all_relations=False):
    with sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(integrity)
        relations = [dict(r) for r in conn.execute("""SELECT r.id,r.canonical_name name,
            r.description,count(c.id) uses FROM relation_types r JOIN claims c
            ON c.relation_type_id=r.id GROUP BY r.id ORDER BY r.id""")]
        eligible = [r for r in relations if r["uses"] <= 9]
        if size < 20 or size > len(eligible):
            raise ValueError("sample size must be between 20 and eligible relation count")
        # Half stratified by frequency, half lexical neighbours. No model-selected sample.
        rng = random.Random(20260905)
        chosen = {}
        buckets = [[r for r in eligible if lo <= r["uses"] <= hi] for lo, hi in [(1, 1), (2, 3), (4, 9)]]
        for bucket in buckets:
            rng.shuffle(bucket)
        for i in range(size // 2):
            bucket = buckets[i % len(buckets)]
            r = bucket.pop()
            chosen[r["id"]] = r
        stratified_ids = sorted(chosen)
        pairs = sorted(
            ((SequenceMatcher(None, a["name"], b["name"]).ratio(), a["id"], b["id"])
             for a in chosen.values() for b in eligible if b["id"] not in chosen),
            reverse=True,
        )
        records = {r["id"]: r for r in eligible}
        for _, _, rid in pairs:
            chosen[rid] = records[rid]
            if len(chosen) == size:
                break
        if all_relations:
            chosen = {r["id"]: r for r in relations}
            stratified_ids = []
        ids = sorted(chosen)
        marks = ",".join("?" for _ in ids)
        claims = [dict(r) for r in conn.execute(f"""SELECT c.*,s.canonical_name subject,
            o.canonical_name object FROM claims c JOIN entities s ON s.id=c.subject_id
            JOIN entities o ON o.id=c.object_id WHERE relation_type_id IN ({marks}) ORDER BY c.id""", ids)]
        facts = [dict(r) for r in conn.execute(f"""SELECT a.id,a.claim_id,c.subject_id,c.object_id,
            c.relation_type_id,s.canonical_name subject,o.canonical_name object,
            a.normalized_text statement,a.scope_text scope,a.scope_is_restrictive,a.polarity
            FROM assertions a JOIN claims c ON c.id=a.claim_id
            JOIN entities s ON s.id=c.subject_id JOIN entities o ON o.id=c.object_id
            WHERE c.relation_type_id IN ({marks}) ORDER BY a.id""", ids)]
        if {f["claim_id"] for f in facts} != {c["id"] for c in claims}:
            raise ValueError("sample has a claim without Assertion")
        evidence = [dict(r) for r in conn.execute(f"""SELECT e.* FROM evidence e
            WHERE e.claim_id IN (SELECT id FROM claims WHERE relation_type_id IN ({marks}))
            OR e.assertion_id IN (SELECT a.id FROM assertions a JOIN claims c ON c.id=a.claim_id
            WHERE c.relation_type_id IN ({marks})) ORDER BY e.id""", ids + ids)]
    return dict(relations=[chosen[i] for i in ids], claims=claims, facts=facts, evidence=evidence,
                source_used_types=len(relations), source_claims=sum(r["uses"] for r in relations),
                stratified_relation_ids=stratified_ids, integrity=integrity)


def fact_payload(f):
    # Old relation label is intentionally hidden from coarse assignment and review.
    return {k: f[k] for k in ("id", "subject", "object", "statement", "scope", "scope_is_restrictive")}


def cards(sample, fine_map=None):
    fine_map = fine_map or {r["id"]: r["id"] for r in sample["relations"]}
    records = {r["id"]: r for r in sample["relations"]}
    result = []
    for target in sorted(set(fine_map.values())):
        members = [i for i, j in fine_map.items() if j == target]
        facts = [f for f in sample["facts"] if f["relation_type_id"] in members]
        # Two distinct claims; full evidence is reserved for pair/fact validation.
        examples, seen = [], set()
        for f in facts:
            if f["claim_id"] not in seen:
                seen.add(f["claim_id"])
                examples.append(fact_payload(f))
            if len(examples) == 2:
                break
        result.append(dict(id=target, name=records[target]["name"],
                           original_definition=records[target]["description"],
                           uses=len(seen) if not facts else len({f["claim_id"] for f in facts}),
                           examples=examples))
    return result


def parse_groups(result, ids):
    groups = result.get("groups")
    if not isinstance(groups, list):
        raise ValueError("groups must be a list")
    seen = set()
    for g in groups:
        members, target = g.get("member_ids", []), g.get("canonical_id")
        if (type(target) is not int or target not in members or len(members) < 2
                or any(type(i) is not int or i not in ids for i in members)
                or len(set(members)) != len(members) or seen.intersection(members)):
            raise ValueError("invalid or overlapping synonym group")
        seen.update(members)
    return groups


def fine_stage(llm, sample, out):
    source_cards = cards(sample)
    result = llm.complete(kind=VERSION+":synonym-discovery", system=SYSTEM, user="""第一阶段仅找同义表达。
找出可以在相同主宾和语境下相互替换的关系名称。严格双向同义，不是上下位粗化。
例如可转化为/可转换为可同义；可无损转换为/可转换为不等价；产生/可能产生不等价。
方向相反不可归一，不能因为同一对实体上两种关系碰巧都真就判同义。
定义可能是首次抽取的局部概括，请结合例子识别义项。多义标签可提案，后续逐事实验证。
只输出候选同义组，不必覆盖所有名称。每组选择一个已有 ID 为规范关系，组之间不得重叠。
返回 {"groups":[{"canonical_id":1,"member_ids":[1,2],"reason":""}]}。
关系=""" + json.dumps(source_cards, ensure_ascii=False))
    save_json(out/"fine_proposals.json", result)
    groups = parse_groups(result, {r["id"] for r in sample["relations"]})
    by_id = {r["id"]: r for r in source_cards}
    jobs = [(i, g["canonical_id"]) for g in groups for i in g["member_ids"] if i != g["canonical_id"]]

    def judge(pair):
        source, target = pair
        relevant = [f for f in sample["facts"] if f["relation_type_id"] in (source, target)]
        answer = llm.complete(kind=VERSION+":synonym-review", system=SYSTEM, user="""独立复核同义替换。
不要默认提案正确。判断两个关系表达在这些事实所涉及的义项上是否双向同义，保留方向、
核心动作、否定、可能性、程度和限制。上位概括不是同义，事实碰巧同时成立也不是同义。
分别检查每条 Assertion 所用的旧表达替换成另一表达是否不改变意思。
关系定义仅为线索，不把偶然实体限制当词义。对多义标签允许逐条指出不适合替换的事实。
返回 {"lexically_equivalent":true,"reason":"", "checks":[{"id":1,"equivalent":true,"reason":""}]}。
必须覆盖全部 Assertion ID，一条都不能遗漏。
两个关系=%s\n全部事实=%s""" % (json.dumps([by_id[source], by_id[target]], ensure_ascii=False),
                                                   json.dumps(relevant, ensure_ascii=False)))
        checked = keyed(answer.get("checks"), "id", [f["id"] for f in relevant])
        return dict(source_id=source, target_id=target, result=answer,
                    accepted=answer.get("lexically_equivalent") is True,
                    accepted_assertion_ids=[f["id"] for f in relevant
                      if f["relation_type_id"] == source and answer.get("lexically_equivalent") is True
                      and checked[f["id"]].get("equivalent") is True])

    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as pool:
        reviews = list(pool.map(judge, jobs))
    # Only whole Claim replacements: all its Assertions must agree on the same synonym.
    assertion_map = {f["id"]: f["relation_type_id"] for f in sample["facts"]}
    for review in reviews:
        for aid in review["accepted_assertion_ids"]:
            assertion_map[aid] = review["target_id"]
    claim_map = {}
    for c in sample["claims"]:
        targets = {assertion_map[f["id"]] for f in sample["facts"] if f["claim_id"] == c["id"]}
        claim_map[c["id"]] = targets.pop() if len(targets) == 1 else c["relation_type_id"]
    fine_sample = dict(sample)
    fine_sample["facts"] = [{**f, "relation_type_id": claim_map[f["claim_id"]]} for f in sample["facts"]]
    fine_sample["relations"] = [r for r in sample["relations"] if r["id"] in set(claim_map.values())]
    save_json(out/"fine_reviews.json", reviews)
    save_json(out/"fine_claim_mapping.json", claim_map)
    return fine_sample, claim_map, reviews


def parse_catalog(result, max_types):
    items = result.get("relations")
    if not isinstance(items, list) or not 1 <= len(items) <= max_types:
        raise ValueError("invalid coarse catalog size")
    if any(not isinstance(x.get("name"), str) or not x["name"].strip()
           or not isinstance(x.get("definition"), str) or not x["definition"].strip() for x in items):
        raise ValueError("empty coarse definition")
    if len({x["name"].strip() for x in items}) != len(items):
        raise ValueError("duplicate coarse name")
    return [dict(id=i, name=x["name"].strip(), definition=x["definition"].strip()) for i, x in enumerate(items, 1)]


def coarse_stage(llm, sample, out, max_types=35):
    result = llm.complete(kind=VERSION+":coarse-catalog", system=SYSTEM, user="""第二阶段：根据规范细关系
及实际 Assertion，归纳一张粗关系表。只输出目录，不输出逐成员分配。
尽量形成20至%d种可理解且可复用的粗二元谓词，可少于20，不为凑数细分。
保留核心参与者、方向；程度、机制、比较维度和适用条件由完整 Assertion 保存。
例：可无损转换为/可有损转换为可归入可转换为；不同维度的高于可用在某方面高于。
粗关系允许单向蕴含，不要求与细关系同义。不要使用相关于/有关/涉及等万能兜底。
目标/可能/否定不等于实际肯定效果；为这些事实选择仍然成立的措辞。
每种只需名称和一句定义，定义交代主客体角色及方向，不罗列互不相干的动作。
定义不要写仅来自个别例子的领域限制。无需照搬细关系定义，按全部给定事实归纳。
目录是带 Assertion 语境的导航概括，不能独立作无条件推理。
返回 {"relations":[{"name":"","definition":""}]}。
细关系=%s""" % (max_types, json.dumps(cards(sample), ensure_ascii=False)))
    save_json(out/"coarse_catalog_raw.json", result)
    catalog = parse_catalog(result, max_types)
    save_json(out/"coarse_catalog.json", catalog)
    batches = [sample["facts"][i:i+6] for i in range(0, len(sample["facts"]), 6)]

    def assign(batch):
        answer = llm.complete(kind=VERSION+":coarse-map", system=SYSTEM, user="""给每条 Assertion
选择一个最合适的粗关系，候选是完整目录。只需在原 Assertion 的条件/时间/比较维度下
蕴含粗关系，不要求语义等价；丢失程度/机制/细分本身不能拒绝。
主宾不得交换或变为它的属性/输出等隐藏对象。不得把目标、可能、否定升级为实际肯定。
没有合适候选则 coarse_id=null，不要改目录，不要强行覆盖。
返回 {"mappings":[{"id":1,"coarse_id":null,"projection":"用原主宾写出带必要限定的粗命题",
"omitted_detail":"被概括的细节","reason":""}]}，必须逐条覆盖所有ID。
目录=%s\n事实=%s""" % (json.dumps(catalog, ensure_ascii=False),
                             json.dumps([fact_payload(f) for f in batch], ensure_ascii=False)))
        indexed = keyed(answer.get("mappings"), "id", [f["id"] for f in batch])
        for x in indexed.values():
            target = x.get("coarse_id")
            if target is not None and (type(target) is not int or target not in {c["id"] for c in catalog}):
                raise ValueError("unknown coarse relation")
        return list(indexed.values())

    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as pool:
        mappings = []
        for i, rows in enumerate(pool.map(assign, batches), 1):
            mappings.extend(rows)
            save_json(out/"coarse_mapping_checkpoint.json", mappings)
            log(f"coarse mapping batches {i}/{len(batches)}")
    by_id = {f["id"]: f for f in sample["facts"]}
    cats = {c["id"]: c for c in catalog}
    mapped = [m for m in mappings if m["coarse_id"] is not None]
    review_batches = [mapped[i:i+6] for i in range(0, len(mapped), 6)]

    def review(batch):
        # Do not show the mapper's explanation or original fine label to the reviewer.
        payload = [dict(fact=fact_payload(by_id[m["id"]]), candidate=cats[m["coarse_id"]]) for m in batch]
        prompt = """独立审查粗关系映射。
在原 Assertion 的条件、时间和比较维度下，原主语和宾语通过候选关系是否仍成立？
只检验单向蕴含，信息更少/不保留机制程度不构成拒绝理由；不能要求同义或双向蕴含。
严格保留端点角色和方向，不得省略隐藏的真正参与对象；目标/可能/否定不等于实际肯定。
不要因为映射已提出就认可。若原 Assertion 真而语境中的粗命题仍可能假，valid=false。
返回 {"checks":[{"id":1,"valid":true,"reason":""}]}，覆盖每条ID。
待审=%s""" % json.dumps(payload, ensure_ascii=False)
        return checked_completion(llm, kind=VERSION+":coarse-review", prompt=prompt,
                                  field="checks", ids=[m["id"] for m in batch])

    with ThreadPoolExecutor(max_workers=llm.max_concurrency) as pool:
        checks = []
        for i, rows in enumerate(pool.map(review, review_batches), 1):
            checks.extend(rows)
            save_json(out/"coarse_review_checkpoint.json", checks)
            log(f"coarse review batches {i}/{len(review_batches)}")
    checked = {x["id"]: x for x in checks}
    for m in mappings:
        m["review"] = checked.get(m["id"])
        m["accepted"] = m["id"] in checked and checked[m["id"]].get("valid") is True
    save_json(out/"coarse_mappings.json", mappings)
    return catalog, mappings


def materialize(path, sample, claim_map, mappings):
    """Every source Claim/Assertion/evidence remains available alongside derived edges."""
    if path.exists():
        raise ValueError("derived database already exists")
    with sqlite3.connect(path) as c:
        c.execute("PRAGMA foreign_keys=ON")
        c.executescript("""CREATE TABLE source_rows(kind TEXT,id INTEGER,payload TEXT,PRIMARY KEY(kind,id));
        CREATE TABLE fine_claims(id INTEGER PRIMARY KEY,subject_id INTEGER,relation_id INTEGER,object_id INTEGER,
          UNIQUE(subject_id,relation_id,object_id));
        CREATE TABLE claim_lineage(source_claim_id INTEGER PRIMARY KEY,fine_claim_id INTEGER REFERENCES fine_claims(id));
        CREATE TABLE coarse_lineage(assertion_id INTEGER PRIMARY KEY,source_claim_id INTEGER REFERENCES claim_lineage(source_claim_id),
          fine_claim_id INTEGER REFERENCES fine_claims(id),coarse_relation_id INTEGER,accepted INTEGER,version TEXT,judgment TEXT);""")
        for kind in ("relations", "claims", "facts", "evidence"):
            c.executemany("INSERT INTO source_rows VALUES(?,?,?)", [(kind, r["id"], json.dumps(r, ensure_ascii=False)) for r in sample[kind]])
        fine_ids = {}
        for claim in sample["claims"]:
            key = (claim["subject_id"], claim_map[claim["id"]], claim["object_id"])
            c.execute("INSERT OR IGNORE INTO fine_claims(subject_id,relation_id,object_id) VALUES(?,?,?)", key)
            fid = c.execute("SELECT id FROM fine_claims WHERE subject_id=? AND relation_id=? AND object_id=?", key).fetchone()[0]
            fine_ids[claim["id"]] = fid
            c.execute("INSERT INTO claim_lineage VALUES(?,?)", (claim["id"], fid))
        facts = {f["id"]: f for f in sample["facts"]}
        keyed(mappings, "id", facts)
        for m in mappings:
            claim_id = facts[m["id"]]["claim_id"]
            c.execute("INSERT INTO coarse_lineage VALUES(?,?,?,?,?,?,?)", (m["id"], claim_id, fine_ids[claim_id], m["coarse_id"], int(m["accepted"]), VERSION, json.dumps(m, ensure_ascii=False)))
        if c.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("broken lineage")
        return c.execute("SELECT count(*) FROM fine_claims").fetchone()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=80)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.db)
    sample = load_sample(args.db, args.sample_size)
    manifest = dict(version=VERSION, source_db=str(args.db.resolve()), source_sha256=source_hash,
                    script_sha256=digest(__file__), sample_size=args.sample_size,
                    selection="half frequency-stratified; half lexical neighbours; uses<=9; seed=20260905",
                    model="MiniMax-M3", workers=6, catalog_max=35,
                    relation_ids=[r["id"] for r in sample["relations"]],
                    assertions=len(sample["facts"]), claims=len(sample["claims"]))
    if (out/"manifest.json").exists() and json.loads((out/"manifest.json").read_text()) != manifest:
        raise ValueError("manifest changed; use a new output directory")
    save_json(out/"manifest.json", manifest)
    save_json(out/"input.json", sample)
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return
    if (out/"summary.json").exists():
        raise ValueError("completed run exists; use a new output directory")
    config = LLMConfig.from_env()
    llm = CachedLLM(base_url="https://api.minimaxi.com/v1/text/chatcompletion_v2", api_key=config.api_key,
                    model="MiniMax-M3", max_concurrency=6, cache_path=out/"llm_cache.jsonl")
    log(f"Starting {VERSION}: {manifest['sample_size']} types, {manifest['assertions']} Assertions")
    fine_sample, claim_map, reviews = fine_stage(llm, sample, out)
    save_json(out/"fine_catalog.json", cards(fine_sample))
    log(f"fine stage: {len(sample['relations'])} -> {len(fine_sample['relations'])} used types")
    catalog, mappings = coarse_stage(llm, fine_sample, out)
    if digest(args.db) != source_hash:
        raise ValueError("source changed during run")
    fine_edges = materialize(out/"derived.db", sample, claim_map, mappings)
    accepted = [m for m in mappings if m["accepted"]]
    by_id = {f["id"]: f for f in sample["facts"]}
    residual_types = {claim_map[by_id[m["id"]]["claim_id"]] for m in mappings if not m["accepted"]}
    distribution = Counter(m["coarse_id"] for m in accepted)
    summary = dict(status="complete", validation="same-model separate-prompt review; not independent accuracy",
                   source_unchanged=True, original_types=len(sample["relations"]), fine_types=len(fine_sample["relations"]),
                   source_claims=len(sample["claims"]), fine_claims=fine_edges, assertions=len(mappings),
                   fine_replaced_claims=sum(claim_map[c["id"]] != c["relation_type_id"] for c in sample["claims"]),
                   synonym_proposals=len(reviews), coarse_catalog_types=len(catalog), used_coarse_types=len(distribution),
                   coarse_accepted_assertions=len(accepted), preliminary_mapped=sum(m["coarse_id"] is not None for m in mappings),
                   assertion_coverage=len(accepted)/len(mappings), residual_fine_types=len(residual_types),
                   effective_types=len(distribution)+len(residual_types), distribution=dict(distribution), runtime=llm.metrics)
    save_json(out/"summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
