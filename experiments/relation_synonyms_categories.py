"""Read-only relation synonym canonicalization and independent coarse classification.

No coarse predicates are substituted into facts. All LLM calls share one limiter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import time

from experiments.relation_coarsening_experiment import CachedLLM, log, save_json
from experiments.relation_two_stage_pilot import cards as pilot_cards, digest, keyed, load_sample
from kg.llm import LLMConfig

VERSION = "relation-synonyms-categories-v1"
SYSTEM = "你是关系词表分析员。只输出JSON，根据提供的关系和Assertion判断，不补充事实。"
MODEL = "MiniMax-M3"
ENDPOINT = "https://api.minimaxi.com/v1/text/chatcompletion_v2"


def dumps(value):
    return json.dumps(value, ensure_ascii=False)


def request(llm, stage, payload, instruction, validate):
    prompt = instruction + "\n输入=" + dumps(payload)
    for attempt in range(3):
        result = llm.complete(kind=VERSION+":"+stage, system=SYSTEM, user=prompt)
        try:
            return validate(result)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            if attempt == 2:
                raise ValueError(f"{stage}: invalid structured response: {error}") from error
            prompt += "\n格式纠正：" + str(error) + "。请重新输出全部真实ID，不能用批次序号。"
            log(f"{stage}: structural retry {attempt+1}")


def parallel(jobs, worker, out, stage, workers=6):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(worker, job): i for i, job in enumerate(jobs)}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
            save_json(out/(stage+"_checkpoint.json"), [results[i] for i in sorted(results)])
            save_json(out/"progress.json", dict(status="running", stage=stage,
                      completed=len(results), total=len(jobs), updated_at=time.time()))
            log(f"{stage}: {len(results)}/{len(jobs)}")
    return [results[i] for i in range(len(jobs))]


def context(f):
    return {k: f[k] for k in ("id", "subject", "object", "statement", "scope",
                              "scope_is_restrictive", "polarity") if k in f}


def cards(sample):
    result = pilot_cards(sample)
    for card in result:
        distinct = {}
        for fact in sample["facts"]:
            if fact["relation_type_id"] == card["id"]:
                distinct.setdefault(fact["claim_id"], fact)
        facts = list(distinct.values())
        indexes = sorted({0, len(facts)//2, len(facts)-1}) if facts else []
        card["examples"] = [context(facts[i]) for i in indexes]
    return result


def retrieval(source_cards, k=12):
    """Union of independent character TF-IDF and E5 rankings, never a merge rule."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from kg.embeddings import EMBEDDING_MODEL, _encode

    texts = [c["name"] + "。" + c["original_definition"] + "。" +
             "；".join(f["statement"] for f in c["examples"]) for c in source_cards]
    lexical = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit_transform(texts)
    lexical_scores = (lexical @ lexical.T).toarray()
    query = np.asarray(_encode(["query: "+s for s in texts]))
    passages = np.asarray(_encode(["passage: "+s for s in texts]))
    semantic_scores = query @ passages.T
    result = []
    for i, card in enumerate(source_cards):
        def top(scores):
            return [int(j) for j in np.argsort(-scores[i], kind="stable") if j != i][:k]
        lex, sem = top(lexical_scores), top(semantic_scores)
        result.append(dict(id=card["id"], lexical_ids=[source_cards[j]["id"] for j in lex],
                           semantic_ids=[source_cards[j]["id"] for j in sem],
                           candidate_ids=sorted({source_cards[j]["id"] for j in lex+sem})))
    return dict(model=EMBEDDING_MODEL, per_channel_k=k, method="char-tfidf union contextual-e5", rows=result)


def discovery(llm, sample, recalled, out):
    by_id = {c["id"]: c for c in cards(sample)}

    def discover(row):
        allowed = set(row["candidate_ids"])
        def validate(result):
            pairs = result["matches"]
            if not isinstance(pairs, list):
                raise ValueError("matches must be list")
            ids = [p["target_id"] for p in pairs]
            if (any(type(i) is not int or i not in allowed for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("invalid or duplicate target_id")
            return dict(source_id=row["id"], matches=pairs)
        return request(llm, "synonym-discovery", dict(source=by_id[row["id"]],
                       candidates=[by_id[i] for i in row["candidate_ids"]]), """找出源关系的同义表达候选。
必须是相同意思换说法、相同主客体方向可以互换。上下位、作用相反、用途与实际效果、
可能与实际、同一对实体上碰巧都真，均不是同义。定义仅作义项线索，不把局部实体限制
当成词义。疑似多义可提候选供后续全Assertion检查。没有同义候选则返回空列表。
返回 {"matches":[{"target_id":真实候选ID,"reason":"同义依据"}]}。""", validate)

    proposed = parallel(recalled["rows"], discover, out, "synonym_discovery")
    pairs = sorted({tuple(sorted((row["source_id"], m["target_id"])))
                    for row in proposed for m in row["matches"]})
    save_json(out/"synonym_proposals.json", dict(proposals=proposed, pairs=pairs))
    return pairs


def review_synonyms(llm, sample, pairs, out):
    by_id = {c["id"]: c for c in cards(sample)}
    facts = defaultdict(list)
    for f in sample["facts"]:
        facts[f["relation_type_id"]].append(f)
    jobs = [(pair, batch) for pair in pairs
            for all_facts in [facts[pair[0]]+facts[pair[1]]]
            for start in range(0, len(all_facts), 12)
            for batch in [all_facts[start:start+12]]]

    def review(job):
        pair, batch = job
        def validate(result):
            checked = keyed(result["checks"], "id", [f["id"] for f in batch])
            if any(type(x.get("equivalent")) is not bool for x in checked.values()):
                raise ValueError("equivalent must be boolean")
            return dict(pair=pair, checks=list(checked.values()))
        return request(llm, "synonym-review", dict(relations=[by_id[i] for i in pair],
                       facts=[dict(**context(f), original_relation_id=f["relation_type_id"]) for f in batch]),
                       """独立核验两个关系是否为相同意思的不同表达，不要默认候选正确。
逐Assertion判断当前旧关系换成另一关系是否保留同一含义：双向同义，不能只是新三元组也真。
保持核心动作、主宾角色、方向、否定、可能性与明确的程度差异；目的不等于实际效果。
“用于”是用途，并不要求已经实际使用。定义可能局部化，结合实际义项判断。
对于多义关系逐条判断，不把一个用法的同义扩展到别的用法。
返回 {"checks":[{"id":真实AssertionID,"equivalent":true或false,"reason":""}]}，覆盖全部ID。""", validate)

    return parallel(jobs, review, out, "synonym_review")


def canonicalize(sample, reviews):
    """Only directly reviewed source-to-root mappings; mixed Claims remain unchanged."""
    pair_checks = defaultdict(dict)
    for row in reviews:
        pair_checks[tuple(row["pair"])].update({x["id"]: x["equivalent"] for x in row["checks"]})
    facts = defaultdict(list)
    for f in sample["facts"]:
        facts[f["claim_id"]].append(f)
    order = sorted(sample["relations"], key=lambda r: (-r["uses"], r["id"]))
    roots = []
    claim_map = {}
    for relation in order:
        rid = relation["id"]
        own = [c for c in sample["claims"] if c["relation_type_id"] == rid]
        for claim in own:
            target = rid
            for root in roots:
                checks = pair_checks.get(tuple(sorted((rid, root))), {})
                # All uses of the canonical name must support this equivalence too.
                target_facts = [f for f in sample["facts"] if f["relation_type_id"] == root]
                if all(checks.get(f["id"]) is True for f in facts[claim["id"]]+target_facts):
                    target = root
                    break
            claim_map[claim["id"]] = target
        if any(claim_map[c["id"]] == rid for c in own):
            roots.append(rid)
    fine = dict(sample)
    fine["facts"] = [{**f, "relation_type_id": claim_map[f["claim_id"]]} for f in sample["facts"]]
    fine["claims"] = [{**c, "relation_type_id": claim_map[c["id"]]} for c in sample["claims"]]
    fine["relations"] = [r for r in sample["relations"] if r["id"] in set(claim_map.values())]
    return fine, claim_map


def validate_catalog(result, fine_ids, maximum=35):
    rows = result["categories"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError("category count out of bounds")
    names = set()
    for i, row in enumerate(rows, 1):
        for field in ("name", "definition", "inclusion_rule", "exclusion_rule"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("missing " + field)
        if row["name"] in names:
            raise ValueError("duplicate category name")
        names.add(row["name"])
        ids = row.get("support_relation_ids")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x not in fine_ids for x in ids):
            raise ValueError("invalid supporting relation IDs")
        row["id"] = i
    return rows


def categories(llm, fine, out, *, fact_limit=None):
    source_cards = cards(fine)
    def propose(batch):
        return request(llm, "category-induction-local", batch, """从这些细关系及Assertion提出候选粗类别，最多15类。
粗类别只是检索分类，不替换谓词。提高/降低可同属影响，高于/低于可同属比较。
每类给纳入/排除与冲突优先规则。保留细关系所有含义，不用相关/其他兜底。
返回 {"categories":[{"name":"", "definition":"", "inclusion_rule":"",
"exclusion_rule":"", "support_relation_ids":[本批真实细关系ID]}]}。""",
                       lambda x: validate_catalog(x, {r["id"] for r in batch}, 15))
    proposals = parallel([source_cards[i:i+60] for i in range(0, len(source_cards), 60)],
                         propose, out, "category_proposals")
    induction_input = dict(local_candidates=proposals,
                           fine_relations=[dict(id=r["id"], name=r["name"]) for r in source_cards])
    catalog = request(llm, "category-induction", induction_input, """根据全部细关系和带来源的局部候选，整合一张全局单层粗类别表。
局部候选来自细关系及实际Assertion；合并重复候选、统一抽象层次，不直接照搬局部目录。
目标是分类、导航和检索，不是改写谓词，不要求粗类别替换原谓词后组成三元组。
细关系及Assertion全部保留。提高/降低/促进/抑制可以同属影响；高于/低于可同属比较。
保持类别抽象层次一致，优先形成10至30类，最多35类，不为凑数细分。
每类写清纳入规则和与邻近类别的排除/优先规则。用途与实际影响可分开。
不要用“其他/相关/涉及”兜底吞掉难例；无法归类可以保留待定。
分类不能改变细关系、端点、否定、可能性或Assertion。支持ID只是目录生成来源。
返回 {"categories":[{"name":"", "definition":"", "inclusion_rule":"",
"exclusion_rule":"与哪些类区分，冲突时如何优先", "support_relation_ids":[真实细关系ID]}]}。""",
                      lambda x: validate_catalog(x, {r["id"] for r in fine["relations"]}))
    save_json(out/"coarse_catalog.json", catalog)
    # Provenance is not an assignment hint.
    public = [{k: v for k, v in c.items() if k != "support_relation_ids"} for c in catalog]
    cards_by_id = {r["id"]: r for r in source_cards}
    ids = {c["id"] for c in catalog}

    def classify(batch):
        def validate(result):
            rows = keyed(result["classifications"], "id", [f["id"] for f in batch])
            for row in rows.values():
                cid = row.get("category_id")
                if cid is not None and (type(cid) is not int or cid not in ids):
                    raise ValueError("unknown category_id")
            return list(rows.values())
        payload = dict(catalog=public, facts=[dict(**context(f),
                       fine_relation=cards_by_id[f["relation_type_id"]]["name"],
                       fine_definition=cards_by_id[f["relation_type_id"]]["original_definition"]) for f in batch])
        return request(llm, "category-assignment", payload, """按完整目录为每条关系用法选择一个主要粗类别。
这是分类标注，不是三元组替换或单向蕴含判断。具体细关系及Assertion仍然保留。
允许相反作用归入同类；按目录纳入/排除及优先规则处理用途、实际作用等边界。
结合细关系与Assertion消歧，不根据端点主题分类。类别缺失或有无法排除的歧义则category_id=null。
返回 {"classifications":[{"id":真实AssertionID,"category_id":整数或null,"reason":""}]}。覆盖全部ID。""", validate)

    assignment_facts = fine["facts"][:fact_limit]
    batches = [assignment_facts[i:i+10] for i in range(0, len(assignment_facts), 10)]
    mappings = [x for batch in parallel(batches, classify, out, "category_assignment") for x in batch]
    by_fact = {f["id"]: f for f in fine["facts"]}
    cats = {c["id"]: c for c in public}
    assigned = [m for m in mappings if m["category_id"] is not None]

    def audit(batch):
        def validate(result):
            rows = keyed(result["checks"], "id", [m["id"] for m in batch])
            if any(type(r.get("valid")) is not bool for r in rows.values()):
                raise ValueError("valid must be boolean")
            return list(rows.values())
        return request(llm, "category-review", dict(catalog=public, items=[dict(
            fact=context(by_fact[m["id"]]), fine_relation=cards_by_id[by_fact[m["id"]]["relation_type_id"]]["name"],
            proposed_category=cats[m["category_id"]]) for m in batch]), """独立核验分类是否符合目录边界。
粗类别只是组织关系的标签，不改写事实，不要求它是原事实蕴含的二元谓词。
正负作用可以同类；细节、方向、条件在细关系和Assertion中保留。判断是否分错类别或违反优先规则。
不查看、不沿用分配者理由，不因已分配就接受。无法确定则valid=false。
返回 {"checks":[{"id":真实AssertionID,"valid":true或false,"reason":""}]}。覆盖全部ID。""", validate)

    checks = [x for batch in parallel([assigned[i:i+10] for i in range(0, len(assigned), 10)],
              audit, out, "category_review") for x in batch]
    checked = {r["id"]: r for r in checks}
    for row in mappings:
        row["review"] = checked.get(row["id"])
        row["accepted"] = checked.get(row["id"], {}).get("valid") is True
        row["fine_relation_id"] = by_fact[row["id"]]["relation_type_id"]
    save_json(out/"category_mappings.json", mappings)
    return catalog, mappings


def materialize(path, sample, fine, claim_map, catalog, mappings):
    if path.exists():
        raise ValueError("output database exists")
    keyed(mappings, "id", [f["id"] for f in sample["facts"]])
    with sqlite3.connect(path) as c:
        c.execute("PRAGMA foreign_keys=ON")
        c.executescript("""CREATE TABLE source_rows(kind TEXT,id INTEGER,payload TEXT,PRIMARY KEY(kind,id));
        CREATE TABLE fine_relations(id INTEGER PRIMARY KEY,name TEXT,definition TEXT);
        CREATE TABLE coarse_categories(id INTEGER PRIMARY KEY,name TEXT,definition TEXT,payload TEXT);
        CREATE TABLE fine_claims(id INTEGER PRIMARY KEY,subject_id INTEGER,relation_id INTEGER REFERENCES fine_relations(id),
          object_id INTEGER,UNIQUE(subject_id,relation_id,object_id));
        CREATE TABLE claim_lineage(source_claim_id INTEGER PRIMARY KEY,fine_claim_id INTEGER REFERENCES fine_claims(id));
        CREATE TABLE assertion_classification(assertion_id INTEGER PRIMARY KEY,source_claim_id INTEGER REFERENCES claim_lineage(source_claim_id),
          fine_relation_id INTEGER REFERENCES fine_relations(id),category_id INTEGER REFERENCES coarse_categories(id),
          proposed_category_id INTEGER REFERENCES coarse_categories(id),status TEXT,judgment TEXT);
        CREATE VIEW classified_facts AS SELECT a.*,f.subject_id,f.object_id,r.name fine_relation,k.name coarse_category
          FROM assertion_classification a JOIN claim_lineage l ON l.source_claim_id=a.source_claim_id
          JOIN fine_claims f ON f.id=l.fine_claim_id JOIN fine_relations r ON r.id=a.fine_relation_id
          LEFT JOIN coarse_categories k ON k.id=a.category_id;""")
        for kind in ("relations", "claims", "facts", "evidence"):
            c.executemany("INSERT INTO source_rows VALUES(?,?,?)", [(kind, r["id"], dumps(r)) for r in sample[kind]])
        c.executemany("INSERT INTO fine_relations VALUES(?,?,?)", [(r["id"], r["name"], r["description"]) for r in fine["relations"]])
        c.executemany("INSERT INTO coarse_categories VALUES(?,?,?,?)", [(r["id"], r["name"], r["definition"], dumps(r)) for r in catalog])
        for row in sample["claims"]:
            key = row["subject_id"], claim_map[row["id"]], row["object_id"]
            c.execute("INSERT OR IGNORE INTO fine_claims(subject_id,relation_id,object_id) VALUES(?,?,?)", key)
            fid = c.execute("SELECT id FROM fine_claims WHERE subject_id=? AND relation_id=? AND object_id=?", key).fetchone()[0]
            c.execute("INSERT INTO claim_lineage VALUES(?,?)", (row["id"], fid))
        facts = {f["id"]: f for f in sample["facts"]}
        for row in mappings:
            f = facts[row["id"]]
            c.execute("INSERT INTO assertion_classification VALUES(?,?,?,?,?,?,?)", (f["id"], f["claim_id"],
                      claim_map[f["claim_id"]], row["category_id"] if row["accepted"] else None,
                      row["category_id"], "accepted" if row["accepted"] else "pending", dumps(row)))
        if c.execute("PRAGMA foreign_key_check").fetchall() or c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("derived database integrity failure")
        for kind in ("relations", "claims", "facts", "evidence"):
            stored = [json.loads(r[0]) for r in c.execute("SELECT payload FROM source_rows WHERE kind=? ORDER BY id", (kind,))]
            if stored != sorted(sample[kind], key=lambda r: r["id"]):
                raise ValueError("source lineage mismatch")
        return c.execute("SELECT count(*) FROM fine_claims").fetchone()[0]


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args.db, args.sample_size or 80, all_relations=args.sample_size == 0)
    manifest = dict(version=VERSION, source_db=str(args.db.resolve()), source_sha256=digest(args.db),
                    model=MODEL, endpoint=ENDPOINT, max_concurrency=6, sample_size=args.sample_size, smoke_only=args.smoke_only,
                    used_types=len(sample["relations"]), claims=len(sample["claims"]), assertions=len(sample["facts"]),
                    source_hashes={p: digest(p) for p in [__file__, "experiments/relation_two_stage_pilot.py",
                       "experiments/relation_coarsening_experiment.py", "kg/llm.py", "kg/embeddings.py"]},
                    semantics="synonym replacement plus category annotation, never coarse predicate replacement")
    if (out/"manifest.json").exists() and json.loads((out/"manifest.json").read_text()) != manifest:
        raise ValueError("manifest differs; use a fresh output directory")
    if (out/"summary.json").exists():
        raise ValueError("completed output exists")
    save_json(out/"manifest.json", manifest)
    save_json(out/"input.json", sample)
    for name in manifest["source_hashes"]:
        target = out/"source"/Path(name).name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    log(f"prepared {manifest['used_types']} relations, {manifest['assertions']} Assertions; MiniMax-M3 cap=6")
    if args.prepare_only:
        return
    llm = CachedLLM(base_url=ENDPOINT, api_key=LLMConfig.from_env().api_key, model=MODEL,
                    max_concurrency=6, cache_path=out/"llm_cache.jsonl")
    recalled = retrieval(cards(sample))
    save_json(out/"retrieval.json", recalled)
    if args.smoke_only:
        recalled["rows"] = recalled["rows"][:2]
    pairs = discovery(llm, sample, recalled, out)
    reviews = review_synonyms(llm, sample, pairs, out)
    fine, claim_map = canonicalize(sample, reviews)
    save_json(out/"fine_claim_mapping.json", claim_map)
    save_json(out/"fine_catalog.json", cards(fine))
    if args.smoke_only:
        catalog, mappings = categories(llm, fine, out, fact_limit=4)
        save_json(out/"smoke_summary.json", dict(status="complete", discovery_items=2,
                  proposed_pairs=len(pairs), categories=len(catalog), classified_assertions=len(mappings), metrics=llm.metrics))
        return
    catalog, mappings = categories(llm, fine, out)
    if digest(args.db) != manifest["source_sha256"]:
        raise ValueError("source changed during execution")
    staging = out/"derived.building.db"
    if staging.exists():
        staging.unlink()  # Incomplete derived output only; all expensive work is cached.
    edges = materialize(staging, sample, fine, claim_map, catalog, mappings)
    staging.replace(out/"derived.db")
    accepted = [r for r in mappings if r["accepted"]]
    distribution = Counter(r["category_id"] for r in accepted)
    relation_categories = defaultdict(set)
    type_counts = defaultdict(Counter)
    for row in mappings:
        type_counts[row["fine_relation_id"]]["accepted" if row["accepted"] else "pending"] += 1
    for row in accepted:
        relation_categories[row["fine_relation_id"]].add(row["category_id"])
    save_json(out/"fine_category_mapping.json", [dict(fine_relation_id=r["id"],
              category_ids=sorted(relation_categories[r["id"]]),
              assertion_counts=dict(type_counts[r["id"]]),
              status="mixed_contexts" if len(relation_categories[r["id"]]) > 1 else
                     "partly_pending" if type_counts[r["id"]]["pending"] and relation_categories[r["id"]] else
                     "classified" if relation_categories[r["id"]] else "pending") for r in fine["relations"]])
    summary = dict(status="complete", original_types=len(sample["relations"]), fine_types=len(fine["relations"]),
                   coarse_categories=len(catalog), used_categories=len(distribution), source_claims=len(sample["claims"]),
                   fine_claims=edges, assertions=len(mappings), accepted_assertions=len(accepted),
                   pending_assertions=len(mappings)-len(accepted), coverage=len(accepted)/len(mappings),
                   fine_replaced_claims=sum(claim_map[c["id"]] != c["relation_type_id"] for c in sample["claims"]),
                   synonym_candidate_pairs=len(pairs), mixed_category_fine_types=sum(len(v)>1 for v in relation_categories.values()),
                   distribution=dict(distribution), source_unchanged=True, metrics=llm.metrics,
                   quality="same-model separate-prompt review; coverage is not independently measured accuracy")
    save_json(out/"summary.json", summary)
    names = {r["id"]: r["name"] for r in sample["relations"]}
    cats = {r["id"]: r["name"] for r in catalog}
    facts = {f["id"]: f for f in sample["facts"]}
    lines = ["# 细关系与粗类别实验", "", "自动复核覆盖率不是独立准确率。", "", "```json", dumps(summary), "```",
             "", "| Assertion | 原关系 | 规范细关系 | 粗类别 | Assertion 原文 |", "|---|---|---|---|---|"]
    for row in sorted(mappings, key=lambda r: r["id"]):
        f = facts[row["id"]]
        values = [str(f["id"]), names[f["relation_type_id"]], names[claim_map[f["claim_id"]]],
                  cats[row["category_id"]] if row["accepted"] else "待定", f["statement"]]
        lines.append("| " + " | ".join(v.replace("|", "／").replace("\n", " ") for v in values) + " |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    save_json(out/"progress.json", summary)
    log(dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="0=all used types; otherwise old fixed pilot sampling")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save_json(args.output_dir/"failure.json", dict(status="failed", error_type=type(error).__name__,
                  message=str(error), time=time.time()))
        raise


if __name__ == "__main__":
    main()
