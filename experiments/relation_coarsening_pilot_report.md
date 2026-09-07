# Relation coarsening pilot report

Date: 2026-08-31

## Scope

- Read-only source: `tmp/d2l-full1105-c6-20260817-193600.db`
- Full graph: 919 used relation types and 3,139 claims.
- Frozen seed catalog: 48 relation types used at least 10 times.
- Pilot: 200 stratified low-frequency relation types, representing 545 claims.
- Model: `MiniMax-M3` through the local API Gateway.
- Global request concurrency: at most 8.

The shared seed-mapping pass mapped 61/200 pilot relation types and 182/545
claims. This is the baseline for both arms.

## Results

| Arm | Candidates | New candidates | Mapped types | Mapped claims | Type coverage | Claim coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared 48 seeds | 48 | 0 | 61/200 | 182/545 | 30.50% | 33.39% |
| Embedding, cosine distance 0.075 | 48 | 0 | 61/200 | 182/545 | 30.50% | 33.39% |
| Embedding, cosine distance 0.060 | 48 | 0 | 61/200 | 182/545 | 30.50% | 33.39% |
| Name-only Embedding, partial subsets, minimum 3 | 48 | 0 | 61/200 | 182/545 | 30.50% | 33.39% |
| Name-only Embedding, partial subsets, minimum 2 | 58 | 10 | 88/200 | 263/545 | 44.00% | 48.26% |
| Name-only Embedding, manually trusted subset | 56 | 8 | 82/200 | 246/545 | 41.00% | 45.14% |
| Direct LLM, unfiltered | 61 | 13 | 116/200 | 318/545 | 58.00% | 58.35% |
| Direct LLM, candidate gate | 52 | 4 | 77/200 | 240/545 | 38.50% | 44.04% |
| Direct LLM, manually trusted subset | 50 | 2 | 69/200 | 205/545 | 34.50% | 37.61% |

The last row is not a new run. It removes two candidate-gate false positives
from the strict run and counts one salvageable candidate under a single atomic
parent predicate.

## Embedding arm

At distance 0.075, 12 eligible clusters were proposed and MiniMax rejected all
of them. At 0.060, four smaller clusters were proposed and all were again
rejected. The rejections were semantically justified. Examples include:

- `可表示`, `可体现于`, and `是…的表示形式`: capability, occurrence context,
  and representation-form roles are different.
- `训练`, `训练以最大化为目标`, and `学习`: trainer-to-model,
  model-to-objective, and learner-to-item have different endpoint roles.
- `用于调整`, `用于确定`, and `调整输入以适配`: related vocabulary does not
  imply a shared binary predicate with the same direction.

Raw relation-name/description embeddings therefore organize lexical or topical
similarity better than predicate argument structure. This arm was conservative
but produced no compression gain in the pilot.

## Simplified name-only Embedding rerun

A user-directed rerun simplified the Embedding arm in two ways:

1. Embed only the relation name. Definitions and evidence do not influence
   clustering.
2. After clustering, let MiniMax inspect definitions, all sampled Assertions,
   and actual subject/object examples, then select only compatible subsets.
   A five-member cluster may therefore yield a three-member candidate while
   the other two members remain residual.

At distance 0.075 the method produced nine name clusters; at 0.10 it produced
13. Requiring at least three compatible members still produced no candidate.
Inspection showed that the name clusters had correctly recalled several exact
or near-exact pairs, including `可转换为/可转化为`, `以…为例/以…为示例模型`,
and the two `重新解释` relations. They were rejected only because each group had
two members.

A final run therefore used distance 0.10, minimum two member types, and minimum
two total uses. Across three residual rounds it retained 10 candidates and
mapped 27 additional relation types. Surface coverage reached 88/200 types and
263/545 claims under 58 candidates.

Endpoint-level manual review rejected six mapped relation types:

- Remove `表示...的轴向尺寸` and `可表示` from the generated `表示` parent.
  The former represents the dimensions of the object rather than the object;
  the latter is capability rather than an actual representation relation.
- Reject the `创建/开发了` candidate. In particular, the real endpoint pair
  `Chainer -> 深度学习框架` does not entail that Chainer developed the class of
  deep-learning frameworks.
- Reject `学习得到`, which mixes learner-to-content and information-source-to-
  learnable-content subject roles.

The remaining eight candidates cover 21 new relation types and 64 new claims:

- `存在于`
- `可转换为`
- `以…为例`
- `以…为框架重新解释`
- a reduced `表示` candidate after removing the two invalid members
- `可表现为/体现于`
- `在…中有效`
- `是…的应用任务` (renamed from `是…的任务/问题`)

The manually trusted result is therefore 56 candidates, 82/200 mapped relation
types, and 246/545 mapped claims: 41.00% type coverage and 45.14% claim
coverage. This is materially better than both the original Embedding arm and
the manually trusted direct-LLM result (34.50% types and 37.61% claims).

## Full-triple-validation rerun

A subsequent rerun retained the name-only, distance-0.10, minimum-two setup but
added a final gate over every stored claim triple for each proposed type
mapping. Candidate discovery still used at most three representative examples;
accepted mappings then had to return one numbered judgment for every claim.

The database loader recovered all 3,139 claims across 919 used relation types,
with no difference between each type's claim count and loaded full-example
count.

The full gate changed the shared baseline before candidate discovery:

- Previous seed baseline: 61/200 types and 182/545 claims.
- Full-validated seed baseline: 51/200 types and 147/545 claims.
- Ten prior seed mappings returned to the residual pool.

Across three Embedding rounds, the rerun generated nine retained new
candidates and mapped 24 additional types. The final structural result was 57
candidates, 75/200 mapped relation types, and 224/545 mapped claims: 37.50% type
coverage and 41.10% claim coverage.

In the first discovery round, representative-example mapping accepted 21
types. Full validation rejected two:

- `表示...的轴向尺寸 -> 表示` was correctly rejected. The exact endpoint
  projection changes an axis or tensor into its hidden dimension information.
- `是…的机制 -> 存在于` exposed an over-strict gate. MiniMax explicitly said
  that `自注意力存在于Transformer编码器` and `推迟初始化存在于深度学习框架`
  were true, but rejected them because `存在于` loses mechanism information.

The second case shows that the full gate currently mixes two independent
questions: whether every projected coarse triple is true, and whether the
coarse relation preserves enough information. Assertion already preserves the
detail, so a truth-only mapping policy should not reject a true projection only
for being coarser. Conversely, the gate still accepted `可表示 -> 表示`, treating
capability as part of the representation family. Full data visibility therefore
does not by itself settle the intended abstraction policy.

Before a full-book run, the final gate should be split into a truth check and a
separate granularity-quality check. The former should reject endpoint changes
and false triples; the latter should report information loss without silently
overriding a truth-preserving coarse-family mapping.

## Truth-only full-triple-validation rerun

The full gate was then rewritten to ask only whether every projected triple is
true when its original Assertion is true. Losing detail is explicitly allowed,
because Assertion remains the lossless record. The gate still rejects hidden
endpoint changes: for example, a relation to an object's dimension, property,
or output cannot be projected as a relation to the object itself. It also runs
the counterexample test directly: can the Assertion be true while the projected
triple is false?

A focused regression confirmed both sides of the intended boundary:

- `表示...的轴向尺寸 -> 表示` is rejected because it removes the hidden
  dimension endpoint.
- `是…的机制 -> 存在于` is accepted because all original triples still entail
  the coarser location triples, while mechanism detail remains in Assertion.

The final 200-type, three-round rerun retained 57 candidates and mapped 81/200
relation types and 246/545 claims: 40.50% type coverage and 45.14% claim
coverage. The shared seed stage mapped 59 types; nine new Embedding candidates
added 22 types. The new candidates were:

- `是学习的基础`
- `以...为框架重新解释`
- `以…为例`
- `可转化为`
- `不包含`
- `出现于`
- `X表示Y`
- `X在Y中有效`
- `X的成员均属于Y`

Full validation rejected two old seed mappings and one newly proposed mapping:

- `融合了思想 -> 采用` was correctly rejected because absorbing one idea does
  not entail adopting the whole method.
- `是…的体现 -> 是…的具体体现` was rejected because one stored Assertion only
  says attention weights are used to illustrate attention aggregation, not
  that the weights are themselves its concrete manifestation.
- `互为同一示例的替代实现 -> 与…并列` was conservatively rejected because one
  Assertion says the two algorithms are not essentially different algorithms;
  this is defensible but remains the least certain of the three rejections.

This result essentially reproduces the manually trusted name-only result:
81 versus 82 mapped types and exactly the same 246 mapped claims, without the
known invalid dimension-to-object member. It is therefore the best calibrated
automated result in this pilot, though it is still a 200-type experiment rather
than evidence that the complete book will fit below 100 candidates.

## Direct LLM arm

The unfiltered run found useful structure, but 58.35% is not a trustworthy
quality result. Manual review of all 13 new candidates found:

- Clearly acceptable: 3 candidates, covering 13 relation types and 33 claims.
- Borderline or definition-dependent: 5 candidates, covering 24 types and 58
  claims.
- Reject: 5 candidates, covering 18 types and 45 claims.

Representative failures:

- `设计影响/沿用` reverses the semantic role between `沿用做法` and
  `为…提供灵感`.
- `不具有/无法保证` merges absence, part-whole exclusion, guarantee, and
  `可能不提供`; the last item does not entail definite absence.
- `在…语境下定义/成立` incorrectly included an architecture construction
  pattern that is not a context relation.
- `创新性推动/变革` treats `率先推出` as entailing transformative impact.

## Candidate-gate rerun

A second pass required each candidate to be a single atomic binary predicate,
to preserve endpoint roles, direction, polarity, and modality, and to be
entailed by every proposed source relation. The gate correctly rejected all
three broad second-round proposals, including a catch-all causal-effect type
and a definite-negative type containing `可能不提供`.

The same MiniMax model nevertheless approved three problematic or mixed first-
round abstractions. After reviewing the actual retained members:

- Keep `在…维度上多于/超过`: 5 types, 13 claims.
- Salvage `度量或表征` by naming its single common parent `刻画/表征`: the
  retained `量化度量`, `是…的指标`, and `描述` relations all entail that parent
  (3 types, 10 claims).
- Reject `表示`: it mixes representation form, representational capability,
  active representation, encoding, and `可学习得到`.
- Reject `构成先决条件`: historical predecessor and structural foundation do
  not necessarily entail a necessary prerequisite.

This leaves a manually trusted total of 69/200 relation types and 205/545
claims under 50 candidates.

## Decision

Do not yet launch the 871-relation full run, but continue from the simplified
name-only Embedding arm with the truth-only, all-triple gate.

- The original name-plus-definition Embedding arm discovers no usable new
  candidate. Name-only recall plus LLM subset selection discovers useful,
  mostly narrow candidates.
- The truth-only full gate removes the known endpoint-changing false positive
  while retaining valid coarse projections that lose only Assertion-preserved
  detail. Its 246-claim coverage matches the manually trusted result.
- Direct LLM has higher surface coverage, but neither ordinary membership
  judgment nor same-model self-review is reliable enough for autonomous
  mapping.
- The fewer-than-100 target is no longer obviously impossible, but remains
  uncertain. After the trusted name-only run, 56 candidates cover 82/200 pilot
  types and leave 118 residual types. The remaining 44 candidate slots would
  need to absorb about 2.68 pilot relation types each; the eight trusted new
  candidates currently absorb 2.63 each. Later residual rounds are likely
  harder, so this near-equality should not be treated as a successful forecast.

The next experiment should scale this exact setup beyond the 200-type pilot and
measure whether later residual rounds continue producing candidates with at
least two valid member types. The stored experimental object remains simpler
than a relation card: a name, one-sentence predicate definition, member IDs,
all-triple truth judgments, and the audit decision.

## Artifacts

- Implementation: `experiments/relation_coarsening_experiment.py`
- Tests: `tests/test_relation_coarsening_experiment.py`
- Embedding pilot: `tmp/relation-coarsening-pilot200-m3-gateway-20260831/`
- Tight embedding sweep: `tmp/relation-coarsening-pilot200-embedding060-m3-gateway-20260831/`
- Name-only, partial-subset, minimum-three sweep: `tmp/relation-coarsening-pilot200-embedding-names-subsets100-m3-gateway-20260831/`
- Name-only, partial-subset, minimum-two run: `tmp/relation-coarsening-pilot200-embedding-names-subsets100-min2-m3-gateway-20260831/`
- Full-triple-validation rerun: `tmp/relation-coarsening-pilot200-embedding-names-subsets100-min2-fulltriples-m3-gateway-20260831/`
- Truth-only full-triple-validation v2 rerun: `tmp/relation-coarsening-pilot200-embedding-names-subsets100-min2-truthonly-v2-m3-gateway-20260901/`
- Focused truth-boundary regression cache: `tmp/relation-coarsening-pilot200-embedding-names-subsets100-min2-truthonly-m3-gateway-20260901/focused_validation_cache.jsonl`
- Corrected direct-LLM pilot: `tmp/relation-coarsening-pilot200-llm-corrected-m3-gateway-20260831/`
- Candidate-gate rerun: `tmp/relation-coarsening-pilot200-llm-strict-m3-gateway-20260831/`
