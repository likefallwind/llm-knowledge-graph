# AI Knowledge Graph Agent Guide

## 1. 项目目标

本项目持续阅读人工维护目录中的高质量 AI 语料，从中抽取并逐步合并形成覆盖人工智能主要知识体系的知识图谱，为知识导航和个性化教学提供基础。

核心闭环只有一条：

```text
Read → Extract → Resolve → Merge → Repeat
```

衡量进展时，不以节点和边的数量代替质量。首先保证知识来自可定位语料，其次才扩大覆盖范围。

注意，本仓库目的是为了做整个AI领域知识图谱，想要做好这个事情，规则一定要简单，代码易维护和拓展，不要加入过多的工程代码，也不要加入过于琐碎的细节。算法决策不确定时候可以和我讨论

## 2. 必须遵守的知识边界

LLM 可以阅读、抽取、归类、消歧、规范命名和裁判证据，但不能成为知识来源。

必须始终满足：

1. 每个 Entity 至少有一段能在 Source 正文中定位的 Evidence。
2. 每个 Claim 至少有一段能在 Source 正文中定位、且确实表达该关系的 Evidence。
3. LLM 凭参数记忆补充、改写或推断出的知识不能入库。
4. 翻译和规范命名不是新来源，必须保留原始 Source、LLM quote 和程序取得的真实原文。
5. 共现、章节顺序、超链接和“尤其、涉及、用于”等弱表达不能单独证明类型化关系。

一句话原则：

> LLM 负责理解和判断语料，语料负责提供知识。

## 3. 最小知识模型

只保留四类知识对象：

- `Source`：语料的一个不可变内容版本。
- `Entity`：可稳定复指、可独立定义或学习的知识对象。
- `Claim`：两个规范 Entity 之间的关系三元组。
- `Evidence`：支持或反对 Entity/Claim 的完整溯源记录，同时保存 LLM quote、Passage 引用和程序取得的真实原文。

别名是 Entity 的属性，EntityObservation/ClaimObservation 是语料观察记录，
`source_progress` 是断点续跑记录；它们都不是新的知识对象。

Entity 主类型只能是：

```text
resource
criterion
data
task
solution
concept
```

类型根据 definition 判断，不按名称字面猜测；无法归入前五类时才使用 `concept`。

**类型是 mention 级的观察，不是 Entity 的单值属性。** 每次抽取判出的类型记在 `evidence.observed_entity_type` 上；Entity 层的类型表示是这些观察的汇总（type profile，见 `store.type_profile`），`entities` 表没有 `entity_type` 列。

这样处理是因为一个词确实可能同时属于多个类型，而且取决于语境：「深度学习」既是一族做法（`solution`），也是一个研究方向（`concept`）。这类词在语言学上叫 dot object，两个义项不互斥、同时成立，所以「给它选一个正确类型」这个问题本身就是错的。

由此产生的约束：

- 不要把 profile 折叠成单一类型（argmax、多数派、首次观察都不行），它不是投票。
- profile 同时给 `observations` 和 `sources`：前者反映语料分布，一本书反复使用会刷高；后者反映有多少独立来源这样判。两者含义不同，不要混用。
- 历史 Evidence 的类型留空，不得用实体旧类型回填——我们不知道当时那次观察判的是什么，回填等于编造观察记录。
- 需要单值类型的场景（例如将来的导航查询）应当在查询层定阈值，并明确它是一个可调策略，不是数据库里的既成事实。

Claim 关系只能是：

```text
is_a
part_of
prerequisite_of
```

不要添加 `related_to`。新增关系前必须证明它反复出现、语义边界清楚、对导航或教学确实有用，并同步更新 schema、提示词、验证、测试和文档。

## 4. 定义只有一个来源

六种类型和三种关系的定义写在 `kg/ontology.py`，它是唯一来源。抽取、关系裁判和身份裁决三处提示词都从它渲染，本文和 `design/algorithm.md` 只做摘要和说明，不再各写一遍。

定义一律写成**判定测试 + 排除项 + 正反例**，不写同义改写。同义改写（“是……的一种”“是……的组成部分”）在边界样本上给不出确定答案，正是早期假阳性的来源。

要点摘要：

- `is_a`：实例测试——任取一个 subject，它本身就是一个 object。领域归属和构件关系都不是 `is_a`。两端 `entity_type` 通常相同，但只作自查线索，不作否决理由（见 4.1）。
- `part_of`：构件、正文明确列出的阶段，或**原文明确陈述的领域归属**（三者之一即可）。
- `prerequisite_of`：原文明确陈述不先掌握 subject 就无法学习 object。教材先后顺序本身不是证据。

关于领域归属：`深度学习 / 人工智能` 这类子领域关系既过不了实例测试，也不是构件，本体论上通常单列为 `subfield_of`。第一版**不新增这种关系**，暂并入 `part_of`——因为并入后它是 `claims` 表里的一等数据，将来要拆分可以纯本地重判，而拒掉则只剩 `source_progress.result` 里的字符串，恢复必须重跑全部抽取。等真实语料统计出它的实际频次，再按第 3 节的门槛决定是否单列。

修改 `kg/ontology.py` 的语义时，必须同时 bump `extraction.EXTRACTION_PROMPT_VERSION`、`resolution.RESOLUTION_PROMPT_VERSION` 和 `validation.VALIDATION_PROMPT_VERSION`，否则旧的 `done` 片段会被错误跳过。

### 4.1 不要用类型一致性做门槛

`is_a` 在本体论上要求两端同类型，但**不得**把它实现为硬约束——无论是提示词里的“必须”，还是 `kg check` 的检查项。原因有二：

1. 判别力低。`深度学习 is_a 人工智能`（两端多半都是 `concept`）和 `卷积层 is_a 卷积神经网络`（两端都是 `solution`）这两个典型错误，类型一致性一个都拦不住；它只能拦住本来就不会发生的离谱组合。
2. 误伤代价高且不可见。类型判断本身不可靠，一旦作为门槛，正确的 `is_a` 会被静默丢弃，连 `rejected` 记录都不留。

因此类型一致性只作为模型的自查线索出现在定义里，判定始终以实例测试和排除项为准。

### 4.2 已知弱证据风险

真实 MiniMax M3 冒烟曾把“人工智能，特别是神经网络与深度学习的发展”接受为 `神经网络/深度学习 part_of 人工智能`。**注意这条至今仍是假阳性**：`part_of` 放宽的是“领域归属这种语义可以入图”，不是“弱表达可以入图”。“特别是”只做强调，没有陈述任何归属关系，正确判定是 `insufficient`。该句已作为反例写进 `kg/ontology.py` 的 `part_of` 定义。

## 5. 实体对齐规则

实体对齐只允许三个结果：

```text
same       同一个对象，使用已有 Entity
new        不同对象，新建 Entity
uncertain  信息不足，新建独立 Entity，暂不合并
```

工作原则：

- 精确规范名或别名唯一命中时直接复用。
- 字符串相似度只召回候选，不能直接决定合并。
- LLM 必须同时看到新观察、候选定义和已累计 Evidence。
- `uncertain` 不是错误或拒绝；宁可保留重复实体，也不要误合并。
- `reconcile` 只能在模型明确返回 `same` 时合并。
- canonical name 可由 LLM 根据当前观察规范化，但不能借机添加语料没有提供的知识。

## 6. 数据与运行安全

- 当前最小实现默认写入 `data/knowledge.db`。
- `data/kg.db` 是旧复杂 schema 的历史数据库，不得自动迁移、覆盖或修改。
- Source 以 `(source_key, content_hash)` 版本化；相同内容重复运行必须幂等。
- 教材、Markdown 和 HTML 优先按标题 Section 切分，过大 Section 才继续切
  Chunk；无标题语料回退到 Passage 分块。目录层级只用于 Source 结构、定位和
  覆盖检查，不能单独证明知识 Claim。
- 通过 Passage 校验的 EntityObservation 必须在身份解析前保存，并记录后续
  `same/new/uncertain` 结果；默认每个 Chunk 最多抽取 50 个 Entity。
- 相同 `(subject, relation, object)` 只能有一条 Claim；新来源只追加 Evidence。
- LLM 必须同时输出 `model_quote` 和 1–3 个当前片段中的 Passage ID。
- `model_quote` 原样保留；程序根据 Passage ID 取得的 `source_text` 也必须保留，两者不得互相覆盖。
- 当前不实现 quote 与 source text 的精确或模糊匹配；Passage ID 有效即可保留二者。
- Evidence 必须记录 Source 版本、内容哈希、位置、抽取模型和提示词版本；Claim Evidence 还要记录裁判模型和提示词版本。
- Claim Evidence 还要原样保留当时的裁判 verdict 和 reason，供未来争议复核。
- 当前不实现校准队列。未来 LLM 或人类校准应追加引用原始 Evidence 的新记录，不能改写原始 Evidence；稳定引用方式留到校准实验设计时确定。
- Claim 两端必须是已存在 Entity，不允许自环。
- `is_a` 和 `prerequisite_of` 不允许形成循环。
- 失败片段记录为 `failed`，下次运行可重试；不得因为 API 失败写入半条 Claim。
- 不执行大规模语料运行、删除数据库或覆盖历史结果，除非用户明确要求。

## 7. 默认模型

默认直接使用 MiniMax M3：

```text
model: MiniMax-M3
endpoint: https://api.minimaxi.com/v1/text/chatcompletion_v2
api key: MINIMAX_API_KEY
```

除非用户明确指定其他模型或 endpoint，否则不要切换 provider。不得把 API key 写入代码、文档、日志、测试或记忆。

## 8. 开发方式

每次修改前：

1. 阅读 `plan.md`、`design/algorithm.md` 和相关代码。
2. 查看真实数据库、测试或运行结果后再判断问题。
3. 区分语料问题、模型判断问题和算法实现问题。
4. 先问：不增加这个机制，核心闭环是否真的无法工作？

实现时：

- 保持 SQLite 加少量 Python 模块的学术项目形态。
- 每个新增字段和算法步骤都应对应一个可验证问题、可复现实验条件或核心不变量。
- 优先使用可直接阅读的函数、SQL、JSON 导出和小型回归样本，不引入服务层、插件层、任务编排框架或管理后台。
- 失败样本、模型原始判断和来源文本应可直接检查；不要用自动修补链隐藏实验失败。
- 不引入 `proposed/published/shadow` 状态机、复杂审核队列、置信度累乘或生产级并发。
- Parser 对 Passage ID 的存在性和范围检查保持确定性，不查询 LLM。
- Evidence 的正式 `source_text` 只能由程序从 Passage 取得，不能直接采用模型 quote。
- 不用提示词掩盖可以机械验证的错误。
- 不因单条坏数据添加面向个案的规则；优先修正普遍语义边界。
- 代码行为变化时同步更新 `design/algorithm.md` 和相关测试。

## 9. 验证要求

代码修改至少运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q kg tests
git diff --check
```

涉及数据库或流水线时，还应运行：

```bash
python -m kg --db data/knowledge.db status
python -m kg --db data/knowledge.db check
```

涉及 LLM 接口、提示词或响应解析时，应在 `MINIMAX_API_KEY` 可用时做有界真实测试，例如：

```bash
python -m kg --db data/knowledge.db run sources/catalog.json \
  --source-limit 1 --max-chunks 1
```

汇报时必须区分确定性测试、真实模型冒烟和全量语料运行；不能把其中一个说成另一个。

## 10. 第一版范围

第一版需要持续证明：

1. 能批量读取不同来源和版本的语料。
2. Entity 和 Claim 都保存 LLM quote、有效 Passage 引用和程序取得的真实原文。
3. 能用 LLM 判断相似名称是 `same/new/uncertain`。
4. 相同 Claim 能累计多个来源的 Evidence。
5. 允许孤立和暂时重复的 Entity，并能在后续重判合并。
6. 核心闭环可幂等、可续跑地持续运行。

主动搜索、覆盖规划、生产并发、权限系统和复杂人工审核都不属于当前核心。
