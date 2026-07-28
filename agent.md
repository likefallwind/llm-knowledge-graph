# AI Knowledge Graph Agent Guide

## 1. 项目目标

本项目持续阅读人工维护目录中的高质量 AI 语料，从中抽取并逐步合并形成覆盖人工智能主要知识体系的知识图谱，为知识导航和个性化教学提供基础。

核心闭环只有一条：

```text
Read → Extract → Resolve → Merge → Repeat
```

衡量进展时，不以节点和边的数量代替质量。首先保证知识来自可定位语料，其次才扩大覆盖范围。

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

别名是 Entity 的属性，`source_progress` 只是断点续跑记录；它们不是新的知识对象。

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

Claim 关系只能是：

```text
is_a
part_of
prerequisite_of
```

不要添加 `related_to`。新增关系前必须证明它反复出现、语义边界清楚、对导航或教学确实有用，并同步更新 schema、提示词、验证、测试和文档。

## 4. 关系语义不可放宽

- `is_a`：subject 是 object 的一种。相关、使用、作用于、属于同一领域都不是 `is_a`。
- `part_of`：subject 是 object 的真实组成部分或正文明确列出的阶段。领域相关、研究分支、用途和共现默认都不是 `part_of`。
- `prerequisite_of`：理解 subject 是学习 object 的实质性前提。教材先后顺序本身不是证据。

真实 MiniMax M3 冒烟曾把“人工智能，特别是神经网络与深度学习的发展”接受为 `神经网络/深度学习 part_of 人工智能`。这种“特别是”表达并没有明确陈述组成关系，应视为关系裁判的已知弱证据风险，不能把一次模型判定当成关系语义本身。

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
