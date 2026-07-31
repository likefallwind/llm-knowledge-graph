# 核心算法

本文描述当前最小实现实际执行的算法。产品边界见 `plan.md`，开发约束见 `agent.md`。代码行为变化时必须同步更新本文。

本文按学术验证项目组织：算法步骤对应可检验的不变量，模型判断和失败均保留为可观察结果；不把核心流程包装成服务、任务平台或复杂状态机。

## 1. 问题定义

输入是人工维护语料目录中的本地文件或远程 URI，输出是 SQLite 中四类可溯源知识对象：

```text
Source → Entity
       → Claim(Entity, relation, Entity)
       → Evidence(Source, Entity | Claim)
```

知识增长的唯一入口是：

```text
Read → Extract → Resolve → Merge → Repeat
```

模型默认使用 MiniMax M3。模型输出只是候选观察，只有引用当前片段中的有效 Passage、完成实体解析和关系证据裁决后才能写入知识图谱。

通过 Passage 校验的 ClaimObservation 是持久研究记录。Entity/Claim 是当前
已落实的图谱，Observation 端点暂时无法解析时保留为 pending，后续只重放
解析和物化，不重新抽取 Source。

## 2. 核心不变量

数据库在任何完整片段处理后都应满足：

1. 每个 Entity 至少有一条 support Evidence。
2. 每个 Claim 至少有一条 support Evidence。
2.1 Entity 没有单值类型；类型只存在于 Evidence 上，Entity 层按需汇总为 type profile。
3. Evidence 必须且只能指向一个 Entity 或 Claim。
4. Evidence 同时保存 LLM 原始 quote 和程序从 Source Passage 取得的真实原文。
5. Passage ID 必须属于对应 Source 版本和当前抽取片段。
6. Claim 两端必须存在且不同。
7. Claim 唯一键为 `(subject_id, relation, object_id)`。
8. `is_a` 和 `prerequisite_of` 不形成有向循环。
9. 不确定的实体身份不触发合并；不确定的关系不写入。
10. 有效 ClaimObservation 不因端点未解析或语义证据不足而删除。
11. 相同模型与提示词版本不重复裁判同一 Observation。

`kg check` 检查外键、无证据对象和关系循环。语义正确性仍需要语料约束和关系裁判共同保证。

## 3. Source 读取与版本化

### 3.1 人工目录

`kg.sources.load_catalog` 读取 JSON，或在安装 PyYAML 后读取 YAML。每项至少包含：

```text
name
type
path 或 uri
```

可选字段包括稳定 `key`、`version`、`language` 和原始 `uri`。本地相对路径以目录文件所在位置解析；没有 URI 时使用绝对 `file://` URI。

### 3.2 正文提取

当前支持：

- PDF：优先执行 `pdftotext -layout`，否则使用可选依赖 `pypdf`。
- HTML：删除 script/style 等不可见内容、标签和多余空白。
- JSONL：逐行读取字符串，或对象中的 `text/content` 字段。
- 其他文本：按 UTF-8 读取。
- 远程 URI：HTTP 下载后按 Content-Type 处理，远程 PDF 保持二进制解析。

删除 NUL 字符后正文不能为空。

### 3.3 Source 版本

正文哈希为：

```text
content_hash = SHA256(UTF-8 content)
```

数据库唯一键为：

```text
(source_key, content_hash)
```

目录未声明 version 时使用 `content_hash[:12]`。同一逻辑来源内容不变时复用 Source；内容变化时新增不可变版本。

## 4. Passage 与确定性文本分块

### 4.1 稳定 Passage

`kg.sources.segment_text` 先按空行和 PDF 换页符切分自然段。超过 1200 字的长段落，优先在后半段的换行、中文句号、问号、感叹号或英文句号处继续切分。

每个 Passage 保存：

```text
passage_id       P000001、P000002……
text             程序从 Source 取得的真实文本
page/location    页码和字符范围
content_hash     Passage 文本 SHA256
start/end        Source 正文字符位置
```

Passage ID 只在一个不可变 Source 版本内解释。当前分段算法版本为：

```text
source-passages-1
```

### 4.2 带 Passage ID 的 Chunk

`kg.sources.chunk_text` 默认参数：

```text
max_chars     = 8000
overlap_chars = 500
max_passage_chars = 1200
```

Chunk 只能在 Passage 边界切分，尾部 Passage 在不超过 `overlap_chars` 时带入下一块。发送给 LLM 的文本为：

```text
[P000001] 第一段真实原文……

[P000002] 第二段真实原文……
```

每块记录：

```text
index
passages
带 ID 的 prompt text
page/character location
SHA256(chunk)
```

分段和组块时间复杂度均为 `O(|text|)`。重叠块复用相同 Passage ID，不产生另一个原文身份。

## 5. 有据抽取

### 5.1 MiniMax M3 输入

每个未完成片段进行一次抽取调用。系统提示词明确：

- 只能依据当前片段；
- 不能用模型记忆补充知识；
- 没有有效 Passage 依据的对象或关系必须省略；
- 只输出 JSON。

六种 `entity_type` 和三种 `relation` 的定义由 `kg.ontology` 渲染进用户提示词，格式为**判定测试 + 排除项 + 正反例**。`kg/ontology.py` 是这些定义的唯一来源，抽取、关系裁判（第 8 节）和身份裁决（第 6.4 节）共用它，避免同一条定义在多处分叉。渲染后的定义块约占提示词 3000 字符。

抽取结果包含：

```json
{
  "entities": [
    {
      "name": "...",
      "definition": "...",
      "entity_type": "...",
      "aliases": [],
      "evidence": {
        "passage_ids": ["P000001"],
        "quote": "LLM 认为最关键的引文"
      }
    }
  ],
  "claims": [
    {
      "subject": "...",
      "relation": "...",
      "object": "...",
      "stance": "support|oppose",
      "evidence": {
        "passage_ids": ["P000002"],
        "quote": "LLM 认为最关键的关系引文"
      }
    }
  ]
}
```

默认每个 Chunk 最多输出 30 个 Entity 和 30 个 Claim。Claim 端点具有实质
定义时应同时输出 Entity；否则仍允许输出有 Passage 关系证据的 Claim，端点
作为待定引用保存。因此 Entity 上限不会再删除 ClaimObservation。

### 5.2 Evidence Passage 解析

LLM 对每个 Entity/Claim 同时输出：

```text
passage_ids   当前 Chunk 中的 1–3 个 Passage ID
quote         LLM 选择的关键引文，允许轻微省略或改写
```

程序确定性检查：

1. `passage_ids` 是非空数组且最多 3 个；
2. 每个 ID 都存在于当前 Chunk；
3. ID 不重复。

通过后，程序根据 ID 取得并拼接真实 Passage 文本：

```text
source_text = "\n\n".join(passage.text for selected passage)
```

最终 Evidence 同时保留：

```text
model_quote   LLM 原样输出
source_text   程序从 Source 取得的真实原文
passage_ids
location
```

当前不执行 quote 与 source text 的精确匹配或模糊相似度计算。有效
Passage ID 负责定位真实原文，`model_quote` 和 `source_text` 原样并存，
差异留给未来的 LLM 或人工校准研究。

### 5.3 Entity 机械校验

Entity 必须同时满足：

1. `name` 非空。
2. `definition` 至少四个字符。
3. `entity_type` 属于六种主类型。
4. Evidence 引用当前 Chunk 中 1–3 个有效 Passage。
5. 同一片段内规范化名称不重复。

类型由模型根据 definition 判定；机械层只验证类型词表，不根据名称重新猜类型。

模型意外返回超过上限的 Entity 时按原顺序截断；ClaimObservation 单独保存，
不要求为了即时物化而把端点强行包装为无定义的 Entity。

判出的类型写进该次观察对应的 `evidence.observed_entity_type`，**不写进 Entity**。详见第 6.6 节。

### 5.4 Claim 机械校验

Claim 必须同时满足：

1. relation 属于三种核心关系。
2. stance 属于 `support/oppose`。
3. subject 与 object 非空且不同。
4. Evidence 引用当前 Chunk 中 1–3 个有效 Passage。

这一步只证明 LLM 指向了真实原文段落，还没有证明原文真的表达该关系。

### 5.5 Observation 持久化与裁判缓存

Claim 通过 5.2 和 5.4 的机械校验后，先写入不可丢失的
`claim_observations`，再进行实体解析。关系裁判结果按
`(observation_id, validator_model, validator_prompt_version)` 保存；端点尚未
解析也不妨碍裁判。这样新增 Entity 后可直接使用已有裁判结果落实 Claim。

Observation 没有手写状态机：端点列为空、当前版本裁判缺失、`claim_id`
存在等字段直接导出 pending/materialized 等审计口径。

旧数据库中的 Claim Evidence 只在升级时执行一次 Observation 回填；新数据库
直接保存原生 Observation，后续 `status/check/audit` 重新连接数据库时不得把
已落实 Evidence 再导入为第二条 legacy Observation。schema 6 使用独立回填
标记保证该迁移只执行一次，并清理由早期 schema 5 行为产生、且已有完整原生
Observation 对应的重复 legacy 行。

### 5.6 待定端点与 Entity 晋升

未解析 subject/object 按保留标点、忽略空白的 reference key 聚合。至少三个
独立 `(Source, Passage ID)` 出现且原文明确出现名称时，进入晋升审核。审核
只能依据这些 Source 文本：能够形成稳定定义则创建 Entity；能够确认现有
Entity 则增加 alias；否则保留 uncertain。相似度只召回候选，不自动建立身份。
审核返回的证据必须用 `(source_id, passage_id)` 精确引用，避免不同 Source 中
同名 `P000001` 造成来源歧义。

晋升审核以证据集合指纹、模型和审核版本缓存，证据未变化时不重复调用模型。

## 6. 实体解析

设新 Entity observation 为 `o`。

### 6.1 名称规范化

数据库名称键：

```python
normalize(name) = NFKC(name).casefold().strip()
normalize(name) = collapse_whitespace(normalize(name))
```

规范化不删除连字符、加号、括号或其他可能承载语义的标点。

### 6.2 精确匹配

先在 canonical name 和 aliases 上查询规范化名称：

```text
0 个命中：进入候选召回
1 个命中：直接 same
多个命中：视为歧义，进入候选召回
```

精确唯一匹配不调用 LLM。

### 6.3 候选召回

当前第一版对所有 Entity 的 canonical name 和 aliases 计算 `SequenceMatcher` 相似度：

```text
score(o, e) = max(similarity(normalize(o.name), normalize(name))
                  for name in canonical_and_aliases(e))
```

若一方去空格后的名称包含另一方，score 至少提升到 `0.55`。

普通解析保留：

```text
score >= 0.35 的前 5 个候选
```

相似度只负责召回，绝不自动合并。

### 6.4 LLM 身份裁决

MiniMax M3 看到：

- 新观察的名称、definition、类型、model quote 和程序取得的 source text；
- 每个候选的 canonical name、aliases、definition、类型；
- 候选最近最多三条 Entity Evidence。

输出：

```text
same(candidate_id)
new(canonical_name)
uncertain(canonical_name)
```

执行规则：

- `same` 只有 candidate_id 确实来自候选集合时才复用实体。
- `new` 新建实体。
- `uncertain` 也新建独立实体，不建立合并状态或审核队列。
- 非法 decision 或非法 candidate_id 降级为 `uncertain`。
- `new/uncertain` 返回的 canonical name 若已指向现有实体，退回观察名，避免通过名称冲突偷偷合并。

### 6.5 后续重判

`reconcile` 对现有实体执行较窄的相似候选扫描：

```text
score >= 0.55
```

模型再次看到双方 definition、aliases 和后来累计的 Evidence。只有明确返回 `same` 才合并；`new` 保持分离，`uncertain` 保持分离。
有界 `--limit` 优先检查字符串召回分数最高的候选对，避免实体插入顺序决定
本轮审查对象；相似度仍只负责排序，不能决定合并。

当前候选扫描是全表扫描：

```text
单个 observation：O(N × S)
全量 reconcile：最坏 O(N² × S)
```

其中 `S` 是短名称字符串相似度成本。第一版通过 `--limit` 控制重判规模，不提前引入向量数据库。

### 6.6 类型下沉与 type profile

类型是 mention 级的观察，不是 Entity 的属性。每条 Entity Evidence 记录本次观察判出的类型：

```text
evidence.observed_entity_type
```

Entity 层不存类型，需要时用一条聚合查询得到 profile（`store.type_profile`）：

```sql
SELECT observed_entity_type,
       COUNT(*)                  AS observations,
       COUNT(DISTINCT source_id) AS sources
FROM evidence
WHERE entity_id=? AND polarity='support' AND observed_entity_type<>''
GROUP BY observed_entity_type
```

结果形如：

```text
深度学习   solution: 17 个来源 / 31 条观察
           concept:   9 个来源 / 14 条观察
```

**profile 不是投票，不取 argmax。** 一个词确实可能同时属于多个类型——「深度学习」既是一族做法也是一个研究方向。语言学上这叫 inherent polysemy / dot object：两个义项不互斥，同时成立。因此「实体只有一个类型」这个前提本身是错的，profile 如实保留分布而不折叠。

两个计数口径含义不同，都要给出：`observations` 会被单一来源反复使用刷高，反映的是语料分布；`sources` 反映有多少独立来源这样判。默认展示口径留待真实数据出来后再定。

这个设计顺带消解了两处旧缺陷，不需要单独修补：

1. 复用已有实体时，新观察的类型判断此前被完全丢弃（`entities.entity_type` 只在建实体时写一次）；现在它作为一条 Evidence 自然进入 profile。
2. `merge_entities` 此前不处理类型冲突，`reconcile` 按 `min(id)` 选 target，等于按建库顺序随机保留一边；现在没有单值列可冲突，Evidence 一转移，两边 profile 自然并集。

需要单值类型的下游场景（如导航查询）应在查询层定阈值，并明确那是可调策略而非既成事实。

## 7. Claim 端点物化

片段内已解析 Entity 建立：

```text
reference key(observed name/alias) → entity_id
```

Claim 端点按以下顺序解析：

1. 当前片段的 Entity 映射；
2. 数据库中 canonical name/alias 的唯一精确匹配。

无法唯一解析任一端点时，Claim 暂不写入，但 ClaimObservation、原文证据和
裁判结果继续保留。系统不根据相似度或 LLM 猜测缺失端点。新增 Entity、验证
alias 或 Entity 合并后，确定性重放会重新做唯一精确匹配，并在两端齐备时落实
Claim；不重新抽取 Source，也不重复当前版本的关系裁判。

## 8. 关系证据裁判

Passage 解析后的每个 Claim Evidence 单独交给 MiniMax M3。裁判同时看到 model quote 和 source text，但提示词明确规定 source text 是唯一权威证据。裁判只能返回：

```text
supports
contradicts
insufficient
```

裁判看到的关系定义由 `kg.ontology.relation_detail(relation)` 渲染，与抽取阶段完全同源，包含该关系的判定测试、全部排除项和正反例。提示词要求先做判定测试，再逐条核对排除项，冲突时以排除项为准。

摘要（完整定义见 `kg/ontology.py`）：

- `is_a`：实例测试成立。两端 `entity_type` 通常相同，但只作自查线索，不作否决理由。
- `part_of`：构件、正文明确列出的阶段，或原文明确陈述的领域归属。
- `prerequisite_of`：原文明确陈述 subject 是学习 object 的实质性前提。

执行规则：

```text
stance=support 且 verdict=supports
    → 新建/复用 Claim，追加 support Evidence

stance=oppose 且 verdict=contradicts 且 Claim 已存在
    → 追加 oppose Evidence

其他组合
    → 不写入关系
```

反对证据不会凭空创建一条只有反对 Evidence 的 Claim。

同一 Chunk 中通过机械校验的 ClaimObservation 可以通过 `--judge-workers` 并行裁判，
不要求端点已经解析；
裁判结果按原顺序返回，Claim 去重、循环检查、Evidence 和数据库写入仍串行
执行。Entity 对齐依赖当前图谱状态，也保持串行。

### 8.1 已知语义风险

有效 Passage ID 只能证明模型选择了真实原文范围，不能证明关系判得正确。真实 MiniMax M3 冒烟中：

```text
人工智能，特别是神经网络与深度学习的发展
```

曾被接受为：

```text
神经网络 part_of 人工智能
深度学习 part_of 人工智能
```

该句只有强调和共现，没有明确归属陈述，正确判定是 `insufficient`。这两条在旧
实验结果中是假阳性——`part_of` 接受领域归属这种**语义**，但不接受“特别是”
这种**弱表达**。该句现已作为反例写进 `kg/ontology.py` 的 `part_of` 定义，
直接送进抽取和裁判两处提示词。

在此之前，最强的 `part_of` 措辞（带“领域相关、研究分支、用途和共现默认都不是”排除项）只存在于 `agent.md`，抽取和裁判看到的都是剥掉排除项的弱版本。`tests/test_ontology.py` 的 `DefinitionsReachThePromptsTest` 就是为防止这种分叉复发而存在的：它断言每条定义的每个排除项都真的出现在送给 LLM 的提示词里。

2026-08-01 的五 Chunk 有界运行又暴露出三类可重复边界：把“建议先浏览以便
更顺畅理解”判成必需前提；把目录中的同名算法当作章节构件；把解决同一问题
的可替换途径当作共存构件。它们没有被写成字符串拦截规则，而是作为通用排除
项和反例加入同一 ontology。更新后的 MiniMax M3 对三条原始 Source Evidence
独立复判，均返回 `insufficient`。旧实验库保留原始判断用于对照，不当作正式
知识库继续扩张。

### 8.2 领域归属暂并入 part_of

`深度学习 / 人工智能` 这类子领域关系既过不了 `is_a` 的实例测试，也不是构件，本体论上通常单列为 `subfield_of`。第一版不新增该关系，理由有二：

1. 表面触发词不互斥——中文教材大量用“组成部分”表述子领域，要正确路由必须先裁定某个词是不是“领域”，而 `entity_type` 是单值列，装不下 `深度学习`（既是方法族又是研究方向）这类随上下文变化的类型；
2. 当前没有多跳遍历功能，混合语义的代价尚未发生，按第 14 节的门槛应当推迟。

因此领域归属暂记为 `part_of`。它作为 `claims` 表的一等数据，将来要拆分只需
对存量 Claim 重判，不必重跑抽取。只有真实的跨来源合并和知识导航反复受这种
混合语义阻碍时才考虑拆分，不以关系频次本身作为决策依据。

## 9. Claim 合并和循环检查

### 9.1 唯一 Claim

写入前查询：

```text
(subject_id, relation, object_id)
```

已有 Claim 直接复用，新的来源 Evidence 追加。

Evidence 唯一键：

```text
(target_key, source_id, excerpt_hash, polarity)
```

`excerpt_hash` 对 `source_text + model_quote + passage_ids` 计算 SHA256。因此完全相同的抽取观察不会重复写入；同一 Passage 上不同的 LLM quote 会分别保留，不同来源也可以持续支持同一 Claim。

### 9.2 Evidence 溯源与未来校准

Evidence 还保存：

```text
source_id                   可关联文档名称、URI、版本、内容哈希和语言
passage_version
extraction_model
extraction_prompt_version
validator_model
validator_prompt_version
validator_verdict
validator_reason
created_at
```

当前不实现校准代码、审核队列或 reviewer 状态。以上原始字段保持不可变，
为未来 LLM 或人类校准保留输入。未来校准结果应采用追加记录并引用原始
Evidence，不能改写历史 `model_quote/source_text`；稳定引用方式留到校准
实验设计时确定。

未入图 Claim 以 `claim_observations` 和追加式
`claim_observation_judgments` 保留三元组引用、原始端点名、source/model quote、
Source/Passage、模型、提示词版本和裁判理由。端点未解析、证据不足和闭环检查
失败都不会删除这些研究记录。旧 `rejected/rejection_details` 继续兼容历史运行，
schema 迁移会在字段完整时导入对应 Observation。`kg status` 和 `kg audit`
分别报告待定端点、待裁判、语义未通过、阻塞及可晋升候选。

### 9.3 循环

添加 `subject → object` 前，对 `is_a/prerequisite_of` 检查现有图中是否存在：

```text
object ⇢ subject
```

使用 DFS；若存在路径，新边会闭合循环，因此拒绝。复杂度为 `O(V + E)`。

`part_of` 当前不做循环检查，与 `plan.md` 的第一版规则保持一致。

## 10. Entity 合并

将 source Entity 合并到 target Entity 时，在一个事务中：

1. source canonical name 和 aliases 加入 target aliases。
2. Entity Evidence 转移到 target 并按唯一键去重。
3. ClaimObservation 已解析端点从 source 重定向到 target。
4. 所有 Claim 端点从 source 改为 target。
5. 重定向后自环的 Claim 丢弃，但 Observation 和证据仍保留。
6. 重复 Claim 合为一条，Evidence 全部转移并去重。
7. 可能产生关系循环的重定向 Claim 不写回。
8. 删除 source Entity。
9. 若模型给出无冲突的更优 canonical name，则更新 target。
10. 使用已缓存裁判重放未落实 Observation。

第一版没有 merge event、撤销或复杂审核状态，因此合并必须坚持“宁可不合并，也不误合并”。

## 11. 断点续跑和失败语义

`source_progress` 主键为：

```text
(source_id, chunk_index, chunk_hash)
```

这里的 `chunk_hash` 实际是处理指纹，包含正文 Chunk 哈希、Passage 版本、**抽取/身份裁决/关系裁判三个提示词版本**、模型和每块实体/Claim 上限。算法或参数变化后，同一文本会重新处理，不会被旧 `done` 错误跳过。

三个提示词都从 `kg/ontology.py` 取得所需定义。Entity 类型语义变化时 bump
抽取和身份裁决版本；relation 语义变化时 bump 抽取和关系裁判版本。这样受影响
的已有 `done` 片段会重新处理，而无关判断不会仅因版本联动被重复执行。

状态只有：

```text
done
failed
```

运行规则：

- `done` 的相同片段直接跳过。
- `failed` 的片段下次重新执行。
- `--start-chunk N` 忽略 index 小于 N 的片段，且不消耗 `--max-chunks` 额度。
- Entity/Evidence/ClaimObservation/Judgment/Claim 都有确定性唯一键，因此失败后的重试保持幂等。
- 机械校验通过的 ClaimObservation 先提交；后续 Entity 解析、关系裁判或物化失败
  只回滚未提交部分，已保存的原文证据不丢失，然后记录失败原因。
- LLM 没回答不等于知识为假；失败不能产生拒绝关系或虚假知识。

## 12. 默认 LLM 调用

默认配置：

```text
model = MiniMax-M3
endpoint = https://api.minimaxi.com/v1/text/chatcompletion_v2
api_key = MINIMAX_API_KEY
temperature = 0.0
timeout = 600 seconds
retries = 3
```

HTTP 429、5xx 和连接错误执行指数退避：

```text
1s, 2s, 4s，单次等待上限 8s
```

响应必须包含 `choices[0].message.content`，MiniMax `base_resp.status_code`
非零时视为失败。内容允许 Markdown JSON fence 或 JSON 前后有解释文字，
但顶层必须能解析出一个 JSON 对象。标准 JSON 解析失败时，使用
`json-repair` 严格模式处理字符串中未转义引号等常见语法错误；不启用会补值、
转换类型或删除字段的 Schema repair，修复结果仍走原有字段与业务规则校验。
严格修复也失败时，用完全相同的请求重新生成一次；第二次仍无法解析才记为
本片段失败，不引入额外提示词或第三次调用。

## 13. 一轮批处理伪代码

```python
for spec in load_catalog(catalog):
    loaded = load_source(spec)
    source_id = upsert_source_by_content_hash(loaded)

    for chunk in chunk_text(loaded.content):
        if progress_done(source_id, chunk):
            continue

        try:
            batch = extract_with_minimax_m3(chunk)
            batch = validate_passage_references(batch, chunk.passages)

            observations = persist_claim_observations(batch.claims)
            commit()  # 后续失败也保留已定位到 Source 的关系证据

            local_entities = {}
            for observed_entity in batch.entities:
                entity_id = resolve(observed_entity)
                source_text = read_selected_passages(
                    chunk.passages, observed_entity.passage_ids
                )
                add_entity_evidence(
                    entity_id,
                    source_id,
                    model_quote=observed_entity.quote,
                    source_text=source_text,
                    passage_ids=observed_entity.passage_ids,
                )
                local_entities[observed_entity.names] = entity_id

            resolve_observation_endpoints(observations, local_entities)
            for observation in observations:
                verdict = cached_or_judge_relation_evidence(observation)
                save_append_only_judgment(observation, verdict)
                if endpoints_ready(observation) and verdict_matches_stance(
                    verdict, observation.stance
                ):
                    claim_id = upsert_claim_with_cycle_check(
                        observation.subject_id,
                        observation.relation,
                        observation.object_id,
                    )
                    add_claim_evidence_from_observation(claim_id, observation)

            replay_cached_observations_unlocked_by_new_entities()

            mark_done(source_id, chunk)
        except Exception as error:
            rollback()
            mark_failed(source_id, chunk, error)
```

## 14. 当前明确不做

- 不设计 `proposed/published/shadow` 状态机。
- 不做置信度累乘、多级别名状态或复杂审核队列。
- 不让主动搜索代替广泛阅读。
- 不要求新 Entity 首次出现时连接到主图。
- 不建设生产级并发、权限、分布式任务或向量检索。
- 不因 LLM 的单次判断绕过有效 Passage 引用和程序提取的真实 source text。
- 暂不实现校准代码，但不得丢弃未来校准所需的 quote、原文、来源版本和模型/提示词信息。

任何新增机制都必须先回答：

> 不加它，当前核心闭环是否真的无法工作？

## 15. 图谱审计视图

`kg viz --out out/graph.html` 生成单个自包含 HTML，不引入服务或前端框架。
页面内嵌 Entity、Claim、Evidence 和拒绝审计数据，默认只画高连接实体的有限
邻域，避免把大图渲染成不可读的全量“毛线团”。支持：

- 搜索实体并查看一至三跳邻域；
- 按 `is_a / part_of / prerequisite_of` 开关关系；
- 点击节点查看定义、alias、type profile 和 Entity Evidence；
- 点击边查看真实原文、Passage 位置和关系裁判；
- 搜索、筛选并查看未落实 ClaimObservation、端点解析和晋升候选；
- 查看拒绝类别和代表样本。

待定端点和晋升候选只进入右侧审计面板，不画成正式 Entity 或 Claim，避免把
研究线索误表示为已经入图的知识。

可视化是只读的质量审计入口，不引入新的知识对象或发布状态。
