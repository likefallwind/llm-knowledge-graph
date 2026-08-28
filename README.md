# AI 全领域知识图谱（最小实现）

这是 `plan.md` 的可运行实现。正式 Claim、Assertion 和语料特有事实只以语料为来源；LLM
负责抽取、实体身份判断和关系证据裁决。Entity identity 与用于身份识别的规范概念解释可
使用可靠通用知识判断术语义项、对象边界和通常含义，但不能据此新增语料未表达的关系或
具体事实。

vNext 把 KGGen 的开放抽取与 Tree-KG 的教材结构先验合并到同一条、仍然可审计的闭环中：

```text
Read → Structure → Extract Entities → Extract Relations → Normalize → Verify → Merge
```

## 设计边界

- 正式图仍只有四类知识对象：`Source`、`Entity`、`Claim`、`Evidence`；Section、Observation 和开放词表是结构/审计数据。
- 实体类型和关系类型均开放抽取并全局归一。旧六类实体类型和三个核心关系只作为种子，不是白名单。
- `relation_kind` 保留 `is_a / part_of / prerequisite_of / other` 四种导航类别；`other` 下可以保存任意有原文证据的开放谓词。
- 关系候选不按字符串相似度截断，也不为三种种子关系保留特殊席位：只使用名称精确命中和已有 Claim 支撑的开放 RelationType。精确命中仍须结合完整 Assertion 复判。
- 新 RelationType 和 relation alias 只在最终关系证据裁判支持后写入；无法形成忠实 `subject → predicate → object` 投影的观察留作审计但不入图。
- 教材目录持久化为 Section 树，自底向上生成仅由 Passage 支持的摘要。目录用于上下文、候选召回和展示，不自动生成 Claim。
- Entity 和 Claim 没有 `proposed / published / shadow` 状态机。通过 Passage
  校验的 EntityObservation 和 ClaimObservation 都会先永久保存；实体身份判断
  和 Claim 端点暂时不确定都不会丢失原始观察。
- 每个 Entity 都必须有可定位的来源 Evidence；没有被证据裁判确认的 Claim 不入图。
- LLM 同时输出关键 quote 和 Passage ID；程序按 ID 取得真实原文，两者都会保存在 Evidence 中。
- 当前不做 quote 与原文的字符匹配；Passage ID 负责定位，二者留待以后校准。
- Evidence 记录文档版本、位置、模型和提示词版本，为未来 LLM/人类校准保留可能性；当前不实现校准队列。
- 相同 `(subject, relation, object)` 只保存一个 Claim，不同 Source 的 Evidence 自动累计。
- 实体对齐只有 `same / new / uncertain`。`uncertain` 会保留独立实体，之后可用 `reconcile` 重新判断。
- 名称的语义相似度、局部类型和 definition 都只是 identity 候选与义项线索；即使唯一
  精确同名也由 LLM 基于通用知识做最终判断，原文主要用于确定当前提及的义项。
- identity 候选召回使用中英多语言 Embedding，只编码 observation 的名称、待审核 aliases
  和 definition，并对已有 Entity 的规范名、aliases 和 definition 取最相近的前 10 个；
  不把三元组或原文放入 Embedding。这些文本只负责召回。resolver 可补充最多五个
  可靠通用知识中的标准翻译、英文全称、通行缩写或名称变体，明确审核后才注册。
- `Entity.definition` 不由第一次抽取永久决定。主流程结束时，同一 Entity 的全部
  EntityObservation 用于锚定当前义项和具体语料事实，定义整理器可使用可靠通用知识补全
  通常含义、上位类别和跨场景稳定特征，形成用于身份识别的规范概念解释。聚合结果保存所
  引用的 Observation、Passage、模型和提示词版本；原始观察不覆盖。
- `is_a` 和 `prerequisite_of` 写入前检查循环；孤立 Entity 合法。

旧项目数据位于被忽略的 `data/kg.db`，schema 8 试验数据也采用了不同的抽取语义。vNext 不迁移或修改旧库；新数据库默认是 `data/knowledge-vnext.db`。

## 环境

Python 3.11+。安装项目及必需依赖：

```bash
python -m pip install -e .
```

实体候选召回默认使用 `intfloat/multilingual-e5-small`，支持中英文跨语言匹配。模型在首次
召回时下载；可通过 `KG_EMBEDDING_MODEL` 覆盖模型名。

PDF 读取优先使用系统的 `pdftotext`。没有该命令时可安装可选依赖：

```bash
python -m pip install -e '.[pdf,yaml]'
```

流水线的实体/关系抽取、实体消歧、关系证据裁判、定义聚合、关系补抽、目录摘要及
开放类型/关系词表归一默认全部使用 MiniMax-M3。各角色共用同一个兼容客户端和 API key：

```bash
export MINIMAX_API_KEY='...'
```

默认配置为：

```text
endpoint: https://api.minimaxi.com/v1/text/chatcompletion_v2
complex model: MiniMax-M3
simple model: MiniMax-M3
```

如需临时经过兼容网关，可设置 `KG_LLM_BASE_URL`。`KG_COMPLEX_LLM_MODEL` 和
`KG_SIMPLE_LLM_MODEL` 可分别覆盖两个角色；`KG_LLM_MODEL` 会同时覆盖两者，适合
临时只使用一个模型。API key 也兼容旧变量 `MINIMAX_API` 和 `minimax_api`。

## 运行

初始化并查看状态：

```bash
python -m kg init
python -m kg status
```

处理人工维护的语料目录：

```bash
python -m kg run examples/sources.json
```

仓库同时提供了针对现有 `data/docs/*.pdf` 教材语料的目录。建议第一次只跑一个片段：

```bash
python -m kg run sources/catalog.json \
  --source-limit 1 --max-chunks 1
```

长批次可以用 `--source-limit` 和 `--max-chunks` 控制本轮规模；已完成的相同版本片段会自动跳过：

```bash
python -m kg run sources.json \
  --source-limit 2 --max-chunks 20 \
  --summary-workers 2 --chunk-workers 2 --judge-workers 2
```

默认每个 Chunk 最多抽取 50 个 Entity 和 30 个 Claim。达到 Entity 上限的 Chunk
会在运行结果的 `entity_cap_hit_chunks` 中列出，供进一步细分审计。Claim 端点满足 Entity
标准且有定义证据时应同时输出 Entity；否则 ClaimObservation 和关系证据仍会
保存，等待后续 Entity、已验证 alias 或实体合并使端点唯一可解析。

如需只处理指定下界之后的片段，可使用 `--start-chunk`。它按 Chunk index
过滤，不会消耗 `--max-chunks` 的处理额度：

```bash
python -m kg run sources/catalog.json \
  --source-limit 1 --start-chunk 120 --max-chunks 17 \
  --max-entities 50 --judge-workers 4
```

`--summary-workers` 在同一目录深度内并行 Section 摘要，并在进入父级前等待该层
全部完成；LLM 请求并发执行，SQLite 仍由主线程串行写入。

`--chunk-workers` 有界并行不同 Chunk 的无状态 LLM 抽取，并严格按原 Chunk
顺序消费结果；`--judge-workers` 并行同一片段内彼此独立的关系证据裁判。
Entity 对齐、Claim 物化和全部 SQLite 写入仍保持串行，避免改变图谱合并语义。
若连续 3 个 Chunk 失败，主调度会暂停 10 分钟再继续；任一 Chunk 成功后连续
失败计数清零，避免服务限额错误瞬间扩散到全部剩余 Chunk。
两者默认均为 1；小批次建议先使用 `--chunk-workers 2 --judge-workers 2`。

`kg run` 默认在片段处理结束后，为拥有至少一条 EntityObservation 且观察集合发生
变化的 Entity 聚合概念解释。原文负责锚定义项和具体事实，可靠通用知识只补全通常含义、
上位类别和跨场景稳定特征；用途、性质、实现方式和典型比较可以作为辅助解释，但不能把
一次局部场景写成概念身份边界。结果必须引用当前 Entity 的真实 Observation ID 和
Passage ID；无效引用或模型失败不会覆盖旧定义。相同观察指纹、模型和提示词版本会直接跳过。
长实验可用 `--definition-limit N` 限制本轮数量，之后继续运行即可断点续做；仅在明确
需要跳过该阶段时使用 `--skip-definition-synthesis`。

也可以单独聚合全部待更新定义，或只处理指定 Entity：

```bash
python -m kg synthesize-definitions
python -m kg synthesize-definitions --entity-id 38
```

随着 Evidence 增加，重新判断相似但尚未合并的 Entity：

```bash
python -m kg reconcile --limit 20
```

使用已保存的 Observation 重放待定端点，不重新抽取 Source。默认同一端点在
至少 3 个独立 Passage 出现后进入有据 Entity 晋升审核；只有原文足以形成稳定
定义时才创建 Entity，字符串相似仅用于召回候选：

```bash
python -m kg replay-pending \
  --promote-threshold 3
```

关系裁判按模型和提示词版本缓存。新增 Entity、确认 alias 或合并 Entity 后，
已裁判且两端齐全的 Observation 会直接落实为 Claim 并累计 Evidence。

检查硬约束并导出：

```bash
python -m kg expand-relations --limit 50
python -m kg check
python -m kg export --out out/graph.json
```

查看图结构、拒绝原因，并生成不依赖外部 JavaScript 的交互式审计页面：

```bash
python -m kg audit
python -m kg viz --view mixed --out out/graph.html
python -m kg viz --view document --out out/document.html
python -m kg viz --view semantic --out out/semantic.html
```

`status` 和 `audit` 同时报告 EntityObservation 的身份解析分布，以及
ClaimObservation 总数、待定端点、未裁判记录、已落实记录、被语义裁判拒绝的
记录和三次以上的 Entity 晋升候选。

混合 HTML 默认展示高连接实体的局部子图和教材目录，可搜索实体、切换核心或开放关系、调整邻域层数，
并点击节点或边查看 Source、Passage、真实原文和模型裁判理由。右侧
Observation 审计可按待定端点、待裁判、支持但未落实、物化阻塞和语义裁判
结果筛选，点击记录可查看原始端点、解析 Entity、证据和模型/提示词版本；
三次以上的 Entity 晋升候选也单独列出，不混入正式图。拒绝统计将
旧运行中的端点未解析、非法 Passage 等算法性损失与语义拒绝分开显示；新运行
的待定端点及 `insufficient/contradicts` 主要以 Observation 统计呈现，不再把
可恢复记录归入终止性拒绝。

## 语料目录

推荐使用无需额外依赖的 JSON：

```json
{
  "sources": [
    {
      "key": "stable-logical-key",
      "name": "语料名称",
      "type": "textbook",
      "path": "data/docs/book.pdf",
      "uri": "https://example.org/book.pdf",
      "language": "zh",
      "version": "2026"
    }
  ]
}
```

`path` 相对目录文件解析；省略 `path` 时从 `uri` 下载正文。支持 PDF、HTML、
纯文本、Markdown 和 JSONL。Markdown/HTML 标题及常见教材编号标题会形成
Section 路径；Chunk 不跨 Section，小节过长时才在该 Section 内继续按 Passage
切分。没有标题结构的网页或纯文本自动退化为原来的 Passage 分块。标题路径只
用于切分、定位和抽取上下文，不自动生成知识 Claim。Source 正文按内容哈希
版本化，同一逻辑来源内容变化后会产生新版本。

YAML 目录也受支持，但需要 PyYAML。

## 测试

测试使用确定性的假 LLM，不调用外部 API：

```bash
python -m unittest discover -s tests -v
```
