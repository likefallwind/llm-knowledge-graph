# AI 全领域知识图谱（最小实现）

这是 `plan.md` 的可运行实现。系统只把语料作为知识来源，LLM 只负责抽取、实体身份判断和关系证据裁决。

核心闭环：

```text
Read → Extract → Resolve → Merge → Repeat
```

## 设计边界

- 只有四类知识对象：`Source`、`Entity`、`Claim`、`Evidence`。
- 类型只有 `resource / criterion / data / task / solution / concept` 六种。类型是每次观察的判断，记在 Evidence 上；Entity 层汇总为 type profile，允许一个实体同时属于多个类型。
- Claim 只有 `is_a / part_of / prerequisite_of` 三种关系。
- Entity 和 Claim 没有 `proposed / published / shadow` 状态机。通过 Passage
  校验的 ClaimObservation 会永久保存；端点暂时不存在只是 pending，不是拒绝。
- 每个 Entity 都必须有可定位的来源 Evidence；没有被证据裁判确认的 Claim 不入图。
- LLM 同时输出关键 quote 和 Passage ID；程序按 ID 取得真实原文，两者都会保存在 Evidence 中。
- 当前不做 quote 与原文的字符匹配；Passage ID 负责定位，二者留待以后校准。
- Evidence 记录文档版本、位置、模型和提示词版本，为未来 LLM/人类校准保留可能性；当前不实现校准队列。
- 相同 `(subject, relation, object)` 只保存一个 Claim，不同 Source 的 Evidence 自动累计。
- 实体对齐只有 `same / new / uncertain`。`uncertain` 会保留独立实体，之后可用 `reconcile` 重新判断。
- `is_a` 和 `prerequisite_of` 写入前检查循环；孤立 Entity 合法。

旧项目数据位于被忽略的 `data/kg.db`。它使用旧版复杂 schema，本项目不会修改它；新数据库默认是 `data/knowledge.db`。

## 环境

Python 3.11+，没有必需的第三方 Python 依赖。

PDF 读取优先使用系统的 `pdftotext`。没有该命令时可安装可选依赖：

```bash
python -m pip install -e '.[pdf,yaml]'
```

LLM 默认直接使用 MiniMax M3。只需配置 MiniMax API key：

```bash
export MINIMAX_API_KEY='...'
```

默认配置为：

```text
endpoint: https://api.minimaxi.com/v1/text/chatcompletion_v2
model: MiniMax-M3
```

如需临时经过兼容网关，可显式设置 `KG_LLM_BASE_URL`；如需临时覆盖模型，可设置 `KG_LLM_MODEL`。API key 也兼容旧变量 `MINIMAX_API` 和 `minimax_api`。

## 运行

初始化并查看状态：

```bash
python -m kg --db data/knowledge.db init
python -m kg --db data/knowledge.db status
```

处理人工维护的语料目录：

```bash
python -m kg --db data/knowledge.db run examples/sources.json
```

仓库同时提供了针对现有 `data/docs/*.pdf` 教材语料的目录。建议第一次只跑一个片段：

```bash
python -m kg --db data/knowledge.db run sources/catalog.json \
  --source-limit 1 --max-chunks 1
```

长批次可以用 `--source-limit` 和 `--max-chunks` 控制本轮规模；已完成的相同版本片段会自动跳过：

```bash
python -m kg --db data/knowledge.db run sources.json \
  --source-limit 2 --max-chunks 20 \
  --chunk-workers 2 --judge-workers 2
```

默认每个 Chunk 最多抽取 30 个 Entity 和 30 个 Claim。Claim 端点满足 Entity
标准且有定义证据时应同时输出 Entity；否则 ClaimObservation 和关系证据仍会
保存，等待后续 Entity、已验证 alias 或实体合并使端点唯一可解析。

如需只处理指定下界之后的片段，可使用 `--start-chunk`。它按 Chunk index
过滤，不会消耗 `--max-chunks` 的处理额度：

```bash
python -m kg --db data/knowledge.db run sources/catalog.json \
  --source-limit 1 --start-chunk 120 --max-chunks 17 \
  --max-entities 30 --judge-workers 4
```

`--chunk-workers` 有界并行不同 Chunk 的无状态 LLM 抽取，并严格按原 Chunk
顺序消费结果；`--judge-workers` 并行同一片段内彼此独立的关系证据裁判。
Entity 对齐、Claim 物化和全部 SQLite 写入仍保持串行，避免改变图谱合并语义。
两者默认均为 1；小批次建议先使用 `--chunk-workers 2 --judge-workers 2`。

随着 Evidence 增加，重新判断相似但尚未合并的 Entity：

```bash
python -m kg --db data/knowledge.db reconcile --limit 20
```

使用已保存的 Observation 重放待定端点，不重新抽取 Source。默认同一端点在
至少 3 个独立 Passage 出现后进入有据 Entity 晋升审核；只有原文足以形成稳定
定义时才创建 Entity，字符串相似仅用于召回候选：

```bash
python -m kg --db data/knowledge.db replay-pending \
  --promote-threshold 3
```

关系裁判按模型和提示词版本缓存。新增 Entity、确认 alias 或合并 Entity 后，
已裁判且两端齐全的 Observation 会直接落实为 Claim 并累计 Evidence。

检查硬约束并导出：

```bash
python -m kg --db data/knowledge.db check
python -m kg --db data/knowledge.db export --out out/graph.json
```

查看图结构、拒绝原因，并生成不依赖外部 JavaScript 的交互式审计页面：

```bash
python -m kg --db data/knowledge.db audit
python -m kg --db data/knowledge.db viz --out out/graph.html
```

`status` 和 `audit` 同时报告 Observation 总数、待定端点、未裁判记录、已落实
记录、被语义裁判拒绝的记录和三次以上的 Entity 晋升候选。

HTML 默认展示高连接实体的局部子图，可搜索实体、切换关系、调整邻域层数，
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

`path` 相对目录文件解析；省略 `path` 时从 `uri` 下载正文。支持 PDF、HTML、纯文本、Markdown 和 JSONL。Source 正文按内容哈希版本化，同一逻辑来源内容变化后会产生新版本。

YAML 目录也受支持，但需要 PyYAML。

## 测试

测试使用确定性的假 LLM，不调用外部 API：

```bash
python -m unittest discover -s tests -v
```
