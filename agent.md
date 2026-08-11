# AI Knowledge Graph Agent Guide

## 1. 目标与原则

本项目从人工维护的高质量 AI 语料中持续构建可追溯知识图谱，用于知识导航和个性化教学。

当前主流程：

```text
Read → Structure → Extract → Normalize → Resolve → Verify → Merge → Synthesize
```

始终遵守：

- 正式知识内容以语料为唯一来源；Entity identity 裁判可以使用可靠通用知识判断两个提及
  是否指向同一对象，但不能据此新增 Claim、Assertion 或语料未表达的事实。
- 质量优先于节点、边和完成 chunk 的数量。
- 保持 SQLite 加少量 Python 模块的学术项目形态，优先简单、可审计、可复现的实现。
- 不为单个坏样本堆叠专用规则；先修正可泛化的语义边界。
- 当前实验状态、未完成工作和接手步骤记录在 `TODO.md`，不要写进本文件。

## 2. 当前数据模型

正式图的核心对象仍是：

- `Source`：不可变语料版本。
- `Entity`：可稳定复指、可独立定义或学习的知识对象。
- `Claim`：两个规范 Entity 之间便于导航的紧凑关系投影。
- `Evidence`：Entity、Claim 或 Assertion 的可定位原文证据。

schema 10 还保存以下结构与审计数据：

- `Section` / `Passage`：教材结构与原文定位。
- `EntityObservation` / `ClaimObservation`：抽取时的原始观察和后续处理状态。
- `Assertion`：Claim 背后的完整命题，保留条件、范围、数量、时间、否定和极性。
- 开放 Entity 类型词表、RelationType 词表及其归一记录。

Entity 类型和关系谓词均为开放词表：

- `resource / criterion / data / task / solution / concept` 是兼容旧语料的种子类型，不是白名单。
- `is_a / part_of / prerequisite_of` 是种子关系和导航类别，不是 Claim 关系白名单。
- `relation_kind` 为 `is_a / part_of / prerequisite_of / other`；开放谓词可归入 `other`。
- 类型是 mention 级观察；Entity 没有单值类型，查询时使用 type profile。

## 3. 语料与证据边界

- 每个 Entity 必须有至少一条可定位的 support Evidence。
- 每个正式 Claim 和 Assertion 必须有原文证据，并通过关系裁判。
- 抽取、Claim 和 Assertion 不能凭模型记忆补充原文没有表达的知识；identity 判断和
  `Entity.definition` 聚合可使用可靠通用知识。原文在 identity 中用于确定指代与义项；
  在定义聚合中用于锚定义项并支持语料特有的事实、数字、版本、历史事件和应用结果。
- `model_quote` 保留模型选择的关键引文；正式 `source_text` 必须由程序根据 Passage ID 从 Source 取得。
- Section 标题、目录距离和 Section 摘要只用于结构、上下文、召回和展示，不能单独证明 Claim。
- 共现、章节顺序、超链接、主题相近和模型常识不能单独证明关系。
- 删除代码块、练习题和界面操作后，正文仍应能解释被抽取的知识。
- 教材临时函数、演示类、局部变量和操作步骤不入图；框架、工具及与领域概念一对一对应的 API 可以作为独立 Entity，但不能冒充概念 alias。

## 4. Entity、alias 与定义

Entity 对齐只有三种结果：

```text
same       当前观察与已有 Entity 是同一知识对象
new        是不同知识对象，创建新 Entity
uncertain  证据不足，保留独立对象，等待后续重判
```

执行约束：

- 字符串相似度只用于召回候选，不能直接决定合并。
- 同名和唯一精确匹配也只用于候选召回，最终 identity 必须由模型明确判断。
- identity 裁判必须看到观察定义、原文、候选定义、类型画像和既有 Evidence，并允许使用
  可靠通用知识判断通常含义、同义关系、翻译、缩写及概念/实现/子类/实例边界。
- 原文用于确定当前名称实际指向哪个义项，不要求原文重新证明两个通用术语同义。
- Observation 或 Entity 的 definition 可能只是局部、不完整或带场景的概括，只作义项线索，
  不能直接充当身份边界。
- 同名不自动等于同一对象；不同使用场景也不自动等于不同对象。
- 当前 passage 中的简称、类比或角色映射可以解析到已有 Entity，但不自动成为全局 alias。
- 只有模型明确接受、且脱离当前上下文仍安全互换的名称才能注册为 alias。resolver 可用可靠
  通用知识补充标准翻译、英文全称、通行缩写、正式名/简称和拼写变体；不得补充普通近义词、
  相关概念、实现/API、实例或局部角色。
- `Entity.definition` 是帮助身份识别的规范概念解释，不要求是严格词典定义。它可由该 Entity
  的全部 Observation 聚合更新，并用可靠通用知识补全通常含义、上位类别和跨场景稳定特征；
  用途、性质、实现方式和典型比较可以辅助解释，但不能把一次局部场景写成身份边界。聚合
  必须记录所引用的 Observation 和 Passage，失败不能覆盖旧定义。
- 合并只在模型明确返回 `same` 时执行；不得用批量相似度自动合并。

## 5. Relation、Assertion 与物化

- 抽取阶段保存完整 `statement`、`scope`、极性、原始谓词和两个原始端点。
- RelationType 是开放的，但规范名必须简洁、可复用，并具有稳定语义和固定方向。
- 当前 observation 能映射到某个 RelationType，不代表原始谓词可以注册为全局 relation alias；两项必须独立判断。
- `uncertain` relation 保持 pending，不创建空、`None` 或猜测出的 RelationType。
- 关系裁判必须分别判断：
  1. 原文是否支持完整 Assertion；
  2. `subject → canonical relation → object` 是否忠实保留核心参与者、含义和方向。
- Claim 可以省略已由 Assertion 保存的限制，但不能偷换端点、遗漏真正参与关系的第三个对象或改变方向。
- 只有端点唯一解析、Assertion 得到支持且投影忠实时才物化 Claim。
- 相同 `(subject, relation_type, object)` 只保存一条 Claim；新证据追加到同一 Claim。
- Claim 不允许自环；`is_a` 和 `prerequisite_of` 不允许形成有向循环。

## 6. 数据库与运行安全

- 当前 schema 为 10，默认正式数据库是 `data/knowledge-vnext.db`。
- `data/knowledge.db` 和 `data/kg.db` 属于旧实验或旧 schema，不自动迁移、覆盖或与 vNext 混用。
- 新提示词、schema 或高风险算法实验使用 `tmp/` 下的独立数据库；验证通过不等于正式库已更新。
- Source 按 `(source_key, content_hash)` 版本化；同一处理指纹的已完成 chunk 必须可幂等跳过。
- 通过 Passage 机械校验的 Observation 在身份解析和关系裁判前持久化。失败 chunk 可以留下未解析 Observation，但不能留下无证据的正式 Claim；统计质量时必须区分 `done`、`failed` 和正式图对象。
- `summary-workers`、`chunk-workers`、`judge-workers` 是不同阶段的并发设置；SQLite 解析、合并和写入仍由主线程串行执行。
- `--llm-max-concurrency` 是复杂模型与简单模型共享的单进程请求上限，不能把各 worker 数简单相加当作真实并发。
- 同一数据库同时只允许一个写进程。长任务使用独立 tmux、日志、数据库和 `.started/.finished/.exit` 标记。
- 未经用户明确授权，不启动大规模外部 API 运行，不覆盖正式数据库，不删除历史实验。

## 7. 模型与配置

默认模型分工：

```text
complex model: MiniMax-M3
simple model:  MiniMax-M2.7
endpoint:      https://api.minimaxi.com/v1/text/chatcompletion_v2
api key:       MINIMAX_API_KEY
```

复杂模型用于抽取、Entity 消歧、Claim 裁判、定义聚合和关系补抽；简单模型用于 Section 摘要及开放类型/关系词表归一。可通过 `KG_COMPLEX_LLM_MODEL`、`KG_SIMPLE_LLM_MODEL`、`KG_LLM_MODEL` 和 `KG_LLM_BASE_URL` 临时覆盖。

不得把 API key 写入代码、文档、日志、测试、提交或记忆；除非用户明确指定，不切换 provider。

## 8. 开发与验证

修改前先阅读相关代码、`design/algorithm.md`、真实数据库和实验输出，区分语料问题、模型判断问题和实现问题。

实现时：

- 不引入复杂审核状态机、置信度累乘、额外分类器或生产级任务框架，除非已有实验明确证明必要。
- Passage ID、外键、空名称、自环、指纹等可机械检查的约束必须由代码验证，不能交给提示词兜底。
- 行为变化同步更新 `design/algorithm.md` 和相关测试。
- 提示词或语义规则变化时，更新所有受影响的 prompt/version 常量，确保旧 `done` chunk 不会被错误复用。
- 保留模型原始判断、失败原因和来源文本，不用自动修补隐藏实验失败。

代码修改至少运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q kg tests
git diff --check
```

涉及数据库或流水线时运行：

```bash
python -m kg --db data/knowledge-vnext.db status
python -m kg --db data/knowledge-vnext.db check
```

涉及 LLM、提示词或响应解析时，在用户授权外部数据发送且 `MINIMAX_API_KEY` 可用后，使用 `tmp/` 独立数据库做有界真实测试。汇报必须区分确定性测试、真实模型冒烟、小规模实验和全量运行。
