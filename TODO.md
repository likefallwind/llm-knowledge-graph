# TODO 与探索记录

最后更新：2026-09-07

以下正式库状态是 2026-08-11 的历史记录，本次关系实验未重新审计正式库。

正式库状态：《动手学深度学习》(D2L) 单本书已跑完，数据仍在 `data/knowledge-vnext.db`
（schema v9，3817 实体 / 6901 Claim / 17297 Evidence / 1105 片段全部 done，`kg check` 通过）。
2026-08-08 的修改和重跑全部位于实验分支 `experiment/assertion-layer-pilot` 和 `tmp/`
下的独立 schema v10 数据库，**没有修改或迁移正式库**。

> 注意：`data/knowledge.db`（默认路径）仍是 6 个实体的旧库。跑 CLI 记得带
> `--db data/knowledge-vnext.db`。

---

## 2026-09-07：关系研究初步收敛

- 确定采用“规范细关系＋独立粗类别＋完整 Assertion”：细关系合并同义表达，粗类别用于
  检索组织与编排，不替换事实谓词。不再以细关系必须压缩到指定数量为目标。
- 全919类实验已完成：细关系919→873，27个粗类别全部有实际使用；3218/3309条
  Assertion分类通过同模型复核（覆盖率97.25%），91条待定。861/873个细关系至少
  有一条用法分类通过，814个细关系的全部用法通过；覆盖率不是独立准确率。
- 对79条替换Claim的全量审阅：65条建议保留、10条建议撤回、4条待定。建议尚未应用，
  当前结果可用于检索原型；不声称所有873种细关系均已通过质量审计。
- 接受“大部分正确、支持实际使用”的目标；后续用代表性抽查和检索任务评估质量，
  不针对单个坏例持续追加提示词限制，不以零错误或100%分类覆盖为门槛。
- [收敛报告](experiments/relation_synonyms_categories_report.md)；
  [逐条细关系审计](experiments/relation_synonyms_categories_fine_audit.md)。
- 本次合并保留独立实验代码、测试和研究记录；未将实验派生表接入正式schema，未修改正式库。

## 2026-09-05：同义归一后粗化的小样本

- 已完成 MiniMax-M3 两阶段实验：80种关系、203条Claim、209条Assertion。
- 同义归一80→79种；生成34种粗关系，同模型分离提示词复核接受183/209条Assertion。
  残差17种细关系仍保留，有效类型数51；87.6%是自动接受覆盖率，不是准确率。
- 源库只读且哈希未变；派生库完整性通过，证据和来源映射全部保留。136项本地测试通过。
- 目录中“替代”“结构上连接”“引发或防止停滞”存在混合角色或拼接语义；另有期望
  效果被升级为实际效果的漏判。暂不扩大全书或替换正式关系。
- 下一步优先用简单规则修订目录、补足方向，在同一样本上重映射；本轮没有“跳过同义
  阶段直接粗化”的消融，不能据此声称两阶段优于直接粗化。
- [报告](experiments/relation_two_stage_pilot_report.md)；完成目录：
  `tmp/relation-two-stage-pilot80-20260905-idfix/`。原目录因模型返回错误ID中止，缓存
  已在新目录复用，两个目录均保留。

## 2026-08-09 至 2026-08-11：Entity resolution、None 边与关系投影修复

这一轮是在 schema v10 Assertion 实验之上继续修质量问题。所有真实实验仍写入 `tmp/`
独立数据库，**没有覆盖 `data/knowledge-vnext.db` 正式库**。

### 1. 已完成：Entity resolution 与语义命名

最初的明确错误是把不同概念错误合并，例如随机梯度下降、小批量随机梯度下降和批量梯度
下降。讨论后确定的原则不是建立全局身份证明、机械枚举否决规则或全图二次重判，而是：

- 继续使用现有 LLM identity/alias 判断，不增加一轮独立 alias 审查和额外 API 调用。
- 第一次 identity 判断更严格，`same` 必须是同一概念，而不是相关、上下位、组件、参数、
  实现方式或仅在某场景中一起出现。
- 当前上下文中的简称可以解析到正确 Entity，但只有模型明确接受的 alias 建议才注册为永久
  全局 alias；“本句指的是某 Entity”不等于“这个裸称在全局永远是它的 alias”。
- semantic naming 使用观察定义与当前上下文补全名称。例如某处裸称“梯度下降”，如果语义
  实际是批量梯度下降，就应命名为“批量梯度下降”，而不是先创建一个歧义的“梯度下降”。
- 当拟创建的新 Entity 规范名已经碰撞时，结合定义和场景重试语义命名；一般概念在不同
  使用场景中出现时，不应只因场景不同而机械拆分。

当前相关版本与提交：

- `3340ccc`：只提升模型明确接受的 alias 建议，并改进 identity/relation 处理。
- `397bda1`：扩充候选召回与 identity 消歧上下文。
- `f48d21a`：规范名碰撞时重试 semantic naming。
- 当前 `kg/resolution.py`：`entity-identity-ontology-10-knowledge-aliases-top10`。v9 修复了
  identity 请求遗漏 `EntityObservation.aliases` 的问题；v10 进一步让 observation aliases
  参与候选召回、将候选上限从 5 提到 10，并允许 resolver 补充严格限定的标准知识名称变体。

47 个完整 chunk 的冻结分析见 `tmp/fresh47-quality-analysis-20260810.md`。关键结论：

- 260 个 EntityObservation，198 个 Entity；198 `new`、62 `same`、0 unresolved。
- 多层感知机在多个 chunk 中正确合并。
- 数学卷积与深度学习卷积、一元/二元/三元语法、一般多层感知机与单隐藏层多层感知机
  均保持了合理区分。
- 随机梯度下降、小批量随机梯度下降、批量梯度下降没有再错误合并；上下文裸称可以指向
  批量梯度下降，但没有注册成污染全局的 alias。
- 47 个 chunk 已足以确认 Entity 修改总体有效。剩余的 3 个失败 chunk 不影响这个主结论。

剩余边界：`全连接层` 曾因 chunk 523 先以 LSTM 使用场景出现、chunk 140 后出现通用定义而
形成两个 Entity。后续提示已加入“同一 Entity 的不同使用场景仍可能是同一概念”的说明；
小规模复测中 chunk 523 没有再次单独抽取 `全连接层`，因此这个特定顺序案例尚未形成一次
完整的正向回归证据，但也没有观察到新的错误拆分。

### 2. 已完成：None/null relation 输入 bug

此前确认的 bug 是：MiniMax 没有返回可用的 relation-normalizer 结果时，不应把它当成
合法的 `null`/`none` relation 继续输入或物化。当前行为是拒绝空/`none`/`null` 等关系名，
没有可靠归一结果的 ClaimObservation 保持 pending/uncertain，不创建空 RelationType 和 Claim。

47-chunk 实验的证据：

- 空 RelationType：0；空 Entity 名称：0。
- 19 条未归一化边明确停留在 pending/uncertain，没有被错误物化。
- endpoint pending：0；SQLite integrity 与 KG consistency check 通过。

因此 None 边 bug 当前可视为已修复。不要把 pending relation 数量误报成 None 边数量。

### 3. 已完成：关系上下文、alias 注册与最终投影裁判

47-chunk 人工抽查暴露的主要剩余问题已经从 Entity 转到边：自然语言 statement 有原文支持，
但投影成 `subject - canonical relation - object` 后可能改变参与者、含义或方向。典型旧反例：

- 把“卷积层的权重称为卷积核”投影成“卷积层是卷积核的别称”。
- 把“LSTM 包含遗忘门”投影成 `LSTM is_member_of 遗忘门`。
- 把图模式/符号式编程、交叉熵损失/掩蔽语言模型的方向投反。

按“不额外增加审查调用”的原则，修改了现有 relation-normalizer 和 judge 调用：

- relation normalizer 在 Entity resolution 之后运行，获得完整 statement、scope、原文引文、
  source text、规范化后的主宾端点和候选关系描述。
- 当前 `kg/vocabulary.py` 版本为
  `open-relation-normalizer-3-contextual-alias`。
- `decision=same` 与 `register_alias=true` 分开判断。当前 observation 可以映射到已有关系，
  不代表原始短语适合作为全局 alias；只有明确返回 `register_alias=true` 才注册。
- judge 同时判断 `assertion_verdict` 和 `projection_faithful`。只有原文支持完整 Assertion，
  且最终三元组投影的核心参与者、关系含义和方向均忠实，才能物化。
- Claim 是导航用的紧凑投影，Assertion 保存条件、范围、数量、时间等限定。不能只因 Claim
  没有重复 Assertion 中已经保存的限定就判为不忠实。
- `14366a7` 已提交上述 relation description、处理顺序和投影上下文的主体修改。
- 当前工作区还有尚未提交的 judge v5 修改：
  `canonical-assertion-judge-5-scoped-projection`，涉及 `kg/validation.py` 与
  `tests/test_ontology.py`。新对话开始后先看 `git status --short`，不要丢掉这两处修改。

此前验证结果：全套测试 `106 passed, 28 subtests passed`，`git diff --check` 通过。

### 4. 已完成：6 个针对性小规模关系实验

实验组数据库：

```text
tmp/relation-context-v3-parallel-20260810-215900.g353.db
tmp/relation-context-v3-parallel-20260810-215900.g523.db
tmp/relation-context-v3-parallel-20260810-215900.g740.db
tmp/relation-context-v3-parallel-20260810-215900.g981.db
```

退出标记 `tmp/relation-context-v3-parallel-20260810-215900.exit` 为
`final_status=0 failures=none`。人工检查结果：

- relation aliases 全部为空，没有再次把上下文短语（尤其“包含”）扩散成全局 alias。
- 标量案例正确接受紧凑投影：Assertion 保留“只有一个元素”的限制，Claim 可保持
  `标量 -> 由...表示 -> 张量`。
- LSTM 到输入门、遗忘门、输出门的三条边全部归一为“包含组成部分”，且均被支持、物化。
- 卷积核案例正确生成“卷积核是卷积层的可学习参数/组成部分”。
- 图模式案例生成 `tf.function -> 启用 -> 图模式`，没有错误地把图模式和符号式编程合并。
- 掩蔽语言模型案例生成“掩蔽语言模型使用交叉熵损失”，含义与方向正确。
- 候选记忆元的权重关系这次没有被抽取，因此该特定目标没有获得覆盖；这是抽取随机性，
  不是已观察到的错误边。

自动门禁 `tmp/relation-context-v3-parallel-20260810-215900.gate.json` 返回失败，但人工复核确认
主要是门禁写得过于机械：它要求固定实体名称和固定方向，因而把上述语义正确的反向表述判为
“未覆盖”。`tmp/relation-gate-then-fresh50-20260810.exit` 因此阻止了 50-chunk 启动。
**这个 gate failure 不是模型修复失败，不能据此回滚提示词。**

当前结论：小规模实验没有看到 alias 扩散、None 边、错误合并和旧的四类投影错误复发，
已经达到扩大到 100 chunk 的条件；但这不等于全部边界情况都已完美验证。

### 5. 已停止：旧 identity 版本的 50-chunk 实验

用户随后决定先跑 50 个 chunk，并将并发上限调整为 4。任务已于
2026-08-11 10:56:35+08:00 在后台启动：

- 启动脚本：`tmp/run_fresh50_relation_v1_20260811.sh`
- 分析器：`tmp/analyze_fresh50.py`
- tmux：`kg-fresh50-rel-v1-20260811`
- 数据库：`tmp/fresh50-relation-v1-20260811.db`
- 日志：`tmp/fresh50-relation-v1-20260811.log`
- 分析：`tmp/fresh50-relation-v1-20260811.analysis.json`
- 完成标记：`tmp/fresh50-relation-v1-20260811.finished`
- 退出标记：`tmp/fresh50-relation-v1-20260811.exit`
- 配置：`chunk-workers=4`、`judge-workers=4`、共享
  `llm-max-concurrency=4`；同一时间只运行一个 `kg` 进程。
- 样本：30 个跨全书的分散新样本（6 个窗口，每个 5 个）+ 上一轮 20 个回归样本，
  共 50 个、无重复，范围 chunk 39–1082。
- 最终核查：SQLite integrity 为 `ok`，`source_progress` 为 `44 done / 2 failed`；日志停在
  regression chunk 681–683 开始处，没有 `.finished`、`.exit`、存活 tmux 或 Python worker。
  因此它是未完成的旧提示词样本，不得继续当作当前版本实验，也不要在同一数据库混入新版
  identity 结果。

### 6. 已完成：知识优先的 Entity identity 与 10 组真实测试

Entity identity、概念解释与语料事实抽取的知识边界已经拆开：抽取、Claim 和 Assertion
继续严格依据原文；identity 裁判可以使用可靠通用知识判断通常含义、同义/翻译/缩写，
以及概念、实现、子类、实例和配置之间的对象边界。`Entity.definition` 聚合也可在全部
Observation 锚定的义项上，用可靠通用知识补全通常含义、上位类别和跨场景稳定特征；
语料特有的事实仍须原文支持。局部 definition 只是线索，不能因应用场景或定义详略不同
机械拆分 Entity。

实现同步修改了首次 identity、独立 same 复核、同名冲突重命名、`reconcile` 和待定端点
晋升；删除“同名且类型兼容就直接 same”的机械捷径。该轮 10 组测试使用版本：

```text
entity-identity-ontology-8-knowledge-first
endpoint-promotion-2-knowledge-identity
```

真实 MiniMax-M3 小规模测试使用 4 路并发、10 组正反例，结果 `10/10`：

- 正例全部 same：GoogLeNet/通用语境的全连接层、pandas/通用语境的张量、BERT/通用语境
  的掩蔽语言模型、Inception/通用语境的卷积层、CNN/卷积神经网络。
- 负例全部 new：注意力机制中的值/感官输入、数学卷积/深度学习互相关、Vocab 类/词表、
  SGD/小批量 SGD、BERT/BERT-base。
- CNN 正确注册标准缩写和英文全称；五个负例没有给新 Entity 注册冲突 alias。

结果文件：`tmp/identity-knowledge-v8-eval10-20260811-153042.json`。确定性验证为
`108 passed, 22 subtests passed`，`compileall` 和 `git diff --check` 通过。

### 7. 已完成：知识辅助的 Entity 概念解释与旧坏样本冒烟

`Entity.definition` 已明确为帮助 identity 判断的规范概念解释，不要求是严格词典定义。
当前 `kg/definitions.py` 版本为
`entity-definition-observations-3-knowledge-assisted`：

- 全部 Observation 用于锚定当前义项和语料特有的具体事实；可靠通用知识可补全通常含义、
  上位类别和跨场景稳定特征。
- 用途、性质、实现方式、典型比较和应用场景可以进入解释，但一次局部场景不能成为概念
  身份边界。
- 拥有一条 Observation 的 Entity 也会进入整理；首次抽取留下的局部 definition 不再因为
  观察数量不足而长期保留。
- prompt version 已升级，旧 `entity-definition-observations-2` 缓存不会阻止新版重算。

真实 MiniMax-M3 冒烟在原 30-chunk 数据库的独立可写副本中复测 7 个 Entity：批量规范化、
LSTM、MXNet、全连接层、多层感知机、卷积神经网络和注意力机制。结果 `7/7` 成功、0 失败，
再次运行全部按观察指纹缓存跳过；SQLite integrity 与 `kg check` 均通过。关键旧问题已改善：

- 多层感知机从“二维图像输入、四阶权重张量”的局部描述恢复为由多个全连接层组成的前馈
  神经网络。
- 批量规范化不再被收窄为 GoogLeNet 后续版本中的专属层。
- 全连接层的通用含义保持在第一句，二维图像、RNN 和注意力对比只作为场景补充。
- 只有一条 Observation 的卷积神经网络也成功生成概念解释。

冒烟数据库：`tmp/fresh30-definition-v3-smoke-20260811-191617.db`。原始数据库
`tmp/fresh30-identity-v8-20260811-154811.db` 未修改，仍为 0 条 definition synthesis。
确定性验证更新为 `111 passed, 22 subtests passed`，`compileall` 和 `git diff --check` 通过。

### 8. 已完成：EntityObservation alias 传递修复

30-chunk 样本中 `飞桨` 与 `Paddle（飞桨）` 被拆分的直接原因不是 identity 模型不知道
两者的翻译关系：Observation #53 已抽取出 `aliases=["PaddlePaddle"]`，但 v8 构造
resolution 请求时遗漏了 `EntityObservation.aliases`。模型一方面被要求审核新观察 aliases，
另一方面实际看不到这些候选名称，导致 `PaddlePaddle` 没有注册；后续 `paddle` 因而无法
召回已有的 `飞桨` Entity。

当前 `kg/resolution.py` 版本升级为 `entity-identity-ontology-9-alias-visible`，请求中明确
传入待审核 aliases，机械层仍只接受抽取阶段提出、且 resolver 明确返回在
`accepted_aliases` 中的名称。真实 MiniMax-M3 顺序重放结果：

```text
飞桨 + alias PaddlePaddle -> new Entity #1
paddle                     -> same Entity #1
最终 Entity                -> 飞桨（PaddlePaddle）
最终 aliases               -> 飞桨 / PaddlePaddle / paddle
```

冒烟脚本与数据库分别为 `tmp/alias-resolution-v9-smoke.py` 和
`tmp/alias-resolution-v9-smoke.db`。确定性全套验证仍为
`111 passed, 22 subtests passed`，`compileall` 和 `git diff --check` 通过。

### 9. 已完成：知识 alias、incoming alias 召回与 Top 10 回归

v9 冒烟之后的 6 组并发 6 测试确认了一个剩余边界：只在语料分别出现“支持向量机/SVM”
或“主成分分析/PCA”、且抽取没有给第一条 Observation 提出 alias 时，模型虽知道两者相同，
但字符串召回为空，旧流程仍会创建两个 Entity。当前 v10 做了三项有界修改：

- observation 的 name 与待审核 aliases 都可用于候选召回；这只扩大 LLM 看到的候选，不
  直接证明 `same`，也不绕过 alias 审核。
- 普通 identity 候选上限从 5 提高到 10。
- resolver 可返回最多五个 `knowledge_aliases`，仅限可靠通用知识中的标准翻译、英文全称、
  通行缩写、正式名/简称和拼写变体；原文 alias 仍单独通过 `accepted_aliases` 审核。

同一批真实 MiniMax-M3 顺序测试在 `workers=6` 下由 v9 的 `4/6` 提升为 v10 的 `6/6`：

```text
飞桨                    -> paddle                    same
批量规范化              -> BN                        same
Gated Recurrent Unit    -> 门控循环单元              same
Convolutional Neural Network -> 卷积神经网络          same
支持向量机（无原文 alias） -> SVM                    same
主成分分析（无原文 alias） -> PCA                    same
```

最后两组在第二条到来前已由 resolver 注册标准全称与缩写，候选分数均为 `1.0`。
结果文件：`tmp/alias-resolution-v10-eval6-20260811-201559.json`。确定性验证为
`114 passed, 22 subtests passed`，`compileall` 和 `git diff --check` 通过。

下一步不要立即全量运行；应先在全新独立数据库重跑同一批 50 个 chunk，确认真实抽取顺序、
候选召回和 identity 组合仍稳定，再决定是否全书重跑。

### 10. 当前仓库与实验状态摘要

- 当前分支：`experiment/assertion-layer-pilot`。
- 当前 HEAD：`14366a7`，与 `origin/experiment/assertion-layer-pilot` 一致。
- 未提交代码还包括本节 Entity identity 修改；`kg/validation.py` 中原有 judge v5 scoped
  projection 修改必须保留，不要混淆或覆盖。
- `tmp/` 实验文件被忽略，不会进入 Git；不要覆盖旧数据库。
- 旧 fresh50-v6 实验实际只完成 47/50：
  `tmp/fresh50-v6-20260810.exit` 为 `done=47/50`、`final_status=1`，没有 `.finished`。
- 对这 47 个完整 chunk 的结论应读取 `tmp/fresh47-quality-analysis-20260810.md`，不要把
  47/50 的运行状态误写为完整 50-chunk 成功。

---

## 2026-08-08：人工质量审计、Assertion 层与抽取实验

### 1. 人工审计得到的问题

建立并人工标注了 `audit/quality_audit_50.csv`（Windows 下编辑，GBK 编码）。主要问题：

1. 一批代码局部符号、辅助类、API、UI 操作词不应成为 Entity，例如 `train_ch6`、
   `fancy_func`、`d2l.Timer`、`d2l.Animator`、`Stopping` 等。
2. Claim 经常只保留结论，忽略凸性、学习率、实现版本、时间或适用范围等必要条件。
3. 部分 Claim 虽然能在原文找到依据，但只是当前代码如何调用辅助工具、练习题中的提问
   或界面操作步骤，并不是希望进入图谱的可复用教学知识。
4. 只审计已经生成的 Entity/Claim 会漏掉“正常知识被过滤”的召回问题；后续必须增加
   从原始 chunk 反向检查遗漏的审计。

人工审计中的典型问题已定位回 D2L chunk：

| Chunk | 典型问题 |
| ---: | --- |
| 80 | `numerical_lim` 等局部示例符号 |
| 290 | 围棋/国际象棋与强化学习关系的语义泛化 |
| 388 | `train_ch6`、`d2l.*` 等代码实现对象 |
| 593 | 练习题及参数 `w` |
| 675 | SGD 收敛结论遗漏凸问题、学习率等必要条件 |
| 736 | `fancy_func` 等局部函数 |
| 891 | 当前实现语境被泛化 |
| 1082 | EC2 界面操作和示例对象 |

### 2. 当前确定的数据模型

保留 Claim 的三元组图结构，同时增加一层轻量 Assertion：

```text
Claim = (canonical subject, canonical relation, canonical object)
  └─ Assertion = 可独立判断真假的完整命题 + 限制语境 + polarity
       └─ Evidence = 原始 Source Passage、模型引文、裁判结果与版本信息
```

- Claim 仍是图上的边和关系合并单元，不把全部语境塞进边键。
- Assertion 是同一 Claim 下带条件的完整命题；多个 Assertion 可能是不同条件、不同粒度，
  也可能是重复表述。
- `statement` 是忠于原文、经过端点规范化的完整关系表述，不是原句全文，也不是脱离
  原文的自由改写。
- `scope_text` 保存会限制命题成立范围的条件；`scope_is_restrictive` 标记删除该条件是否
  会使命题变假或明显扩大适用范围。
- 当前只按最终规范化命题做精确去重，尚未做 LLM 语义级 Assertion 合并。
- 最终裁判必须在 Entity 与 relation 都规范化之后执行，旧端点上的判断不能复用。

### 3. 实验分支实现

分支：`experiment/assertion-layer-pilot`

主要修改：

- schema 升至 v10，新增 `assertions` 表；`claim_observations` 保存 statement、scope、
  fingerprint、最终 Claim/Assertion ID；Evidence 可指向 Assertion，同时保留 Claim ID
  以兼容 Claim 级聚合。
- Entity 提示词加入教学知识准入、代码/练习/UI 排除和叙述正文端点召回保护。
- Relation 提示词要求完整 Assertion、必要条件、精确端点名称、关系方向和可复用知识
  检查；代码只能补充证明正文已介绍的知识，不能单独生成关系。
- judge 同时检查原文支持、端点语义扩大、必要条件遗漏，以及实现事实/操作步骤是否被
  错当成一般知识。
- 新抽取的完整否定命题仍属于原文 `support`；否定含义写入 predicate/statement，不能
  因为出现“不收敛”等词误标成 `oppose`。
- 修复 canonical name 包含原名称时反复替换的问题，例如
  `随机梯度下降 -> 随机梯度下降（SGD）（SGD）`，保证 Assertion fingerprint 幂等。
- 端点不匹配的解析拒绝现在会在日志中记录 subject、object 与 statement 摘要，方便审计。
- audit/export/check 已适配 Assertion；`kg check` 会检查没有 Evidence 的 Assertion。

验证：全套测试 `92 passed, 28 subtests passed`，`git diff --check` 通过。

### 4. 三轮 5-chunk 实验

三轮均使用独立数据库，测试 chunk `80、388、593、675、1082`，关闭章节摘要与定义综合，
总 LLM 并发不超过 6。

| 轮次 | Entity obs | Claim obs | 物化 Assertion | 结论 |
| --- | ---: | ---: | ---: | --- |
| v1 | 35 | 27 | 23 | 条件保留改善，但大量代码/API/UI 内容仍进入图谱 |
| v2 | 18 | 13 | 11 | 代码/UI 精度大幅改善，但过度过滤叙述正文，chunk 675 丢失正常关系 |
| v3 | 24 | 17 | 16 | 恢复正常正文召回，同时继续挡住代码实现关系；当前最平衡 |

v3 的关键结果：

- chunk 388 只保留 `LeNet-5` Entity，0 条代码实现关系；`train_ch6`、`d2l.Timer`、
  `d2l.Animator`、`d2l.Accumulator` 均未形成知识。
- chunk 593 为 0 Entity / 0 Claim，练习题没有生成知识。
- chunk 675 恢复 `梯度下降`、`深度学习` 等端点，共生成 7 条 Claim，全部通过 judge；
  恢复了“大样本时 SGD 相比梯度下降更适合”和“非凸情况下最优性保证不可用”等关系。
- chunk 675 没有端点名称不匹配拒绝，fingerprint mismatch 为 0。
- chunk 1082 保留了“停止实例”和“终止实例”的区别：它来自叙述正文而非点击步骤，
  是否属于目标图谱仍是一个范围选择。
- `matplotlib`、`d2l 包` 可能作为正文明确介绍的软件资源 Entity 出现，但没有物化关系。
- v3 数据库 `tmp/assertion-pilot-v3-20260808.db` 的 `kg check` 通过。

当前边界问题：图谱是覆盖 D2L 教材中的全部可复用知识，还是只保留深度学习概念、方法、
模型和数据集？前者可接纳软件资源及云服务知识，后者应进一步排除。这不是简单的原文
支持问题，需要在大样本审计后决定。

### 5. 正在运行：200-chunk 扩大实验

2026-08-08 22:53 启动后台任务，目前仍在运行，完成状态**尚未验证**。

- tmux：`kg-assertion-sample200-v3-20260808`
- 数据库：`tmp/assertion-sample200-v3-20260808.db`
- 日志：`tmp/assertion-sample200-v3-20260808.log`
- 成功标记：`tmp/assertion-sample200-v3-20260808.finished`
- 退出标记：`tmp/assertion-sample200-v3-20260808.exit`
- `chunk-workers=3`，`judge-workers=3`，共享 `llm-max-concurrency=6`
- 关闭 section summaries 与 definition synthesis，只测试核心抽取质量
- 单个 chunk 失败会记录并继续，避免整夜任务被一个样本中断

抽样为 10 个分散窗口，每个连续 20 个 chunk，共 200 个新 chunk：

```text
20–39, 130–149, 240–259, 350–369, 460–479,
570–589, 680–699, 790–809, 900–919, 1020–1039
```

### 6. 200-chunk 完成后的检查清单

- [ ] 检查 `.finished`、`.exit`、目标 200 个 `source_progress` 和 `kg check`，不能只看 tmux
- [ ] 汇总 Entity / Claim / Assertion / Evidence 数量、失败与解析拒绝原因
- [ ] 从生成结果抽约 50 个 Entity/Assertion 做精度审计
- [ ] 从 10–15 个原始 chunk 反向检查重要 Entity/Assertion 是否漏抽
- [ ] 专门检查代码、练习题、UI/云服务操作、软件资源四类边界样本
- [ ] 检查同一 Claim 下多个 Assertion 是重复、不同条件，还是需要语义合并
- [ ] 大样本通过后再考虑跑完整 D2L；暂不迁移或覆盖正式 schema v9 数据库

---

## 一、优先要做的（按用户 2026-08-07 的判断排序）

关系归并**暂停**。先把上游做好，否则在脏数据上归并没有意义。

### 1. 实体噪声：不该进来的东西被抽成了实体

教材里的函数名、示例中的地名、图表引用等被当成实体。粗略模式匹配的量级：

| 模式 | 命中 | 例 |
| --- | ---: | --- |
| 带点的 API 路径 | 179 | `tf.range`、`np.random.normal` |
| 公式 / 数学表达式 | 69 | `$\mathcal{X}$`、`$\mathbb{R}$` |
| 图表引用 | 19 | `图 fig_Neuron：真实的神经元` |
| 章节标题 / 超长串 | 238 | `localhost:8888` |

注意：另有 798 个命中「代码标识符」模式，但里面混着 `TensorFlow`、`LeNet`、`AlexNet`
这些**正当实体**，不能直接按模式清理。3817 个实体里约 620 个是孤立点，其中 163 个命中可疑模式。

要做的：
- [ ] 定义「什么算 Entity」的判定测试，写进 `kg/ontology.py`，并在抽取提示词里给排除项
- [ ] 决定已入库噪声怎么处理（打标记 vs 删除）。删除会牵动 Evidence 与 Claim，需先想清楚
- [ ] 区分「代码标识符」的两种情况：`nn.Linear` 作为 API 是噪声，但 `全连接层` 的别名里
      有 `nn.Linear` 是**正确的别名合并**（见下面第五节，别名合并是当前做得最好的一环）

### 2. 三元组的关系不准

抽样验证发现，claim 的关系与其自身原文对不上的比例不低。已确认的两类：

- **方向错误**：`ResNet-50 --用于训练--> ImageNet`（原文是"ResNet-50 在 ImageNet 上训练"，
  主宾反了）。在 `用于训练` 的 36 条里有 4 条属于此类。
- **原文根本不支持**：schema C 下有 5.6% 的 claim 被判为 `none`（换成表达力更强的
  schema H 后降到 0.5%，说明其中大部分是 schema 表达力不足而非抽取错误，但仍有残留）。

要做的：
- [ ] 全库筛查反向边候选（如 subject 是模型、object 是数据集、关系却是 `用于训练`），
      统计规模后再决定修法
- [ ] 在抽取提示词里加方向约束与自查
- [ ] 考虑在 `kg/validation.py` 的裁判里增加方向核对

### 3. 1573 条孤儿观察（重试路径 bug）

`entity_observations` 里有 1573 条 `resolution_outcome=''`、`entity_id IS NULL`、
`resolver_model=''`——从未进入实体对齐。

成因：片段第一次处理到一半崩溃，观察已落库但未对齐，片段标 `failed`；重试时重新抽取，
LLM 输出不同 → 新的 `observation_key` → 存成新行并正常对齐，旧的那批永久孤儿。
例：chunk 340 的 progress 记录写着 `entity_observations: 9`，但库里该 chunk 有 63 条。

知识本身没丢（重试重抽过），但 `entity_observations = 11671` 这个数里有 1573 条死行，
凡是按观察数算的统计都会偏高。

要做的：
- [ ] 修重试逻辑：重试前清理该 (source, chunk) 下未对齐的观察，或让重试复用已有观察
- [ ] 清理存量的 1573 条（或至少标记）
- [ ] **跑第二本书之前必须修**，否则继续累积

### 4. `relation_types.description` 只在创建时写一次

`kg/vocabulary.py:128` 的 INSERT 是唯一写 description 的地方。一个关系的定义是它
**第一次出现**时由模型看着那一条 claim 写的，之后几百条 claim 归进来，定义从不更新。

例：`应用于` 有 296 条 claim，定义只反映第 1 条；`涉及` 有 56 条，定义写着
"主语通常为书籍、网站、文献等资源"，显然来自某条"某书涉及某主题"。

这是关系归并做不好的**直接原因之一**——归并时喂给模型的定义本身就是脱离上下文的产物。

要做的：
- [ ] 参照 `kg/definitions.py` 给实体做的定义聚合（`synthesize_pending`），
      给关系也做一套：从该关系的全部 claim 重新合成定义

---

## 二、关系归并探索的结论（2026-08-07，避免重做）

目标：把开放抽取产生的 **2093 种关系**（在 `relation_types` 表里是 2365 行，其中 272 种
的 claim 未通过裁判）归并成几十种可复用的 schema。

### 验收标准（用户定）

- 条数不追求精确，**只避开数量级误差**：几十条可以，几百或个位数不行
- 不能出现某一族吞掉绝大多数成员的分布失衡
- 「`improves` 和 `solves` 该不该合并」这类问题**本来就没有唯一答案**，
  不要按吹毛求疵的逻辑设计实验

### 试过的八种做法

| 做法 | 结果 | 结论 |
| --- | --- | --- |
| A 分批归纳（8 批）→ 合并 | 15 条 | 跨批看不见彼此，造出 `composed_of` / `structural_composition` 这类重复；合并阶段脱离语料，滑向按主题分类（`训练过程`、`部署环境` 不是谓词）。**淘汰** |
| B 单次归纳（全部 2093，带定义） | 35 条 | 可用 |
| C 单次归纳（全部 2093，仅名字+频次） | 22 条 | 可用，且**去掉定义反而更好** |
| D 一步到位（schema + 全量成员，带定义） | ❌ | schema 塌成 4 条（`part_of` 一族吞 1639 个），成员漏 314 / 重 380 / 编造 230。模型在 rationale 里自己承认"结构错误，下面是修正版" |
| D-lite 一步到位（不带定义） | ❌ | schema 36 条尚可，但漏 150 / **重 796** / 编造 245。单次调用耗时约 25 分钟 |
| E 分批分配（每批完整 schema + 40 个关系，53 批） | ✅ | 三份 schema 全部 **0 漏 0 重 0 造 0 空族**。方法本身可靠，对 schema 选择不敏感 |
| G 迭代归纳（标签配真实原文，schema 逐批累积） | 75 条 | 前 300 个高频标签（覆盖 65.8% 观察）。轨迹 39→47→52→58→75，**新增量不再衰减**，跑满 14 批外推到 150–200 条，故在第 5 批停止 |
| H = G 的独立合并（只许合并，不许新增） | 71 条 | 75→71，**只合并了 4 对**。模型自报 7 对判定测试重叠却拒绝合并，理由是"排除条款能区分"——它钻了提示词的口子 |

**核心结论：归纳与分配必须是两次调用。** 分界线不在输入塞不塞得下（2093 个标签+频次
只有约 21k token），而在**输出要不要逐项枚举**。输出二三十条 schema 模型游刃有余；
输出 2093 个 id 会漏、会重、会编，且为了让枚举可行会主动把 schema 压塌。

### 质量测量（这才是关键，机械检查全过不代表分得对）

**a) 原文独立重判**（给 claim 的 `source_text` 原文 + 主宾 + schema，**不给原关系标签**，
让模型独立判该是哪条，再和标签级分配比一致率，同一批 198 条）：

| | schema C（22 条，从标签归纳） | schema H（71 条，从原文归纳） |
| --- | ---: | ---: |
| 与分配一致 | 46.5% | **56.1%** |
| 判为 `none`（原文不支持） | 5.6% | **0.5%** |
| 标签多义率 | 52.4% | 50.8% |

把语料放进归纳，一致率涨约 10 个点，`none` 大幅下降。但**标签多义率几乎不变**——
多义是标签自身的属性，换 schema 治不了。

**b) 标签纯度**（同一标签下抽 4 条 claim，直接问"是不是同一种关系"，不经过 schema）：

- **70% 纯**。按频次分层：50+ 次的标签纯度 **80%**，10–49 次仅 50%，2–3 次 79%。
  **高频标签反而最纯**，问题集中在中频段。
- 前面那个 50.8% 虚高约 20 个点，差额是相邻 key 边界糊造成的裁判抖动，不是真多义。

**c) schema 条目纯度**（同一 schema 条目下，从 4 个**不同**原始标签各抽 1 条）：

- 二值评分（4 条全同才算纯）：**40%**（27/67）
- 放宽为「≥3/4 同类即算可用」：**63%**
- 分级评分（平均最大组占比）：**75%**
- 完全散架（每条各自成组）：5/67

**同一批判断，打分规则不同，结论从 40% 到 75%。** 大族反而干净（`uses`/`acts_on`/
`covers`/`applies_to`/`produces` 五个最大的都判纯，合计占全图 40% 的 Claim），
脏的集中在中小族：`has_attribute`（吞 99 个标签）、`interacts_with_affects`（127 个）、
`depends_on`（87 个）、`improves`（46 个）。

**结论：聚类这一步确实引入了新的混杂**（schema 条目纯度低于原始标签纯度）。这个定性
结论稳，但具体数值高度依赖打分规则和抽样设计，不宜当作硬指标。

### 试过但不成立的想法

- **类型签名**（把映射单位从「标签」改成「标签 + 两端实体类型」）：粒度扫描显示无论
  怎么粗化都没有甜蜜点——最粗时 7213 条 claim 仍被切成 3969 个组合（复用度 1.8），
  而判别力已掉到 72%。原因是实体类型本身也是没归并的开放词表（1400 种），且
  2093 个标签 / 7213 条观察的密度下，加任何区分特征都会打散到接近 1:1。
- **让 LLM 做归并**：H 那次只合并 4 对。但**这个结论证据不足**——只跑了一次、一种提示词，
  而且模型明显是钻了提示词的口子，不能据此断言"LLM 不擅长合并"。

### 待验证（工作表已生成，用户暂缓）

人工盲标 20 组（每组 4 条真实语句 + 目标关系名与定义，模型判定隐藏），用来定标尺：
你与模型的一致率、模型偏松还是偏紧、40% 还是 75% 更接近真相、以及**「说不清」的比例**。
最后这个数最关键：如果大量卡住，说明"存在唯一正确答案"这个假设不成立，
后面就不该继续优化聚类。

---

## 三、副产品（已确认，与归并无关）

- **关系归并会让相同三元组塌缩累积证据**：多证据 Claim 从 260 涨到 452（+74%），
  但占比仍只有 6.9%。单本书内的天花板就这么高，真正的多来源互证要靠第二本书。
- **别名合并是当前做得最好的一环**：11671 条观察 → 3817 个实体
  （`same` 6236 / `new` 3812 / `uncertain` 50），2053 个实体带多别名，
  且能跨 LaTeX 符号、英文术语、框架 API 名、中文变体合并。例：
  `权重` ← `$\mathbf{w}$` / `synaptic weights` / `W^(1)` / `突触权重`（25 个别名）。

---

## 四、产物位置

**注意 `tmp/` 在 `.gitignore` 里，以下内容不进版本库。** 如果需要长期保留，
应移入仓库跟踪目录。

```
tmp/relschema/
  schema-*.json        五份候选 schema（A/B/C/D-lite/G/H）
  assign-*.json        四份全量分配结果，每份 2093 行
  validation-*.json    两次原文验证的模型原始判断
  relations-2093.json  喂进去的全部 2093 个关系（含定义、频次、样例）
  raw/                 所有中间产物的原始 JSON（含失败的 D）
  scripts/             全部实验脚本（24 个 .py）
  pages/               生成的三个 HTML 页面与模板
```

主要脚本：

| 脚本 | 作用 |
| --- | --- |
| `relcluster.py` | 分批归纳 → 合并 → 分配 → 报告（做法 A） |
| `induce_once.py` | 单次归纳（做法 B / C） |
| `exp_oneshot.py` | 一步到位实验（D）与分批分配（E），`e <schema> <out>` 可指定 schema |
| `induce_iter.py` | 迭代归纳（做法 G） |
| `consolidate.py` | 独立合并（G → H） |
| `validate.py` / `validate_h.py` | 原文独立重判 |
| `label_purity.py` / `schema_purity.py` | 标签纯度 / schema 条目纯度 |
| `typesig.py` / `typesig2.py` | 类型签名判别力扫描 |
| `build_worksheet.py` | 生成人工盲标工作表 |

脚本依赖 `kg.llm`，通过 `sys.path.insert` 引入仓库根目录；并发统一由
`EXP_CONCURRENCY` 环境变量控制（默认 6，用户要求不要超过）。
运行前需 `MINIMAX_API_KEY`，以及 `pip install json-repair`（当前环境缺这个依赖）。

生成的页面（Artifact，私有）：

- 开放词表体检 https://claude.ai/code/artifact/5c3bffc8-f6e3-4e4f-8c29-a378b2775c69
- 关系映射对照表 https://claude.ai/code/artifact/0dd20881-ee6c-444f-8d98-8323e5e80e76
- 人工盲标工作表 https://claude.ai/code/artifact/0233e57b-bd08-41ce-852a-f71f1709549f
- 知识图谱总览（力导向图，可筛选、可点查证据）
  https://claude.ai/code/artifact/6f0f5d28-f549-4ff9-a38d-2c37dadf75ba
  由 `graphdata.py` 导出数据 + `pages/graph.template.html` 生成。
  仓库自带的 `kg viz` 产出的是审计列表且体积达 56MB，不适合看图结构。

---

## 五、下次继续时的建议顺序

1. 先做第一节的 1–3（实体噪声、关系方向、孤儿观察）——这三个都有明确对错，
   不像关系归并那样欠定
2. 再做第 4（关系定义聚合），它是归并做不好的直接原因之一
3. 上游干净之后，再回来看关系归并。届时可以直接复用
   `tmp/relschema/scripts/` 里的脚本，schema H 可作为起点
4. **跑第二本书之前**必须先修孤儿观察的重试 bug

未验证的方向：把 schema 写进抽取提示词做受控抽取（用户最初三阶段计划的第三步）。
它绕开了"标签"这一中间层——模型读句子直接选 schema key，带完整上下文，
理论上不继承标签多义问题。但它继承 schema 质量，所以要等 schema 质量确认后再做。
## 2026-09-05：细关系归一＋粗类别后台实验（已完成）

- 用户确认保留规范细关系，将剩余关系标注到粗类别，供不同检索与编排逻辑使用。
- 新脚本 `experiments/relation_synonyms_categories.py`：词面/E5候选并集、直接同义核验、
  单层粗分类与独立提示词复核；MiniMax-M3官方API，共享总并发6。
- 本轮输入全919类、3139 Claim、3309 Assertion，源库只读。旧80类单向粗化结果保留。
- 运行目录 `tmp/relation-synonyms-categories-full919-20260905-1300/`；先有界冒烟，成功再全量。
  tmux独立socket `kg-relcat-20260905-1300`，session `relations-m3-c6`。
  本轮已完成（`.exit=0`、`summary.json: complete`），收敛结论见本文2026-09-07记录。
  新运行仍须以退出码与结果文件确认完成，不能根据tmux存在宣称完成。
- 关键结果：`fine_claim_mapping.json`、`fine_catalog.json`、`coarse_catalog.json`、
  `category_mappings.json`、`fine_category_mapping.json`、`derived.db`、`report.md`。
  细关系数和粗类别数分开统计，同模型复核覆盖率不是独立准确率。
