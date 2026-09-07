# 实验结果登记册

大型实验产物保存在被 Git 忽略的 `data/experiments/`，本目录只提交体积较小的实验清单。

## 保存规则

1. 每次全量实验使用新的递增编号：`EXP-001`、`EXP-002`、`EXP-003`……
2. 已登记实验的数据库和 JSON 导出均视为只读快照，不迁移 schema、不原地续跑、不覆盖。
3. 新实验必须写入新的目录，并保存 source version、schema、核心计数、文件大小和 SHA-256。
4. 实验比较优先使用各自的 `graph.json`；需要完整审计信息时使用对应的 `graph.db`。
5. 若产物需要迁移到其他机器，应整体复制 `data/experiments/<experiment-slug>/`，并用清单中的 SHA-256 验证。

## 已登记实验

| ID | 名称 | 状态 | Artifact | Manifest |
| --- | --- | --- | --- | --- |
| EXP-001 | D2L 全书开放抽取基线 | Frozen | `data/experiments/exp-001-d2l-fullbook-open-extraction-baseline/` | [EXP-001 manifest](exp-001-d2l-fullbook-open-extraction-baseline.json) |

## 同义归一后粗化的小样本实验

`relation_two_stage_pilot.py` 先提出同义候选，结合全部相关 Assertion 复核；只有
同一 Claim 的全部 Assertion 支持替换时才在派生图中使用规范细关系。之后由 AI
根据规范细关系和代表事实生成最多 35 种粗关系，使用完整目录逐 Assertion 分配，
再用独立提示词复核单向蕴含。两个阶段都保留模型原始判断。

默认 80 种关系限于使用次数不超过 9 的类型，一半按频次分层，一半补入词面邻居；
这是有意增加同义候选的诊断样本，不是全书压缩率的无偏估计。模型为 MiniMax-M3，
官方 endpoint，并发上限 6；需 `MINIMAX_API_KEY`。`--prepare-only` 不调用模型。

```bash
python -m experiments.relation_two_stage_pilot \
  --db tmp/d2l-full1105-c6-20260817-193600.db \
  --output-dir tmp/relation-two-stage-pilot80-new-run
```

源库只读。`input.json` 保存样本事实及原证据；`derived.db` 是实验专用 schema，
其中 `fine_claims` 为替换去重后的细图，`claim_lineage` 保留每条旧 Claim 的去向，
`coarse_lineage` 记录每条 Assertion 的粗化提案、接受状态及版本。只有
`accepted=1` 才进入粗视图，粗关系的名称和定义在 `coarse_catalog.json` 中。
此数据库不能作为生产 `kg --db` 的输入。关系含义依赖相应 Assertion 的条件，
粗图不能用于脱离原文语境的无条件推理。

必须以 `summary.json: status=complete` 和派生库完整性确认完成；同模型的分离提示词
复核仍不是独立模型准确率验证。输出目录支持失败后的请求缓存复用，但输入或脚本变化
必须使用新目录，已完成结果禁止覆盖。

2026-09-05 首轮已完成，结果与已发现的语义问题见
[首轮报告](relation_two_stage_pilot_report.md)。
## 同义归一与粗类别标注

`relation_synonyms_categories.py` 是 2026-09-05 确认的新目标：保留规范细关系，额外标注粗类别，
不以粗谓词替换细关系。旧 `relation_two_stage_pilot.py` 保留作为单向粗化对照。

```bash
.venv/bin/python -m experiments.relation_synonyms_categories \
  --db tmp/d2l-full1105-c6-20260817-193600.db \
  --output-dir tmp/relation-synonyms-categories-full919-RUN
```

默认处理全部使用过的关系；`--sample-size 80 --smoke-only` 用固定80类输入，只测试两项候选发现、
这些候选的核验及4条事实分类。MiniMax-M3 官方API，总并发固定6。长任务使用独立tmux。
关键输出：`manifest.json`、`progress.json`、`retrieval.json`、`fine_claim_mapping.json`、
`fine_catalog.json`、`coarse_catalog.json`、`category_mappings.json`、`fine_category_mapping.json`、
`derived.db`、`summary.json` 和逐条 `report.md`。派生库是实验schema，不用于生产 `kg --db`。
结束以 `summary.json` 的 complete、退出码0和数据库完整性检查为准；自动复核覆盖率不是准确率。

2026-09-07 已初步收敛为规范细关系与独立粗类别两层表示：全919类实验保留873种细关系，
生成27个粗类别，3218/3309条Assertion分类通过同模型复核（覆盖率97.25%），91条待定。
实验完成不等于正式库已接入；少量已定位错误的撤回建议尚未应用。
类别分布、使用边界及后续工作见[收敛报告](relation_synonyms_categories_report.md)，
79条细关系替换的逐条审阅见[审计报告](relation_synonyms_categories_fine_audit.md)。
