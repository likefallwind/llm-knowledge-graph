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

