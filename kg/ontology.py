"""六种 Entity 类型和三种 Claim 关系的唯一定义源。

抽取、关系裁判和身份裁决三处提示词都从这里渲染，避免同一条定义在多个
文件里各写一遍并逐渐分叉。定义写成「判定测试 + 排除项 + 正反例」，而不是
同义改写，因为同义改写无法在边界样本上给出确定答案。

修改本文件的语义时，必须同时 bump 引用它的提示词版本常量
（extraction.EXTRACTION_PROMPT_VERSION、validation.VALIDATION_PROMPT_VERSION），
否则旧的 done 片段会被错误跳过。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Definition:
    name: str
    summary: str
    test: str
    excludes: tuple[str, ...]
    positive: tuple[str, ...]
    negative: tuple[str, ...]


ENTITY_TYPE_DEFS: tuple[Definition, ...] = (
    Definition(
        name="resource",
        summary="知识载体本身",
        test="它是一份可被阅读或引用的材料实体（论文、教材、课程、文档、代码库、章节），"
        "而不是材料里讲述的内容。",
        excludes=("材料中描述的方法、数据或概念，按它们各自的类型判定",),
        positive=("《神经网络与深度学习》", "CS229 讲义", "附录 A"),
        negative=("反向传播（是做法，应为 solution）",),
    ),
    Definition(
        name="criterion",
        summary="评判或优化的标准",
        test="它给出好坏的度量或优化目标：损失函数、评价指标、优化目标、评测协议。",
        excludes=(
            "计算或优化该度量的算法，应为 solution",
            "被评测的数据集合，应为 data",
        ),
        positive=("交叉熵损失", "准确率", "最大似然目标"),
        negative=("梯度下降（是求解方法，应为 solution）",),
    ),
    Definition(
        name="data",
        summary="具体的数据集合",
        test="它是一组具体的样本或记录的集合，可被用于训练、验证或评测。",
        excludes=(
            "数据的数学表示或结构，应为 concept",
            "产生或变换数据的方法，应为 solution",
        ),
        positive=("ImageNet", "MNIST", "训练集"),
        negative=("张量（数学对象，应为 concept）", "数据增强（做法，应为 solution）"),
    ),
    Definition(
        name="task",
        summary="要解决的问题",
        test="它有明确的输入、输出或成功条件，回答「要做什么」。",
        excludes=(
            "完成该任务的方法，应为 solution",
            "衡量完成质量的指标，应为 criterion",
            "没有明确输入输出的研究方向，应为 concept",
        ),
        positive=("图像分类", "机器翻译", "目标检测"),
        negative=(
            "计算机视觉（研究方向，没有单一输入输出，应为 concept）",
            "卷积神经网络（做法，应为 solution）",
        ),
    ),
    Definition(
        name="solution",
        summary="解决问题的做法",
        test="它回答「怎么做」：算法、模型、架构、过程、系统或工具，可以被执行或实例化。",
        excludes=(
            "它所解决的问题，应为 task",
            "它所优化的目标，应为 criterion",
            "它依赖的数学性质或现象，应为 concept",
        ),
        positive=("反向传播", "卷积神经网络", "Adam", "Transformer"),
        negative=("图像分类（问题，应为 task）", "交叉熵损失（目标，应为 criterion）"),
    ),
    Definition(
        name="concept",
        summary="兜底类型",
        test="前五类都不成立，且原文给出了实质性定义或知识含义的对象："
        "数学对象、性质、规律、现象、研究方向。",
        excludes=(
            "离开上下文没有稳定身份的通用词，例如「定义」「公式」「方法」「过程」「性质」本身",
            "指代不明的短语，例如「上述方法」「该模型」",
        ),
        positive=("梯度", "凸性", "过拟合", "计算机视觉", "人工智能"),
        negative=("「定义」", "「上述方法」（无稳定身份，不应成为 Entity）"),
    ),
)


RELATION_DEFS: tuple[Definition, ...] = (
    Definition(
        name="is_a",
        summary="类属：subject 是 object 的一个子类或实例",
        test="实例测试——任取一个 subject，它本身就是一个 object，并且继承 object 的一般性陈述。"
        "同层的兄弟节点应当是互斥可替换的选项。"
        "两端的 entity_type 通常相同，可用作自查线索；"
        "但类型本身也可能判错，因此类型不同不构成否决理由，判定仍以实例测试和排除项为准。",
        excludes=(
            "领域归属（subject 是 object 范围内的分支），应为 part_of",
            "构件关系（subject 是 object 的零件），应为 part_of",
            "仅仅相关、使用、作用于或属于同一领域",
            "端点被截短限定词后才成立的关系。"
            "「深度学习是一种人工智能方法」中的 object 是「人工智能方法」，不是「人工智能」",
        ),
        positive=(
            "卷积神经网络 is_a 神经网络",
            "交叉熵损失 is_a 损失函数",
            "深度学习 is_a 人工智能方法（原文确实写了「人工智能方法」时）",
        ),
        negative=(
            "深度学习 is_a 人工智能（领域归属，应为 part_of）",
            "卷积层 is_a 卷积神经网络（构件，应为 part_of）",
        ),
    ),
    Definition(
        name="part_of",
        summary="整体归属：subject 属于 object，但 subject 本身不是一个 object",
        test="下列三种之一，且原文明确陈述："
        "(a) 构件——subject 是 object 的组成部件，同层部件是共存协作而非互斥选项；"
        "(b) 阶段——subject 是正文明确列出的流程步骤；"
        "(c) 领域归属——原文明确说 subject 是 object 的分支、子领域或组成部分。",
        excludes=(
            "共现、并列举例、超链接、章节先后顺序",
            "解决同一任务或过程的若干方法、途径和可替换选项；它们不是共存构件，"
            "不能仅因原文使用「包括」就判为 part_of",
            "「特别是」「尤其」「涉及」「用于」「是……的重要进展」这类强调或相关性表述，"
            "它们没有陈述任何归属关系",
            "目录说明「以 subject 为标题的章节」属于某书或某章时，不能推出同名算法、"
            "模型或概念本身 part_of 该书或该章；若表达目录结构，subject 必须保留"
            "章节编号或「章、节、附录」等完整 resource 身份",
            "仅凭常识成立、但当前原文没有说出来的归属",
        ),
        positive=(
            "卷积层 part_of 卷积神经网络（构件）",
            "附录 part_of 《神经网络与深度学习》（构件）",
            "机器学习 part_of 人工智能（领域归属，需原文写出「分支」「子领域」或「组成部分」）",
        ),
        negative=(
            "神经网络 part_of 人工智能，依据「人工智能，特别是神经网络与深度学习的发展」。"
            "「特别是」只是强调，没有陈述归属，应判 insufficient",
            "玻尔兹曼机 part_of 第 15 章 深度信念网络，仅因目录列出「15.1 玻尔兹曼机」。"
            "目录中的 resource 是「15.1 玻尔兹曼机」这一节，算法本身不是章节构件，"
            "应判 insufficient",
            "特征提取 part_of 降维，仅因原文说降维包括特征提取和特征选择两种途径。"
            "两者是可替换途径而非共存构件，应判 insufficient",
        ),
    ),
    Definition(
        name="prerequisite_of",
        summary="学习前提：理解 subject 是学习 object 的实质性前提",
        test="原文明确陈述不先掌握 subject 就无法理解或学习 object。",
        excludes=(
            "教材、章节或讲授的先后顺序本身",
            "「在……基础上」「结合……」这类措辞，若原文没有说明必需性",
            "「建议先浏览」「可先了解」「有助于」「便于」或「更顺畅地理解」只说明"
            "推荐和帮助，没有陈述不掌握 subject 就无法学习 object",
            "仅有引用或提及关系",
        ),
        positive=("线性代数 prerequisite_of 主成分分析（原文写明需先掌握）",),
        negative=(
            "第二章 prerequisite_of 第三章（仅有先后顺序）",
            "概率图模型 prerequisite_of 深度生成模型，仅因原文建议先浏览前者以便"
            "更顺畅地理解后者（推荐而非必需，应判 insufficient）",
        ),
    ),
)


ENTITY_TYPE_BY_NAME = {item.name: item for item in ENTITY_TYPE_DEFS}
RELATION_BY_NAME = {item.name: item for item in RELATION_DEFS}


def _render(item: Definition, *, with_examples: bool = True) -> str:
    lines = [f"- {item.name}（{item.summary}）", f"  判定：{item.test}"]
    for text in item.excludes:
        lines.append(f"  排除：{text}")
    if with_examples:
        for text in item.positive:
            lines.append(f"  正例：{text}")
        for text in item.negative:
            lines.append(f"  反例：{text}")
    return "\n".join(lines)


def entity_type_block(*, with_examples: bool = True) -> str:
    """六种 entity_type 的完整定义，供抽取提示词使用。"""
    return "\n".join(
        _render(item, with_examples=with_examples) for item in ENTITY_TYPE_DEFS
    )


def relation_block(*, with_examples: bool = True) -> str:
    """三种 relation 的完整定义，供抽取提示词使用。"""
    return "\n".join(
        _render(item, with_examples=with_examples) for item in RELATION_DEFS
    )


def relation_detail(relation: str) -> str:
    """单条关系的完整定义，供关系裁判使用。"""
    return _render(RELATION_BY_NAME[relation])


def entity_type_summary() -> str:
    """六种类型的一行摘要，供身份裁决使用（它只需要读懂类型标签）。"""
    return "\n".join(
        f"- {item.name}：{item.summary}——{item.test}" for item in ENTITY_TYPE_DEFS
    )
