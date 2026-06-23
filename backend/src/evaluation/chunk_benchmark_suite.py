"""切分策略微基准测试套件

为每个策略提供标准测试文档，输出可复现的对比报告。
"""
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional
from .chunk_quality_metrics import ChunkQualityEvaluator, ChunkQualityReport


# ============ 标准测试文档 ============

# 1. 普通叙事文本（中文）
NARRATIVE_ZH = """
机器学习是人工智能的一个重要分支。它通过算法让计算机从数据中学习和改进。深度学习是机器学习的一个子领域，使用多层神经网络来处理复杂模式。

在图像识别任务中，卷积神经网络（CNN）表现优异。它通过卷积层提取特征，池化层降低维度，最终通过全连接层输出结果。自然语言处理则常使用循环神经网络（RNN）或Transformer架构。

强化学习是另一种重要的学习范式。智能体通过与环境交互获得奖励信号，逐步学习最优策略。AlphaGo就是一个著名的强化学习成功案例。

迁移学习允许将在一个任务上学到的知识应用到相关任务中。这大大减少了训练所需的数据量和计算资源。联邦学习则关注在保护数据隐私的前提下进行协作训练。

大语言模型如GPT和BERT改变了自然语言处理的格局。它们通过在海量文本上预训练，然后在下游任务上微调，取得了突破性进展。
"""

# 2. 医疗报告
MEDICAL_REPORT = """
【病史摘要】
患者男性，45岁。因"反复上腹痛3月余，加重1周"入院。患者3月前无明显诱因出现上腹部隐痛，伴反酸、烧心。近1周疼痛加重，夜间明显。

【体格检查】
生命体征平稳。腹平软，上腹部轻压痛，无反跳痛及肌紧张。肠鸣音正常。

【辅助检查】
胃镜示：胃窦部可见一处约0.5cm×0.6cm溃疡，底部覆白苔，周围黏膜充血水肿。
病理检查提示：慢性炎症伴轻度不典型增生。
13C呼气试验：阳性。

【诊断】
1. 胃溃疡（A1期）
2. 幽门螺杆菌感染

【治疗方案】
1. 四联疗法：奥美拉唑20mg bid + 阿莫西林1g bid + 克拉霉素0.5g bid + 胶体果胶铋220mg bid，疗程14天
2. 避免辛辣刺激食物
3. 定期复查胃镜

【预后】
经规范治疗后预后良好。建议根除Hp治疗后复查13C呼气试验，并定期随访。
"""

# 3. Markdown 文档
MARKDOWN_DOC = """# GraphRAG 系统架构

## 系统概述

GraphRAG 是一个基于知识图谱的检索增强生成系统。

### 核心组件

- **数据接入层**：支持多种文档格式的导入
- **知识图谱层**：使用 Neo4j 存储实体和关系
- **检索层**：支持向量检索和全文检索
- **生成层**：基于图谱上下文的增强生成

### 技术栈

GraphRAG 的技术栈包括：

1. **后端框架**：FastAPI
2. **图数据库**：Neo4j
3. **嵌入模型**：text-embedding-3-small
4. **大语言模型**：Qwen 系列

## 数据流程

### 文档处理

文档处理流程包括加载、分块、实体抽取和关系构建。

### 查询处理

查询处理流程包括：查询理解、图谱检索、上下文构建和答案生成。
"""

# 4. 中英混合代码文档
MIXED_CODE_DOC = """
def calculate_patient_risk(age: int, symptoms: list[str]) -> float:
    \"\"\"计算患者风险评分。
    Args:
        age: 患者年龄
        symptoms: 症状列表
    Returns:
        风险评分 (0-1)
    \"\"\"
    base_score = 0.0
    if age > 60:
        base_score += 0.3
    elif age > 40:
        base_score += 0.1

    high_risk_symptoms = ["chest_pain", "dyspnea", "hemoptysis"]
    for symptom in symptoms:
        if symptom in high_risk_symptoms:
            base_score += 0.2

    return min(base_score, 1.0)


class PatientRecord:
    \"\"\"患者记录类\"\"\"

    def __init__(self, name: str, age: int, diagnosis: str):
        self.name = name
        this.age = age  # 故意错误
        self.diagnosis = diagnosis
        self.medications = []

    def add_medication(self, drug: str, dosage: str) -> None:
        \"\"\"添加用药记录\"\"\"
        self.medications.append({"drug": drug, "dosage": dosage})
"""

# 5. 大段学术文本（测试语义边界检测）
ACADEMIC_TEXT = """
知识图谱是一种用图结构描述实体及其关系的技术。它在搜索引擎、推荐系统和问答系统中得到了广泛应用。知识图谱的构建涉及实体识别、关系抽取和知识融合等关键步骤。

实体识别的目标是识别文本中具有特定意义的实体名词。传统方法主要基于规则和字典，而现代方法则广泛采用深度学习技术。基于BiLSTM-CRF的序列标注模型是目前最常用的方法之一。

关系抽取旨在识别实体之间的语义关系。根据是否预先定义关系类型，可以分为限定域关系抽取和开放域关系抽取。远程监督方法通过将知识库与文本对齐，可以自动生成大量训练数据。

知识融合解决的是不同来源知识的一致性整合问题。实体对齐是知识融合的核心任务，它判断来自不同知识图谱的实体是否指向现实世界中的同一对象。近年来，基于图神经网络的方法在实体对齐任务上取得了最优性能。

然而，当前知识图谱技术仍面临诸多挑战。知识图谱的完备性不足是一个普遍问题，大量真实世界的知识尚未被收录。知识图谱的时效性也是一个重要问题，过时的知识会影响下游任务的性能。

未来研究趋势包括大语言模型与知识图谱的融合、多模态知识图谱、以及动态知识图谱的构建与推理。这些方向有望推动知识图谱技术在更广泛场景中的应用。
"""

# 6. 短文本（边界测试）
SHORT_TEXT = "这是一个短文本。只有两句话。测试边界情况。"

# 7. 空文本
EMPTY_TEXT = ""

# 8. 纯段落无分隔
LONG_PARA = "患者，男性，56岁，因\"右侧肢体无力伴言语不清3小时\"入院。患者于3小时前无明显诱因突发右侧肢体无力，表现为右手持物不稳，右下肢行走拖曳，伴言语含糊不清，无恶心呕吐，无意识障碍。" * 20


@dataclass
class BenchmarkCase:
    name: str
    text: str
    description: str
    expected_min_chunks: int = 1
    expected_max_chunks: int = 50
    min_acceptable_score: float = 0.0  # 每个策略的最低分数要求


BENCHMARK_CASES = [
    BenchmarkCase("narrative_zh", NARRATIVE_ZH, "中文叙事文本（5段，混合主题）", 1, 10),
    BenchmarkCase("medical_report", MEDICAL_REPORT, "医疗报告（含5节标准结构）", 1, 10),
    BenchmarkCase("markdown_doc", MARKDOWN_DOC, "Markdown 文档（多级标题）", 1, 10),
    BenchmarkCase("mixed_code", MIXED_CODE_DOC, "中英混排代码文档", 1, 10),
    BenchmarkCase("academic_text", ACADEMIC_TEXT, "学术文本（语义渐近变化）", 1, 15),
    BenchmarkCase("short_text", SHORT_TEXT, "极短文本边界测试", 1, 3),
    BenchmarkCase("empty_text", EMPTY_TEXT, "空文档安全测试", 0, 1),
    BenchmarkCase("long_para", LONG_PARA, "超大段落压力测试", 5, 100),
]


class BenchmarkRunner:
    def __init__(self):
        self.evaluator = ChunkQualityEvaluator()

    def run_all(
        self,
        splitter_factory,
        strategy_name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 75,
    ) -> List[ChunkQualityReport]:
        """对所有测试用例运行指定策略"""
        results = []
        for case in BENCHMARK_CASES:
            splitter = splitter_factory(chunk_size, chunk_overlap)
            start = time.perf_counter()
            chunks = splitter(case.text)
            elapsed = (time.perf_counter() - start) * 1000

            chunk_texts = []
            for c in chunks:
                chunk_texts.append(c.content if hasattr(c, "content") else c)

            report = self.evaluator.evaluate(
                chunk_texts,
                case.text,
                f"{strategy_name}/{case.name}",
                elapsed,
                chunk_overlap,
            )

            # 添加用例信息
            report.violations = report.violations or []
            if not (case.expected_min_chunks <= len(chunk_texts) <= case.expected_max_chunks):
                report.violations.append(
                    f"chunk数量超出预期范围 [{case.expected_min_chunks}, {case.expected_max_chunks}]"
                )

            results.append(report)

        return results

    def print_report(self, results: List[ChunkQualityReport]):
        """打印格式化的对比报告"""
        header = (
            f"{'用例':<20} {'chunks':>6} {'总分':>6} {'完整':>5} {'边界':>5} "
            f"{'召回':>5} {'稳定':>6} {'耗时ms':>7}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for r in results:
            passed = "P" if r.passed else "F"
            lines.append(
                f"{r.strategy_name.split('/')[-1]:<20} "
                f"{r.num_chunks:>6} "
                f"{r.overall_score:>6.3f} "
                f"{r.consistency:>5.2f} "
                f"{r.boundary:>5.2f} "
                f"{r.retrieval_recall:>5.2f} "
                f"{r.length_stab:>6.3f} "
                f"{r.split_time_ms:>7.1f}"
            )

        lines.append(sep)
        print("\n".join(lines))

    def compare_strategies(self, strategy_factories: dict, chunk_size: int = 512, chunk_overlap: int = 75):
        """对比多个策略并输出汇总"""
        print(f"\n{'='*70}")
        print(f"切分策略对比报告  chunk_size={chunk_size} overlap={chunk_overlap}")
        print(f"{'='*70}")

        all_results = {}
        for name, factory in strategy_factories.items():
            print(f"\n>>> 策略: {name}")
            results = self.run_all(factory, name, chunk_size, chunk_overlap)
            all_results[name] = results
            self.print_report(results)

        # 汇总平均分
        print(f"\n{'='*70}")
        print("策略综合排名（平均总分）:")
        print(f"{'='*70}")
        scores = []
        for name, results in all_results.items():
            avg = sum(r.overall_score for r in results) / len(results)
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            scores.append((name, avg, passed, total))
        scores.sort(key=lambda x: x[1], reverse=True)

        for name, avg, passed, total in scores:
            print(f"  {name:<20} avg_score={avg:.3f}  passed={passed}/{total}")

        return all_results
