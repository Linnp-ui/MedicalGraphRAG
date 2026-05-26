# 离线评估系统

GRAPHRAG 医疗知识图谱系统的离线评估框架，用于在发布前对模型进行全面评估。

## 架构概览

```
src/evaluation/
├── __init__.py                  # 模块初始化
├── evaluator.py                 # 核心评估器
├── metrics_engine.py            # NLP指标计算引擎
├── benchmark_dataset.py         # 基准数据集管理
├── llm_judge.py                 # LLM自动评分器
└── threshold_checker.py         # 阈值检查器
```

## 功能特性

### 1. 基准数据集
- **医疗领域测试用例**: 30个覆盖疾病、药物、症状等类别的测试问题
- **多维度验证**: 包含意图类型、实体列表、关键词和参考回答
- **类别细分**: disease, drug, symptom, prevention, treatment, examination, health_advice

### 2. 自动指标计算
- **NLP指标**: Exact Match, F1-score, BLEU, ROUGE-1, ROUGE-2, ROUGE-L
- **关键词匹配**: 检查答案是否包含预期关键词
- **意图准确率**: 验证问题意图分类是否正确
- **实体召回率**: 检查命名实体识别的召回

### 3. LLM-as-Judge
- 基于GPT-4的自动评分
- 多维度打分: 正确性、完整性、相关性、安全性
- 双评估一致性检查

### 4. 阈值校验
- 综合评分 ≥ 75%
- 意图准确率 ≥ 80%
- 实体召回率 ≥ 70%
- 回答相关性 ≥ 70%

## 快速开始

### 1. 运行演示评估

```bash
cd backend
python scripts/run_demo_evaluation.py
```

这个脚本会使用模拟数据运行完整的评估流程，并生成JSON和Markdown报告。

### 2. 运行完整评估（需要配置好LLM）

```bash
cd backend
python scripts/run_offline_eval.py
```

### 3. 运行框架测试

```bash
cd backend
python scripts/test_evaluation_framework.py
```

## 阈值配置

在 `ThresholdConfig` 中可以自定义评估阈值：

```python
config = ThresholdConfig(
    overall_score=0.75,      # 综合评分阈值
    intent_accuracy=0.80,    # 意图准确率阈值
    entity_recall=0.70,      # 实体召回率阈值
    answer_relevance=0.70,   # 回答相关性阈值
    harmful_rate=0.05,       # 有害内容率阈值
    error_rate=0.02,         # 错误率阈值
    p95_latency_ms=3000.0    # P95延迟阈值
)
```

## 报告格式

### 1. JSON报告
包含完整的测试用例、评分细节和执行信息

### 2. Markdown报告
可视化的评估报告，包括：
- 评估概览
- 综合指标
- 类别评分
- 性能指标
- 阈值检查结果
- 评估结论

## 指标说明

### NLP指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| Exact Match | 完全匹配率 | 越高越好 |
| F1-score | 精确率和召回率调和平均 | > 80% |
| BLEU | 翻译质量评分 | > 60% |
| ROUGE-L | 最长公共子序列 | > 70% |
| Keyword Matching | 关键词覆盖率 | > 70% |

### 业务指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| Intent Accuracy | 意图分类准确率 | ≥ 80% |
| Entity Recall | 实体识别召回率 | ≥ 70% |
| Answer Relevance | 回答相关性 | ≥ 70% |

### 性能指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| Avg Response Time | 平均响应时间 | < 3s |
| P95 Latency | 95%分位延迟 | < 3s |
| Throughput | 吞吐量 | > 1 req/s |

## 扩展开发

### 添加新的测试用例

在 `MedicalBenchmarkLoader` 中添加更多测试用例：

```python
@dataclass
class BenchmarkItem:
    question: str
    reference_answer: str
    expected_intent: str
    expected_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
```

### 自定义指标计算

继承或扩展 `MetricsEngine`：

```python
class CustomMetricsEngine(MetricsEngine):
    def custom_score(self, prediction, reference):
        # 实现自定义评分逻辑
        pass
```

## 集成指南

### 在CI/CD中使用

```yaml
# .github/workflows/evaluation.yml
name: Offline Evaluation
on: [pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run offline evaluation
        run: python backend/scripts/run_offline_eval.py
      - name: Upload evaluation report
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-report
          path: backend/test_results/
```

### 与现有QA系统集成

```python
from src.evaluation import OfflineEvaluator, ThresholdConfig

# 创建评估器
evaluator = OfflineEvaluator(
    threshold_config=ThresholdConfig(
        overall_score=0.75,
        intent_accuracy=0.80,
        entity_recall=0.70,
        answer_relevance=0.70
    )
)

# 加载数据集（或使用自定义数据集）
evaluator.load_dataset()

# 运行评估
report = evaluator.run_evaluation()

# 打印和保存报告
evaluator.print_report(report)
evaluator.save_report(report)
```

## 常见问题

### 1. 如何调整评分权重？
可以在 `OfflineEvaluator._calculate_aggregate_metrics` 中调整各指标的权重。

### 2. 如何添加自定义的LLM评分提示词？
可以在 `LLMJudge._build_prompt` 中自定义提示词模板。

### 3. 评估报告保存在哪里？
默认保存在 `backend/test_results/` 目录下，可以通过 `save_report` 的 `output_dir` 参数自定义。

## 更新日志

- v1.0 - 初始版本，包含完整的离线评估框架、30个医疗测试用例、多维度指标计算
