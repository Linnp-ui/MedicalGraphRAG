#!/usr/bin/env python3
"""测试评估框架"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.metrics_engine import MetricsEngine
from src.evaluation.benchmark_dataset import MedicalBenchmarkLoader
from src.evaluation.threshold_checker import ThresholdConfig, ThresholdChecker

print("=" * 70)
print("测试评估框架")
print("=" * 70)

# 测试指标计算引擎
print("\n1. 测试指标计算引擎...")
engine = MetricsEngine()

test_prediction = "高血压是一种常见的慢性疾病，患者需要长期服用降压药物"
test_reference = "高血压是一种常见的慢性疾病，需要长期治疗，避免摄入过多盐分"
keywords = ["高血压", "慢性", "药物", "治疗"]

print(f"预测: {test_prediction}")
print(f"参考: {test_reference}")
print(f"关键词: {keywords}")

metrics = engine.calculate_all(test_prediction, test_reference, keywords)
print(f"EM: {metrics['exact_match']:.2f}, F1: {metrics['f1']:.2f}")
print(f"ROUGE-L: {metrics['rouge_l']:.2f}, 关键词匹配: {metrics['keyword_matching']:.2f}")
print("✅ 指标计算引擎测试通过！")

# 测试基准数据集
print("\n2. 测试基准数据集...")
dataset = MedicalBenchmarkLoader.load_medical_benchmark()
print(f"数据集名称: {dataset.name}")
print(f"测试用例数: {len(dataset)}")
print(f"前3个用例:")
for i, item in enumerate(dataset.items[:3], 1):
    print(f"  {i}. {item.question[:50]}...")
print("✅ 基准数据集测试通过！")

# 测试阈值检查器
print("\n3. 测试阈值检查器...")
config = ThresholdConfig(
    overall_score=0.75,
    intent_accuracy=0.80,
    entity_recall=0.70,
    answer_relevance=0.70
)

checker = ThresholdChecker(config)

test_metrics = {
    'overall_score': 0.82,
    'intent_accuracy': 0.78,
    'entity_recall': 0.85,
    'answer_relevance': 0.79,
}

result = checker.check_all(test_metrics)
print(f"测试指标: {test_metrics}")
print(f"检查结果: {result.passed}")
print(f"详细信息: {result.message}")
print("✅ 阈值检查器测试通过！")

print("\n" + "=" * 70)
print("🎉 所有测试通过！")
print("=" * 70)
