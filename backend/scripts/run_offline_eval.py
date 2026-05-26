#!/usr/bin/env python3
"""离线评估执行脚本"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation import OfflineEvaluator, ThresholdConfig


def main():
    print("=" * 70)
    print("GRAPHRAG 离线评估工具")
    print("=" * 70)
    
    config = ThresholdConfig(
        overall_score=0.75,
        intent_accuracy=0.80,
        entity_recall=0.70,
        answer_relevance=0.70,
        harmful_rate=0.05,
        error_rate=0.02,
        p95_latency_ms=3000.0
    )
    
    evaluator = OfflineEvaluator(threshold_config=config)
    evaluator.load_dataset()
    
    print(f"\n数据集: {evaluator.dataset.name}")
    print(f"测试用例数: {len(evaluator.dataset)}")
    print(f"阈值配置已加载")
    print("-" * 70)
    
    report = evaluator.run_evaluation()
    
    evaluator.print_report(report)
    
    report_path = evaluator.save_report(report, output_dir="test_results")
    
    if report.threshold_result and report.threshold_result.passed:
        print("\n✅ 所有阈值检查通过，评估完成！")
    else:
        print("\n⚠️ 部分阈值检查未通过，请查看报告详情")
    
    return report


if __name__ == "__main__":
    main()