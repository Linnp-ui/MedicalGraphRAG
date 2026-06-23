#!/usr/bin/env python3
"""分层评估运行脚本

用法:
    python scripts/run_layered_eval.py                  # 全量评估
    python scripts/run_layered_eval.py --layer L1        # 仅运行组件层
    python scripts/run_layered_eval.py --layer L2        # 仅运行检索层
    python scripts/run_layered_eval.py --layer L3        # 仅运行生成层
    python scripts/run_layered_eval.py --layer L4        # 仅运行系统层
    python scripts/run_layered_eval.py --demo            # 演示模式（模拟数据）
"""

import sys
import os
import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation import (
    LayerLevel,
    LayeredEvaluationFramework,
    LayeredEvaluationReport,
    ThresholdConfig,
    MetricsEngine,
    BenchmarkDataset,
    MedicalBenchmarkLoader,
    MedicalGoldenCase,
    MEDICAL_GOLDEN_CASES,
)
from src.evaluation.benchmark_dataset import BenchmarkItem


# ──────────────────────────────────────────────
# 演示模式：使用模拟数据，无需外部服务
# ──────────────────────────────────────────────

def run_demo_evaluation() -> LayeredEvaluationReport:
    """演示模式：用模拟数据生成完整分层评估报告"""
    print("=" * 70)
    print("  GRAPHRAG 分层评估 - 演示模式")
    print("=" * 70)

    dataset = MedicalBenchmarkLoader.load_medical_benchmark()
    total_start = time.time()
    report = LayeredEvaluationReport(
        timestamp=datetime.now().isoformat(),
        total_duration_s=0.0,
    )

    # ── L1 组件层 ──
    print("\n── L1 组件层评估 ──")
    from src.evaluation.layered_framework import LayerReport
    l1 = LayerReport(layer=LayerLevel.COMPONENT, layer_name="组件层 (NER / 意图分类 / 实体提取)")
    t0 = time.time()

    intent_correct = 0
    total_expected = 0
    total_found = 0
    ner_tp = 0
    ner_fp = 0
    for item in dataset.items:
        if random.random() > 0.12:
            intent_correct += 1
        match_rate = random.uniform(0.72, 0.95)
        found = int(len(item.expected_entities) * match_rate)
        found = min(found, len(item.expected_entities))
        total_expected += len(item.expected_entities)
        total_found += found
        ner_tp += found
        ner_fp += random.randint(0, 1)

    n = len(dataset.items)
    l1.add_metric("意图分类准确率", intent_correct / n, 0.80, weight=0.35)
    l1.add_metric("实体识别召回率", total_found / total_expected if total_expected else 0, 0.70, weight=0.35)
    l1.add_metric("NER精确率", ner_tp / (ner_tp + ner_fp) if (ner_tp + ner_fp) else 0, 0.75, weight=0.30)
    l1.score = l1.weighted_score()
    l1.duration_s = time.time() - t0
    l1.details = {"total_items": n, "intent_correct": intent_correct}
    report.layer_reports[LayerLevel.COMPONENT] = l1
    print(f"  评分: {l1.score:.2%} | 通过: {'是' if l1.passed else '否'} | 耗时: {l1.duration_s:.1f}s")

    # ── L2 检索层 ──
    print("\n── L2 检索层评估 ──")
    l2 = LayerReport(layer=LayerLevel.RETRIEVAL, layer_name="检索层 (向量/图谱检索 + 路由策略)")
    t0 = time.time()

    routing_correct = sum(1 for _ in range(n) if random.random() > 0.15)
    kw_coverages = [random.uniform(0.65, 0.92) for _ in range(n)]
    rec_scores = [random.uniform(0.60, 0.88) for _ in range(n)]

    l2.add_metric("路由策略准确率", routing_correct / n, 0.80, weight=0.30)
    l2.add_metric("关键词覆盖率", sum(kw_coverages) / n, 0.65, weight=0.30)
    l2.add_metric("检索召回率", sum(rec_scores) / n, 0.70, weight=0.40)
    l2.score = l2.weighted_score()
    l2.duration_s = time.time() - t0
    l2.details = {"total_items": n, "routing_correct": routing_correct}
    report.layer_reports[LayerLevel.RETRIEVAL] = l2
    print(f"  评分: {l2.score:.2%} | 通过: {'是' if l2.passed else '否'} | 耗时: {l2.duration_s:.1f}s")

    # ── L3 生成层 ──
    print("\n── L3 生成层评估 ──")
    l3 = LayerReport(layer=LayerLevel.GENERATION, layer_name="生成层 (回答质量 / 医疗安全 / 忠实度)")
    t0 = time.time()

    metrics_engine = MetricsEngine()
    f1_scores = []
    sem_sims = []
    for item in dataset.items:
        answer_type = random.randint(0, 1)
        if answer_type == 0:
            demo_answer = item.reference_answer
        else:
            words = item.reference_answer.split("，")
            demo_answer = "，".join(words[: len(words) * 4 // 5]) if len(words) > 1 else item.reference_answer
        m = metrics_engine.calculate_all(demo_answer, item.reference_answer, item.keywords)
        f1_scores.append(m["f1"])
        sem_sims.append(m["semantic_similarity"])

    n_safety = min(10, len(MEDICAL_GOLDEN_CASES))
    safety_scores = [random.uniform(0.82, 0.98) for _ in range(n_safety)]
    faith_scores = [random.uniform(0.70, 0.92) for _ in range(n_safety)]
    rel_scores = [random.uniform(0.72, 0.93) for _ in range(n_safety)]

    l3.add_metric("F1分数", sum(f1_scores) / n, 0.50, weight=0.20)
    l3.add_metric("语义相似度", sum(sem_sims) / n, 0.60, weight=0.15)
    l3.add_metric("医疗安全评分", sum(safety_scores) / n_safety, 0.85, weight=0.30)
    l3.add_metric("忠实度", sum(faith_scores) / n_safety, 0.70, weight=0.20)
    l3.add_metric("回答相关性", sum(rel_scores) / n_safety, 0.70, weight=0.15)
    l3.score = l3.weighted_score()
    l3.duration_s = time.time() - t0
    l3.details = {"generation_items": n, "safety_eval_items": n_safety}
    report.layer_reports[LayerLevel.GENERATION] = l3
    print(f"  评分: {l3.score:.2%} | 通过: {'是' if l3.passed else '否'} | 耗时: {l3.duration_s:.1f}s")

    # ── L4 系统层 ──
    print("\n── L4 系统层评估 ──")
    l4 = LayerReport(layer=LayerLevel.SYSTEM, layer_name="系统层 (端到端性能 / 延迟 / 阈值合规)")
    t0 = time.time()

    latencies = [random.uniform(0.5, 4.0) for _ in range(n)]
    errors = sum(1 for _ in range(n) if random.random() < 0.03)
    avg_lat = sum(latencies) / n
    sorted_lat = sorted(latencies)
    p95_lat = sorted_lat[min(int(n * 0.95), n - 1)]
    err_rate = errors / n
    throughput = n / sum(latencies)

    overall_score = (
        0.25 * sum(f1_scores) / n
        + 0.25 * sum(rec_scores) / n
        + 0.20 * sum(sem_sims) / n
        + 0.30 * sum(f1_scores) / n
    )

    lat_norm = max(0, 1 - avg_lat / 5.0)
    p95_norm = max(0, 1 - p95_lat / 10.0)
    l4.add_metric("平均延迟(s)", lat_norm, 0.5, weight=0.15,
                  detail=f"实际={avg_lat:.2f}s, 阈值<=5.00s",
                  passed=avg_lat <= 5.0)
    l4.add_metric("P95延迟(s)", p95_norm, 0.5, weight=0.10,
                  detail=f"实际={p95_lat:.2f}s, 阈值<=10.00s",
                  passed=p95_lat <= 10.0)
    l4.add_metric("错误率", 1 - err_rate, 0.5, weight=0.15,
                  detail=f"实际={err_rate:.4f}, 阈值<=0.0500",
                  passed=err_rate <= 0.05)
    l4.add_metric("综合评分", overall_score, 0.75, weight=0.30)
    l4.add_metric("吞吐量(次/秒)", throughput, 0.1, weight=0.10)
    l4.add_metric("阈值合规", 1.0, 1.0, weight=0.20)
    l4.passed = all(m.passed for m in l4.metrics)
    l4.score = l4.weighted_score()
    l4.duration_s = time.time() - t0
    l4.details = {
        "total_items": n, "errors": errors,
        "avg_latency_s": avg_lat, "p95_latency_s": p95_lat,
        "throughput_rps": throughput, "overall_score": overall_score,
    }
    report.layer_reports[LayerLevel.SYSTEM] = l4
    print(f"  评分: {l4.score:.2%} | 通过: {'是' if l4.passed else '否'} | 耗时: {l4.duration_s:.1f}s")

    report.total_duration_s = time.time() - total_start
    report.compute_overall()
    return report


# ──────────────────────────────────────────────
# 真实模式：调用实际服务
# ──────────────────────────────────────────────

def run_real_evaluation(layers=None) -> LayeredEvaluationReport:
    """真实模式：调用 NER / 意图分类 / QA Chain 等实际服务"""
    print("=" * 70)
    print("  GRAPHRAG 分层评估 - 真实模式")
    print("=" * 70)

    framework = LayeredEvaluationFramework()
    report = framework.run(layers=layers)
    return report


# ──────────────────────────────────────────────
# 报告输出
# ──────────────────────────────────────────────

def print_report(report: LayeredEvaluationReport):
    print("\n" + "=" * 70)
    print("  分层评估报告")
    print("=" * 70)
    print(f"  评估时间:   {report.timestamp}")
    print(f"  总耗时:     {report.total_duration_s:.1f}s")
    print(f"  总体结果:   {'通过' if report.overall_passed else '未通过'}")
    print(f"  总体评分:   {report.overall_score:.2%}")

    for level in LayerLevel:
        lr = report.layer_reports.get(level)
        if lr is None:
            continue
        status = "PASS" if lr.passed else "FAIL"
        print(f"\n  ── {level.value} {lr.layer_name} [{status}] "
              f"评分: {lr.score:.2%}  耗时: {lr.duration_s:.1f}s ──")
        for m in lr.metrics:
            if "延迟" in m.name or m.name == "错误率":
                sym = "Y" if m.value <= m.threshold else "N"
            else:
                sym = "Y" if m.passed else "N"
            val = f"{m.value:.4f}" if m.value < 1 else f"{m.value:.2f}"
            thr = f"{m.threshold:.4f}" if m.threshold < 1 else f"{m.threshold:.2f}"
            detail_str = f"  ({m.detail})" if m.detail else ""
            print(f"    [{sym}] {m.name}: {val}  (阈值: {thr}, 权重: {m.weight}){detail_str}")
        if lr.errors:
            for err in lr.errors:
                print(f"    !! 错误: {err}")

    print("\n" + "=" * 70)


def save_report(report: LayeredEvaluationReport, output_dir: str = "test_results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"layered_eval_{timestamp}.json")
    md_path = os.path.join(output_dir, f"layered_eval_{timestamp}.md")

    # JSON 报告
    data = {
        "timestamp": report.timestamp,
        "total_duration_s": report.total_duration_s,
        "overall_passed": report.overall_passed,
        "overall_score": report.overall_score,
        "layers": {},
    }
    for level, lr in report.layer_reports.items():
        data["layers"][level.value] = {
            "layer_name": lr.layer_name,
            "passed": lr.passed,
            "score": lr.score,
            "duration_s": lr.duration_s,
            "metrics": [
                {"name": m.name, "value": m.value, "threshold": m.threshold,
                 "passed": m.passed, "weight": m.weight}
                for m in lr.metrics
            ],
            "errors": lr.errors,
            "details": lr.details,
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    md = _generate_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n报告已保存:")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    return json_path


def _generate_markdown(report: LayeredEvaluationReport) -> str:
    md = "# GRAPHRAG 分层评估报告\n\n"
    md += f"**评估时间**: {report.timestamp}\n\n"
    md += f"**总体结果**: {'通过' if report.overall_passed else '未通过'} | "
    md += f"**总体评分**: {report.overall_score:.2%} | "
    md += f"**总耗时**: {report.total_duration_s:.1f}s\n\n"

    md += "---\n\n"

    for level in LayerLevel:
        lr = report.layer_reports.get(level)
        if lr is None:
            continue
        status = "PASS" if lr.passed else "FAIL"
        md += f"## {level.value} {lr.layer_name} [{status}]\n\n"
        md += f"- **评分**: {lr.score:.2%}\n"
        md += f"- **耗时**: {lr.duration_s:.1f}s\n\n"

        md += "| 指标 | 值 | 阈值 | 状态 | 权重 |\n"
        md += "|------|-----|------|------|------|\n"
        for m in lr.metrics:
            if "延迟" in m.name or m.name == "错误率":
                sym = "Y" if m.value <= m.threshold else "N"
            else:
                sym = "Y" if m.passed else "N"
            val = f"{m.value:.4f}" if m.value < 1 else f"{m.value:.2f}"
            thr = f"{m.threshold:.4f}" if m.threshold < 1 else f"{m.threshold:.2f}"
            md += f"| {m.name} | {val} | {thr} | {sym} | {m.weight} |\n"
        md += "\n"

        if lr.errors:
            md += "**错误**:\n"
            for err in lr.errors:
                md += f"- {err}\n"
            md += "\n"

    md += "---\n\n"
    if report.overall_passed:
        md += "**结论**: 所有层级评估通过，系统整体表现达标。\n"
    else:
        md += "**结论**: 部分层级评估未通过，请根据上述报告定位问题。\n"

    return md


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRAPHRAG 分层评估工具")
    parser.add_argument("--demo", action="store_true", help="演示模式（模拟数据，无需外部服务）")
    parser.add_argument("--layer", type=str, default=None,
                        choices=["L1", "L2", "L3", "L4"],
                        help="仅运行指定层级评估")
    parser.add_argument("--output", type=str, default="test_results", help="报告输出目录")
    args = parser.parse_args()

    if args.demo:
        report = run_demo_evaluation()
    else:
        layers = None
        if args.layer:
            layer_map = {"L1": LayerLevel.COMPONENT, "L2": LayerLevel.RETRIEVAL,
                         "L3": LayerLevel.GENERATION, "L4": LayerLevel.SYSTEM}
            layers = [layer_map[args.layer]]
        report = run_real_evaluation(layers=layers)

    print_report(report)
    save_report(report, output_dir=args.output)

    if report.overall_passed:
        print("\n所有层级评估通过！")
    else:
        print("\n部分层级评估未通过，请查看报告详情")

    return report


if __name__ == "__main__":
    main()
