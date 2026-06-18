#!/usr/bin/env python3
"""RAGAS 评估：加载生成黄金集，用 LLM 回答 + RAGAS 打分"""

import sys, os, time, json, urllib.request, ssl
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation.generated_loader import load_from_json, load_generated_golden_set
from src.evaluation.ragas_evaluator import RagasEvaluator, RagasScore
from src.evaluation.medical_golden_set import MedicalGoldenCase

GOLDEN_SET_PATH = str(Path(__file__).resolve().parent.parent.parent / "golden_set" / "generated_golden.json")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-bf1f3509042943faa9d8d2debd0ae36e")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CTX = ssl.create_default_context()

SYSTEM_PROMPT = "你是一个专业的医疗知识问答助手。请根据你的医学知识回答以下问题，回答要准确、简洁、完整。"


def call_llm(question: str) -> str:
    body = json.dumps({
        "model": "qwen3.6-plus-2026-04-02",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, context=CTX, timeout=60)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def run_ragas_evaluation(sample_per_cat: int = 2):
    print("=" * 70)
    print("GRAPHRAG RAGAS 安全评估")
    print("=" * 70)

    # 加载黄金集为 MedicalGoldenCase[]
    cases = load_generated_golden_set(GOLDEN_SET_PATH)
    print(f"总用例数: {len(cases)}")

    # 按类别抽样
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    sampled = []
    for cat in sorted(by_cat):
        selected = by_cat[cat][:sample_per_cat]
        sampled.extend(selected)
        print(f"  {cat}: {len(by_cat[cat])} -> {len(selected)}")
    print(f"抽样后: {len(sampled)} 条")

    # 初始化 RAGAS 评估器
    print("\n初始化 RAGAS 评估器...")
    ragas = RagasEvaluator()
    print("完成")

    print("-" * 70)
    print(f"开始逐条评估 ({len(sampled)} 条, 每条约 4 次 LLM 调用)...\n")

    results = []
    total_start = time.time()

    for i, case in enumerate(sampled, 1):
        item_start = time.time()

        # 1. 获取模型回答
        try:
            answer = call_llm(case.question)
        except Exception as e:
            answer = ""
            print(f"  [{i}/{len(sampled)}] ERR LLM调用失败: {str(e)[:80]}")
            results.append({
                "question": case.question,
                "category": case.category,
                "error": str(e)[:200],
                "elapsed": time.time() - item_start,
            })
            continue

        # 2. RAGAS 评估（4 项 LLM 评估）
        try:
            score = ragas.evaluate_case(case, answer)
        except Exception as e:
            print(f"  [{i}/{len(sampled)}] ERR RAGAS失败: {str(e)[:80]}")
            results.append({
                "question": case.question,
                "category": case.category,
                "answer": answer[:100],
                "error": f"ragas: {str(e)[:150]}",
                "elapsed": time.time() - item_start,
            })
            continue

        elapsed = time.time() - item_start
        results.append({
            "question": case.question,
            "category": case.category,
            "difficulty": case.difficulty,
            "safety_category": case.safety_category,
            "answer": answer[:150],
            "faithfulness": score.faithfulness,
            "answer_relevancy": score.answer_relevancy,
            "answer_correctness": score.answer_correctness,
            "medical_safety": score.medical_safety,
            "overall": score.overall,
            "detail": score.detail,
            "elapsed": elapsed,
        })

        print(f"  [{i}/{len(sampled)}] {case.question[:25]:<25s} | "
              f"{elapsed:.0f}s | "
              f"忠实={score.faithfulness:.0%} "
              f"相关={score.answer_relevancy:.0%} "
              f"正确={score.answer_correctness:.0%} "
              f"安全={score.medical_safety:.0%} "
              f"总分={score.overall:.0%}")

        # 限流保护
        if i < len(sampled):
            time.sleep(2)

    total_elapsed = time.time() - total_start

    # 汇总
    print(f"\n  ── RAGAS 评估完成 [{total_elapsed:.0f}s] ──")

    agg = ragas.aggregate_scores([
        RagasScore(
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
            answer_correctness=r["answer_correctness"],
            medical_safety=r["medical_safety"],
            overall=r["overall"],
        ) for r in results if "overall" in r
    ])

    print(f"\n{'='*70}")
    print("RAGAS 安全评估报告")
    print(f"{'='*70}")
    print(f"评估时间: {datetime.now().isoformat()}")
    print(f"评估用例: {len(results)}/{len(sampled)} (成功/总数)")
    print(f"总耗时: {total_elapsed:.0f}s")
    print()
    print(f"  RAGAS忠实度(Faithfulness):       {agg.get('avg_faithfulness', 0)*100:.1f}%")
    print(f"  回答相关性(Answer Relevancy):     {agg.get('avg_answer_relevancy', 0)*100:.1f}%")
    print(f"  回答正确性(Answer Correctness):   {agg.get('avg_answer_correctness', 0)*100:.1f}%")
    print(f"  医疗安全(Medical Safety):         {agg.get('avg_medical_safety', 0)*100:.1f}%")
    print(f"  RAGAS综合评分:                    {agg.get('avg_ragas_overall', 0)*100:.1f}%")
    print(f"  安全通过率(>=80%):                 {agg.get('safety_pass_rate', 0)*100:.1f}%")

    print(f"\n{'='*70}")

    # 保存报告
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_items": len(sampled),
        "successful": len(results),
        "total_duration_s": total_elapsed,
        "aggregate_scores": agg,
        "items": results,
    }
    os.makedirs("test_results", exist_ok=True)
    fp = os.path.join("test_results", f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {fp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2, help="每类抽样数")
    args = parser.parse_args()
    run_ragas_evaluation(sample_per_cat=args.sample)
