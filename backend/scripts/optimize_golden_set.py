#!/usr/bin/env python3
"""用 LLM 优化黄金集参考回答：保留原始事实，改写为自然流畅的医疗问答风格"""

import sys, os, time, json, urllib.request, ssl
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation.generated_loader import load_from_json, GeneratedCase

GOLDEN_SET_PATH = str(Path(__file__).resolve().parent.parent.parent / "golden_set" / "generated_golden.json")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-bf1f3509042943faa9d8d2debd0ae36e")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CTX = ssl.create_default_context()

OPTIMIZE_PROMPT = """你是一个医学写作专家。你的任务是基于提供的参考信息，生成一个专业、准确、流畅的医疗回答。

要求：
1. 保留参考信息中的所有医学事实，不得添加虚构信息
2. 用自然流畅的中文完整句表达，不使用列表或结构化格式
3. 回答要完整、准确、条理清晰
4. 保持医学专业术语的准确性
5. 长度与参考信息相当，不可过度精简

问题：{question}

参考信息：{reference}

请生成优化后的回答："""


def call_llm(question: str, reference: str) -> str:
    prompt = OPTIMIZE_PROMPT.format(question=question, reference=reference)
    body = json.dumps({
        "model": "qwen3.6-plus-2026-04-02",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, context=CTX, timeout=60)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def optimize_golden_set(sample_per_cat: int = None):
    print("=" * 70)
    print("黄金集参考回答优化 (LLM 改写)")
    print("=" * 70)

    # 加载
    cases = load_from_json(GOLDEN_SET_PATH)
    print(f"总用例数: {len(cases)}")

    if sample_per_cat:
        by_cat = defaultdict(list)
        for c in cases:
            by_cat[c.category].append(c)
        to_process = []
        for cat in sorted(by_cat):
            selected = by_cat[cat][:sample_per_cat]
            to_process.extend(selected)
            print(f"  {cat}: {len(by_cat[cat])} -> {len(selected)}")
    else:
        to_process = cases
        cat_dist = defaultdict(int)
        for c in cases:
            cat_dist[c.category] += 1
        for cat, cnt in sorted(cat_dist.items()):
            print(f"  {cat}: {cnt}")

    print(f"\n实际处理: {len(to_process)} 条")
    print("-" * 70)

    results = []
    total_start = time.time()
    success = 0
    fail = 0

    for i, case in enumerate(to_process, 1):
        item_start = time.time()
        try:
            new_answer = call_llm(case.question, case.reference_answer)
            success += 1
        except Exception as e:
            new_answer = case.reference_answer
            fail += 1
            print(f"  [{i}/{len(to_process)}] ERR 跳过: {str(e)[:60]}")
            continue

        optimized = GeneratedCase(
            question=case.question,
            reference_answer=new_answer,
            expected_intent=case.expected_intent,
            expected_entities=case.expected_entities,
            keywords=case.keywords,
            category=case.category,
            difficulty=case.difficulty,
            safety_category=case.safety_category,
            forbidden_content=case.forbidden_content,
        )
        results.append(optimized)

        elapsed = time.time() - item_start
        old_len = len(case.reference_answer)
        new_len = len(new_answer)
        delta = new_len - old_len
        print(f"  [{i}/{len(to_process)}] OK | {case.question[:25]:<25s} | "
              f"{elapsed:.0f}s | {old_len}->{new_len}字 ({delta:+d})")

        if i < len(to_process):
            time.sleep(2)

    total_elapsed = time.time() - total_start

    # 保存优化后的黄金集
    output_items = []
    for c in results:
        output_items.append({
            "question": c.question,
            "reference_answer": c.reference_answer,
            "expected_intent": c.expected_intent,
            "expected_entities": c.expected_entities,
            "keywords": c.keywords,
            "category": c.category,
            "difficulty": c.difficulty,
            "safety_category": c.safety_category,
            "forbidden_content": c.forbidden_content,
        })

    # 统计
    cat_dist2 = defaultdict(int)
    intent_dist2 = defaultdict(int)
    diff_dist2 = defaultdict(int)
    source_dist2 = defaultdict(int)
    for c in results:
        cat_dist2[c.category] += 1
        intent_dist2[c.expected_intent] += 1
        diff_dist2[c.difficulty] += 1

    output = {
        "total": len(output_items),
        "summary": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_by": "optimize_golden_set.py (LLM qwen3.6-plus-2026-04-02)",
            "description": "基于黄金集原参考回答，用LLM优化为自然医疗问答风格",
            "total_items": len(output_items),
            "success": success,
            "fail": fail,
            "total_time_s": round(total_elapsed, 1),
            "category_distribution": dict(cat_dist2),
            "intent_distribution": dict(intent_dist2),
            "difficulty_distribution": dict(diff_dist2),
        },
        "items": output_items,
    }

    output_dir = "golden_set"
    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, "generated_golden_optimized.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"优化完成! 成功={success}, 失败={fail}, 总耗时={total_elapsed:.0f}s")
    print(f"保存至: {fp}")
    print(f"{'='*70}")

    # 展示效果对比
    print("\n效果对比 (前3条):")
    for c in results[:3]:
        idx = to_process.index(c) if c in to_process else 0
        orig = to_process[idx].reference_answer if idx < len(to_process) else ""
        print(f"\n  Q: {c.question}")
        print(f"  原: {orig[:100]}...")
        print(f"  新: {c.reference_answer[:100]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="每类抽样数(默认全量)")
    args = parser.parse_args()
    optimize_golden_set(sample_per_cat=args.sample)
