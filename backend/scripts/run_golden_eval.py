#!/usr/bin/env python3
"""RAG管线评估：通过完整 DRIFT检索 + QAChain生成 评估检索召回质量"""

import sys, os, time, json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation.generated_loader import load_from_json, GeneratedCase
from src.evaluation.metrics_engine import MetricsEngine
from src.evaluation.threshold_checker import ThresholdConfig, ThresholdChecker

GOLDEN_SET_PATH = str(Path(__file__).resolve().parent.parent.parent / "golden_set" / "generated_golden_v2.json")


def check_intent(answer: str, expected_intent: str, keywords: list) -> bool:
    """简易意图验证：检查回答是否包含答案关键词"""
    if not keywords:
        return True
    matched = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return matched >= max(1, len(keywords) * 0.5)


def check_entities(answer: str, expected: list) -> int:
    return sum(1 for e in expected if e.lower() in answer.lower())


def run_rag_evaluation():
    print("=" * 70)
    print("GRAPHRAG 黄金评估集离线评估 (RAG管线版)")
    print("=" * 70)

    # 初始化
    cases = load_from_json(GOLDEN_SET_PATH)
    metrics_engine = MetricsEngine()
    threshold_config = ThresholdConfig(
        overall_score=0.75, intent_accuracy=0.80, entity_recall=0.70,
        keyword_matching=0.70, answer_relevance=0.30, harmful_rate=0.05, error_rate=0.02, p95_latency_ms=3000.0,
    )
    threshold_checker = ThresholdChecker(threshold_config)

    # 初始化 RAG 管线
    print("\n初始化 RAG 管线...")
    from src.workflow.graph import run_workflow
    from src.core.neo4j_client import get_neo4j_client

    # 验证 Neo4j 连接
    try:
        neo4j_client = get_neo4j_client()
        count_result = neo4j_client.execute_query("MATCH (n) RETURN count(n) as cnt")
        node_count = count_result[0]["cnt"] if count_result else 0
        print(f"  [Neo4j 连接成功] 节点数={node_count}")
    except Exception as e:
        print(f"  [Neo4j 连接失败] {e}")
        return

    # 验证 LLM 配置
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    llm_model = os.getenv("DASHSCOPE_MODEL", "qwen3.6-plus-2026-04-02")
    llm_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    print(f"  [LLM 配置] model={llm_model}, url={llm_url}")

    print(f"\n数据集: generated_golden_v2.json")
    print(f"总用例数: {len(cases)} (将按类别抽样)")

    # 每类抽取 5 条
    from collections import defaultdict
    sampled = []
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    for cat, items in sorted(by_cat.items()):
        selected = items[:5]
        sampled.extend(selected)
        print(f"  {cat}: {len(items)} -> {len(selected)}")
    print(f"抽样后总计: {len(sampled)} 条")
    print("-" * 70)
    print("开始评估 (RAG管线: DRIFT检索 + QAChain生成)...\n")

    results = []
    total_start = time.time()

    total = len(sampled)
    for i, case in enumerate(sampled, 1):
        item_start = time.time()
        error = False
        error_msg = ""
        model_answer = ""
        routing = ""
        doc_count = 0
        entity_count = 0

        try:
            result = run_workflow(case.question)
            model_answer = result.get("answer", "")
            routing = result.get("routing", "unknown")
            doc_count = len(result.get("documents", []))
            entity_count = len(result.get("entities", []))

            if not model_answer:
                error = True
                error_msg = "empty_answer"

        except Exception as e:
            error_msg_full = str(e)
            # 限流则重试一次
            if "429" in error_msg_full or "RateLimit" in error_msg_full:
                time.sleep(5)
                try:
                    result = run_workflow(case.question)
                    model_answer = result.get("answer", "")
                    routing = result.get("routing", "unknown")
                    doc_count = len(result.get("documents", []))
                    entity_count = len(result.get("entities", []))
                except Exception as e2:
                    model_answer = ""
                    error = True
                    error_msg = f"rate_limit retry fail: {str(e2)[:100]}"
            else:
                error = True
                error_msg = error_msg_full[:200]

        metrics = metrics_engine.calculate_all(
            prediction=model_answer,
            reference=case.reference_answer,
            keywords=case.keywords,
        ) if model_answer else {}

        intent_correct = check_intent(model_answer, case.expected_intent, case.keywords) if model_answer else False
        entities_found = check_entities(model_answer, case.expected_entities) if model_answer else 0

        elapsed = time.time() - item_start

        # 限流保护
        if i < total:
            time.sleep(1)

        results.append({
            "question": case.question,
            "reference_answer": case.reference_answer,
            "model_answer": model_answer,
            "metrics": metrics,
            "intent_correct": intent_correct,
            "entities_found": entities_found,
            "expected_entities": len(case.expected_entities),
            "response_time": elapsed,
            "error_occurred": error,
            "error_message": error_msg,
            "category": case.category,
            "difficulty": case.difficulty,
            "routing": routing,
            "doc_count": doc_count,
            "retrieved_entity_count": entity_count,
        })

        progress = "ERR" if error else "OK"
        err_info = f" [{error_msg[:50]}]" if error else ""
        route_info = f" [{routing}]" if routing else ""
        print(f"  [{i}/{total}] {progress} | {case.question[:25]:<25s} | "
              f"{elapsed:.1f}s | F1={metrics.get('f1', 0):.2%} | "
              f"意图={'Y' if intent_correct else 'N'} | "
              f"实体={entities_found}/{len(case.expected_entities)} | "
              f"检索={doc_count}doc+{entity_count}ent{route_info}{err_info}")

    total_elapsed = time.time() - total_start
    print(f"\n  ── 评估完成 [{total_elapsed:.1f}s] ──")

    # 生成报告
    report = _build_report(results, threshold_config, threshold_checker)
    _print_report(report)
    _save_report(report)

    if report["threshold_passed"]:
        print("\n所有阈值检查通过！")
    else:
        print("\n部分阈值检查未通过")

    return report


def _build_report(results, threshold_config, threshold_checker):
    successful = [r for r in results if not r["error_occurred"] and r["model_answer"]]

    overall = {}
    primary = ["exact_match", "f1", "keyword_matching", "retrieval_recall", "semantic_similarity"]
    secondary = ["bleu", "rouge_1", "rouge_2", "rouge_l"]
    for mn in primary + secondary:
        vals = [r["metrics"].get(mn, 0) for r in successful]
        overall[mn] = sum(vals) / len(vals) if vals else 0

    intent_acc = sum(1 for r in successful if r["intent_correct"]) / len(successful) if successful else 0
    total_exp = sum(r["expected_entities"] for r in successful)
    total_fnd = sum(r["entities_found"] for r in successful)
    entity_rec = total_fnd / total_exp if total_exp > 0 else 0
    keyword_match = overall.get("keyword_matching", 0)
    retrieval_rec = overall.get("retrieval_recall", 0)
    ans_rel = overall.get("semantic_similarity", 0)

    overall.update({
        "intent_accuracy": intent_acc,
        "entity_recall": entity_rec,
        "keyword_matching": keyword_match,
        "retrieval_recall": retrieval_rec,
        "answer_relevance": ans_rel,
        "overall_score": 0.25 * intent_acc + 0.25 * entity_rec + 0.20 * keyword_match + 0.20 * retrieval_rec + 0.10 * ans_rel,
    })

    cat_metrics = {}
    for cat in set(r["category"] for r in results):
        cr = [r for r in results if r["category"] == cat and not r["error_occurred"]]
        if not cr:
            continue
        cm = {}
        for mn in primary + secondary:
            vals = [r["metrics"].get(mn, 0) for r in cr]
            cm[mn] = sum(vals) / len(vals)
        c_intent = sum(1 for r in cr if r["intent_correct"]) / len(cr)
        c_exp = sum(r["expected_entities"] for r in cr)
        c_fnd = sum(r["entities_found"] for r in cr)
        c_ent = c_fnd / c_exp if c_exp > 0 else 0
        c_kw = cm.get("keyword_matching", 0)
        c_rr = cm.get("retrieval_recall", 0)
        cm.update({
            "intent_accuracy": c_intent, "entity_recall": c_ent,
            "keyword_matching": c_kw, "retrieval_recall": c_rr,
            "overall_score": 0.25 * c_intent + 0.25 * c_ent + 0.20 * c_kw + 0.20 * c_rr + 0.10 * cm.get("semantic_similarity", 0),
        })
        cat_metrics[cat] = cm

    # 检索策略分布
    routing_dist = {}
    for r in results:
        rt = r.get("routing", "unknown")
        routing_dist[rt] = routing_dist.get(rt, 0) + 1

    times = [r["response_time"] for r in successful]
    times_all = [r["response_time"] for r in results]
    sorted_t = sorted(times_all)
    p95 = sorted_t[int(len(sorted_t) * 0.95)] if sorted_t else 0

    exec_summary = {
        "avg_response_time": sum(times) / len(times) if times else 0,
        "max_response_time": max(times) if times else 0,
        "min_response_time": min(times) if times else 0,
        "total_execution_time": sum(times_all),
        "throughput": len(results) / sum(times_all) if sum(times_all) > 0 else 0,
        "p95_latency_ms": p95 * 1000,
        "routing_distribution": routing_dist,
        "avg_doc_count": sum(r.get("doc_count", 0) for r in successful) / len(successful) if successful else 0,
        "avg_entity_count": sum(r.get("retrieved_entity_count", 0) for r in successful) / len(successful) if successful else 0,
    }

    thr = threshold_checker.check_all(overall)

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": "generated_golden_set_rag",
        "eval_mode": "rag_pipeline",
        "total_items": len(results),
        "passed": len(successful),
        "failed": len(results) - len(successful),
        "overall_metrics": overall,
        "category_metrics": cat_metrics,
        "threshold_passed": thr.passed if thr else False,
        "threshold_details": thr.details if thr else {},
        "execution_summary": exec_summary,
        "results": results,
    }


def _print_report(report):
    import io
    out = io.StringIO()

    print("\n" + "=" * 70, file=out)
    print("离线评估报告 (RAG管线)", file=out)
    print("=" * 70, file=out)
    print(f"\n评估时间: {report['timestamp']}", file=out)
    print(f"评估模式: {report.get('eval_mode', 'unknown')}", file=out)
    print(f"总测试数: {report['total_items']} | 通过: {report['passed']} | 失败: {report['failed']}", file=out)

    om = report["overall_metrics"]
    print(f"\n【综合指标】", file=out)
    print(f"  综合评分: {om.get('overall_score', 0) * 100:.1f}%", file=out)
    print(f"  意图分类准确率: {om.get('intent_accuracy', 0) * 100:.1f}%", file=out)
    print(f"  实体识别召回率: {om.get('entity_recall', 0) * 100:.1f}%", file=out)
    print(f"  关键词匹配率: {om.get('keyword_matching', 0) * 100:.1f}%", file=out)
    print(f"  检索召回率: {om.get('retrieval_recall', 0) * 100:.1f}%", file=out)
    print(f"  回答相关性: {om.get('answer_relevance', 0) * 100:.1f}%", file=out)
    print(f"\n【NLP指标（参考）】", file=out)
    print(f"  F1: {om.get('f1', 0) * 100:.1f}% | BLEU: {om.get('bleu', 0) * 100:.1f}%", file=out)
    print(f"  ROUGE-L: {om.get('rouge_l', 0) * 100:.1f}%", file=out)

    print(f"\n【类别评分】", file=out)
    for cat, m in sorted(report["category_metrics"].items()):
        s = m.get("overall_score", 0) * 100
        tag = "Y" if s >= 70 else "N"
        print(f"  {tag} {cat}: {s:.1f}%", file=out)

    es = report["execution_summary"]
    print(f"\n【性能指标】", file=out)
    print(f"  总耗时: {es.get('total_execution_time', 0):.1f}s", file=out)
    print(f"  平均响应: {es.get('avg_response_time', 0):.2f}s/条", file=out)
    print(f"  P95延迟: {es.get('p95_latency_ms', 0):.0f}ms", file=out)
    print(f"  吞吐量: {es.get('throughput', 0):.2f} 条/秒", file=out)
    print(f"\n【检索指标】", file=out)
    print(f"  平均检索文档数: {es.get('avg_doc_count', 0):.1f}", file=out)
    print(f"  平均检索实体数: {es.get('avg_entity_count', 0):.1f}", file=out)
    routing = es.get("routing_distribution", {})
    if routing:
        print(f"  检索策略分布: {dict(routing)}", file=out)

    if report.get("threshold_details", {}).get("checks"):
        print(f"\n【阈值检查】", file=out)
        print(f"  结果: {'通过' if report['threshold_passed'] else '未通过'}", file=out)
        for name, actual, threshold, passed in report["threshold_details"]["checks"]:
            tag = "Y" if passed else "N"
            formatted = f"{actual:.2%}" if isinstance(actual, float) else str(actual)
            print(f"    {tag} {name}: {formatted}", file=out)

    print(file=out)
    content = out.getvalue()
    print(content, end="")


def _save_report(report):
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(output_dir, f"golden_eval_rag_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {fp}")
    return fp


if __name__ == "__main__":
    run_rag_evaluation()
