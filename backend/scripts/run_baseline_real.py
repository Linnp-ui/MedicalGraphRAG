"""Real-world baseline benchmark using actual project documents and large stress tests."""
import sys, os, textwrap, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.text_splitter import TextSplitter, SplitStrategy
from src.evaluation.chunk_quality_metrics import ChunkQualityEvaluator, ChunkQualityReport

BASELINE_CS = 768
BASELINE_OL = 128

SKIP_STRATEGIES = {SplitStrategy.SEMANTIC}

class CallableSplitter:
    def __init__(self, chunk_size, chunk_overlap, strategy):
        self._inner = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
    def __call__(self, text):
        return self._inner.split_text(text)

def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "input")
DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")

REAL_CASES = [
    # (name, text, description)
    ("med_kb_10k",    load(os.path.join(DATA, "medical_knowledge_base.md")),           "真实医学知识库(10k字符,结构化Markdown)"),
    ("arch_review_8k", load(os.path.join(DOCS, "architecture-review-2026-05-05.md")), "架构评审文档(8.7k字符,层级标题)"),
    ("proj_transform_10k", load(os.path.join(DOCS, "project-transformation.md")),     "项目转型文档(10.7k字符,多节结构)"),
    ("chunk_plan_5k",  load(os.path.join(DOCS, "chunking-improvement-plan.md")),      "分块优化计划(5.4k字符,技术文档)"),
    ("opt_report_3k",  load(os.path.join(DOCS, "optimization-report.md")),            "优化报告(3.3k字符,短文档)"),
]

# ── Massive stress tests ──────────────────────────────────────────────
med_base = load(os.path.join(DATA, "medical_knowledge_base.md"))
LARGE_CASES = [
    ("med_kb_x3",     med_base * 3,       "医学KB×3(~31k) 测试大批量同结构"),
    ("med_kb_x5",     med_base * 5,       "医学KB×5(~52k) 测试超大文档"),
    ("mixed_x3",      med_base * 3 + "\n\n" + load(os.path.join(DOCS, "architecture-review-2026-05-05.md")) * 2,
                                             "混合多文档(~50k) 医学+架构跨领域"),
]

ALL_CASES = REAL_CASES + LARGE_CASES

def evaluate_strategy(strategy: SplitStrategy, cases: list) -> list[ChunkQualityReport]:
    evaluator = ChunkQualityEvaluator()
    splitter = CallableSplitter(BASELINE_CS, BASELINE_OL, strategy)
    results = []
    for name, text, desc in cases:
        t0 = time.perf_counter()
        chunks = splitter(text)
        elapsed = (time.perf_counter() - t0) * 1000
        texts = [c.content if hasattr(c, "content") else c for c in chunks]
        report = evaluator.evaluate(texts, text, f"{strategy.value}/{name}", elapsed, BASELINE_OL)
        results.append((name, chunks, report))
    return results

def fmt_header(text):
    w = 72
    print(f"\n{'='*w}")
    print(f"  {text}")
    print(f"{'='*w}")

def print_detail(strategy_name, results):
    """Print per-case detail table."""
    h = f"{'用例':<22} {'chunks':>6} {'总分':>7} {'完整':>6} {'边界':>6} {'密度':>6} {'重叠':>6} {'稳定':>7} {'召回':>6} {'耗时ms':>8} {'chunk大小':>10}"
    print(f"\n  >>> {strategy_name}")
    print("  " + "-" * len(h))
    print("  " + h)
    print("  " + "-" * len(h))
    for name, chunks, r in results:
        sizes = [len(c.content if hasattr(c, "content") else c) for c in chunks]
        size_range = f"{min(sizes):,}-{max(sizes):,}" if sizes else "N/A"
        print(f"  {name:<22} {r.num_chunks:>6} {r.overall_score:>7.3f} {r.consistency:>6.2f} "
              f"{r.boundary:>6.2f} {r.density:>6.2f} {r.overlap_eff:>6.2f} "
              f"{r.length_stab:>7.3f} {r.retrieval_recall:>6.2f} {r.split_time_ms:>8.2f} "
              f"{size_range:>10}")
    print("  " + "-" * len(h))

def main():
    print(f"\n{'='*72}")
    print(f"  GRAPHRAG BASELINE ── 真实文档 + 大规模压力测试")
    print(f"  chunk_size={BASELINE_CS}, overlap={BASELINE_OL}")
    print(f"  策略: {', '.join(s.value for s in SplitStrategy if s not in SKIP_STRATEGIES)}")
    print(f"  用例: {len(ALL_CASES)}个 (含{sum(1 for _ in LARGE_CASES)}个大文档)")
    print(f"{'='*72}")

    # ── Collect all results ──────────────────────────────────────────
    all_data: dict[str, list] = {}
    for strategy in SplitStrategy:
        if strategy in SKIP_STRATEGIES:
            print(f"\n  [SKIP] {strategy.value}")
            continue
        print(f"\n  [RUN]  {strategy.value} ...", end="")
        r = evaluate_strategy(strategy, ALL_CASES)
        all_data[strategy.value] = r
        print(f" done")

    # ── Per-strategy detail ──────────────────────────────────────────
    fmt_header("各策略详细评分")
    for sname, results in all_data.items():
        print_detail(sname, results)

    # ── Strategy ranking ─────────────────────────────────────────────
    fmt_header("策略综合排名（全部用例平均）")
    h = f"{'策略':<18} {'总分':>7} {'完整':>6} {'边界':>6} {'密度':>6} {'重叠':>6} {'稳定':>7} {'召回':>6} {'耗时ms':>8} {'达标率':>7}"
    print("  " + h)
    print("  " + "-" * len(h))

    ranked = []
    for sname, results in all_data.items():
        nr = len(results)
        avg_score = sum(r.overall_score for _, _, r in results) / nr
        avg_cons = sum(r.consistency for _, _, r in results) / nr
        avg_bound = sum(r.boundary for _, _, r in results) / nr
        avg_dens = sum(r.density for _, _, r in results) / nr
        avg_olap = sum(r.overlap_eff for _, _, r in results) / nr
        avg_stab = sum(r.length_stab for _, _, r in results) / nr
        avg_recall = sum(r.retrieval_recall for _, _, r in results) / nr
        avg_time = sum(r.split_time_ms for _, _, r in results) / nr
        passed = sum(1 for _, _, r in results if r.passed)
        ranked.append((avg_score, sname, avg_cons, avg_bound, avg_dens, avg_olap, avg_stab, avg_recall, avg_time, passed))

    ranked.sort(key=lambda x: x[0], reverse=True)
    for avg_score, sname, avg_cons, avg_bound, avg_dens, avg_olap, avg_stab, avg_recall, avg_time, passed in ranked:
        print(f"  {sname:<18} {avg_score:>7.3f} {avg_cons:>6.2f} {avg_bound:>6.2f} "
              f"{avg_dens:>6.2f} {avg_olap:>6.2f} {avg_stab:>7.3f} {avg_recall:>6.2f} "
              f"{avg_time:>8.2f} {passed}/{len(ALL_CASES):>4}")
    print("  " + "-" * len(h))

    # ── Best per case ────────────────────────────────────────────────
    fmt_header("各用例最佳策略")
    h2 = f"{'用例':<22} {'最佳策略':<16} {'总分':>7} {'完整':>6} {'边界':>6} {'召回':>6} {'chunks':>6} {'耗时ms':>8}"
    print("  " + h2)
    print("  " + "-" * len(h2))
    for case_name, _, _ in ALL_CASES:
        best_score = -1
        best_info = None
        for sname, results in all_data.items():
            for name, chunks, r in results:
                if name == case_name and r.overall_score > best_score:
                    best_score = r.overall_score
                    best_info = (sname, r, chunks)
        if best_info:
            sname, r, chunks = best_info
            print(f"  {case_name:<22} {sname:<16} {best_score:>7.3f} {r.consistency:>6.2f} "
                  f"{r.boundary:>6.2f} {r.retrieval_recall:>6.2f} {r.num_chunks:>6} {r.split_time_ms:>8.2f}")
    print("  " + "-" * len(h2))

    # ── Large-doc specific ranking ───────────────────────────────────
    fmt_header("大文档专项排名（>20k字符）")
    h3 = f"{'策略':<18} {'总分(大文档)':>13} {'完整':>6} {'边界':>6} {'召回':>6} {'chunks':>6}"
    print("  " + h3)
    print("  " + "-" * len(h3))
    large_ranked = []
    for sname, results in all_data.items():
        large_results = [(name, chunks, r) for name, chunks, r in results if any(name == l[0] for l in LARGE_CASES)]
        if not large_results:
            continue
        avg_large = sum(r.overall_score for _, _, r in large_results) / len(large_results)
        avg_cons = sum(r.consistency for _, _, r in large_results) / len(large_results)
        avg_bound = sum(r.boundary for _, _, r in large_results) / len(large_results)
        avg_recall = sum(r.retrieval_recall for _, _, r in large_results) / len(large_results)
        avg_chunks = sum(r.num_chunks for _, _, r in large_results) / len(large_results)
        large_ranked.append((avg_large, sname, avg_cons, avg_bound, avg_recall, avg_chunks))
    large_ranked.sort(key=lambda x: x[0], reverse=True)
    for avg_large, sname, avg_cons, avg_bound, avg_recall, avg_chunks in large_ranked:
        print(f"  {sname:<18} {avg_large:>13.3f} {avg_cons:>6.2f} {avg_bound:>6.2f} {avg_recall:>6.2f} {avg_chunks:>6.0f}")
    print("  " + "-" * len(h3))

    if ranked:
        best = ranked[0]
        print(f"\n  >>> BASELINE 最佳全局策略: {best[1]} (总分 {best[0]:.3f})")
    if large_ranked:
        lbest = large_ranked[0]
        print(f"  >>> 大文档推荐策略: {lbest[1]} (总分 {lbest[0]:.3f})")
    print()

if __name__ == "__main__":
    main()
