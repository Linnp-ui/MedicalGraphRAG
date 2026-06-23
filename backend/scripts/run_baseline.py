"""Run baseline benchmark for all chunking strategies and output a report."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.text_splitter import TextSplitter, SplitStrategy
from src.evaluation.chunk_benchmark_suite import BenchmarkRunner, BENCHMARK_CASES

BASELINE_CHUNK_SIZE = 768
BASELINE_OVERLAP = 128

ALL_STRATEGIES = list(SplitStrategy)

class CallableSplitter:
    """Adapter: wraps TextSplitter so it's callable like splitter(text)."""
    def __init__(self, chunk_size, chunk_overlap, strategy):
        self._inner = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
    def __call__(self, text):
        return self._inner.split_text(text)

def make_factory(strategy: SplitStrategy):
    def factory(chunk_size, chunk_overlap):
        return CallableSplitter(chunk_size, chunk_overlap, strategy)
    return factory

def main():
    runner = BenchmarkRunner()

    factories = {}
    for s in ALL_STRATEGIES:
        label = s.value
        if s == SplitStrategy.SEMANTIC:
            print(f"\n  [SKIP] {label} — requires SentenceTransformer model download")
            continue
        factories[label] = make_factory(s)

    print("=" * 72)
    print("  GRAPHRAG 切分策略 BASELINE 基准测试")
    print(f"  参数: chunk_size={BASELINE_CHUNK_SIZE}, overlap={BASELINE_OVERLAP}")
    print(f"  策略: {', '.join(factories.keys())}")
    print("=" * 72)

    all_results = runner.compare_strategies(factories, BASELINE_CHUNK_SIZE, BASELINE_OVERLAP)

    # Per-case comparison table
    print(f"\n{'='*72}")
    print("  各用例最佳策略一览")
    print(f"{'='*72}")
    header = f"{'用例':<20} {'最佳策略':<18} {'总分':>6} {'完整':>5} {'边界':>5} {'召回':>5} {'耗时ms':>7}"
    print(header)
    print("-" * len(header))
    for case in BENCHMARK_CASES:
        best_name = None
        best_score = -1
        best_report = None
        for sname, reports in all_results.items():
            for r in reports:
                if r.strategy_name.endswith(case.name):
                    if r.overall_score > best_score:
                        best_score = r.overall_score
                        best_name = sname
                        best_report = r
        if best_report:
            print(f"{case.name:<20} {best_name:<18} {best_score:>6.3f} "
                  f"{best_report.consistency:>5.2f} {best_report.boundary:>5.2f} "
                  f"{best_report.retrieval_recall:>5.2f} {best_report.split_time_ms:>7.1f}")

    # Summary table with all six metrics per strategy
    print(f"\n{'='*72}")
    print("  策略综合评分（8用例平均）")
    print(f"{'='*72}")
    h2 = f"{'策略':<18} {'总分':>6} {'完整':>5} {'边界':>5} {'密度':>5} {'重叠':>5} {'稳定':>6} {'召回':>5} {'耗时ms':>7} {'达标':>5}"
    print(h2)
    print("-" * len(h2))

    ranked = []
    for sname, reports in all_results.items():
        nr = len(reports)
        avg_score = sum(r.overall_score for r in reports) / nr
        avg_cons = sum(r.consistency for r in reports) / nr
        avg_bound = sum(r.boundary for r in reports) / nr
        avg_dens = sum(r.density for r in reports) / nr
        avg_olap = sum(r.overlap_eff for r in reports) / nr
        avg_stab = sum(r.length_stab for r in reports) / nr
        avg_recall = sum(r.retrieval_recall for r in reports) / nr
        avg_time = sum(r.split_time_ms for r in reports) / nr
        passed = sum(1 for r in reports if r.passed)
        ranked.append((avg_score, sname, avg_cons, avg_bound, avg_dens, avg_olap, avg_stab, avg_recall, avg_time, passed))

    # total cases per strategy (should be same for all)
    total_cases = len(BENCHMARK_CASES)
    ranked.sort(key=lambda x: x[0], reverse=True)
    for avg_score, sname, avg_cons, avg_bound, avg_dens, avg_olap, avg_stab, avg_recall, avg_time, passed in ranked:
        print(f"{sname:<18} {avg_score:>6.3f} {avg_cons:>5.2f} {avg_bound:>5.2f} "
              f"{avg_dens:>5.2f} {avg_olap:>5.2f} {avg_stab:>6.3f} {avg_recall:>5.2f} "
              f"{avg_time:>7.1f} {passed}/{total_cases}")

    print("-" * len(h2))
    print(f"\n  指标权重: 完整(25%) + 召回(25%) + 边界(20%) + 密度(10%) + 重叠(10%) + 稳定(10%)")

    # Best overall strategy
    if ranked:
        best = ranked[0]
        print(f"\n  >>> BASELINE 最佳策略: {best[1]} (总分 {best[0]:.3f})")
        print(f"  >>> 建议设为默认策略")

    print(f"\n{'='*72}")
    print("  BASELINE 完成")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
