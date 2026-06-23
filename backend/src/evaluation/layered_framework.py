"""分层评估框架

四层评估架构，从组件到系统逐层诊断问题来源：
  L1 组件层 — NER / 意图分类 / 实体提取
  L2 检索层 — 向量检索 / 图谱检索 / 路由策略
  L3 生成层 — 回答质量 / 医疗安全 / 忠实度
  L4 系统层 — 端到端性能 / 延迟 / 阈值合规
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .metrics_engine import MetricsEngine
from .llm_judge import LLMJudge
from .ragas_evaluator import RagasEvaluator, RagasScore, MedicalSafetyChecker
from .threshold_checker import ThresholdChecker, ThresholdConfig, CheckResult
from .benchmark_dataset import BenchmarkDataset, BenchmarkItem
from .medical_golden_set import MedicalGoldenCase, MEDICAL_GOLDEN_CASES


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

class LayerLevel(str, Enum):
    COMPONENT = "L1"
    RETRIEVAL = "L2"
    GENERATION = "L3"
    SYSTEM = "L4"


@dataclass
class LayerMetric:
    """单条指标"""
    name: str
    value: float
    threshold: float
    passed: bool
    weight: float = 1.0
    detail: str = ""


@dataclass
class LayerReport:
    """单层评估报告"""
    layer: LayerLevel
    layer_name: str
    metrics: List[LayerMetric] = field(default_factory=list)
    passed: bool = True
    score: float = 0.0
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float, threshold: float,
                   weight: float = 1.0, detail: str = "",
                   passed: Optional[bool] = None):
        if passed is not None:
            is_passed = passed
        else:
            is_passed = value >= threshold
        self.metrics.append(LayerMetric(
            name=name, value=value, threshold=threshold,
            passed=is_passed, weight=weight, detail=detail,
        ))
        if not is_passed:
            self.passed = False

    def weighted_score(self) -> float:
        total_weight = sum(m.weight for m in self.metrics)
        if total_weight == 0:
            return 0.0
        return sum(m.value * m.weight for m in self.metrics) / total_weight


@dataclass
class LayeredEvaluationReport:
    """完整分层评估报告"""
    timestamp: str
    total_duration_s: float
    layer_reports: Dict[LayerLevel, LayerReport] = field(default_factory=dict)
    overall_passed: bool = True
    overall_score: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)

    def compute_overall(self):
        layer_weights = {
            LayerLevel.COMPONENT: 0.25,
            LayerLevel.RETRIEVAL: 0.25,
            LayerLevel.GENERATION: 0.30,
            LayerLevel.SYSTEM: 0.20,
        }
        total = 0.0
        for level, report in self.layer_reports.items():
            w = layer_weights.get(level, 0.0)
            total += w * report.weighted_score()
            if not report.passed:
                self.overall_passed = False
        self.overall_score = total


# ──────────────────────────────────────────────
# L1 组件层评估
# ──────────────────────────────────────────────

class ComponentLayerEvaluator:
    """L1 组件层：评估 NER、意图分类、实体提取的独立准确性"""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "intent_accuracy": 0.80,
            "entity_recall": 0.70,
            "ner_precision": 0.75,
        }
        self._intent_classifier = None
        self._ner_engine = None

    def _get_intent_classifier(self):
        if self._intent_classifier is None:
            from ..chains.medical_intent import MedicalIntentClassifier
            self._intent_classifier = MedicalIntentClassifier()
        return self._intent_classifier

    def _get_ner_engine(self):
        if self._ner_engine is None:
            from ..ingestion.medical_ner import MedicalNER
            self._ner_engine = MedicalNER(use_bert=True)
        return self._ner_engine

    def evaluate(self, dataset: BenchmarkDataset) -> LayerReport:
        report = LayerReport(
            layer=LayerLevel.COMPONENT,
            layer_name="组件层 (NER / 意图分类 / 实体提取)",
        )
        start = time.time()

        intent_correct = 0
        total_entities_expected = 0
        total_entities_found = 0
        ner_true_positives = 0
        ner_false_positives = 0

        for item in dataset.items:
            # 意图分类
            predicted_intent = self._classify_intent(item.question)
            if predicted_intent == item.expected_intent:
                intent_correct += 1

            # 实体提取
            found_entities = self._extract_entities(item.question)
            expected_set = set(item.expected_entities)
            found_set = set(found_entities)

            total_entities_expected += len(expected_set)
            total_entities_found += len(expected_set & found_set)
            ner_true_positives += len(expected_set & found_set)
            ner_false_positives += len(found_set - expected_set)

        n = len(dataset.items)
        if n == 0:
            report.errors.append("数据集为空")
            report.duration_s = time.time() - start
            return report

        intent_acc = intent_correct / n
        entity_rec = total_entities_found / total_entities_expected if total_entities_expected > 0 else 0.0
        ner_prec = ner_true_positives / (ner_true_positives + ner_false_positives) if (ner_true_positives + ner_false_positives) > 0 else 0.0

        report.add_metric("意图分类准确率", intent_acc,
                          self.thresholds["intent_accuracy"], weight=0.35)
        report.add_metric("实体识别召回率", entity_rec,
                          self.thresholds["entity_recall"], weight=0.35)
        report.add_metric("NER精确率", ner_prec,
                          self.thresholds["ner_precision"], weight=0.30)

        report.score = report.weighted_score()
        report.details = {
            "total_items": n,
            "intent_correct": intent_correct,
            "entities_expected": total_entities_expected,
            "entities_found": total_entities_found,
        }
        report.duration_s = time.time() - start
        return report

    def _classify_intent(self, question: str) -> str:
        try:
            classifier = self._get_intent_classifier()
            result = classifier.classify(question)
            return result.intent.value if hasattr(result.intent, "value") else str(result.intent)
        except Exception:
            return "unknown"

    def _extract_entities(self, question: str) -> List[str]:
        try:
            ner = self._get_ner_engine()
            entities = ner.extract(question)
            return [e.name for e in entities]
        except Exception:
            return []


# ──────────────────────────────────────────────
# L2 检索层评估
# ──────────────────────────────────────────────

class RetrievalLayerEvaluator:
    """L2 检索层：评估向量检索、图谱检索、路由策略质量"""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "routing_accuracy": 0.80,
            "retrieval_recall": 0.70,
            "retrieval_precision": 0.60,
            "keyword_coverage": 0.65,
        }
        self._drift_searcher = None

    def _get_drift_searcher(self):
        if self._drift_searcher is None:
            from ..retrieval.drift_search import DRIFTSearch
            self._drift_searcher = DRIFTSearch()
        return self._drift_searcher

    def evaluate(self, dataset: BenchmarkDataset) -> LayerReport:
        report = LayerReport(
            layer=LayerLevel.RETRIEVAL,
            layer_name="检索层 (向量/图谱检索 + 路由策略)",
        )
        start = time.time()

        metrics_engine = MetricsEngine()
        routing_correct = 0
        keyword_coverages = []
        retrieval_recalls = []

        for item in dataset.items:
            # 路由策略评估
            predicted_route = self._predict_route(item.question)
            expected_route = self._expected_route(item.expected_intent)
            if predicted_route == expected_route:
                routing_correct += 1

            # 检索质量评估：通过关键词覆盖率和检索召回率间接衡量
            retrieved_context = self._retrieve(item.question, predicted_route)
            kw_cov = self._keyword_coverage(retrieved_context, item.keywords)
            keyword_coverages.append(kw_cov)

            rec = metrics_engine.retrieval_recall(
                retrieved_context, item.keywords, item.reference_answer
            )
            retrieval_recalls.append(rec)

        n = len(dataset.items)
        if n == 0:
            report.errors.append("数据集为空")
            report.duration_s = time.time() - start
            return report

        routing_acc = routing_correct / n
        avg_kw_cov = sum(keyword_coverages) / n
        avg_rec = sum(retrieval_recalls) / n

        report.add_metric("路由策略准确率", routing_acc,
                          self.thresholds["routing_accuracy"], weight=0.30)
        report.add_metric("关键词覆盖率", avg_kw_cov,
                          self.thresholds["keyword_coverage"], weight=0.30)
        report.add_metric("检索召回率", avg_rec,
                          self.thresholds["retrieval_recall"], weight=0.40)

        report.score = report.weighted_score()
        report.details = {
            "total_items": n,
            "routing_correct": routing_correct,
            "avg_keyword_coverage": avg_kw_cov,
            "avg_retrieval_recall": avg_rec,
        }
        report.duration_s = time.time() - start
        return report

    def _predict_route(self, question: str) -> str:
        try:
            searcher = self._get_drift_searcher()
            result = searcher.classify_intent(question)
            return result if isinstance(result, str) else "local"
        except Exception:
            return "local"

    @staticmethod
    def _expected_route(intent: str) -> str:
        global_intents = {"disease_query", "prevention_query", "health_advice"}
        if intent in global_intents:
            return "global"
        return "local"

    def _retrieve(self, question: str, route: str) -> str:
        try:
            searcher = self._get_drift_searcher()
            if route == "global":
                result = searcher.global_search(question)
            else:
                result = searcher.local_search(question)
            return self._extract_context(result)
        except Exception:
            return ""

    @staticmethod
    def _extract_context(result: dict) -> str:
        """从检索结果中提取文本上下文"""
        parts = []
        if "global_summary" in result and result["global_summary"]:
            parts.append(str(result["global_summary"]))
        if "community_summaries" in result:
            for s in result["community_summaries"].values():
                if s:
                    parts.append(str(s))
        if "results" in result:
            for r in result["results"]:
                if isinstance(r, dict):
                    if r.get("summary"):
                        parts.append(str(r["summary"]))
                    if r.get("entity"):
                        parts.append(str(r["entity"]))
                    for rel in r.get("relationships", []):
                        if isinstance(rel, dict):
                            parts.append(rel.get("description", ""))
                        elif isinstance(rel, str):
                            parts.append(rel)
        if "context" in result and result["context"]:
            parts.append(str(result["context"]))
        if "answer" in result and result["answer"]:
            parts.append(str(result["answer"]))
        return " ".join(p for p in parts if p)

    @staticmethod
    def _keyword_coverage(text: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched / len(keywords)


# ──────────────────────────────────────────────
# L3 生成层评估
# ──────────────────────────────────────────────

class GenerationLayerEvaluator:
    """L3 生成层：评估回答质量、医疗安全、忠实度"""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None,
                 use_llm_judge: bool = False):
        self.thresholds = thresholds or {
            "f1_score": 0.50,
            "semantic_similarity": 0.60,
            "medical_safety": 0.85,
            "faithfulness": 0.70,
            "answer_relevancy": 0.70,
        }
        self.use_llm_judge = use_llm_judge
        self.metrics_engine = MetricsEngine()
        self.ragas = RagasEvaluator()
        self._qa_chain = None
        self._drift_searcher = None

    def _get_qa_chain(self):
        if self._qa_chain is None:
            from ..chains.qa_chain import QAChain
            self._qa_chain = QAChain()
        return self._qa_chain

    def _get_drift_searcher(self):
        if self._drift_searcher is None:
            from ..retrieval.drift_search import DRIFTSearch
            self._drift_searcher = DRIFTSearch()
        return self._drift_searcher

    def evaluate(self, dataset: BenchmarkDataset,
                 golden_cases: Optional[List[MedicalGoldenCase]] = None) -> LayerReport:
        report = LayerReport(
            layer=LayerLevel.GENERATION,
            layer_name="生成层 (回答质量 / 医疗安全 / 忠实度)",
        )
        start = time.time()

        f1_scores = []
        sem_sims = []
        safety_scores = []
        faith_scores = []
        relevancy_scores = []

        items = dataset.items
        for item in items:
            answer = self._get_answer(item.question)
            metrics = self.metrics_engine.calculate_all(
                answer, item.reference_answer, item.keywords
            )
            f1_scores.append(metrics["f1"])
            sem_sims.append(metrics["semantic_similarity"])

        # RAGAS 安全评估（使用黄金数据集）
        eval_cases = golden_cases or MEDICAL_GOLDEN_CASES[:10]
        for case in eval_cases:
            answer = self._get_answer(case.question)
            try:
                score = self.ragas.evaluate_case(case, answer)
                safety_scores.append(score.medical_safety)
                faith_scores.append(score.faithfulness)
                relevancy_scores.append(score.answer_relevancy)
            except Exception:
                safety_scores.append(0.5)
                faith_scores.append(0.5)
                relevancy_scores.append(0.5)

        n_gen = len(f1_scores)
        n_safety = len(safety_scores)

        avg_f1 = sum(f1_scores) / n_gen if n_gen else 0.0
        avg_sem = sum(sem_sims) / n_gen if n_gen else 0.0
        avg_safety = sum(safety_scores) / n_safety if n_safety else 0.0
        avg_faith = sum(faith_scores) / n_safety if n_safety else 0.0
        avg_rel = sum(relevancy_scores) / n_safety if n_safety else 0.0

        report.add_metric("F1分数", avg_f1,
                          self.thresholds["f1_score"], weight=0.20)
        report.add_metric("语义相似度", avg_sem,
                          self.thresholds["semantic_similarity"], weight=0.15)
        report.add_metric("医疗安全评分", avg_safety,
                          self.thresholds["medical_safety"], weight=0.30)
        report.add_metric("忠实度", avg_faith,
                          self.thresholds["faithfulness"], weight=0.20)
        report.add_metric("回答相关性", avg_rel,
                          self.thresholds["answer_relevancy"], weight=0.15)

        report.score = report.weighted_score()
        report.details = {
            "generation_items": n_gen,
            "safety_eval_items": n_safety,
            "avg_f1": avg_f1,
            "avg_semantic_similarity": avg_sem,
            "avg_medical_safety": avg_safety,
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_rel,
        }
        report.duration_s = time.time() - start
        return report

    def _get_answer(self, question: str) -> str:
        try:
            searcher = self._get_drift_searcher()
            search_result = searcher.search(question)
            context = RetrievalLayerEvaluator._extract_context(search_result)
            chain = self._get_qa_chain()
            return chain.answer(question, context=context)
        except Exception as e:
            return f"模型调用失败: {e}"


# ──────────────────────────────────────────────
# L4 系统层评估
# ──────────────────────────────────────────────

class SystemLayerEvaluator:
    """L4 系统层：端到端性能、延迟、阈值合规"""

    def __init__(self, threshold_config: Optional[ThresholdConfig] = None,
                 custom_thresholds: Optional[Dict[str, float]] = None):
        self.threshold_config = threshold_config or ThresholdConfig()
        self.custom_thresholds = custom_thresholds or {
            "avg_latency_s": 5.0,
            "p95_latency_s": 10.0,
            "error_rate": 0.05,
            "throughput_rps": 0.1,
        }
        self.metrics_engine = MetricsEngine()
        self._qa_chain = None
        self._drift_searcher = None

    def _get_qa_chain(self):
        if self._qa_chain is None:
            from ..chains.qa_chain import QAChain
            self._qa_chain = QAChain()
        return self._qa_chain

    def _get_drift_searcher(self):
        if self._drift_searcher is None:
            from ..retrieval.drift_search import DRIFTSearch
            self._drift_searcher = DRIFTSearch()
        return self._drift_searcher

    def evaluate(self, dataset: BenchmarkDataset) -> LayerReport:
        report = LayerReport(
            layer=LayerLevel.SYSTEM,
            layer_name="系统层 (端到端性能 / 延迟 / 阈值合规)",
        )
        start = time.time()

        latencies = []
        errors = 0
        overall_metrics_accum = {}

        for item in dataset.items:
            t0 = time.time()
            try:
                answer = self._get_answer(item.question)
                latency = time.time() - t0
                latencies.append(latency)

                m = self.metrics_engine.calculate_all(
                    answer, item.reference_answer, item.keywords
                )
                for k, v in m.items():
                    overall_metrics_accum.setdefault(k, []).append(v)
            except Exception:
                errors += 1
                latencies.append(time.time() - t0)

        n = len(dataset.items)
        if n == 0:
            report.errors.append("数据集为空")
            report.duration_s = time.time() - start
            return report

        avg_latency = sum(latencies) / n
        sorted_lat = sorted(latencies)
        p95_idx = min(int(n * 0.95), n - 1)
        p95_latency = sorted_lat[p95_idx]
        error_rate = errors / n
        throughput = n / sum(latencies) if sum(latencies) > 0 else 0.0

        # 计算综合指标均值
        avg_metrics = {k: sum(v) / len(v) for k, v in overall_metrics_accum.items()}
        overall_score = (
            0.25 * avg_metrics.get("keyword_matching", 0) +
            0.25 * avg_metrics.get("retrieval_recall", 0) +
            0.20 * avg_metrics.get("semantic_similarity", 0) +
            0.30 * avg_metrics.get("f1", 0)
        )

        # 延迟指标（反向：值越小越好，归一化为 0~1 的通过率）
        latency_norm = max(0, 1 - avg_latency / self.custom_thresholds["avg_latency_s"])
        p95_norm = max(0, 1 - p95_latency / self.custom_thresholds["p95_latency_s"])
        report.add_metric("平均延迟(s)", latency_norm,
                          0.5, weight=0.15,
                          detail=f"实际={avg_latency:.2f}s, 阈值<={self.custom_thresholds['avg_latency_s']}s",
                          passed=avg_latency <= self.custom_thresholds["avg_latency_s"])
        report.add_metric("P95延迟(s)", p95_norm,
                          0.5, weight=0.10,
                          detail=f"实际={p95_latency:.2f}s, 阈值<={self.custom_thresholds['p95_latency_s']}s",
                          passed=p95_latency <= self.custom_thresholds["p95_latency_s"])

        report.add_metric("错误率", 1 - error_rate,
                          0.5, weight=0.15,
                          detail=f"实际={error_rate:.4f}, 阈值<={self.custom_thresholds['error_rate']}",
                          passed=error_rate <= self.custom_thresholds["error_rate"])

        report.add_metric("综合评分", overall_score,
                          self.threshold_config.overall_score, weight=0.30)

        throughput_norm = min(1.0, throughput / self.custom_thresholds["throughput_rps"])
        report.add_metric("吞吐量(次/秒)", throughput_norm,
                          1.0, weight=0.10,
                          detail=f"{throughput:.2f}/s")

        # 阈值合规检查
        threshold_checker = ThresholdChecker(self.threshold_config)
        threshold_result = threshold_checker.check_all(avg_metrics)
        report.add_metric("阈值合规", 1.0 if threshold_result.passed else 0.0,
                          1.0, weight=0.20,
                          detail=threshold_result.message)

        report.score = report.weighted_score()
        report.details = {
            "total_items": n,
            "errors": errors,
            "avg_latency_s": avg_latency,
            "p95_latency_s": p95_latency,
            "throughput_rps": throughput,
            "overall_score": overall_score,
            "threshold_passed": threshold_result.passed,
            "avg_metrics": avg_metrics,
        }
        report.duration_s = time.time() - start
        return report

    def _get_answer(self, question: str) -> str:
        try:
            searcher = self._get_drift_searcher()
            search_result = searcher.search(question)
            context = RetrievalLayerEvaluator._extract_context(search_result)
            chain = self._get_qa_chain()
            return chain.answer(question, context=context)
        except Exception as e:
            return f"模型调用失败: {e}"


# ──────────────────────────────────────────────
# 分层评估框架主入口
# ──────────────────────────────────────────────

class LayeredEvaluationFramework:
    """分层评估框架

    用法:
        framework = LayeredEvaluationFramework()
        report = framework.run(dataset)
        framework.print_report(report)
        framework.save_report(report, "eval_results/")
    """

    def __init__(
        self,
        component_thresholds: Optional[Dict[str, float]] = None,
        retrieval_thresholds: Optional[Dict[str, float]] = None,
        generation_thresholds: Optional[Dict[str, float]] = None,
        system_threshold_config: Optional[ThresholdConfig] = None,
        system_custom_thresholds: Optional[Dict[str, float]] = None,
        use_llm_judge: bool = False,
    ):
        self.component_evaluator = ComponentLayerEvaluator(component_thresholds)
        self.retrieval_evaluator = RetrievalLayerEvaluator(retrieval_thresholds)
        self.generation_evaluator = GenerationLayerEvaluator(
            generation_thresholds, use_llm_judge=use_llm_judge
        )
        self.system_evaluator = SystemLayerEvaluator(
            system_threshold_config, system_custom_thresholds
        )

    def run(
        self,
        dataset: Optional[BenchmarkDataset] = None,
        golden_cases: Optional[List[MedicalGoldenCase]] = None,
        layers: Optional[List[LayerLevel]] = None,
    ) -> LayeredEvaluationReport:
        """执行分层评估

        Args:
            dataset: 评估数据集，默认使用医疗基准集
            golden_cases: 黄金数据集（用于L3安全评估），默认使用内置集
            layers: 指定要运行的层，默认全部运行
        """
        if dataset is None:
            from .benchmark_dataset import MedicalBenchmarkLoader
            dataset = MedicalBenchmarkLoader.load_medical_benchmark()

        if layers is None:
            layers = list(LayerLevel)

        total_start = time.time()
        report = LayeredEvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_duration_s=0.0,
        )

        layer_runners = {
            LayerLevel.COMPONENT: lambda: self.component_evaluator.evaluate(dataset),
            LayerLevel.RETRIEVAL: lambda: self.retrieval_evaluator.evaluate(dataset),
            LayerLevel.GENERATION: lambda: self.generation_evaluator.evaluate(dataset, golden_cases),
            LayerLevel.SYSTEM: lambda: self.system_evaluator.evaluate(dataset),
        }

        for level in layers:
            if level not in layer_runners:
                continue
            print(f"\n{'='*60}")
            print(f"  运行 {level.value} 评估...")
            print(f"{'='*60}")
            try:
                layer_report = layer_runners[level]()
                report.layer_reports[level] = layer_report
                status = "通过" if layer_report.passed else "未通过"
                print(f"  {level.value} {layer_report.layer_name}: {status} "
                      f"(评分: {layer_report.score:.2%}, 耗时: {layer_report.duration_s:.1f}s)")
            except Exception as e:
                error_report = LayerReport(
                    layer=level,
                    layer_name=f"{level.value} 评估异常",
                    passed=False,
                    errors=[str(e)],
                )
                report.layer_reports[level] = error_report
                print(f"  {level.value} 评估异常: {e}")

        report.total_duration_s = time.time() - total_start
        report.compute_overall()
        return report

    def run_single_layer(
        self,
        level: LayerLevel,
        dataset: Optional[BenchmarkDataset] = None,
        golden_cases: Optional[List[MedicalGoldenCase]] = None,
    ) -> LayerReport:
        """单独运行某一层评估"""
        if dataset is None:
            from .benchmark_dataset import MedicalBenchmarkLoader
            dataset = MedicalBenchmarkLoader.load_medical_benchmark()

        runners = {
            LayerLevel.COMPONENT: lambda: self.component_evaluator.evaluate(dataset),
            LayerLevel.RETRIEVAL: lambda: self.retrieval_evaluator.evaluate(dataset),
            LayerLevel.GENERATION: lambda: self.generation_evaluator.evaluate(dataset, golden_cases),
            LayerLevel.SYSTEM: lambda: self.system_evaluator.evaluate(dataset),
        }
        return runners[level]()

    # ──────────────────────────────────────────
    # 报告输出
    # ──────────────────────────────────────────

    def print_report(self, report: LayeredEvaluationReport):
        print("\n" + "=" * 70)
        print("  分层评估报告")
        print("=" * 70)
        print(f"  评估时间: {report.timestamp}")
        print(f"  总耗时: {report.total_duration_s:.1f}s")
        print(f"  总体结果: {'通过' if report.overall_passed else '未通过'}")
        print(f"  总体评分: {report.overall_score:.2%}")

        for level in LayerLevel:
            lr = report.layer_reports.get(level)
            if lr is None:
                continue
            status = "PASS" if lr.passed else "FAIL"
            print(f"\n  ── {level.value} {lr.layer_name} [{status}] "
                  f"评分: {lr.score:.2%} 耗时: {lr.duration_s:.1f}s ──")
            for m in lr.metrics:
                symbol = "Y" if m.passed else "N"
                if "延迟" in m.name or m.name == "错误率":
                    symbol = "Y" if m.value <= m.threshold else "N"
                val_str = f"{m.value:.4f}" if m.value < 1 else f"{m.value:.2f}"
                thr_str = f"{m.threshold:.4f}" if m.threshold < 1 else f"{m.threshold:.2f}"
                print(f"    [{symbol}] {m.name}: {val_str}  (阈值: {thr_str}, 权重: {m.weight})")
            if lr.errors:
                for err in lr.errors:
                    print(f"    !! 错误: {err}")

        print("\n" + "=" * 70)

    def save_report(self, report: LayeredEvaluationReport,
                    output_dir: str = "test_results") -> str:
        import os
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"layered_eval_{timestamp}.json")

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
                    {
                        "name": m.name,
                        "value": m.value,
                        "threshold": m.threshold,
                        "passed": m.passed,
                        "weight": m.weight,
                    }
                    for m in lr.metrics
                ],
                "errors": lr.errors,
                "details": lr.details,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n报告已保存: {path}")
        return path

    def compare_reports(self, report_a: LayeredEvaluationReport,
                        report_b: LayeredEvaluationReport) -> Dict[str, Any]:
        """对比两次评估报告，输出各层指标变化"""
        comparison = {}
        for level in LayerLevel:
            lr_a = report_a.layer_reports.get(level)
            lr_b = report_b.layer_reports.get(level)
            if lr_a is None or lr_b is None:
                continue

            metrics_a = {m.name: m.value for m in lr_a.metrics}
            metrics_b = {m.name: m.value for m in lr_b.metrics}

            diffs = {}
            for name in set(metrics_a) | set(metrics_b):
                va = metrics_a.get(name, 0.0)
                vb = metrics_b.get(name, 0.0)
                diffs[name] = {
                    "before": va,
                    "after": vb,
                    "delta": vb - va,
                    "improved": vb > va,
                }

            comparison[level.value] = {
                "score_before": lr_a.score,
                "score_after": lr_b.score,
                "score_delta": lr_b.score - lr_a.score,
                "metric_diffs": diffs,
            }

        return comparison
