import time
import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field

from .benchmark_dataset import BenchmarkDataset, MedicalBenchmarkLoader
from .metrics_engine import MetricsEngine
from .llm_judge import LLMJudge
from .threshold_checker import ThresholdChecker, ThresholdConfig
from .medical_golden_set import MedicalGoldenCase, MEDICAL_GOLDEN_CASES
from .ragas_evaluator import RagasEvaluator, RagasScore
from .generated_loader import load_generated_dataset, load_generated_golden_set


@dataclass
class EvaluationResult:
    question: str
    reference_answer: str
    model_answer: str
    metrics: Dict[str, float]
    intent_correct: bool
    entities_found: int
    expected_entities: int
    response_time: float
    error_occurred: bool = False
    error_message: str = ""


@dataclass
class EvaluationReport:
    timestamp: str
    dataset_name: str
    total_items: int
    passed: int
    failed: int
    overall_metrics: Dict[str, float]
    category_metrics: Dict[str, Dict[str, float]]
    threshold_result: Any
    results: List[EvaluationResult]
    execution_summary: Dict[str, float]


class OfflineEvaluator:
    def __init__(self, threshold_config: ThresholdConfig = None):
        self.dataset = None
        self.metrics_engine = MetricsEngine()
        self.threshold_checker = ThresholdChecker(threshold_config)
        self.llm_judge = LLMJudge()
        self.results: List[EvaluationResult] = []
        self.ragas_scores: List[RagasScore] = []

    def load_dataset(self, dataset: BenchmarkDataset = None):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = MedicalBenchmarkLoader.load_medical_benchmark()

    def load_generated_dataset(self, filepath: str = None):
        """加载从知识库生成的黄金评估集"""
        self.dataset = load_generated_dataset(filepath)

    def evaluate_item(self, item) -> EvaluationResult:
        start_time = time.time()
        
        try:
            model_answer = self._get_model_answer(item.question)
            
            metrics = self.metrics_engine.calculate_all(
                prediction=model_answer,
                reference=item.reference_answer,
                keywords=item.keywords
            )
            
            intent_correct = self._evaluate_intent(item.question, item.expected_intent)
            entities_found = self._evaluate_entities(item.question, item.expected_entities)
            
            response_time = time.time() - start_time
            
            return EvaluationResult(
                question=item.question,
                reference_answer=item.reference_answer,
                model_answer=model_answer,
                metrics=metrics,
                intent_correct=intent_correct,
                entities_found=entities_found,
                expected_entities=len(item.expected_entities),
                response_time=response_time,
                error_occurred=False
            )
        except Exception as e:
            response_time = time.time() - start_time
            return EvaluationResult(
                question=item.question,
                reference_answer=item.reference_answer,
                model_answer="",
                metrics={},
                intent_correct=False,
                entities_found=0,
                expected_entities=len(item.expected_entities),
                response_time=response_time,
                error_occurred=True,
                error_message=str(e)
            )

    def _get_model_answer(self, question: str) -> str:
        try:
            from ..chains.qa_chain import QAChain
            qa_chain = QAChain()
            return qa_chain.answer(question)
        except Exception as e:
            return f"模型调用失败: {str(e)}"

    def _evaluate_intent(self, question: str, expected_intent: str) -> bool:
        try:
            from ..chains.medical_intent import MedicalIntentClassifier
            classifier = MedicalIntentClassifier()
            result = classifier.classify(question)
            actual_intent = result.intent.value if hasattr(result.intent, 'value') else str(result.intent)
            return actual_intent == expected_intent
        except Exception:
            return False

    def _evaluate_entities(self, question: str, expected_entities: List[str]) -> int:
        try:
            from ..chains.medical_intent import MedicalIntentClassifier
            classifier = MedicalIntentClassifier()
            result = classifier.classify(question)
            found = 0
            for entity in expected_entities:
                if entity in result.entities:
                    found += 1
            return found
        except Exception:
            return 0

    def run_evaluation(self, use_llm_judge: bool = False) -> EvaluationReport:
        if self.dataset is None:
            self.load_dataset()

        print(f"开始离线评估 - 数据集: {self.dataset.name}, 测试用例数: {len(self.dataset)}")
        
        for i, item in enumerate(self.dataset.items, 1):
            print(f"  [{i}/{len(self.dataset)}] {item.question[:30]}...")
            result = self.evaluate_item(item)
            self.results.append(result)
            
            if result.error_occurred:
                print(f"     ❌ 错误: {result.error_message}")

        return self._generate_report(use_llm_judge)

    def run_ragas_evaluation(self) -> List[RagasScore]:
        print(f"\n开始RAGAS黄金数据集评估 - {len(MEDICAL_GOLDEN_CASES)} 个高危病例")
        ragas = RagasEvaluator()
        scores = []
        for i, case in enumerate(MEDICAL_GOLDEN_CASES, 1):
            print(f"  [{i}/{len(MEDICAL_GOLDEN_CASES)}] [{case.safety_category}] {case.question[:30]}...")
            try:
                model_answer = self._get_model_answer(case.question)
                score = ragas.evaluate_case(case, model_answer)
                scores.append(score)
            except Exception as e:
                print(f"     ❌ 错误: {e}")
                scores.append(RagasScore())
        self.ragas_scores = scores
        return scores

    def _generate_report(self, use_llm_judge: bool) -> EvaluationReport:
        successful_results = [r for r in self.results if not r.error_occurred]
        
        if not successful_results:
            return EvaluationReport(
                timestamp=datetime.now().isoformat(),
                dataset_name=self.dataset.name,
                total_items=len(self.results),
                passed=0,
                failed=len(self.results),
                overall_metrics={},
                category_metrics={},
                threshold_result=None,
                results=self.results,
                execution_summary={}
            )

        overall_metrics = {}
        primary_metrics = ['exact_match', 'f1', 'keyword_matching', 'retrieval_recall', 'semantic_similarity']
        secondary_metrics = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']
        
        for metric_name in primary_metrics + secondary_metrics:
            values = [r.metrics.get(metric_name, 0) for r in successful_results]
            overall_metrics[metric_name] = sum(values) / len(values)

        intent_accuracy = sum(1 for r in successful_results if r.intent_correct) / len(successful_results)
        total_expected_entities = sum(r.expected_entities for r in successful_results)
        total_found_entities = sum(r.entities_found for r in successful_results)
        entity_recall = total_found_entities / total_expected_entities if total_expected_entities > 0 else 0
        keyword_matching = overall_metrics.get('keyword_matching', 0)
        answer_relevance = overall_metrics.get('semantic_similarity', 0)

        # 检索召回质量为核心：意图准确率 + 实体召回率 + 关键词匹配率 + 检索召回率
        # 文本相似度仅作参考，不作为主要评判标准
        retrieval_recall = overall_metrics.get('retrieval_recall', 0)
        overall_metrics.update({
            'intent_accuracy': intent_accuracy,
            'entity_recall': entity_recall,
            'keyword_matching': keyword_matching,
            'retrieval_recall': retrieval_recall,
            'answer_relevance': answer_relevance,
            'overall_score': 0.25 * intent_accuracy + 0.25 * entity_recall + 0.20 * keyword_matching + 0.20 * retrieval_recall + 0.10 * answer_relevance
        })

        category_metrics = {}
        categories = set(item.category for item in self.dataset.items)
        for category in categories:
            cat_items = [item for item in self.dataset.items if item.category == category]
            cat_results = [self.results[i] for i, item in enumerate(self.dataset.items) if item.category == category]
            cat_success = [r for r in cat_results if not r.error_occurred]
            
            if cat_success:
                cat_metrics = {}
                for metric_name in primary_metrics + secondary_metrics:
                    values = [r.metrics.get(metric_name, 0) for r in cat_success]
                    cat_metrics[metric_name] = sum(values) / len(values)
                
                cat_intent_acc = sum(1 for r in cat_success if r.intent_correct) / len(cat_success)
                cat_total_exp = sum(r.expected_entities for r in cat_success)
                cat_total_found = sum(r.entities_found for r in cat_success)
                cat_entity_recall = cat_total_found / cat_total_exp if cat_total_exp > 0 else 0
                
                cat_metrics.update({
                    'intent_accuracy': cat_intent_acc,
                    'entity_recall': cat_entity_recall,
                    'keyword_matching': cat_metrics.get('keyword_matching', 0),
                    'retrieval_recall': cat_metrics.get('retrieval_recall', 0),
                    'overall_score': 0.25 * cat_intent_acc + 0.25 * cat_entity_recall + 0.20 * cat_metrics.get('keyword_matching', 0) + 0.20 * cat_metrics.get('retrieval_recall', 0) + 0.10 * cat_metrics.get('semantic_similarity', 0)
                })
                category_metrics[category] = cat_metrics

        times = [r.response_time for r in successful_results]
        execution_summary = {
            'avg_response_time': sum(times) / len(times),
            'max_response_time': max(times),
            'min_response_time': min(times),
            'total_execution_time': sum(times),
            'throughput': len(successful_results) / sum(times) if sum(times) > 0 else 0
        }

        threshold_result = self.threshold_checker.check_all(overall_metrics)

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            dataset_name=self.dataset.name,
            total_items=len(self.results),
            passed=len(successful_results),
            failed=len(self.results) - len(successful_results),
            overall_metrics=overall_metrics,
            category_metrics=category_metrics,
            threshold_result=threshold_result,
            results=self.results,
            execution_summary=execution_summary
        )

    def print_report(self, report: EvaluationReport):
        print("\n" + "=" * 70)
        print("离线评估报告")
        print("=" * 70)
        
        print(f"\n【评估概览】")
        print(f"  评估时间: {report.timestamp}")
        print(f"  数据集: {report.dataset_name}")
        print(f"  总测试数: {report.total_items}")
        print(f"  通过: {report.passed} | 失败: {report.failed}")
        
        print("\n【综合指标】")
        print(f"  意图分类准确率: {report.overall_metrics.get('intent_accuracy', 0) * 100:.1f}%")
        print(f"  实体识别召回率: {report.overall_metrics.get('entity_recall', 0) * 100:.1f}%")
        print(f"  关键词匹配率: {report.overall_metrics.get('keyword_matching', 0) * 100:.1f}%")
        print(f"  检索召回率: {report.overall_metrics.get('retrieval_recall', 0) * 100:.1f}%")
        print(f"  回答相关性: {report.overall_metrics.get('answer_relevance', 0) * 100:.1f}%")
        print(f"  综合评分: {report.overall_metrics.get('overall_score', 0) * 100:.1f}%")

        print("\n【NLP指标（参考）】")
        print(f"  F1分数: {report.overall_metrics.get('f1', 0) * 100:.1f}%")
        print(f"  语义相似度: {report.overall_metrics.get('semantic_similarity', 0) * 100:.1f}%")

        bleu = report.overall_metrics.get('bleu', 0)
        rouge_l = report.overall_metrics.get('rouge_l', 0)
        if bleu > 0 or rouge_l > 0:
            print(f"  BLEU: {bleu * 100:.1f}% | ROUGE-L: {rouge_l * 100:.1f}%")
        
        print("\n【性能指标】")
        print(f"  平均响应时间: {report.execution_summary.get('avg_response_time', 0):.2f}s")
        print(f"  最大响应时间: {report.execution_summary.get('max_response_time', 0):.2f}s")
        print(f"  最小响应时间: {report.execution_summary.get('min_response_time', 0):.2f}s")
        print(f"  吞吐量: {report.execution_summary.get('throughput', 0):.2f} 次/秒")
        
        print("\n【类别评分】")
        for category, metrics in report.category_metrics.items():
            score = metrics.get('overall_score', 0) * 100
            status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
            print(f"  {status} {category}: {score:.1f}%")
        
        print("\n【阈值检查】")
        if report.threshold_result:
            print(f"  结果: {'通过' if report.threshold_result.passed else '未通过'}")
            checks = report.threshold_result.details.get("checks", [])
            for name, actual, threshold, passed in checks:
                symbol = "Y" if passed else "N"
                formatted = f"{actual:.2%}" if isinstance(actual, float) else str(actual)
                print(f"    {symbol} {name}: {formatted}")

        if self.ragas_scores:
            print("\n【RAGAS 安全评估】")
            agg = RagasEvaluator().aggregate_scores(self.ragas_scores)
            print(f"  忠实度(Faithfulness):       {agg.get('avg_faithfulness', 0) * 100:.1f}%")
            print(f"  回答相关性(Answer Relevancy): {agg.get('avg_answer_relevancy', 0) * 100:.1f}%")
            print(f"  回答正确性(Answer Correctness): {agg.get('avg_answer_correctness', 0) * 100:.1f}%")
            print(f"  医疗安全(Medical Safety):    {agg.get('avg_medical_safety', 0) * 100:.1f}%")
            print(f"  RAGAS综合评分:               {agg.get('avg_ragas_overall', 0) * 100:.1f}%")
            print(f"  安全通过率:                  {agg.get('safety_pass_rate', 0) * 100:.1f}%")

        print("\n" + "=" * 70)

    def save_report(self, report: EvaluationReport, output_dir: str = "test_results"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"offline_eval_{timestamp}.json")
        
        report_dict = {
            "timestamp": report.timestamp,
            "dataset_name": report.dataset_name,
            "total_items": report.total_items,
            "passed": report.passed,
            "failed": report.failed,
            "overall_metrics": report.overall_metrics,
            "category_metrics": report.category_metrics,
            "threshold_passed": report.threshold_result.passed if report.threshold_result else False,
            "execution_summary": report.execution_summary,
            "results": [
                {
                    "question": r.question,
                    "reference_answer": r.reference_answer,
                    "model_answer": r.model_answer,
                    "metrics": r.metrics,
                    "intent_correct": r.intent_correct,
                    "entities_found": r.entities_found,
                    "expected_entities": r.expected_entities,
                    "response_time": r.response_time,
                    "error_occurred": r.error_occurred,
                    "error_message": r.error_message
                }
                for r in report.results
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {json_path}")
        return json_path