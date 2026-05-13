"""问答功能全面评估框架"""
import json
import time
import sys
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from src.chains.qa_chain import QAChain
from src.chains.medical_intent import MedicalIntentClassifier
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever


@dataclass
class TestCase:
    question: str
    expected_intent: str
    expected_entities: List[str]
    expected_answer_keywords: List[str]
    context: str = ""


@dataclass
class EvaluationResult:
    test_case: TestCase
    intent_correct: bool
    entities_found: int
    answer_relevant: bool
    response_time: float
    error_occurred: bool
    error_message: str = ""


@dataclass
class PerformanceMetrics:
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    p95_response_time: float
    throughput: float


@dataclass
class QualityMetrics:
    intent_accuracy: float
    entity_recall: float
    answer_relevance: float
    overall_score: float


class QAEvaluator:
    def __init__(self):
        self.qa_chain = QAChain()
        self.intent_classifier = MedicalIntentClassifier()
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self.results: List[EvaluationResult] = []

    def load_test_cases(self) -> List[TestCase]:
        return [
            TestCase(
                question="高血压是什么疾病？",
                expected_intent="disease_query",
                expected_entities=["高血压"],
                expected_answer_keywords=["高血压", "血压", "慢性", "疾病"],
                context="高血压是一种常见的慢性疾病，指血液在血管中流动时对血管壁造成的压力持续高于正常水平。"
            ),
            TestCase(
                question="头痛有哪些原因？",
                expected_intent="symptom_query",
                expected_entities=["头痛"],
                expected_answer_keywords=["头痛", "原因", "高血压", "疾病"],
                context="头痛可能由多种原因引起，包括高血压、偏头痛、颈椎病、感冒等。"
            ),
            TestCase(
                question="阿司匹林有什么副作用？",
                expected_intent="drug_query",
                expected_entities=["阿司匹林"],
                expected_answer_keywords=["阿司匹林", "副作用", "胃肠道"],
                context="阿司匹林常见副作用包括胃肠道不适、出血风险、过敏反应等。"
            ),
            TestCase(
                question="我最近经常头痛头晕，可能是什么病？",
                expected_intent="diagnosis_assist",
                expected_entities=["头痛", "头晕"],
                expected_answer_keywords=["头痛", "头晕", "可能", "疾病"],
                context="头痛头晕可能与高血压、颈椎病、贫血、低血糖等多种疾病有关。"
            ),
            TestCase(
                question="糖尿病如何治疗？",
                expected_intent="disease_query",
                expected_entities=["糖尿病"],
                expected_answer_keywords=["糖尿病", "治疗", "胰岛素", "饮食"],
                context="糖尿病治疗包括饮食控制、运动、药物治疗（如胰岛素、二甲双胍）等。"
            ),
            TestCase(
                question="布洛芬能治疗什么？",
                expected_intent="drug_query",
                expected_entities=["布洛芬"],
                expected_answer_keywords=["布洛芬", "治疗", "疼痛", "发热"],
                context="布洛芬用于缓解轻至中度疼痛如头痛、关节痛等，也用于退热。"
            ),
            TestCase(
                question="心肌梗死有什么症状？",
                expected_intent="disease_query",
                expected_entities=["心肌梗死"],
                expected_answer_keywords=["心肌梗死", "症状", "胸痛"],
                context="心肌梗死典型症状为突发胸痛、胸闷、呼吸困难、大汗等。"
            ),
            TestCase(
                question="感冒发烧怎么办？",
                expected_intent="symptom_query",
                expected_entities=["感冒", "发烧"],
                expected_answer_keywords=["感冒", "发烧", "治疗", "休息"],
                context="感冒发烧应注意休息、多喝水，必要时使用退烧药如布洛芬。"
            ),
        ]

    def evaluate_intent(self, question: str, expected_intent: str) -> bool:
        """评估意图分类准确性"""
        try:
            result = self.intent_classifier.classify(question)
            return result.intent.value == expected_intent
        except Exception:
            return False

    def evaluate_entities(self, question: str, expected_entities: List[str]) -> int:
        """评估实体识别召回率"""
        try:
            result = self.intent_classifier.classify(question)
            found = 0
            for entity in expected_entities:
                if entity in result.entities:
                    found += 1
            return found
        except Exception:
            return 0

    def evaluate_answer_relevance(self, answer: str, keywords: List[str]) -> bool:
        """评估回答相关性"""
        if not answer or not keywords:
            return False
        matched = sum(1 for kw in keywords if kw in answer)
        return matched >= len(keywords) // 2

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """评估单个测试用例"""
        start_time = time.time()
        
        try:
            intent_correct = self.evaluate_intent(test_case.question, test_case.expected_intent)
            entities_found = self.evaluate_entities(test_case.question, test_case.expected_entities)
            
            answer = self.qa_chain.answer(test_case.question, test_case.context)
            answer_relevant = self.evaluate_answer_relevance(answer, test_case.expected_answer_keywords)
            
            response_time = time.time() - start_time
            
            return EvaluationResult(
                test_case=test_case,
                intent_correct=intent_correct,
                entities_found=entities_found,
                answer_relevant=answer_relevant,
                response_time=response_time,
                error_occurred=False
            )
        except Exception as e:
            response_time = time.time() - start_time
            return EvaluationResult(
                test_case=test_case,
                intent_correct=False,
                entities_found=0,
                answer_relevant=False,
                response_time=response_time,
                error_occurred=True,
                error_message=str(e)
            )

    def run_evaluation(self) -> Tuple[QualityMetrics, PerformanceMetrics]:
        """运行完整评估"""
        test_cases = self.load_test_cases()
        
        for tc in test_cases:
            print(f"评估: {tc.question}")
            result = self.evaluate_single(tc)
            self.results.append(result)
            
        return self.calculate_metrics()

    def calculate_metrics(self) -> Tuple[QualityMetrics, PerformanceMetrics]:
        """计算评估指标"""
        results = [r for r in self.results if not r.error_occurred]
        
        if not results:
            return (
                QualityMetrics(intent_accuracy=0, entity_recall=0, answer_relevance=0, overall_score=0),
                PerformanceMetrics(avg_response_time=0, max_response_time=0, min_response_time=0, p95_response_time=0, throughput=0)
            )

        # 质量指标
        intent_accuracy = sum(1 for r in results if r.intent_correct) / len(results)
        
        total_expected = sum(len(r.test_case.expected_entities) for r in results)
        total_found = sum(r.entities_found for r in results)
        entity_recall = total_found / total_expected if total_expected > 0 else 0
        
        answer_relevance = sum(1 for r in results if r.answer_relevant) / len(results)
        
        overall_score = (intent_accuracy + entity_recall + answer_relevance) / 3

        # 性能指标
        times = [r.response_time for r in results]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        times_sorted = sorted(times)
        p95_index = int(len(times_sorted) * 0.95)
        p95_time = times_sorted[p95_index] if p95_index < len(times_sorted) else max_time
        throughput = len(results) / sum(times)

        return (
            QualityMetrics(
                intent_accuracy=intent_accuracy,
                entity_recall=entity_recall,
                answer_relevance=answer_relevance,
                overall_score=overall_score
            ),
            PerformanceMetrics(
                avg_response_time=avg_time,
                max_response_time=max_time,
                min_response_time=min_time,
                p95_response_time=p95_time,
                throughput=throughput
            )
        )

    def print_report(self, quality: QualityMetrics, performance: PerformanceMetrics):
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("问答功能评估报告")
        print("=" * 70)
        
        print("\n【质量指标】")
        print(f"  意图分类准确率: {quality.intent_accuracy * 100:.1f}%")
        print(f"  实体识别召回率: {quality.entity_recall * 100:.1f}%")
        print(f"  回答相关性: {quality.answer_relevance * 100:.1f}%")
        print(f"  综合评分: {quality.overall_score * 100:.1f}%")
        
        print("\n【性能指标】")
        print(f"  平均响应时间: {performance.avg_response_time:.2f}s")
        print(f"  最大响应时间: {performance.max_response_time:.2f}s")
        print(f"  最小响应时间: {performance.min_response_time:.2f}s")
        print(f"  P95响应时间: {performance.p95_response_time:.2f}s")
        print(f"  吞吐量: {performance.throughput:.2f} 次/秒")
        
        print("\n【详细结果】")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.intent_correct and result.answer_relevant else "❌"
            print(f"  {status} {i}. {result.test_case.question}")
            if not result.intent_correct:
                print(f"     - 意图错误: 期望 {result.test_case.expected_intent}")
            if result.entities_found < len(result.test_case.expected_entities):
                print(f"     - 实体识别: 找到 {result.entities_found}/{len(result.test_case.expected_entities)}")
            print(f"     - 响应时间: {result.response_time:.2f}s")
            if result.error_occurred:
                print(f"     - 错误: {result.error_message}")
        
        print("\n" + "=" * 70)

    def save_report(self, quality: QualityMetrics, performance: PerformanceMetrics):
        """保存评估报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": {
                "intent_accuracy": quality.intent_accuracy,
                "entity_recall": quality.entity_recall,
                "answer_relevance": quality.answer_relevance,
                "overall_score": quality.overall_score
            },
            "performance_metrics": {
                "avg_response_time": performance.avg_response_time,
                "max_response_time": performance.max_response_time,
                "min_response_time": performance.min_response_time,
                "p95_response_time": performance.p95_response_time,
                "throughput": performance.throughput
            },
            "detailed_results": [
                {
                    "question": r.test_case.question,
                    "intent_correct": r.intent_correct,
                    "entities_found": r.entities_found,
                    "expected_entities": len(r.test_case.expected_entities),
                    "answer_relevant": r.answer_relevant,
                    "response_time": r.response_time,
                    "error_occurred": r.error_occurred,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        report_path = os.path.join(os.path.dirname(__file__), f"qa_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {report_path}")


def main():
    evaluator = QAEvaluator()
    print("开始问答功能评估...")
    
    quality, performance = evaluator.run_evaluation()
    evaluator.print_report(quality, performance)
    evaluator.save_report(quality, performance)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
