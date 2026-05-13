import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.knowledge_fusion import EntityDisambiguator, RelationAligner, KnowledgeFusionEngine
from src.ingestion.medical_processor import MedicalTextProcessor
from src.ingestion.text_splitter import TextSplitter, SplitStrategy
from src.chains.medical_intent import MedicalIntentClassifier


@dataclass
class EvaluationResult:
    test_name: str
    category: str
    passed: bool
    expected: Any
    actual: Any
    score: float
    details: str = ""
    execution_time: float = 0.0


@dataclass
class EvaluationReport:
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    overall_score: float
    category_scores: Dict[str, float]
    results: List[EvaluationResult]
    summary: Dict[str, Any]


class MedicalSystemEvaluator:
    def __init__(self):
        self.entity_disambiguator = EntityDisambiguator()
        self.relation_aligner = RelationAligner()
        self.knowledge_fusion_engine = KnowledgeFusionEngine()
        self.medical_processor = MedicalTextProcessor()
        self.text_splitter = TextSplitter(strategy=SplitStrategy.MEDICAL)
        self.intent_classifier = MedicalIntentClassifier()
        self.results: List[EvaluationResult] = []

    def run_evaluation(self) -> EvaluationReport:
        print("=" * 60)
        print("医疗知识图谱系统评估开始")
        print("=" * 60)

        print("\n[1/6] 运行实体消歧评估...")
        self._eval_entity_disambiguation()

        print("[2/6] 运行关系对齐评估...")
        self._eval_relation_alignment()

        print("[3/6] 运行知识融合评估...")
        self._eval_knowledge_fusion()

        print("[4/6] 运行文本分割评估...")
        self._eval_text_splitting()

        print("[5/6] 运行医疗意图分类评估...")
        self._eval_intent_classification()

        print("[6/6] 运行医疗实体提取评估...")
        self._eval_entity_extraction()

        return self._generate_report()

    def _eval_entity_disambiguation(self):
        test_cases = [
            {
                "name": "疾病同义词-高血压变体",
                "input": "HTN",
                "entity_type": "Disease",
                "expected": "高血压",
            },
            {
                "name": "疾病同义词-糖尿病变体",
                "input": "II型糖尿病",
                "entity_type": "Disease",
                "expected": "糖尿病",
            },
            {
                "name": "症状同义词-头痛变体",
                "input": "脑袋疼",
                "entity_type": "Symptom",
                "expected": "头痛",
            },
            {
                "name": "症状同义词-发热变体",
                "input": "发烧",
                "entity_type": "Symptom",
                "expected": "发热",
            },
            {
                "name": "药物同义词-阿司匹林",
                "input": "乙酰水杨酸",
                "entity_type": "Drug",
                "expected": "阿司匹林",
            },
            {
                "name": "药物同义词-布洛芬",
                "input": "ibuprofen",
                "entity_type": "Drug",
                "expected": "布洛芬",
            },
            {
                "name": "症状消歧-咳嗽",
                "input": "干咳",
                "entity_type": "Symptom",
                "expected": "咳嗽",
            },
            {
                "name": "症状消歧-腹泻",
                "input": "拉肚子",
                "entity_type": "Symptom",
                "expected": "腹泻",
            },
            {
                "name": "实体消歧-多表达合并",
                "input_list": ["高血压病", "HTN", "原发性高血压", "高血压"],
                "entity_type": "Disease",
                "expected_count": 1,
            },
            {
                "name": "相似度计算-高相似",
                "input1": "心肌梗死",
                "input2": "心肌梗塞",
                "expected_min": 0.7,
            },
            {
                "name": "相似度计算-低相似",
                "input1": "肺炎",
                "input2": "胃炎",
                "expected_max": 0.5,
            },
        ]

        for tc in test_cases:
            start_time = time.time()
            try:
                if "input_list" in tc:
                    result = self.entity_disambiguator.disambiguate([
                        {"name": name, "type": tc["entity_type"]}
                        for name in tc["input_list"]
                    ])
                    passed = len(result) == tc["expected_count"]
                    actual = f"合并后{len(result)}个实体"
                    score = 1.0 if passed else 0.0
                elif "input1" in tc:
                    score = self.entity_disambiguator.compute_similarity(tc["input1"], tc["input2"])
                    if "expected_min" in tc:
                        passed = score >= tc["expected_min"]
                        actual = f"相似度={score:.3f}"
                    else:
                        passed = score <= tc["expected_max"]
                        actual = f"相似度={score:.3f}"
                    score = 1.0 if passed else 0.0
                else:
                    result = self.entity_disambiguator.normalize_name(tc["input"], tc["entity_type"])
                    passed = result == tc["expected"]
                    actual = result
                    score = 1.0 if passed else 0.0

                execution_time = time.time() - start_time
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="实体消歧",
                    passed=passed,
                    expected=tc.get("expected", tc.get("expected_count", "")),
                    actual=actual,
                    score=score,
                    execution_time=execution_time,
                ))
            except Exception as e:
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="实体消歧",
                    passed=False,
                    expected=tc.get("expected", ""),
                    actual=str(e),
                    score=0.0,
                    execution_time=time.time() - start_time,
                ))

    def _eval_relation_alignment(self):
        test_cases = [
            {
                "name": "关系映射-表现为",
                "input": "表现为",
                "expected": "HAS_SYMPTOM",
            },
            {
                "name": "关系映射-由...引起",
                "input": "由感染引起",
                "expected": "CAUSED_BY",
            },
            {
                "name": "关系映射-用于治疗",
                "input": "适用于",
                "expected": "DRUG_FOR",
            },
            {
                "name": "关系映射-诊断",
                "input": "通过检查诊断",
                "expected": "DIAGNOSED_BY",
            },
            {
                "name": "关系映射-预防",
                "input": "用于预防",
                "expected": "PREVENTS",
            },
            {
                "name": "关系验证-有效关系",
                "input": ("Disease", "Symptom", "HAS_SYMPTOM"),
                "expected": True,
            },
            {
                "name": "关系验证-无效关系",
                "input": ("Symptom", "Disease", "DRUG_FOR"),
                "expected": False,
            },
            {
                "name": "关系验证-另一种有效关系",
                "input": ("Drug", "Disease", "DRUG_FOR"),
                "expected": True,
            },
            {
                "name": "关系验证-药物-疾病",
                "input": ("Drug", "Disease", "DRUG_FOR"),
                "expected": True,
            },
            {
                "name": "关系验证-疾病-并发症",
                "input": ("Disease", "Disease", "COMPLICATED_BY"),
                "expected": True,
            },
        ]

        for tc in test_cases:
            start_time = time.time()
            try:
                if isinstance(tc["input"], tuple):
                    source_type, target_type, relation = tc["input"]
                    result = self.relation_aligner.validate_relation(source_type, target_type, relation)
                    passed = result == tc["expected"]
                    actual = str(result)
                else:
                    result = self.relation_aligner.align_relation(tc["input"])
                    passed = result == tc["expected"]
                    actual = result

                execution_time = time.time() - start_time
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="关系对齐",
                    passed=passed,
                    expected=tc["expected"],
                    actual=actual,
                    score=1.0 if passed else 0.0,
                    execution_time=execution_time,
                ))
            except Exception as e:
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="关系对齐",
                    passed=False,
                    expected=tc["expected"],
                    actual=str(e),
                    score=0.0,
                    execution_time=time.time() - start_time,
                ))

    def _eval_knowledge_fusion(self):
        test_entities = [
            {"name": "高血压", "type": "Disease", "properties": {"department": "心内科"}},
            {"name": "HTN", "type": "Disease", "properties": {}},
            {"name": "高血压病", "type": "Disease", "properties": {}},
            {"name": "头痛", "type": "Symptom", "properties": {}},
            {"name": "头晕", "type": "Symptom", "properties": {}},
            {"name": "阿司匹林", "type": "Drug", "properties": {"dosage": "100mg"}},
            {"name": "乙酰水杨酸", "type": "Drug", "properties": {}},
            {"name": "二甲双胍", "type": "Drug", "properties": {}},
        ]

        test_relationships = [
            {"source": "高血压", "target": "头痛", "type": "HAS_SYMPTOM", "properties": {}},
            {"source": "HTN", "target": "头晕", "type": "HAS_SYMPTOM", "properties": {}},
            {"source": "阿司匹林", "target": "心肌梗死", "type": "DRUG_FOR", "properties": {}},
            {"source": "二甲双胍", "target": "糖尿病", "type": "DRUG_FOR", "properties": {}},
            {"source": "高血压", "target": "心内科", "type": "BELONGS_TO", "properties": {}},
        ]

        start_time = time.time()
        try:
            fused_entities, fused_relationships = self.knowledge_fusion_engine.fuse(
                test_entities, test_relationships
            )

            entity_count_passed = len(fused_entities) <= len(test_entities)
            relation_dedup_passed = len(fused_relationships) < len(test_relationships)

            hypertension_entities = [e for e in fused_entities if "高血压" in e["name"]]
            has_hypertension = len(hypertension_entities) == 1

            aspirin_entities = [e for e in fused_entities if "阿司匹林" in e["name"]]
            has_aspirin = len(aspirin_entities) == 1

            execution_time = time.time() - start_time

            self.results.append(EvaluationResult(
                test_name="知识融合-实体合并",
                category="知识融合",
                passed=has_hypertension and has_aspirin,
                expected="高血压和阿司匹林各合并为1个",
                actual=f"高血压:{len(hypertension_entities)}个, 阿司匹林:{len(aspirin_entities)}个",
                score=1.0 if (has_hypertension and has_aspirin) else 0.5,
                execution_time=execution_time,
            ))

            self.results.append(EvaluationResult(
                test_name="知识融合-关系去重",
                category="知识融合",
                passed=relation_dedup_passed,
                expected="关系数量减少（去重）",
                actual=f"输入{len(test_relationships)}个，输出{len(fused_relationships)}个",
                score=1.0 if relation_dedup_passed else 0.0,
                execution_time=execution_time,
            ))

            icd10_linked = self.knowledge_fusion_engine.link_to_standard_ontology(fused_entities)
            hypertension_icd = next((e for e in icd10_linked if "高血压" in e.get("name", "")), None)
            has_icd = hypertension_icd and hypertension_icd.get("icd10_code")

            self.results.append(EvaluationResult(
                test_name="知识融合-ICD10链接",
                category="知识融合",
                passed=has_icd,
                expected="高血压链接到I10",
                actual=hypertension_icd.get("icd10_code", "未链接") if hypertension_icd else "未找到",
                score=1.0 if has_icd else 0.0,
                execution_time=execution_time,
            ))

        except Exception as e:
            self.results.append(EvaluationResult(
                test_name="知识融合-整体流程",
                category="知识融合",
                passed=False,
                expected="成功融合",
                actual=str(e),
                score=0.0,
                execution_time=time.time() - start_time,
            ))

    def _eval_text_splitting(self):
        medical_text = """
        【病史摘要】
        患者男性，58岁，因"反复头痛头晕1年，加重1周"入院。
        患者1年前无明显诱因出现头痛头晕，呈阵发性胀痛，伴视物模糊，未重视。
        1周前上述症状加重，为求诊治来我院门诊。

        【既往史】
        既往有高血压病史10年，最高血压180/100mmHg，长期口服硝苯地平控释片治疗。
        否认糖尿病、冠心病病史。

        【体格检查】
        T:36.5℃ P:78次/分 R:18次/分 BP:165/95mmHg。
        神志清，精神可，自动体位，查体合作。

        【辅助检查】
        心电图：窦性心律，左室高电压。
        头颅CT：多发腔隙性脑梗死。

        【诊断】
        1. 高血压病3级（极高危）
        2. 腔隙性脑梗死

        【诊疗计划】
        1. 完善相关检查
        2. 降压治疗：氨氯地平片5mg qd
        3. 抗血小板聚集：阿司匹林肠溶片100mg qd
        """

        start_time = time.time()
        try:
            chunks = self.text_splitter.split_text(medical_text, "test_medical_doc")

            has_history = any("病史" in c.content for c in chunks)
            has_diagnosis = any("诊断" in c.content for c in chunks)
            has_treatment = any("治疗" in c.content for c in chunks)

            reasonable_chunk_size = all(len(c.content) <= 1500 for c in chunks)

            execution_time = time.time() - start_time

            self.results.append(EvaluationResult(
                test_name="医疗文本分割-章节识别",
                category="文本分割",
                passed=has_history and has_diagnosis and has_treatment,
                expected="识别病史、诊断、治疗等章节",
                actual=f"识别章节：{'病史' if has_history else ''}{'诊断' if has_diagnosis else ''}{'治疗' if has_treatment else ''}",
                score=1.0 if (has_history and has_diagnosis) else 0.0,
                execution_time=execution_time,
            ))

            self.results.append(EvaluationResult(
                test_name="医疗文本分割-块大小合理",
                category="文本分割",
                passed=reasonable_chunk_size,
                expected="所有块大小<=1500字符",
                actual=f"最大块大小: {max(len(c.content) for c in chunks) if chunks else 0}",
                score=1.0 if reasonable_chunk_size else 0.0,
                execution_time=execution_time,
            ))

            self.results.append(EvaluationResult(
                test_name="医疗文本分割-块数量",
                category="文本分割",
                passed=3 <= len(chunks) <= 15,
                expected="合理的块数量(3-15)",
                actual=f"生成{len(chunks)}个块",
                score=1.0 if (3 <= len(chunks) <= 15) else 0.5,
                execution_time=execution_time,
            ))

        except Exception as e:
            self.results.append(EvaluationResult(
                test_name="医疗文本分割-整体流程",
                category="文本分割",
                passed=False,
                expected="成功分割",
                actual=str(e),
                score=0.0,
                execution_time=time.time() - start_time,
            ))

    def _eval_intent_classification(self):
        test_cases = [
            {
                "name": "意图分类-疾病查询",
                "input": "高血压是什么原因引起的？",
                "expected": "disease_query",
            },
            {
                "name": "意图分类-症状查询",
                "input": "最近总是头痛头晕，睡不着觉",
                "expected": "symptom_query",
            },
            {
                "name": "意图分类-药物查询",
                "input": "阿司匹林有什么副作用？",
                "expected": "drug_query",
            },
            {
                "name": "意图分类-诊断辅助",
                "input": "我最近总是头痛头晕，可能是什么病？",
                "expected": "diagnosis_assist",
            },
        ]

        for tc in test_cases:
            start_time = time.time()
            try:
                result = self.intent_classifier.classify(tc["input"])
                actual_intent = result.intent.value if hasattr(result.intent, 'value') else str(result.intent)
                passed = actual_intent == tc["expected"]

                execution_time = time.time() - start_time
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="意图分类",
                    passed=passed,
                    expected=tc["expected"],
                    actual=actual_intent,
                    score=1.0 if passed else 0.0,
                    execution_time=execution_time,
                ))
            except Exception as e:
                self.results.append(EvaluationResult(
                    test_name=tc["name"],
                    category="意图分类",
                    passed=False,
                    expected=tc["expected"],
                    actual=str(e),
                    score=0.0,
                    execution_time=time.time() - start_time,
                ))

    def _eval_entity_extraction(self):
        test_text = """
        患者因胸痛入院，既往有高血压病史10年，糖尿病史5年。
        入院后查体：体温38.5℃，脉搏90次/分，呼吸20次/分，血压150/90mmHg。
        辅助检查：心电图显示ST段抬高，心肌酶谱升高。
        诊断：急性心肌梗死。
        治疗：给予阿司匹林300mg口服，氯吡格雷75mg，以及低分子肝素抗凝治疗。
        """

        start_time = time.time()
        try:
            try:
                entities = self.medical_processor.extract_medical_entities(test_text)
            except Exception as nlp_error:
                print(f"    [警告] NLP模型加载失败，使用规则提取: {nlp_error}")
                entities = self.medical_processor._basic_entity_extraction(test_text)

            disease_count = len([e for e in entities if e["type"] == "DISEASE"])
            symptom_count = len([e for e in entities if e["type"] == "SYMPTOM"])
            drug_count = len([e for e in entities if e["type"] == "DRUG"])

            has_hypertension = any("高血压" in e["text"] for e in entities)
            has_diabetes = any("糖尿病" in e["text"] for e in entities)
            has_aspirin = any("阿司匹林" in e["text"] for e in entities)

            execution_time = time.time() - start_time

            self.results.append(EvaluationResult(
                test_name="实体提取-疾病识别",
                category="实体提取",
                passed=disease_count >= 2,
                expected="至少识别2个疾病实体",
                actual=f"识别{disease_count}个疾病: {[e['text'] for e in entities if e['type']=='DISEASE']}",
                score=min(1.0, disease_count / 2),
                execution_time=execution_time,
            ))

            self.results.append(EvaluationResult(
                test_name="实体提取-药物识别",
                category="实体提取",
                passed=drug_count >= 2,
                expected="至少识别2个药物实体",
                actual=f"识别{drug_count}个药物: {[e['text'] for e in entities if e['type']=='DRUG']}",
                score=min(1.0, drug_count / 2),
                execution_time=execution_time,
            ))

            self.results.append(EvaluationResult(
                test_name="实体提取-关键实体",
                category="实体提取",
                passed=has_hypertension and has_diabetes and has_aspirin,
                expected="高血压、糖尿病、阿司匹林",
                actual=f"高血压:{has_hypertension}, 糖尿病:{has_diabetes}, 阿司匹林:{has_aspirin}",
                score=1.0 if (has_hypertension and has_diabetes and has_aspirin) else 0.0,
                execution_time=execution_time,
            ))

        except Exception as e:
            self.results.append(EvaluationResult(
                test_name="实体提取-整体流程",
                category="实体提取",
                passed=False,
                expected="成功提取",
                actual=str(e),
                score=0.0,
                execution_time=time.time() - start_time,
            ))

    def _generate_report(self) -> EvaluationReport:
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        overall_score = sum(r.score for r in self.results) / len(self.results) * 100

        category_scores = {}
        for category in set(r.category for r in self.results):
            cat_results = [r for r in self.results if r.category == category]
            cat_score = sum(r.score for r in cat_results) / len(cat_results) * 100
            category_scores[category] = round(cat_score, 2)

        report = EvaluationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            overall_score=round(overall_score, 2),
            category_scores=category_scores,
            results=self.results,
            summary={
                "avg_execution_time": round(
                    sum(r.execution_time for r in self.results) / len(self.results), 3
                ),
                "slowest_test": max(self.results, key=lambda r: r.execution_time).test_name if self.results else "",
                "fastest_test": min(self.results, key=lambda r: r.execution_time).test_name if self.results else "",
            },
        )

        return report


def print_report(report: EvaluationReport):
    print("\n" + "=" * 60)
    print("医疗知识图谱系统评估报告")
    print("=" * 60)
    print(f"评估时间: {report.timestamp}")
    print(f"总测试数: {report.total_tests}")
    print(f"通过: {report.passed} | 失败: {report.failed}")
    print(f"总分: {report.overall_score}%")

    print("\n--- 按类别评分 ---")
    for category, score in report.category_scores.items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
        print(f"  {status} {category:10s} [{bar}] {score:.1f}%")

    print("\n--- 详细结果 ---")
    for category in sorted(set(r.category for r in report.results)):
        print(f"\n【{category}】")
        cat_results = [r for r in report.results if r.category == category]
        for r in cat_results:
            status = "✅" if r.passed else "❌"
            print(f"  {status} {r.test_name}")
            if not r.passed:
                print(f"     期望: {r.expected}")
                print(f"     实际: {r.actual}")

    print("\n--- 性能统计 ---")
    print(f"  平均执行时间: {report.summary['avg_execution_time']:.3f}s")
    print(f"  最慢测试: {report.summary['slowest_test']}")
    print(f"  最快测试: {report.summary['fastest_test']}")

    print("\n" + "=" * 60)
    return report


def save_report(report: EvaluationReport, output_dir: str = "test_results"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"eval_results_{timestamp}.json"
    md_path = output_path / f"eval_report_{timestamp}.md"

    report_dict = {
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed": report.passed,
        "failed": report.failed,
        "overall_score": report.overall_score,
        "category_scores": report.category_scores,
        "summary": report.summary,
        "results": [
            {
                "test_name": r.test_name,
                "category": r.category,
                "passed": r.passed,
                "expected": str(r.expected),
                "actual": str(r.actual),
                "score": r.score,
                "execution_time": r.execution_time,
            }
            for r in report.results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    md_content = f"""# 医疗知识图谱系统评估报告

## 评估概览

| 指标 | 值 |
|------|-----|
| 评估时间 | {report.timestamp} |
| 总测试数 | {report.total_tests} |
| 通过 | {report.passed} |
| 失败 | {report.failed} |
| **总分** | **{report.overall_score}%** |

## 类别评分

| 类别 | 评分 | 状态 |
|------|------|------|
"""
    for category, score in report.category_scores.items():
        status = "✅ 通过" if score >= 70 else "⚠️ 警告" if score >= 50 else "❌ 失败"
        md_content += f"| {category} | {score}% | {status} |\n"

    md_content += f"""
## 详细测试结果

"""
    for category in sorted(set(r.category for r in report.results)):
        cat_results = [r for r in report.results if r.category == category]
        cat_passed = sum(1 for r in cat_results if r.passed)
        md_content += f"### {category} ({cat_passed}/{len(cat_results)} 通过)\n\n"
        md_content += "| 测试名称 | 状态 | 期望值 | 实际值 | 得分 |\n"
        md_content += "|----------|------|--------|--------|------|\n"
        for r in cat_results:
            status = "✅" if r.passed else "❌"
            md_content += f"| {r.test_name} | {status} | {r.expected} | {r.actual} | {r.score:.1f} |\n"
        md_content += "\n"

    md_content += f"""
## 性能统计

- 平均执行时间: {report.summary['avg_execution_time']:.3f}s
- 最慢测试: {report.summary['slowest_test']}
- 最快测试: {report.summary['fastest_test']}

## 评估结论

"""
    if report.overall_score >= 80:
        md_content += "**系统表现优秀**，各项功能正常运行，建议投入生产使用。\n"
    elif report.overall_score >= 60:
        md_content += "**系统表现良好**，部分功能需要优化，建议根据实际情况改进后使用。\n"
    else:
        md_content += "**系统表现欠佳**，需要重点优化后再进行生产部署。\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n报告已保存至:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    return json_path, md_path


if __name__ == "__main__":
    evaluator = MedicalSystemEvaluator()
    report = evaluator.run_evaluation()
    print_report(report)
    save_report(report)
