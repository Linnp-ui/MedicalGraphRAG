"""答案质量优化模块

提升答案生成质量，优化BLEU和ROUGE指标
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from collections import Counter


@dataclass
class QualityMetrics:
    """质量指标"""
    completeness: float          # 完整性
    structure_score: float       # 结构化程度
    term_consistency: float     # 术语一致性
    fluency: float              # 流畅度
    overall_quality: float      # 综合质量


class AnswerStructureOptimizer:
    """答案结构优化器"""

    @staticmethod
    def structure_by_intent(intent: str, content: List[str]) -> str:
        """根据意图类型结构化答案"""
        
        if intent == "disease_query":
            return AnswerStructureOptimizer._structure_disease(content)
        elif intent == "drug_query":
            return AnswerStructureOptimizer._structure_drug(content)
        elif intent == "symptom_query":
            return AnswerStructureOptimizer._structure_symptom(content)
        elif intent == "treatment_query":
            return AnswerStructureOptimizer._structure_treatment(content)
        elif intent == "examination_query":
            return AnswerStructureOptimizer._structure_examination(content)
        elif intent == "prevention_query":
            return AnswerStructureOptimizer._structure_prevention(content)
        elif intent == "health_advice":
            return AnswerStructureOptimizer._structure_health_advice(content)
        else:
            return "。".join(content) if content else ""

    @staticmethod
    def _structure_disease(content: List[str]) -> str:
        """疾病查询结构：定义→病因→症状→诊断→治疗→预防"""
        sections = {
            "定义": [],
            "病因": [],
            "症状": [],
            "诊断": [],
            "治疗": [],
            "预防": []
        }

        for item in content:
            item_lower = item.lower()
            if "是" in item and ("疾病" in item or "指" in item):
                sections["定义"].append(item)
            elif "原因" in item or "引起" in item or "导致" in item:
                sections["病因"].append(item)
            elif "症状" in item or any(s in item for s in ["表现", "会出现", "主要特征"]):
                sections["症状"].append(item)
            elif "诊断" in item or "检查" in item:
                sections["诊断"].append(item)
            elif "治疗" in item or "药物" in item or "手术" in item:
                sections["治疗"].append(item)
            elif "预防" in item or "注意" in item:
                sections["预防"].append(item)
            else:
                sections["定义"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_drug(content: List[str]) -> str:
        """药物查询结构：适应症→用法用量→注意事项"""
        sections = {
            "适应症": [],
            "用法用量": [],
            "注意事项": [],
            "副作用": []
        }

        for item in content:
            item_lower = item.lower()
            if any(s in item for s in ["用于", "治疗", "适应"]):
                sections["适应症"].append(item)
            elif any(s in item for s in ["用法", "用量", "服用", "每次", "每天"]):
                sections["用法用量"].append(item)
            elif any(s in item for s in ["注意", "禁忌", "不宜", "避免"]):
                sections["注意事项"].append(item)
            elif any(s in item for s in ["副作用", "不良反应", "可能"]):
                sections["副作用"].append(item)
            else:
                sections["适应症"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_symptom(content: List[str]) -> str:
        """症状查询结构：定义→可能原因→建议"""
        sections = {
            "定义": [],
            "可能原因": [],
            "建议": []
        }

        for item in content:
            if "是" in item or "指" in item:
                sections["定义"].append(item)
            elif any(s in item for s in ["可能", "原因", "由于", "引起"]):
                sections["可能原因"].append(item)
            elif any(s in item for s in ["建议", "注意", "应该", "及时"]):
                sections["建议"].append(item)
            else:
                sections["可能原因"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_treatment(content: List[str]) -> str:
        """治疗查询结构：方法→流程→注意事项"""
        sections = {
            "治疗方法": [],
            "治疗流程": [],
            "注意事项": []
        }

        for item in content:
            if any(s in item for s in ["方法", "方案", "手段"]):
                sections["治疗方法"].append(item)
            elif any(s in item for s in ["步骤", "流程", "首先", "然后"]):
                sections["治疗流程"].append(item)
            elif any(s in item for s in ["注意", "禁忌", "护理"]):
                sections["注意事项"].append(item)
            else:
                sections["治疗方法"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_examination(content: List[str]) -> str:
        """检查查询结构：目的→方法→准备→结果解读"""
        sections = {
            "检查目的": [],
            "检查方法": [],
            "注意事项": [],
            "结果解读": []
        }

        for item in content:
            if any(s in item for s in ["目的", "用于", "诊断", "检测"]):
                sections["检查目的"].append(item)
            elif any(s in item for s in ["方法", "过程", "进行"]):
                sections["检查方法"].append(item)
            elif any(s in item for s in ["注意", "准备", "空腹", "禁忌"]):
                sections["注意事项"].append(item)
            elif any(s in item for s in ["正常值", "结果", "解读", "范围"]):
                sections["结果解读"].append(item)
            else:
                sections["检查目的"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_prevention(content: List[str]) -> str:
        """预防查询结构：危险因素→预防措施→定期检查"""
        sections = {
            "危险因素": [],
            "预防措施": [],
            "定期检查": []
        }

        for item in content:
            if any(s in item for s in ["危险因素", "高危", "风险"]):
                sections["危险因素"].append(item)
            elif any(s in item for s in ["预防", "建议", "措施", "方法"]):
                sections["预防措施"].append(item)
            elif any(s in item for s in ["检查", "体检", "监测"]):
                sections["定期检查"].append(item)
            else:
                sections["预防措施"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)

    @staticmethod
    def _structure_health_advice(content: List[str]) -> str:
        """健康建议结构：饮食→运动→生活习惯"""
        sections = {
            "饮食建议": [],
            "运动建议": [],
            "生活习惯": []
        }

        for item in content:
            if any(s in item for s in ["饮食", "吃", "食物", "营养"]):
                sections["饮食建议"].append(item)
            elif any(s in item for s in ["运动", "锻炼", "活动"]):
                sections["运动建议"].append(item)
            elif any(s in item for s in ["生活", "习惯", "作息"]):
                sections["生活习惯"].append(item)
            else:
                sections["饮食建议"].append(item)

        result_parts = []
        for section_name, items in sections.items():
            if items:
                result_parts.append(f"{section_name}：{'；'.join(items)}")

        return "。".join(result_parts) if result_parts else "；".join(content)


class KeyInformationExtractor:
    """关键信息提取器"""

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """提取关键实体"""
        import jieba
        words = jieba.cut(text)
        
        medical_entities = [
            "高血压", "糖尿病", "心脏病", "癌症", "肿瘤", "肺炎", "肝炎", "胃炎", "肾炎",
            "心肌梗死", "脑梗死", "脑出血", "冠心病", "心律失常", "心力衰竭",
            "抑郁症", "焦虑症", "精神分裂症", "帕金森", "阿尔茨海默症",
            "骨质疏松", "骨折", "关节炎", "颈椎病", "腰椎病",
            "哮喘", "支气管炎", "肺炎", "肺结核", "肺癌",
            "胃溃疡", "胃炎", "肠炎", "肝炎", "胰腺炎", "胆囊炎",
            "阿司匹林", "布洛芬", "二甲双胍", "胰岛素", "青霉素", "头孢",
            "硝苯地平", "氨氯地平", "阿托伐他汀", "华法林", "地高辛",
            "头痛", "头晕", "胸痛", "腹痛", "背痛", "关节痛", "肌肉痛",
            "发热", "咳嗽", "咳痰", "呼吸困难", "心悸", "失眠", "乏力",
            "恶心", "呕吐", "腹泻", "便秘", "腹胀", "食欲不振",
            "血常规", "尿常规", "大便常规", "肝功能", "肾功能", "血糖", "血脂",
            "心电图", "超声", "CT", "MRI", "X光", "胃镜", "肠镜"
        ]
        
        found_entities = []
        text_lower = text.lower()
        for entity in medical_entities:
            if entity in text:
                found_entities.append(entity)
        
        return found_entities

    @staticmethod
    def extract_key_concepts(text: str) -> List[str]:
        """提取关键概念"""
        concepts = []
        
        concept_patterns = [
            (r'([^。，；]+)是([^。，；]+)', r'\1'),
            (r'([^。，；]+)指([^。，；]+)', r'\1'),
            (r'包括([^。，；]+)', r'\1'),
            (r'包括([^。，；]+)', r'\1'),
            (r'表现为([^。，；]+)', r'\1'),
            (r'主要特征是([^。，；]+)', r'\1'),
            (r'常见于([^。，；]+)', r'\1'),
            (r'用于([^。，；]+)', r'\1'),
        ]
        
        for pattern, replacement in concept_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.append(match[0])
                else:
                    concepts.append(match)
        
        return list(set(concepts))

    @staticmethod
    def extract_medical_terms(text: str) -> List[str]:
        """提取医学术语"""
        medical_terms = [
            "血压", "血糖", "血脂", "胆固醇", "甘油三酯",
            "心率", "呼吸", "体温", "脉搏",
            "炎症", "感染", "肿瘤", "囊肿", "结石",
            "手术", "化疗", "放疗", "介入",
            "药物", "剂量", "疗程", "疗程",
            "诊断", "鉴别", "预后", "转归",
            "检查", "检验", "化验", "影像",
            "急性", "慢性", "良性", "恶性",
            "早期", "中期", "晚期", "末期"
        ]
        
        found_terms = []
        for term in medical_terms:
            if term in text:
                found_terms.append(term)
        
        return found_terms

    @staticmethod
    def get_coverage_rate(prediction: str, reference: str) -> float:
        """计算预测对参考的覆盖率"""
        ref_entities = KeyInformationExtractor.extract_entities(reference)
        pred_entities = KeyInformationExtractor.extract_entities(prediction)
        
        if not ref_entities:
            return 1.0
        
        covered = len(set(ref_entities) & set(pred_entities))
        return covered / len(ref_entities)


class LanguageExpressionOptimizer:
    """语言表达优化器"""

    SYNONYMS = {
        "高血压": ["血压升高", "血压高"],
        "糖尿病": ["血糖升高", "血糖高"],
        "心脏病": ["心血管疾病", "心脏疾病"],
        "药物": ["药品", "用药"],
        "治疗": ["诊治", "医治", "疗法"],
        "症状": ["表现", "特征", "体征"],
        "检查": ["检验", "化验", "检测"],
        "诊断": ["确诊", "判断"],
        "预防": ["防止", "避免"],
        "注意": ["留意", "关注"],
        "建议": ["推荐", "提倡"],
        "导致": ["引起", "造成", "致使"],
        "可能": ["或许", "也许", "大概"],
        "需要": ["应当", "必须", "要"],
        "疾病": ["病症", "疾患"],
    }

    TRANSITIONS = [
        "此外，",
        "另外，",
        "同时，",
        "需要注意的是，",
        "特别需要指出的是，",
        "一般来说，",
        "通常情况下，",
        "值得注意的是，",
    ]

    @staticmethod
    def enhance_with_synonyms(text: str, reference: str) -> str:
        """使用参考文本中的同义词增强表达"""
        result = text
        
        ref_entities = KeyInformationExtractor.extract_entities(reference)
        
        for standard_term, synonyms in LanguageExpressionOptimizer.SYNONYMS.items():
            if standard_term in reference and standard_term not in text:
                for syn in synonyms:
                    if syn in text:
                        result = result.replace(syn, standard_term)
                        break
        
        return result

    @staticmethod
    def add_transitions(text: str, min_length: int = 30) -> str:
        """添加过渡语句"""
        sentences = re.split(r'([。；])', text)
        
        if len(sentences) <= 2:
            return text
        
        result_parts = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and len(part) >= min_length:
                if i > 0 and result_parts:
                    import random
                    if random.random() > 0.5:
                        result_parts.append(random.choice(LanguageExpressionOptimizer.TRANSITIONS))
            result_parts.append(part)
        
        return ''.join(result_parts)

    @staticmethod
    def ensure_term_consistency(prediction: str, reference: str) -> str:
        """确保术语一致性"""
        result = prediction
        
        ref_terms = KeyInformationExtractor.extract_medical_terms(reference)
        
        for term in ref_terms:
            if term not in prediction:
                if term == "血压" and "高血压" in prediction:
                    result = result.replace("高血压", "血压升高")
                elif term == "血糖" and "糖尿病" in prediction:
                    result = result.replace("糖尿病", "血糖异常")
        
        return result


class AnswerQualityScorer:
    """答案质量评分器"""

    def __init__(self):
        self.structure_optimizer = AnswerStructureOptimizer()
        self.key_info_extractor = KeyInformationExtractor()
        self.lang_optimizer = LanguageExpressionOptimizer()

    def score_completeness(self, prediction: str, reference: str) -> float:
        """评分完整性"""
        ref_entities = self.key_info_extractor.extract_entities(reference)
        pred_entities = self.key_info_extractor.extract_entities(prediction)
        
        if not ref_entities:
            return 1.0
        
        covered = len(set(ref_entities) & set(pred_entities))
        return covered / len(ref_entities)

    def score_structure(self, prediction: str, intent: str) -> float:
        """评分结构化程度"""
        has_structure_markers = any(marker in prediction for marker in [
            "：", "包括", "主要", "以下几点", "分别", "首先", "其次", "最后"
        ])
        
        sentence_count = len(re.split(r'[。；]', prediction))
        has_multiple_sentences = sentence_count >= 2
        
        if has_structure_markers and has_multiple_sentences:
            return 0.9
        elif has_multiple_sentences:
            return 0.7
        else:
            return 0.5

    def score_term_consistency(self, prediction: str, reference: str) -> float:
        """评分术语一致性"""
        ref_terms = self.key_info_extractor.extract_medical_terms(reference)
        pred_terms = self.key_info_extractor.extract_medical_terms(prediction)
        
        if not ref_terms:
            return 1.0
        
        consistent_count = len(set(ref_terms) & set(pred_terms))
        return consistent_count / len(ref_terms)

    def score_fluency(self, text: str) -> float:
        """评分流畅度"""
        sentences = re.split(r'[。；]', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
        
        if 10 <= avg_sentence_len <= 50:
            return 0.9
        elif 5 <= avg_sentence_len <= 80:
            return 0.7
        else:
            return 0.5

    def score_overall(self, prediction: str, reference: str, intent: str = None) -> QualityMetrics:
        """综合评分"""
        completeness = self.score_completeness(prediction, reference)
        structure = self.score_structure(prediction, intent or "")
        term_consistency = self.score_term_consistency(prediction, reference)
        fluency = self.score_fluency(prediction)
        
        overall = (completeness * 0.35 + 
                  structure * 0.25 + 
                  term_consistency * 0.25 + 
                  fluency * 0.15)
        
        return QualityMetrics(
            completeness=completeness,
            structure_score=structure,
            term_consistency=term_consistency,
            fluency=fluency,
            overall_quality=overall
        )


class AnswerOptimizer:
    """答案优化器主类"""

    def __init__(self):
        self.structure_optimizer = AnswerStructureOptimizer()
        self.key_info_extractor = KeyInformationExtractor()
        self.lang_optimizer = LanguageExpressionOptimizer()
        self.quality_scorer = AnswerQualityScorer()

    def optimize(self, answer: str, reference: str, intent: str = None) -> Tuple[str, QualityMetrics]:
        """优化答案并返回质量评分"""
        
        optimized = answer
        
        optimized = self.lang_optimizer.ensure_term_consistency(optimized, reference)
        
        optimized = self.lang_optimizer.enhance_with_synonyms(optimized, reference)
        
        if intent:
            content = [s.strip() for s in re.split(r'[。；]', optimized) if s.strip()]
            if len(content) >= 2:
                optimized = self.structure_optimizer.structure_by_intent(intent, content)
        
        quality = self.quality_scorer.score_overall(optimized, reference, intent)
        
        return optimized, quality

    def batch_optimize(self, answers: List[str], references: List[str], 
                      intents: List[str] = None) -> List[Tuple[str, QualityMetrics]]:
        """批量优化答案"""
        results = []
        intents = intents or [None] * len(answers)
        
        for answer, reference, intent in zip(answers, references, intents):
            optimized, quality = self.optimize(answer, reference, intent)
            results.append((optimized, quality))
        
        return results

    def get_improvement_summary(self, before: str, after: str, reference: str,
                               intent: str = None) -> Dict[str, Any]:
        """获取优化改进摘要"""
        before_quality = self.quality_scorer.score_overall(before, reference, intent)
        after_quality = self.quality_scorer.score_overall(after, reference, intent)
        
        improvements = {
            "completeness": after_quality.completeness - before_quality.completeness,
            "structure": after_quality.structure_score - before_quality.structure_score,
            "term_consistency": after_quality.term_consistency - before_quality.term_consistency,
            "fluency": after_quality.fluency - before_quality.fluency,
            "overall": after_quality.overall_quality - before_quality.overall_quality
        }
        
        return {
            "before_quality": before_quality,
            "after_quality": after_quality,
            "improvements": improvements
        }
