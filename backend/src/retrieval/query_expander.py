from typing import List, Dict, Optional, Set
import re
from loguru import logger


MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    "高血压": ["血压升高", "血压高", "hypertension", "高血压病"],
    "糖尿病": ["血糖升高", "血糖高", "糖尿病mellitus", "DM"],
    "感冒": ["上呼吸道感染", "伤风", "普通感冒", "cold"],
    "流感": ["流行性感冒", "flu", "influenza"],
    "肺炎": ["肺部感染", "肺感染", "pneumonia"],
    "心肌梗死": ["心梗", "心肌梗塞", "heart attack", "MI"],
    "脑梗死": ["脑梗", "缺血性脑卒中", "脑梗塞", "cerebral infarction"],
    "脑出血": ["脑溢血", "脑内出血", "cerebral hemorrhage"],
    "冠心病": ["冠状动脉粥样硬化性心脏病", "冠状动脉疾病", "CAD"],
    "心力衰竭": ["心衰", "心功能衰竭", "心脏衰竭", "heart failure"],
    "心律失常": ["心律不齐", "心脏节律异常", "arrhythmia"],
    "抑郁症": ["抑郁", "忧郁症", "depression"],
    "焦虑症": ["焦虑", "焦虑障碍", "anxiety"],
    "哮喘": ["支气管哮喘", "气喘", "asthma"],
    "慢性咽炎": ["咽炎", "慢性咽炎"],
    "类风湿性关节炎": ["类风湿", "RA", "类风湿关节炎"],
    "骨质疏松": ["骨质流失", "骨松", "osteoporosis"],
    "痛风": ["高尿酸血症", "gout"],
    "贫血": ["贫血症", "anemia"],
    "脂肪肝": ["脂肪性肝病", "fatty liver", "NAFLD"],
    "失眠": ["入睡困难", "睡眠障碍", "insomnia"],
    "头痛": ["头疼", "头痛症", "headache"],
    "头晕": ["眩晕", "头昏", "dizziness", "vertigo"],
    "胸痛": ["胸闷痛", "胸部疼痛", "chest pain"],
    "呼吸困难": ["气短", "气促", "喘不上气", "dyspnea"],
    "心悸": ["心慌", "心跳加速", "palpitation"],
    "发热": ["发烧", "高热", "fever"],
    "咳嗽": ["咳", "干咳", "咳嗽症状", "cough"],
    "恶心": ["想吐", "反胃", "nausea"],
    "呕吐": ["吐", "喷射性呕吐", "vomiting"],
    "腹泻": ["拉肚子", "水样便", "diarrhea"],
    "便秘": ["排便困难", "大便干结", "constipation"],
    "阿司匹林": ["ASA", "乙酰水杨酸", "aspirin", "拜阿司匹林"],
    "布洛芬": ["ibuprofen", "芬必得", "美林"],
    "二甲双胍": ["metformin", "格华止", "盐酸二甲双胍"],
    "胰岛素": ["insulin", "诺和灵", "优泌林"],
    "硝苯地平": ["nifedipine", "拜新同", "心痛定"],
    "氨氯地平": ["amlodipine", "络活喜", "左旋氨氯地平"],
    "阿托伐他汀": ["atorvastatin", "立普妥", "阿托伐他汀钙"],
    "华法林": ["warfarin", "华法令"],
    "青霉素": ["penicillin", "苄青霉素", "阿莫西林"],
    "对乙酰氨基酚": ["扑热息痛", "泰诺", "acetaminophen", "paracetamol"],
    "血常规": ["全血细胞计数", "CBC", "血象"],
    "尿常规": ["尿液分析", "urinalysis"],
    "肝功能": ["肝功", "肝酶", "liver function"],
    "肾功能": ["肾功", "肾小球滤过率", "renal function"],
    "心电图": ["ECG", "EKG", "心电检查"],
    "CT": ["计算机断层扫描", "电子计算机断层扫描"],
    "MRI": ["磁共振", "磁共振成像", "核磁共振"],
    "超声": ["B超", "超声波", "超声检查"],
    "胃镜": ["胃镜检查", "上消化道内镜", "gastroscopy"],
    "肠镜": ["结肠镜", "肠镜检查", "colonoscopy"],
}

INTENT_EXPANSION_TEMPLATES: Dict[str, str] = {
    "disease_query": "疾病 {query} 定义 病因 症状 诊断 治疗 预防 并发症",
    "drug_query": "药物 {query} 适应症 用法 用量 副作用 禁忌 相互作用",
    "drug_interaction": "药物相互作用 {query} 联用 禁忌 不良反应 安全性",
    "diagnosis_assist": "诊断 {query} 症状 可能疾病 检查 鉴别诊断",
    "symptom_query": "症状 {query} 原因 病因 治疗 缓解 注意事项",
    "treatment_query": "治疗 {query} 方案 方法 手术 药物 康复 护理",
    "examination_query": "检查 {query} 目的 方法 结果解读 正常值 注意事项",
    "prevention_query": "预防 {query} 危险因素 预防措施 生活方式 筛查",
    "health_advice": "健康建议 {query} 饮食 运动 生活习惯 注意事项",
    "general": "{query}",
}


class MedicalQueryExpander:
    def __init__(self):
        self._cached_synonyms: Dict[str, List[str]] = MEDICAL_SYNONYMS
        self._intent_templates: Dict[str, str] = INTENT_EXPANSION_TEMPLATES

    def expand(self, query: str, intent: Optional[str] = None) -> List[str]:
        variants: List[str] = [query]
        query_lower = query.lower()

        matched_terms = self._find_matched_terms(query)
        for term, synonyms in matched_terms:
            for syn in synonyms:
                expanded = query.replace(term, syn)
                if expanded != query and expanded not in variants:
                    variants.append(expanded)

        template = self._intent_templates.get(intent or "general", self._intent_templates["general"])
        if intent and intent in self._intent_templates:
            prompted = template.format(query=query)
            if prompted != query and prompted not in variants:
                variants.append(prompted)

        if not intent:
            self._generate_intent_candidates(query, variants)

        return variants[:6]

    def _find_matched_terms(self, query: str) -> List[tuple]:
        matched = []
        for term in sorted(self._cached_synonyms.keys(), key=len, reverse=True):
            if term in query:
                matched.append((term, self._cached_synonyms[term]))
        return matched

    def _generate_intent_candidates(self, query: str, variants: List[str]):
        disease_indicators = ["病", "症", "炎", "癌", "瘤", "肿", "梗死", "出血", "衰竭"]
        symptom_indicators = ["痛", "晕", "热", "咳", "吐", "泻", "悸", "闷"]
        drug_indicators = ["药", "素", "苷", "林", "平", "汀", "普利", "沙坦", "洛尔"]

        has_disease = any(ind in query for ind in disease_indicators)
        has_symptom = any(ind in query for ind in symptom_indicators)
        has_drug = any(ind in query for ind in drug_indicators)

        if has_disease and has_symptom:
            for intent in ["diagnosis_assist", "symptom_query"]:
                prompted = self._intent_templates[intent].format(query=query)
                if prompted not in variants:
                    variants.append(prompted)
        elif has_drug:
            for intent in ["drug_query", "drug_interaction"]:
                prompted = self._intent_templates[intent].format(query=query)
                if prompted not in variants:
                    variants.append(prompted)

    def expand_single(self, query: str, intent: Optional[str] = None) -> str:
        variants = self.expand(query, intent)
        return " ".join(variants)


def expand_query(query: str, intent: Optional[str] = None) -> List[str]:
    expander = MedicalQueryExpander()
    return expander.expand(query, intent)
