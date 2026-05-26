from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from loguru import logger

from ..core.config import get_settings
from ..ingestion.medical_ner import MedicalNER


class MedicalIntent(str, Enum):
    SYMPTOM_QUERY = "symptom_query"
    DISEASE_QUERY = "disease_query"
    DRUG_QUERY = "drug_query"
    TREATMENT_QUERY = "treatment_query"
    DIAGNOSIS_ASSIST = "diagnosis_assist"
    PREVENTION_QUERY = "prevention_query"
    EXAMINATION_QUERY = "examination_query"
    HEALTH_ADVICE = "health_advice"
    MEDICAL_KNOWLEDGE = "medical_knowledge"
    UNKNOWN = "unknown"


class IntentResult(BaseModel):
    intent: MedicalIntent = Field(description="识别的意图类型")
    confidence: float = Field(ge=0.0, le=1.0, description="意图置信度")
    entities: List[str] = Field(default_factory=list, description="提取的医疗实体")
    slots: Dict[str, str] = Field(default_factory=dict, description="意图槽位信息")


# Shared medical keyword lists (used by both entity extraction and rule-based classification)
# Entity extraction uses full lists; classification uses _CLASSIFY subsets to avoid overlap.
_DISEASE_CLASSIFY_LIST = [
    "高血压", "糖尿病", "心肌梗死", "肺炎", "感冒", "癌症", "肿瘤", "颈椎病",
    "肺癌", "乙肝", "胃炎", "脑梗死", "脑出血", "抑郁症", "贫血", "痛风", "肾结石",
    "帕金森", "阿尔茨海默症", "骨质疏松", "咽炎", "失眠", "荨麻疹", "口腔溃疡", "流感",
    "类风湿性关节炎", "支气管哮喘", "脂肪肝", "骨折",
]
_DISEASE_LIST = [
    "肺癌", "乙肝", "胃炎", "脑梗死", "脑出血", "抑郁症", "贫血", "痛风", "肾结石",
    "帕金森", "阿尔茨海默症", "骨质疏松", "咽炎", "失眠", "荨麻疹", "口腔溃疡", "流感",
    "类风湿性关节炎", "支气管哮喘", "脂肪肝", "骨折", "甲状腺", "心血管疾病", "哮喘",
    "冠心病", "心律失常", "高血压危象", "慢性阻塞性肺疾病", "支气管扩张", "肺脓肿",
    "间质性肺病", "呼吸衰竭", "睡眠呼吸暂停", "鼻窦炎", "扁桃体炎", "喉炎", "中耳炎",
    "青光眼", "白内障", "视网膜病变", "黄斑病变", "牙周病", "龋齿", "牙髓炎", "脂溢性皮炎",
    "湿疹", "银屑病", "甲亢", "甲减", "骨关节炎", "偏头痛", "带状疱疹", "手足口病",
    "水痘", "麻疹", "风疹", "流行性腮腺炎", "猩红热", "百日咳", "白喉", "破伤风", "狂犬病",
    "心肌梗死", "心梗",
]
_DRUG_LIST = [
    "阿司匹林", "布洛芬", "二甲双胍", "胰岛素", "硝苯地平", "氨氯地平", "格列齐特",
    "奥美拉唑", "硫糖铝", "秋水仙碱", "沙丁胺醇", "青霉素", "阿莫西林", "头孢菌素",
    "左氧氟沙星", "罗红霉素", "阿奇霉素", "甲硝唑", "替硝唑", "利福平", "异烟肼",
    "泼尼松", "地塞米松", "氢化可的松", "倍他米松", "双氯芬酸", "塞来昔布",
    "氯吡格雷", "替格瑞洛", "华法林", "低分子肝素", "尿激酶", "链激酶",
    "阿托伐他汀", "瑞舒伐他汀", "辛伐他汀", "洛伐他汀", "非诺贝特", "依折麦布",
    "普萘洛尔", "美托洛尔", "阿替洛尔", "比索洛尔", "卡维地洛",
    "左旋氨氯地平", "地尔硫卓", "维拉帕米", "卡托普利", "依那普利",
    "贝那普利", "赖诺普利", "缬沙坦", "氯沙坦", "厄贝沙坦", "替米沙坦", "奥美沙坦",
    "氢氯噻嗪", "呋塞米", "螺内酯", "氨苯蝶啶", "甘露醇", "甘油果糖", "硝酸甘油",
    "单硝酸异山梨酯", "胺碘酮", "普罗帕酮", "美西律",
    "地高辛", "西地兰", "多巴胺", "多巴酚丁胺", "肾上腺素", "去甲肾上腺素",
    "异丙肾上腺素", "特布他林", "布地奈德", "氟替卡松", "倍氯米松",
]
_SYMPTOM_LIST = [
    "头痛", "头晕", "发烧", "咳嗽", "恶心", "呕吐", "胸痛", "乏力", "呼吸困难", "胸闷",
    "腹泻", "失眠", "口腔溃疡", "关节痛", "肌肉痛", "背痛", "颈痛", "肩痛", "腿痛",
    "手臂痛", "麻木", "刺痛", "肿胀", "发红", "瘙痒", "皮疹", "出血", "瘀斑", "黄疸",
    "口干", "口苦", "口臭", "牙龈出血", "鼻出血", "耳痛", "耳鸣", "听力下降", "视力模糊",
    "视力下降", "眼痛", "流泪", "鼻塞", "流涕", "打喷嚏", "咽痛", "声音嘶哑", "吞咽困难",
    "食欲不振", "恶心", "呕吐", "呕血", "黑便", "腹泻", "便秘", "腹痛", "腹胀", "烧心",
    "胸痛", "胸闷", "心悸", "呼吸困难", "咳嗽", "咳痰", "咯血", "气喘", "端坐呼吸", "疼痛",
    "哮喘",
]
_EXAMINATION_LIST = [
    "血常规", "尿常规", "大便常规", "肝功能", "肾功能", "血糖", "血脂", "电解质",
    "凝血功能", "甲状腺功能", "肿瘤标志物", "心电图", "超声", "CT", "MRI", "X光",
    "胃镜", "肠镜", "支气管镜", "病理检查", "核酸检测", "抗体检测", "肺功能检查",
    "动态心电图", "心脏彩超", "腹部超声", "妇科超声", "乳腺超声", "甲状腺超声",
    "头颅CT", "头颅MRI", "胸部X光", "钡餐造影", "血管造影", "腰椎穿刺", "胸腔穿刺",
    "腹腔穿刺", "骨髓穿刺", "基因检测", "血沉", "C反应蛋白", "类风湿因子", "血气分析",
]


class MedicalIntentClassifier:
    """医疗问答意图识别器"""

    def __init__(self):
        self.settings = get_settings()
        self._llm = None
        self._ner = MedicalNER()  # 集成MedicalNER
        
        self._intent_keywords = {
            MedicalIntent.SYMPTOM_QUERY: ["症状", "原因", "为什么", "怎么回事", "怎么了", "出现", "表现", "持续", "怎么处理", "发作"],
            MedicalIntent.DISEASE_QUERY: ["是什么", "什么是", "定义", "病因", "并发症", "诊断", "表现", "高危因素", "后遗症", "传播途径"],
            MedicalIntent.DRUG_QUERY: ["副作用", "用法", "用量", "服用", "药", "药物", "吃什么药", "能吃", "不能吃", "治疗什么", "能治疗", "用于", "能治什么", "禁忌"],
            MedicalIntent.TREATMENT_QUERY: ["治疗", "怎么治", "如何治", "疗法", "手术", "方案", "康复"],
            MedicalIntent.DIAGNOSIS_ASSIST: ["我", "我有", "我最近", "可能", "怀疑", "检查", "诊断", "帮我看看", "怎么回事", "是什么病"],
            MedicalIntent.PREVENTION_QUERY: ["预防", "保健", "避免", "注意", "如何预防", "怎么预防"],
            MedicalIntent.EXAMINATION_QUERY: ["检查", "化验", "检测", "做什么检查", "需要检查", "空腹"],
            MedicalIntent.HEALTH_ADVICE: ["建议", "怎么办", "应该", "注意什么", "怎么调理", "吃什么", "补什么", "饮食"],
            MedicalIntent.MEDICAL_KNOWLEDGE: ["什么是", "解释", "定义", "说明"],
        }
        
        self._intent_descriptions = {
            MedicalIntent.SYMPTOM_QUERY: "询问症状相关信息，如症状表现、原因、持续时间等",
            MedicalIntent.DISEASE_QUERY: "询问疾病相关信息，如疾病定义、病因、症状、诊断等",
            MedicalIntent.DRUG_QUERY: "询问药物相关信息，如药物用途、副作用、用法用量等",
            MedicalIntent.TREATMENT_QUERY: "询问治疗方案、治疗方法、手术方式等",
            MedicalIntent.DIAGNOSIS_ASSIST: "请求辅助诊断，描述症状寻求可能的疾病判断",
            MedicalIntent.PREVENTION_QUERY: "询问疾病预防、健康保健、生活建议等",
            MedicalIntent.EXAMINATION_QUERY: "询问检查检验项目相关信息",
            MedicalIntent.HEALTH_ADVICE: "寻求健康建议、生活方式指导等",
            MedicalIntent.MEDICAL_KNOWLEDGE: "询问医学知识、医学术语解释等",
            MedicalIntent.UNKNOWN: "无法识别的意图",
        }

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.dashscope_model,
                temperature=0,
                api_key=self.settings.dashscope_api_key,
                base_url=self.settings.dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._llm

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """从问题中提取医疗实体 - 结合MedicalNER和规则匹配"""
        entities = []
        
        # 使用MedicalNER提取实体
        ner_entities = self._ner.extract(question)
        for entity in ner_entities:
            entities.append(entity.name)
        
        # 规则匹配（使用共享的模块级列表）
        for disease in _DISEASE_LIST:
            if disease in question and disease not in entities:
                entities.append(disease)
        
        for drug in _DRUG_LIST:
            if drug in question and drug not in entities:
                entities.append(drug)
        
        for symptom in _SYMPTOM_LIST:
            if symptom in question and symptom not in entities:
                entities.append(symptom)
        
        for exam in _EXAMINATION_LIST:
            if exam in question and exam not in entities:
                entities.append(exam)
        
        # 去重
        unique_entities = list(set(entities))
        return unique_entities

    def _rule_based_classify(self, question: str) -> Optional[IntentResult]:
        """基于规则的快速意图分类"""
        entities = self._extract_entities_from_question(question)
        
        drug_list = _DRUG_LIST
        disease_keywords = _DISEASE_CLASSIFY_LIST
        symptom_keywords = _SYMPTOM_LIST
        
        # 药物名称出现在问题中 → 药物查询（优先级最高）
        for drug in drug_list:
            if drug in question:
                return IntentResult(
                    intent=MedicalIntent.DRUG_QUERY,
                    confidence=0.9,
                    entities=entities,
                    slots={}
                )
        
        # 药物相关词汇 → 药物查询
        drug_related_phrases = ["副作用", "用法", "用量", "服用", "药", "药物", "吃什么药", "能吃", "不能吃", "治疗什么", "能治疗", "用于", "能治什么", "禁忌"]
        if any(kw in question for kw in drug_related_phrases):
            return IntentResult(
                intent=MedicalIntent.DRUG_QUERY,
                confidence=0.85,
                entities=entities,
                slots={}
            )
        
        # "康复"相关的问题 → 治疗查询（高优先级）
        if "康复" in question:
            return IntentResult(
                intent=MedicalIntent.TREATMENT_QUERY,
                confidence=0.9,
                entities=entities,
                slots={}
            )
        
        # "怎么办" + 症状 → 症状查询（非常高优先级）
        if "怎么办" in question and any(symptom in question for symptom in symptom_keywords):
            return IntentResult(
                intent=MedicalIntent.SYMPTOM_QUERY,
                confidence=0.95,
                entities=entities,
                slots={}
            )
        
        # "发作" + "怎么处理" → 症状查询（非常高优先级）
        if "发作" in question and "处理" in question:
            return IntentResult(
                intent=MedicalIntent.SYMPTOM_QUERY,
                confidence=0.95,
                entities=entities,
                slots={}
            )
        
        # "发作" + 症状 → 症状查询
        if "发作" in question and any(symptom in question for symptom in symptom_keywords):
            return IntentResult(
                intent=MedicalIntent.SYMPTOM_QUERY,
                confidence=0.9,
                entities=entities,
                slots={}
            )
        
        # 以"我"开头的问题
        if question.startswith("我"):
            # 询问"是什么病"、"怎么回事"、"是什么原因" → 诊断辅助（寻求诊断）
            if any(kw in question for kw in ["是怎么回事", "是什么病", "什么问题", "是什么原因"]):
                return IntentResult(
                    intent=MedicalIntent.DIAGNOSIS_ASSIST,
                    confidence=0.9,
                    entities=entities,
                    slots={}
                )
            # 先检查"注意什么"、"吃什么" → 健康建议
            if any(kw in question for kw in ["注意什么", "吃什么", "怎么补", "饮食"]):
                return IntentResult(
                    intent=MedicalIntent.HEALTH_ADVICE,
                    confidence=0.85,
                    entities=entities,
                    slots={}
                )
            # 描述症状 → 诊断辅助
            if any(kw in question for kw in symptom_keywords):
                return IntentResult(
                    intent=MedicalIntent.DIAGNOSIS_ASSIST,
                    confidence=0.85,
                    entities=entities,
                    slots={}
                )
        
        # "注意什么"、"吃什么"、"怎么补" → 健康建议
        if any(kw in question for kw in ["注意什么", "吃什么", "怎么补", "饮食", "补什么"]):
            return IntentResult(
                intent=MedicalIntent.HEALTH_ADVICE,
                confidence=0.85,
                entities=entities,
                slots={}
            )
        
        # 疾病名称 + "有什么症状"、"高危因素"、"后遗症"、"传播途径" → 疾病查询
        if any(kw in question for kw in ["有什么症状", "高危因素", "后遗症", "传播途径", "表现"]):
            for disease in disease_keywords:
                if disease in question:
                    return IntentResult(
                        intent=MedicalIntent.DISEASE_QUERY,
                        confidence=0.9,
                        entities=entities,
                        slots={}
                    )
        
        # 疾病名称 + "如何延缓"、"怎么治疗" → 疾病查询
        if any(kw in question for kw in ["如何延缓", "怎么治疗"]) and any(disease in question for disease in disease_keywords):
            return IntentResult(
                intent=MedicalIntent.DISEASE_QUERY,
                confidence=0.9,
                entities=entities,
                slots={}
            )
        
        # "治疗"相关问题
        if "治疗" in question or "怎么治" in question or "如何治" in question:
            for disease in disease_keywords:
                if disease in question:
                    return IntentResult(
                        intent=MedicalIntent.DISEASE_QUERY,
                        confidence=0.9,
                        entities=entities,
                        slots={}
                    )
            return IntentResult(
                intent=MedicalIntent.TREATMENT_QUERY,
                confidence=0.8,
                entities=entities,
                slots={}
            )
        
        # 疾病名称单独出现 + 询问相关 → 疾病查询
        for disease in disease_keywords:
            if disease in question and any(kw in question for kw in ["是什么", "什么是", "病因", "诊断", "传染", "如何", "多大"]):
                return IntentResult(
                    intent=MedicalIntent.DISEASE_QUERY,
                    confidence=0.85,
                    entities=entities,
                    slots={}
                )
        
        # "原因"、"为什么" → 症状查询
        if any(kw in question for kw in ["原因", "为什么"]) and any(symptom in question for symptom in symptom_keywords):
            return IntentResult(
                intent=MedicalIntent.SYMPTOM_QUERY,
                confidence=0.8,
                entities=entities,
                slots={}
            )
        
        # 检查相关
        if any(kw in question for kw in ["检查", "空腹", "化验", "需要检查"]):
            return IntentResult(
                intent=MedicalIntent.EXAMINATION_QUERY,
                confidence=0.85,
                entities=entities,
                slots={}
            )
        
        # 预防相关
        if any(kw in question for kw in ["预防", "如何预防"]):
            return IntentResult(
                intent=MedicalIntent.PREVENTION_QUERY,
                confidence=0.85,
                entities=entities,
                slots={}
            )
        
        # 疾病名称单独出现 → 疾病查询
        for disease in disease_keywords:
            if disease in question:
                return IntentResult(
                    intent=MedicalIntent.DISEASE_QUERY,
                    confidence=0.8,
                    entities=entities,
                    slots={}
                )
        
        return None

    def classify(self, question: str) -> IntentResult:
        """对医疗问题进行意图分类（先规则匹配，再LLM）"""
        rule_result = self._rule_based_classify(question)
        if rule_result:
            return rule_result
        
        system_prompt = f"""你是一个医疗意图识别专家。请分析用户的问题并确定其意图类型。

可用的意图类型：
{chr(10).join([f"- {intent.value}: {desc}" for intent, desc in self._intent_descriptions.items()])}

分类优先级规则：
1. 如果问题以"我"开头且描述症状，优先归类为诊断辅助(diagnosis_assist)
2. 如果问题询问"如何治疗"或"治疗方法"，归类为治疗查询(treatment_query)
3. 如果问题询问疾病的治疗或症状，归类为疾病查询(disease_query)
4. 如果问题询问药物相关，归类为药物查询(drug_query)
5. 如果问题描述症状并询问原因，归类为症状查询(symptom_query)
6. 如果问题询问健康建议如"注意什么"、"吃什么"，归类为健康建议(health_advice)

请以JSON格式返回结果，包含以下字段：
- intent: 意图类型（必须是上述列表中的一个）
- confidence: 置信度（0-1之间的浮点数）
- entities: 从问题中提取的医疗实体列表
- slots: 关键槽位信息

只返回JSON，不要有其他解释文字。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "用户问题：{question}"),
        ])

        chain = prompt | self._get_llm()

        try:
            result = chain.invoke({"question": question})
            content = result.content

            import json
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)

                intent_str = data.get("intent", "unknown")
                intent = MedicalIntent(intent_str) if intent_str in [e.value for e in MedicalIntent] else MedicalIntent.UNKNOWN

                slots_data = data.get("slots", {})
                if isinstance(slots_data, dict):
                    for key, value in list(slots_data.items()):
                        if isinstance(value, list):
                            slots_data[key] = ", ".join(str(v) for v in value)
                else:
                    slots_data = {}

                return IntentResult(
                    intent=intent,
                    confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                    entities=data.get("entities", []),
                    slots=slots_data,
                )
            else:
                return IntentResult(intent=MedicalIntent.UNKNOWN, confidence=0.5)

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentResult(intent=MedicalIntent.UNKNOWN, confidence=0.3)

    def get_intent_prompt(self, intent: MedicalIntent) -> str:
        """根据意图类型获取针对性的提示词"""
        prompts = {
            MedicalIntent.SYMPTOM_QUERY: """你是一位专业的医疗专家，请详细分析用户提到的症状。包括：
1. 症状的定义和常见表现
2. 可能的病因
3. 需要进一步了解的信息
4. 建议的行动步骤

请用通俗易懂的语言回答，避免使用过于专业的术语。""",
            MedicalIntent.DISEASE_QUERY: """你是一位专业的医疗专家，请详细介绍用户询问的疾病。包括：
1. 疾病的定义和概述
2. 主要病因和危险因素
3. 常见症状和体征
4. 诊断方法
5. 治疗方案
6. 预后和预防措施

请用通俗易懂的语言回答。""",
            MedicalIntent.DRUG_QUERY: """你是一位专业的药师，请详细介绍用户询问的药物。包括：
1. 药物的通用名和商品名
2. 主要适应症
3. 作用机制
4. 用法用量
5. 不良反应和注意事项
6. 禁忌和慎用人群

请用通俗易懂的语言回答。""",
            MedicalIntent.TREATMENT_QUERY: """你是一位专业的医疗专家，请详细介绍用户询问的治疗方法。包括：
1. 治疗方法的原理和适用范围
2. 治疗流程和步骤
3. 预期效果和风险
4. 术后护理或注意事项

请用通俗易懂的语言回答。""",
            MedicalIntent.DIAGNOSIS_ASSIST: """你是一位专业的诊断专家，请根据用户描述的症状进行初步分析。包括：
1. 可能的疾病诊断
2. 需要排除的疾病
3. 建议的检查项目
4. 下一步建议

重要提示：本分析仅供参考，不能替代专业医生的诊断。""",
            MedicalIntent.PREVENTION_QUERY: """你是一位专业的健康管理专家，请提供疾病预防和健康保健建议。包括：
1. 疾病的危险因素
2. 预防措施和生活方式建议
3. 定期检查建议
4. 疫苗接种建议（如适用）

请用通俗易懂的语言回答。""",
            MedicalIntent.EXAMINATION_QUERY: """你是一位专业的检验医学专家，请详细介绍用户询问的检查项目。包括：
1. 检查项目的目的和意义
2. 检查方法和过程
3. 结果解读
4. 注意事项

请用通俗易懂的语言回答。""",
            MedicalIntent.HEALTH_ADVICE: """你是一位专业的健康顾问，请提供实用的健康建议。包括：
1. 饮食建议
2. 运动建议
3. 生活习惯建议
4. 心理健康建议

请用通俗易懂的语言回答。""",
            MedicalIntent.MEDICAL_KNOWLEDGE: """你是一位专业的医学教育专家，请详细解释用户询问的医学知识。包括：
1. 术语定义和解释
2. 相关生理或病理机制
3. 临床应用
4. 最新研究进展（如适用）

请用通俗易懂的语言回答。""",
        }
        return prompts.get(intent, "")

    def route_to_agent(self, intent: MedicalIntent) -> Literal["symptom_agent", "disease_agent", "drug_agent", "treatment_agent", "general_agent"]:
        """根据意图路由到相应的处理代理"""
        routing_map = {
            MedicalIntent.SYMPTOM_QUERY: "symptom_agent",
            MedicalIntent.DISEASE_QUERY: "disease_agent",
            MedicalIntent.DRUG_QUERY: "drug_agent",
            MedicalIntent.TREATMENT_QUERY: "treatment_agent",
            MedicalIntent.DIAGNOSIS_ASSIST: "general_agent",
            MedicalIntent.PREVENTION_QUERY: "general_agent",
            MedicalIntent.EXAMINATION_QUERY: "general_agent",
            MedicalIntent.HEALTH_ADVICE: "general_agent",
            MedicalIntent.MEDICAL_KNOWLEDGE: "general_agent",
            MedicalIntent.UNKNOWN: "general_agent",
        }
        return routing_map[intent]


class MedicalDialogueManager:
    """医疗对话状态管理器"""

    def __init__(self):
        self.intent_classifier = MedicalIntentClassifier()
        self._dialogue_history: List[Dict[str, str]] = []
        self._current_intent: Optional[MedicalIntent] = None
        self._collected_slots: Dict[str, Any] = {}

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """处理一轮对话"""
        intent_result = self.intent_classifier.classify(user_input)
        self._current_intent = intent_result.intent
        self._collected_slots.update(intent_result.slots)

        self._dialogue_history.append({"role": "user", "content": user_input})

        response = {
            "intent": intent_result.intent.value,
            "confidence": intent_result.confidence,
            "entities": intent_result.entities,
            "slots": self._collected_slots,
            "agent": self.intent_classifier.route_to_agent(intent_result.intent),
            "prompt": self.intent_classifier.get_intent_prompt(intent_result.intent),
        }

        return response

    def reset(self):
        """重置对话状态"""
        self._dialogue_history = []
        self._current_intent = None
        self._collected_slots = {}

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self._dialogue_history

    def add_bot_response(self, response: str):
        """添加机器人响应到历史"""
        self._dialogue_history.append({"role": "bot", "content": response})


def classify_medical_intent(question: str) -> IntentResult:
    """分类医疗问题意图的便捷函数"""
    classifier = MedicalIntentClassifier()
    return classifier.classify(question)


def get_medical_prompt(intent: MedicalIntent) -> str:
    """根据意图获取提示词"""
    classifier = MedicalIntentClassifier()
    return classifier.get_intent_prompt(intent)
