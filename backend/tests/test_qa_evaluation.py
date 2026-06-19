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
            TestCase(
                question="肺炎是什么原因引起的？",
                expected_intent="disease_query",
                expected_entities=["肺炎"],
                expected_answer_keywords=["肺炎", "原因", "感染", "细菌", "病毒"],
                context="肺炎通常由细菌或病毒感染引起，常见病原体包括肺炎链球菌、金黄色葡萄球菌等。"
            ),
            TestCase(
                question="胰岛素的使用方法是什么？",
                expected_intent="drug_query",
                expected_entities=["胰岛素"],
                expected_answer_keywords=["胰岛素", "用法", "注射", "糖尿病"],
                context="胰岛素主要用于糖尿病治疗，需要皮下注射使用。"
            ),
            TestCase(
                question="我咳嗽了半个月了，是怎么回事？",
                expected_intent="diagnosis_assist",
                expected_entities=["咳嗽"],
                expected_answer_keywords=["咳嗽", "可能", "原因", "检查"],
                context="长期咳嗽可能由多种原因引起，如支气管炎、肺炎、哮喘等。"
            ),
            TestCase(
                question="肺癌的高危因素有哪些？",
                expected_intent="disease_query",
                expected_entities=["肺癌"],
                expected_answer_keywords=["肺癌", "高危", "因素", "吸烟"],
                context="肺癌的高危因素包括吸烟、空气污染、职业暴露、遗传因素等。"
            ),
            TestCase(
                question="硝苯地平片的用法用量？",
                expected_intent="drug_query",
                expected_entities=["硝苯地平"],
                expected_answer_keywords=["硝苯地平", "用法", "用量", "高血压"],
                context="硝苯地平是钙通道阻滞剂，用于治疗高血压和心绞痛。"
            ),
            TestCase(
                question="胃炎吃什么药效果好？",
                expected_intent="drug_query",
                expected_entities=["胃炎"],
                expected_answer_keywords=["胃炎", "药物", "治疗", "胃酸"],
                context="胃炎常用药物包括质子泵抑制剂、胃黏膜保护剂等。"
            ),
            TestCase(
                question="如何预防心血管疾病？",
                expected_intent="prevention_query",
                expected_entities=["心血管疾病"],
                expected_answer_keywords=["预防", "心血管", "健康", "生活方式"],
                context="预防心血管疾病需要保持健康生活方式，包括合理饮食、适量运动、戒烟限酒等。"
            ),
            TestCase(
                question="乙肝的传播途径有哪些？",
                expected_intent="disease_query",
                expected_entities=["乙肝"],
                expected_answer_keywords=["乙肝", "传播", "途径", "病毒"],
                context="乙肝主要通过血液、母婴和性接触传播。"
            ),
            TestCase(
                question="我总是恶心想吐，是什么问题？",
                expected_intent="diagnosis_assist",
                expected_entities=["恶心", "呕吐"],
                expected_answer_keywords=["恶心", "呕吐", "可能", "原因"],
                context="恶心呕吐可能由消化系统疾病、神经系统疾病、药物反应等多种原因引起。"
            ),
            TestCase(
                question="二甲双胍有哪些禁忌？",
                expected_intent="drug_query",
                expected_entities=["二甲双胍"],
                expected_answer_keywords=["二甲双胍", "禁忌", "肾功能", "糖尿病"],
                context="二甲双胍禁用于严重肾功能不全、肝功能衰竭、酗酒等患者。"
            ),
            TestCase(
                question="骨折后如何进行康复锻炼？",
                expected_intent="treatment_query",
                expected_entities=["骨折"],
                expected_answer_keywords=["骨折", "康复", "锻炼", "恢复"],
                context="骨折康复需要在医生指导下进行，包括早期活动、功能锻炼等。"
            ),
            TestCase(
                question="脑梗死的后遗症有哪些？",
                expected_intent="disease_query",
                expected_entities=["脑梗死"],
                expected_answer_keywords=["脑梗死", "后遗症", "偏瘫", "语言"],
                context="脑梗死常见后遗症包括偏瘫、语言障碍、认知功能下降等。"
            ),
            TestCase(
                question="甲状腺功能检查需要空腹吗？",
                expected_intent="examination_query",
                expected_entities=["甲状腺"],
                expected_answer_keywords=["甲状腺", "检查", "空腹", "验血"],
                context="甲状腺功能检查通常需要空腹采血，以获得更准确的结果。"
            ),
            TestCase(
                question="我腹泻了三天，应该注意什么？",
                expected_intent="health_advice",
                expected_entities=["腹泻"],
                expected_answer_keywords=["腹泻", "注意", "饮食", "补水"],
                context="腹泻时应注意补充水分和电解质，清淡饮食，必要时就医。"
            ),
            TestCase(
                question="抑郁症的早期症状是什么？",
                expected_intent="disease_query",
                expected_entities=["抑郁症"],
                expected_answer_keywords=["抑郁症", "症状", "情绪", "兴趣"],
                context="抑郁症早期症状包括情绪低落、兴趣减退、睡眠障碍、食欲改变等。"
            ),
            TestCase(
                question="贫血吃什么补血最快？",
                expected_intent="health_advice",
                expected_entities=["贫血"],
                expected_answer_keywords=["贫血", "补血", "食物", "铁"],
                context="贫血患者应多食用富含铁的食物，如红肉、动物肝脏、菠菜等。"
            ),
            TestCase(
                question="痛风发作时怎么处理？",
                expected_intent="symptom_query",
                expected_entities=["痛风"],
                expected_answer_keywords=["痛风", "发作", "处理", "疼痛"],
                context="痛风急性发作时应卧床休息、抬高患肢、可使用秋水仙碱或非甾体抗炎药。"
            ),
            TestCase(
                question="肾结石多大需要手术治疗？",
                expected_intent="disease_query",
                expected_entities=["肾结石"],
                expected_answer_keywords=["肾结石", "手术", "治疗", "大小"],
                context="肾结石的治疗方式取决于结石大小、位置及患者情况，一般大于1cm可能需要手术。"
            ),
            TestCase(
                question="帕金森病有什么症状？",
                expected_intent="disease_query",
                expected_entities=["帕金森"],
                expected_answer_keywords=["帕金森", "症状", "震颤", "运动"],
                context="帕金森病主要症状包括静止性震颤、肌强直、运动迟缓、姿势步态异常等。"
            ),
            TestCase(
                question="阿尔茨海默症如何延缓病情发展？",
                expected_intent="disease_query",
                expected_entities=["阿尔茨海默症"],
                expected_answer_keywords=["阿尔茨海默症", "延缓", "治疗", "认知"],
                context="阿尔茨海默症可通过药物治疗、认知训练、健康生活方式延缓病情进展。"
            ),
            TestCase(
                question="骨质疏松怎么补钙最有效？",
                expected_intent="health_advice",
                expected_entities=["骨质疏松"],
                expected_answer_keywords=["骨质疏松", "补钙", "维生素D", "运动"],
                context="骨质疏松患者应补充钙剂和维生素D，适度负重运动，多晒太阳。"
            ),
            TestCase(
                question="我有慢性咽炎，吃什么药好？",
                expected_intent="drug_query",
                expected_entities=["咽炎"],
                expected_answer_keywords=["咽炎", "药物", "治疗", "慢性"],
                context="慢性咽炎可用含片、中成药治疗，关键在于去除病因和日常护理。"
            ),
            TestCase(
                question="脑出血后遗症康复需要注意什么？",
                expected_intent="treatment_query",
                expected_entities=["脑出血"],
                expected_answer_keywords=["脑出血", "康复", "注意", "功能"],
                context="脑出血康复需要在专业指导下进行，包括肢体功能训练、语言训练等。"
            ),
            TestCase(
                question="我最近总是失眠，是什么原因？",
                expected_intent="diagnosis_assist",
                expected_entities=["失眠"],
                expected_answer_keywords=["失眠", "原因", "睡眠", "压力"],
                context="失眠原因包括精神压力、不良睡眠习惯、疾病因素、药物影响等。"
            ),
            TestCase(
                question="荨麻疹传染吗？",
                expected_intent="disease_query",
                expected_entities=["荨麻疹"],
                expected_answer_keywords=["荨麻疹", "传染", "过敏", "皮肤"],
                context="荨麻疹是一种过敏性疾病，不具有传染性。"
            ),
            TestCase(
                question="我反复口腔溃疡是什么原因？",
                expected_intent="diagnosis_assist",
                expected_entities=["口腔溃疡"],
                expected_answer_keywords=["口腔溃疡", "原因", "免疫", "维生素"],
                context="反复口腔溃疡可能与免疫功能、营养缺乏、精神因素、局部刺激等有关。"
            ),
            TestCase(
                question="如何区分普通感冒和流感？",
                expected_intent="disease_query",
                expected_entities=["感冒", "流感"],
                expected_answer_keywords=["感冒", "流感", "区别", "症状"],
                context="普通感冒和流感在症状严重程度、发热特点、全身症状等方面有区别。"
            ),
            TestCase(
                question="类风湿性关节炎有哪些表现？",
                expected_intent="disease_query",
                expected_entities=["类风湿性关节炎"],
                expected_answer_keywords=["类风湿", "关节炎", "症状", "关节"],
                context="类风湿性关节炎主要表现为对称性关节肿痛、晨僵、功能障碍等。"
            ),
            TestCase(
                question="支气管哮喘急性发作怎么处理？",
                expected_intent="symptom_query",
                expected_entities=["哮喘"],
                expected_answer_keywords=["哮喘", "发作", "处理", "呼吸"],
                context="哮喘急性发作时应使用速效支气管舒张剂，如沙丁胺醇吸入剂。"
            ),
            TestCase(
                question="我有脂肪肝，饮食上要注意什么？",
                expected_intent="health_advice",
                expected_entities=["脂肪肝"],
                expected_answer_keywords=["脂肪肝", "饮食", "注意", "控制"],
                context="脂肪肝患者应控制饮食、减重、戒酒、适度运动。"
            ),
            # ──────────────────────────────────────────────
            # 新增：examination_query 检查查询 (6 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="血常规检查能查出什么？",
                expected_intent="examination_query",
                expected_entities=["血常规"],
                expected_answer_keywords=["血常规", "检查", "红细胞", "白细胞"],
                context="血常规可检测白细胞、红细胞、血小板计数及血红蛋白等。"
            ),
            TestCase(
                question="做CT检查有辐射吗？",
                expected_intent="examination_query",
                expected_entities=["CT"],
                expected_answer_keywords=["CT", "辐射", "安全"],
                context="CT检查有一定辐射，单次常规CT检查辐射剂量在安全范围内。"
            ),
            TestCase(
                question="胃镜检查前需要准备什么？",
                expected_intent="examination_query",
                expected_entities=["胃镜"],
                expected_answer_keywords=["胃镜", "准备", "空腹"],
                context="胃镜检查前需空腹6-8小时，停用抗凝药物。"
            ),
            TestCase(
                question="心电图检查主要查什么？",
                expected_intent="examination_query",
                expected_entities=["心电图"],
                expected_answer_keywords=["心电图", "心脏", "心律"],
                context="心电图用于检测心脏电活动，诊断心律失常、心肌缺血等。"
            ),
            TestCase(
                question="MRI和CT有什么区别？",
                expected_intent="examination_query",
                expected_entities=["MRI", "CT"],
                expected_answer_keywords=["MRI", "CT", "区别", "辐射"],
                context="MRI利用磁场成像无辐射，CT利用X射线有辐射，各有优势。"
            ),
            TestCase(
                question="肿瘤标志物升高一定是癌症吗？",
                expected_intent="examination_query",
                expected_entities=["肿瘤标志物"],
                expected_answer_keywords=["肿瘤标志物", "癌症", "确诊"],
                context="肿瘤标志物升高不能直接确诊癌症，需结合其他检查综合判断。"
            ),
            # ──────────────────────────────────────────────
            # 新增：prevention_query 预防查询 (4 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="如何预防糖尿病？",
                expected_intent="prevention_query",
                expected_entities=["糖尿病"],
                expected_answer_keywords=["糖尿病", "预防", "饮食", "运动"],
                context="预防糖尿病需合理饮食、规律运动、控制体重、定期检测血糖。"
            ),
            TestCase(
                question="怎样预防骨质疏松？",
                expected_intent="prevention_query",
                expected_entities=["骨质疏松"],
                expected_answer_keywords=["骨质疏松", "预防", "钙", "运动"],
                context="预防骨质疏松需补充钙和维生素D、适度运动、避免吸烟酗酒。"
            ),
            TestCase(
                question="如何预防肝癌？",
                expected_intent="prevention_query",
                expected_entities=["肝癌"],
                expected_answer_keywords=["肝癌", "预防", "乙肝", "戒酒"],
                context="预防肝癌需接种乙肝疫苗、戒酒、避免食用霉变食物。"
            ),
            TestCase(
                question="怎样预防流感传播？",
                expected_intent="prevention_query",
                expected_entities=["流感"],
                expected_answer_keywords=["流感", "预防", "疫苗", "卫生"],
                context="预防流感需接种疫苗、勤洗手、避免人群密集场所。"
            ),
            # ──────────────────────────────────────────────
            # 新增：treatment_query 治疗查询 (5 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="冠心病的治疗方法有哪些？",
                expected_intent="treatment_query",
                expected_entities=["冠心病"],
                expected_answer_keywords=["冠心病", "治疗", "药物", "手术"],
                context="冠心病治疗包括药物治疗、介入治疗和冠脉搭桥手术。"
            ),
            TestCase(
                question="胃溃疡怎么治疗？",
                expected_intent="treatment_query",
                expected_entities=["胃溃疡"],
                expected_answer_keywords=["胃溃疡", "治疗", "药物", "抑酸"],
                context="胃溃疡治疗包括抑酸药物、胃黏膜保护剂和根除幽门螺杆菌。"
            ),
            TestCase(
                question="甲亢的治疗方案有哪些？",
                expected_intent="treatment_query",
                expected_entities=["甲亢"],
                expected_answer_keywords=["甲亢", "治疗", "药物", "手术"],
                context="甲亢治疗包括抗甲状腺药物、放射性碘治疗和手术切除。"
            ),
            TestCase(
                question="腰椎间盘突出怎么治疗？",
                expected_intent="treatment_query",
                expected_entities=["腰椎间盘突出"],
                expected_answer_keywords=["腰椎间盘突出", "治疗", "保守", "手术"],
                context="腰椎间盘突出治疗包括保守治疗和手术治疗。"
            ),
            TestCase(
                question="慢性肾衰竭怎么治疗？",
                expected_intent="treatment_query",
                expected_entities=["慢性肾衰竭"],
                expected_answer_keywords=["肾衰竭", "治疗", "透析", "移植"],
                context="慢性肾衰竭治疗包括药物治疗、透析和肾移植。"
            ),
            # ──────────────────────────────────────────────
            # 新增：drug_query 药物查询补充 (8 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="氯吡格雷的作用是什么？",
                expected_intent="drug_query",
                expected_entities=["氯吡格雷"],
                expected_answer_keywords=["氯吡格雷", "抗血小板", "血栓"],
                context="氯吡格雷是抗血小板药物，用于预防血栓形成。"
            ),
            TestCase(
                question="奥美拉唑是治什么病的？",
                expected_intent="drug_query",
                expected_entities=["奥美拉唑"],
                expected_answer_keywords=["奥美拉唑", "胃酸", "溃疡"],
                context="奥美拉唑是质子泵抑制剂，用于治疗胃酸相关疾病。"
            ),
            TestCase(
                question="阿莫西林属于哪类抗生素？",
                expected_intent="drug_query",
                expected_entities=["阿莫西林"],
                expected_answer_keywords=["阿莫西林", "青霉素", "抗生素"],
                context="阿莫西林属于青霉素类抗生素，用于治疗细菌感染。"
            ),
            TestCase(
                question="辛伐他汀的用法用量？",
                expected_intent="drug_query",
                expected_entities=["辛伐他汀"],
                expected_answer_keywords=["辛伐他汀", "降脂", "用量"],
                context="辛伐他汀是降脂药，通常晚间服用。"
            ),
            TestCase(
                question="氨氯地平有什么副作用？",
                expected_intent="drug_query",
                expected_entities=["氨氯地平"],
                expected_answer_keywords=["氨氯地平", "副作用", "水肿"],
                context="氨氯地平常见副作用包括下肢水肿、头痛、面部潮红。"
            ),
            TestCase(
                question="蒙脱石散怎么服用？",
                expected_intent="drug_query",
                expected_entities=["蒙脱石散"],
                expected_answer_keywords=["蒙脱石散", "服用", "腹泻"],
                context="蒙脱石散用于治疗腹泻，需空腹服用。"
            ),
            TestCase(
                question="头孢克洛和头孢克肟有什么区别？",
                expected_intent="drug_query",
                expected_entities=["头孢克洛", "头孢克肟"],
                expected_answer_keywords=["头孢", "区别", "代数"],
                context="头孢克洛属第二代头孢，头孢克肟属第三代头孢，抗菌谱不同。"
            ),
            TestCase(
                question="氯雷他定能长期服用吗？",
                expected_intent="drug_query",
                expected_entities=["氯雷他定"],
                expected_answer_keywords=["氯雷他定", "长期", "过敏"],
                context="氯雷他定是抗过敏药，长期服用需在医生指导下进行。"
            ),
            # ──────────────────────────────────────────────
            # 新增：disease_query 疾病查询补充 (8 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="系统性红斑狼疮是什么病？",
                expected_intent="disease_query",
                expected_entities=["系统性红斑狼疮"],
                expected_answer_keywords=["红斑狼疮", "自身免疫", "全身"],
                context="系统性红斑狼疮是一种自身免疫性疾病，可累及全身多个器官。"
            ),
            TestCase(
                question="慢性阻塞性肺疾病的病因是什么？",
                expected_intent="disease_query",
                expected_entities=["慢性阻塞性肺疾病"],
                expected_answer_keywords=["慢阻肺", "病因", "吸烟"],
                context="慢阻肺主要病因包括吸烟、空气污染和职业粉尘暴露。"
            ),
            TestCase(
                question="肝硬化能治好吗？",
                expected_intent="disease_query",
                expected_entities=["肝硬化"],
                expected_answer_keywords=["肝硬化", "治疗", "不可逆"],
                context="肝硬化是肝脏不可逆的纤维化改变，但可延缓进展。"
            ),
            TestCase(
                question="阑尾炎的典型症状是什么？",
                expected_intent="disease_query",
                expected_entities=["阑尾炎"],
                expected_answer_keywords=["阑尾炎", "症状", "转移性腹痛"],
                context="阑尾炎典型症状为转移性右下腹痛，伴恶心呕吐。"
            ),
            TestCase(
                question="甲状腺功能减退有什么表现？",
                expected_intent="disease_query",
                expected_entities=["甲状腺功能减退"],
                expected_answer_keywords=["甲减", "表现", "乏力", "怕冷"],
                context="甲减表现为乏力、怕冷、体重增加、皮肤干燥等。"
            ),
            TestCase(
                question="胃食管反流病怎么引起的？",
                expected_intent="disease_query",
                expected_entities=["胃食管反流"],
                expected_answer_keywords=["胃食管反流", "原因", "食管括约肌"],
                context="胃食管反流病由下食管括约肌功能障碍导致胃内容物反流。"
            ),
            TestCase(
                question="带状疱疹会传染吗？",
                expected_intent="disease_query",
                expected_entities=["带状疱疹"],
                expected_answer_keywords=["带状疱疹", "传染", "水痘"],
                context="带状疱疹本身不直接传染，但可致未免疫者感染水痘。"
            ),
            TestCase(
                question="慢性肾炎会发展成尿毒症吗？",
                expected_intent="disease_query",
                expected_entities=["慢性肾炎", "尿毒症"],
                expected_answer_keywords=["慢性肾炎", "尿毒症", "进展"],
                context="慢性肾炎如不及时治疗可能进展为尿毒症。"
            ),
            # ──────────────────────────────────────────────
            # 新增：symptom_query 症状查询补充 (4 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="手脚发麻是什么原因？",
                expected_intent="symptom_query",
                expected_entities=["手脚发麻"],
                expected_answer_keywords=["发麻", "原因", "神经", "颈椎"],
                context="手脚发麻可能由颈椎病、糖尿病神经病变、脑血管病等引起。"
            ),
            TestCase(
                question="胸闷气短是怎么回事？",
                expected_intent="symptom_query",
                expected_entities=["胸闷", "气短"],
                expected_answer_keywords=["胸闷", "气短", "心脏", "肺部"],
                context="胸闷气短可能由心脏疾病、肺部疾病或焦虑等引起。"
            ),
            TestCase(
                question="关节红肿热痛是什么原因？",
                expected_intent="symptom_query",
                expected_entities=["关节", "红肿"],
                expected_answer_keywords=["关节", "红肿", "痛风", "感染"],
                context="关节红肿热痛常见于痛风、化脓性关节炎、类风湿等。"
            ),
            TestCase(
                question="早上起来恶心干呕是怎么回事？",
                expected_intent="symptom_query",
                expected_entities=["恶心", "干呕"],
                expected_answer_keywords=["恶心", "干呕", "咽炎", "胃"],
                context="晨起恶心干呕常见于慢性咽炎、胃食管反流等。"
            ),
            # ──────────────────────────────────────────────
            # 新增：diagnosis_assist 诊断辅助补充 (5 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="我右下腹疼痛，伴有发烧，是什么病？",
                expected_intent="diagnosis_assist",
                expected_entities=["腹痛", "发烧"],
                expected_answer_keywords=["右下腹", "疼痛", "阑尾炎"],
                context="右下腹痛伴发热需高度怀疑急性阑尾炎。"
            ),
            TestCase(
                question="皮肤发黄、小便颜色深，可能是什么问题？",
                expected_intent="diagnosis_assist",
                expected_entities=["皮肤发黄", "小便"],
                expected_answer_keywords=["黄疸", "肝", "胆"],
                context="皮肤发黄伴深色尿液提示黄疸，需检查肝胆系统。"
            ),
            TestCase(
                question="我最近体重突然下降了很多，是什么原因？",
                expected_intent="diagnosis_assist",
                expected_entities=["体重下降"],
                expected_answer_keywords=["体重", "下降", "原因", "检查"],
                context="不明原因体重显著下降需排除恶性肿瘤、甲亢、糖尿病等。"
            ),
            TestCase(
                question="我手指关节晨僵，握拳困难，是什么病？",
                expected_intent="diagnosis_assist",
                expected_entities=["晨僵", "关节"],
                expected_answer_keywords=["晨僵", "类风湿", "关节炎"],
                context="手指关节晨僵是类风湿性关节炎的典型表现。"
            ),
            TestCase(
                question="我走路时腿疼，休息后缓解，是什么问题？",
                expected_intent="diagnosis_assist",
                expected_entities=["腿疼"],
                expected_answer_keywords=["间歇性跛行", "血管", "下肢"],
                context="走路时腿疼休息后缓解是间歇性跛行，提示下肢动脉缺血。"
            ),
            # ──────────────────────────────────────────────
            # 新增：health_advice 健康建议补充 (4 cases)
            # ──────────────────────────────────────────────
            TestCase(
                question="高血压患者日常饮食注意什么？",
                expected_intent="health_advice",
                expected_entities=["高血压"],
                expected_answer_keywords=["高血压", "饮食", "低盐", "控制"],
                context="高血压患者应低盐饮食、控制体重、限制饮酒。"
            ),
            TestCase(
                question="糖尿病患者能吃水果吗？",
                expected_intent="health_advice",
                expected_entities=["糖尿病"],
                expected_answer_keywords=["糖尿病", "水果", "血糖", "控制"],
                context="糖尿病患者在血糖控制良好时可适量食用低糖水果。"
            ),
            TestCase(
                question="久坐办公室如何保护腰椎？",
                expected_intent="health_advice",
                expected_entities=["腰椎"],
                expected_answer_keywords=["腰椎", "保护", "坐姿", "运动"],
                context="久坐应保持正确坐姿、定时起身活动、加强腰背肌锻炼。"
            ),
            TestCase(
                question="备孕期间需要补充什么营养？",
                expected_intent="health_advice",
                expected_entities=["备孕"],
                expected_answer_keywords=["备孕", "叶酸", "营养", "补充"],
                context="备孕期间应补充叶酸，保持均衡营养，戒烟戒酒。"
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
            print(f"评估 [{test_cases.index(tc)+1}/{len(test_cases)}]: {tc.question}")
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

        intent_accuracy = sum(1 for r in results if r.intent_correct) / len(results)
        
        total_expected = sum(len(r.test_case.expected_entities) for r in results)
        total_found = sum(r.entities_found for r in results)
        entity_recall = total_found / total_expected if total_expected > 0 else 0
        
        answer_relevance = sum(1 for r in results if r.answer_relevant) / len(results)
        
        overall_score = (intent_accuracy + entity_recall + answer_relevance) / 3

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
            "total_test_cases": len(self.results),
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
                    "expected_intent": r.test_case.expected_intent,
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
    print(f"开始问答功能评估... (共 {len(evaluator.load_test_cases())} 个测试用例)")
    
    quality, performance = evaluator.run_evaluation()
    evaluator.print_report(quality, performance)
    evaluator.save_report(quality, performance)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
