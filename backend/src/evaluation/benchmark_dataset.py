from typing import List, Dict, Any
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class BenchmarkItem:
    question: str
    reference_answer: str
    expected_intent: str
    expected_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class BenchmarkDataset:
    name: str
    items: List[BenchmarkItem] = field(default_factory=list)

    def load_from_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.items = [
            BenchmarkItem(**item) for item in data.get('items', [])
        ]
        self.name = data.get('name', self.name)

    def save_to_json(self, file_path: str):
        data = {
            'name': self.name,
            'items': [
                {
                    'question': item.question,
                    'reference_answer': item.reference_answer,
                    'expected_intent': item.expected_intent,
                    'expected_entities': item.expected_entities,
                    'keywords': item.keywords,
                    'category': item.category,
                    'difficulty': item.difficulty,
                } for item in self.items
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_by_category(self, category: str) -> 'BenchmarkDataset':
        return BenchmarkDataset(
            name=f"{self.name}_{category}",
            items=[item for item in self.items if item.category == category]
        )

    def get_by_difficulty(self, difficulty: str) -> 'BenchmarkDataset':
        return BenchmarkDataset(
            name=f"{self.name}_{difficulty}",
            items=[item for item in self.items if item.difficulty == difficulty]
        )

    def split_train_test(self, train_ratio: float = 0.8) -> tuple:
        import random
        shuffled = self.items.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train = BenchmarkDataset(name=f"{self.name}_train", items=shuffled[:split_idx])
        test = BenchmarkDataset(name=f"{self.name}_test", items=shuffled[split_idx:])
        
        return train, test

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class MedicalBenchmarkLoader:
    @staticmethod
    def load_medical_benchmark() -> BenchmarkDataset:
        dataset = BenchmarkDataset(name="medical_benchmark")
        
        dataset.items = [
            BenchmarkItem(
                question="高血压是什么疾病？",
                reference_answer="高血压是一种常见的慢性疾病，指血液在血管中流动时对血管壁造成的压力持续高于正常水平。长期高血压可导致心脑血管疾病等并发症。",
                expected_intent="disease_query",
                expected_entities=["高血压"],
                keywords=["高血压", "慢性", "血压", "并发症"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="高血压的主要症状有哪些？",
                reference_answer="高血压通常没有明显症状，因此被称为沉默的杀手。部分患者可能出现头痛、头晕、心悸、耳鸣、视力模糊等症状。",
                expected_intent="disease_query",
                expected_entities=["高血压", "症状"],
                keywords=["头痛", "头晕", "心悸", "耳鸣", "症状"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="阿司匹林有什么副作用？",
                reference_answer="阿司匹林常见副作用包括胃肠道不适、恶心、呕吐、出血风险增加等。长期服用可能损伤胃黏膜，严重时可导致胃溃疡。",
                expected_intent="drug_query",
                expected_entities=["阿司匹林"],
                keywords=["阿司匹林", "副作用", "胃肠道", "出血"],
                category="drug",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="我最近经常头痛头晕，可能是什么原因？",
                reference_answer="头痛头晕可能由多种原因引起，包括高血压、颈椎病、贫血、低血糖、睡眠不足、精神压力过大等。建议及时就医检查。",
                expected_intent="diagnosis_assist",
                expected_entities=["头痛", "头晕"],
                keywords=["头痛", "头晕", "原因", "高血压", "颈椎病"],
                category="symptom",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="糖尿病如何治疗？",
                reference_answer="糖尿病治疗包括饮食控制、规律运动、药物治疗和血糖监测。药物包括口服降糖药如二甲双胍、胰岛素注射等。",
                expected_intent="disease_query",
                expected_entities=["糖尿病"],
                keywords=["糖尿病", "治疗", "胰岛素", "饮食", "运动"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="心肌梗死的典型症状是什么？",
                reference_answer="心肌梗死典型症状为突发剧烈胸痛，常位于胸骨后或左胸部，可向左肩、左臂、颈部、下颌部放射，持续时间较长，休息或含服硝酸甘油不能缓解，常伴有大汗、呼吸困难、恶心呕吐等。",
                expected_intent="disease_query",
                expected_entities=["心肌梗死"],
                keywords=["心肌梗死", "症状", "胸痛", "大汗", "呼吸困难"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="感冒和流感有什么区别？",
                reference_answer="普通感冒症状较轻，以上呼吸道症状为主如流涕、鼻塞、咽痛；流感症状较重，全身症状明显如高热、头痛、肌肉酸痛、乏力等，并发症风险更高。",
                expected_intent="disease_query",
                expected_entities=["感冒", "流感"],
                keywords=["感冒", "流感", "区别", "症状", "高热"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="布洛芬能治疗什么疾病？",
                reference_answer="布洛芬属于非甾体抗炎药，用于缓解轻至中度疼痛如头痛、关节痛、牙痛、肌肉痛等，也用于退热和减轻炎症反应。",
                expected_intent="drug_query",
                expected_entities=["布洛芬"],
                keywords=["布洛芬", "治疗", "疼痛", "发热", "抗炎"],
                category="drug",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="如何预防心血管疾病？",
                reference_answer="预防心血管疾病需要保持健康生活方式：合理饮食、规律运动、戒烟限酒、控制体重、定期体检、管理血压血糖血脂等危险因素。",
                expected_intent="prevention_query",
                expected_entities=["心血管疾病"],
                keywords=["预防", "心血管", "健康", "生活方式", "饮食"],
                category="prevention",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="乙肝的传播途径有哪些？",
                reference_answer="乙肝主要通过血液传播、母婴传播和性接触传播。日常接触如握手、拥抱、共餐等不会传播乙肝病毒。",
                expected_intent="disease_query",
                expected_entities=["乙肝"],
                keywords=["乙肝", "传播", "途径", "血液", "母婴"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="二甲双胍的禁忌证有哪些？",
                reference_answer="二甲双胍禁用于严重肾功能不全、肝功能衰竭、严重感染、脱水、酗酒者、孕妇及哺乳期妇女、对二甲双胍过敏者。",
                expected_intent="drug_query",
                expected_entities=["二甲双胍"],
                keywords=["二甲双胍", "禁忌", "肾功能", "肝功能", "过敏"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="骨折后如何进行康复锻炼？",
                reference_answer="骨折康复锻炼应在医生指导下进行，早期进行肌肉等长收缩训练，中期进行关节活动训练，后期进行力量训练和功能恢复训练，循序渐进避免过度。",
                expected_intent="treatment_query",
                expected_entities=["骨折"],
                keywords=["骨折", "康复", "锻炼", "训练", "关节"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="脑梗死的后遗症有哪些？",
                reference_answer="脑梗死常见后遗症包括肢体偏瘫、言语障碍、认知功能下降、吞咽困难、情绪障碍等，康复训练有助于改善功能。",
                expected_intent="disease_query",
                expected_entities=["脑梗死"],
                keywords=["脑梗死", "后遗症", "偏瘫", "语言", "认知"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="甲状腺功能检查需要空腹吗？",
                reference_answer="甲状腺功能检查通常需要空腹采血，建议禁食8-12小时，以免食物影响检测结果准确性。",
                expected_intent="examination_query",
                expected_entities=["甲状腺"],
                keywords=["甲状腺", "检查", "空腹", "验血"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="抑郁症的早期症状是什么？",
                reference_answer="抑郁症早期症状包括情绪低落、兴趣减退、睡眠障碍、食欲改变、注意力不集中、疲劳乏力、自责自罪等，持续两周以上应及时就医。",
                expected_intent="disease_query",
                expected_entities=["抑郁症"],
                keywords=["抑郁症", "症状", "情绪", "兴趣", "睡眠"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="痛风发作时应该怎么办？",
                reference_answer="痛风急性发作时应卧床休息、抬高患肢、避免负重，可使用秋水仙碱、非甾体抗炎药或糖皮质激素缓解疼痛，同时多喝水促进尿酸排泄。",
                expected_intent="symptom_query",
                expected_entities=["痛风"],
                keywords=["痛风", "发作", "处理", "疼痛", "秋水仙碱"],
                category="symptom",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="肾结石多大需要手术治疗？",
                reference_answer="肾结石治疗方式取决于结石大小、位置和患者情况。一般直径大于1cm的结石可能需要手术，小于0.5cm可尝试保守治疗。",
                expected_intent="disease_query",
                expected_entities=["肾结石"],
                keywords=["肾结石", "手术", "治疗", "大小", "直径"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="帕金森病的主要症状是什么？",
                reference_answer="帕金森病主要症状包括静止性震颤、肌强直、运动迟缓、姿势步态异常等，还可能伴有认知障碍、情绪改变、睡眠障碍等非运动症状。",
                expected_intent="disease_query",
                expected_entities=["帕金森"],
                keywords=["帕金森", "症状", "震颤", "运动", "步态"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="如何区分普通感冒和流感？",
                reference_answer="普通感冒症状较轻，以局部症状为主如流涕鼻塞；流感症状重，全身症状明显如高热寒战、肌肉酸痛，发病急，并发症风险高。",
                expected_intent="disease_query",
                expected_entities=["感冒", "流感"],
                keywords=["感冒", "流感", "区别", "症状", "高热"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="类风湿性关节炎有哪些表现？",
                reference_answer="类风湿性关节炎主要表现为对称性关节肿痛、晨僵、关节畸形、功能障碍，常累及手指、手腕、膝盖等关节，可伴有疲劳、低热等全身症状。",
                expected_intent="disease_query",
                expected_entities=["类风湿性关节炎"],
                keywords=["类风湿", "关节炎", "症状", "关节", "晨僵"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="支气管哮喘急性发作怎么处理？",
                reference_answer="哮喘急性发作时应立即使用速效支气管舒张剂如沙丁胺醇吸入剂，保持呼吸道通畅，吸氧，严重时需立即就医或拨打急救电话。",
                expected_intent="symptom_query",
                expected_entities=["哮喘"],
                keywords=["哮喘", "发作", "处理", "沙丁胺醇", "呼吸"],
                category="symptom",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="脂肪肝患者饮食上需要注意什么？",
                reference_answer="脂肪肝患者应控制总热量摄入，减少高脂肪高糖分食物，增加膳食纤维，戒酒，控制体重，适量运动，避免滥用药物。",
                expected_intent="health_advice",
                expected_entities=["脂肪肝"],
                keywords=["脂肪肝", "饮食", "注意", "控制", "戒酒"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="我反复口腔溃疡是什么原因？",
                reference_answer="反复口腔溃疡可能与免疫功能紊乱、营养缺乏、精神压力、局部刺激、遗传因素等有关，也可能是某些全身性疾病的表现。",
                expected_intent="diagnosis_assist",
                expected_entities=["口腔溃疡"],
                keywords=["口腔溃疡", "原因", "免疫", "维生素", "压力"],
                category="symptom",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="骨质疏松怎么补钙最有效？",
                reference_answer="骨质疏松补钙需结合维生素D促进吸收，摄入富含钙的食物如牛奶、豆制品、深绿色蔬菜，适度负重运动，避免过量饮用咖啡和碳酸饮料。",
                expected_intent="health_advice",
                expected_entities=["骨质疏松"],
                keywords=["骨质疏松", "补钙", "维生素D", "运动", "食物"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="脑出血后遗症康复需要注意什么？",
                reference_answer="脑出血康复需在专业指导下进行，包括肢体功能训练、语言训练、认知训练，注意预防并发症，保持积极心态，循序渐进坚持康复。",
                expected_intent="treatment_query",
                expected_entities=["脑出血"],
                keywords=["脑出血", "康复", "注意", "功能", "训练"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="荨麻疹会传染吗？",
                reference_answer="荨麻疹是一种过敏性疾病，不具有传染性。其发病与过敏体质、食物药物过敏、感染等因素有关，不会通过接触传播给他人。",
                expected_intent="disease_query",
                expected_entities=["荨麻疹"],
                keywords=["荨麻疹", "传染", "过敏", "皮肤", "传播"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="我最近总是失眠，是什么原因？",
                reference_answer="失眠原因包括精神压力过大、焦虑抑郁、不良睡眠习惯、疾病因素、药物影响、环境因素等。改善睡眠需要综合调理。",
                expected_intent="diagnosis_assist",
                expected_entities=["失眠"],
                keywords=["失眠", "原因", "睡眠", "压力", "焦虑"],
                category="symptom",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="贫血患者应该吃什么来补血？",
                reference_answer="贫血患者应多食用富含铁的食物如红肉、动物肝脏、动物血制品、豆类、深绿色蔬菜，同时摄入富含维生素C的食物促进铁吸收。",
                expected_intent="health_advice",
                expected_entities=["贫血"],
                keywords=["贫血", "补血", "食物", "铁", "维生素C"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="慢性咽炎吃什么药效果好？",
                reference_answer="慢性咽炎可使用含漱液、含片缓解症状，中成药如利咽解毒颗粒等。关键在于去除病因如戒烟限酒、避免辛辣刺激食物、改善环境。",
                expected_intent="drug_query",
                expected_entities=["咽炎"],
                keywords=["咽炎", "药物", "治疗", "慢性", "含片"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="阿尔茨海默症如何延缓病情发展？",
                reference_answer="阿尔茨海默症可通过药物治疗、认知训练、保持社交活动、健康饮食、规律运动、控制血压血糖等方式延缓病情进展。",
                expected_intent="disease_query",
                expected_entities=["阿尔茨海默症"],
                keywords=["阿尔茨海默症", "延缓", "治疗", "认知", "药物"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="血常规检查能查出什么？",
                reference_answer="血常规可检测白细胞、红细胞、血小板计数及血红蛋白等，用于诊断贫血、感染、炎症、凝血功能障碍等疾病。",
                expected_intent="examination_query",
                expected_entities=["血常规"],
                keywords=["血常规", "检查", "红细胞", "白细胞", "血小板"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="做CT检查有辐射吗？",
                reference_answer="CT检查有一定辐射，单次常规CT检查辐射剂量较低，在安全范围内，但不宜频繁进行CT检查，尤其是孕妇应避免。",
                expected_intent="examination_query",
                expected_entities=["CT"],
                keywords=["CT", "检查", "辐射", "安全", "孕妇"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="MRI和CT有什么区别？",
                reference_answer="MRI利用磁场成像，无辐射，对软组织显影好；CT利用X射线成像，有辐射，对骨骼和肺部显影好，两者各有优势。",
                expected_intent="examination_query",
                expected_entities=["MRI", "CT"],
                keywords=["MRI", "CT", "区别", "辐射", "成像"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="肝功能检查需要空腹吗？",
                reference_answer="肝功能检查需要空腹采血，建议禁食8-12小时，前一天避免高脂饮食和饮酒，以免影响检测结果的准确性。",
                expected_intent="examination_query",
                expected_entities=["肝功能"],
                keywords=["肝功能", "检查", "空腹", "饮食", "酒精"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="心电图检查主要查什么？",
                reference_answer="心电图检查用于检测心脏电活动，可诊断心律失常、心肌缺血、心肌梗死、心脏传导异常等心血管疾病。",
                expected_intent="examination_query",
                expected_entities=["心电图"],
                keywords=["心电图", "检查", "心脏", "心律失常", "心肌缺血"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="超声检查有哪些种类？",
                reference_answer="超声检查包括腹部超声、心脏超声、血管超声、妇科超声、产科超声、乳腺超声、甲状腺超声等多种类型。",
                expected_intent="examination_query",
                expected_entities=["超声"],
                keywords=["超声", "检查", "种类", "心脏", "腹部"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="尿常规检查能发现什么问题？",
                reference_answer="尿常规可检测尿液中的蛋白质、葡萄糖、酮体、红细胞、白细胞等，用于诊断尿路感染、肾病、糖尿病等疾病。",
                expected_intent="examination_query",
                expected_entities=["尿常规"],
                keywords=["尿常规", "检查", "尿液", "感染", "肾"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="胃镜检查前需要准备什么？",
                reference_answer="胃镜检查前需空腹6-8小时，停用抗凝药物，取下假牙，告知医生病史和过敏史，检查后1-2小时再进食。",
                expected_intent="examination_query",
                expected_entities=["胃镜"],
                keywords=["胃镜", "检查", "准备", "空腹", "药物"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="血糖正常值是多少？",
                reference_answer="空腹血糖正常值为3.9-6.1mmol/L，餐后2小时血糖应小于7.8mmol/L，随机血糖应小于11.1mmol/L。",
                expected_intent="examination_query",
                expected_entities=["血糖"],
                keywords=["血糖", "正常值", "空腹", "餐后"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="血脂检查包括哪些项目？",
                reference_answer="血脂检查通常包括总胆固醇、甘油三酯、低密度脂蛋白胆固醇（LDL-C）、高密度脂蛋白胆固醇（HDL-C）等项目。",
                expected_intent="examination_query",
                expected_entities=["血脂"],
                keywords=["血脂", "检查", "胆固醇", "甘油三酯", "脂蛋白"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="做肠镜检查痛苦吗？",
                reference_answer="普通肠镜可能会有腹胀、腹痛等不适，无痛肠镜使用麻醉，检查过程中无痛苦，可根据情况选择。",
                expected_intent="examination_query",
                expected_entities=["肠镜"],
                keywords=["肠镜", "检查", "痛苦", "无痛", "麻醉"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="胸片和胸部CT有什么区别？",
                reference_answer="胸片是二维X线检查，价格低、辐射小，适合初筛；胸部CT是断层扫描，分辨率高，能发现更小的病变，但辐射较大。",
                expected_intent="examination_query",
                expected_entities=["CT", "X光"],
                keywords=["胸片", "CT", "区别", "胸部", "辐射"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="肿瘤标志物检查能确诊癌症吗？",
                reference_answer="肿瘤标志物升高不能直接确诊癌症，也可见于良性疾病，需要结合影像、病理等检查综合判断。",
                expected_intent="examination_query",
                expected_entities=["肿瘤标志物"],
                keywords=["肿瘤标志物", "检查", "癌症", "诊断", "病理"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="动态心电图（Holter）主要查什么？",
                reference_answer="动态心电图24小时监测心脏电活动，用于发现阵发性心律失常、心肌缺血、评估药物疗效等。",
                expected_intent="examination_query",
                expected_entities=["动态心电图", "心电图"],
                keywords=["动态心电图", "Holter", "检查", "心律失常", "24小时"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="腰穿（腰椎穿刺）检查危险吗？",
                reference_answer="腰椎穿刺是相对安全的检查，可能有短暂头痛、局部不适等，但严重并发症罕见，需在专业医生操作下进行。",
                expected_intent="examination_query",
                expected_entities=["腰椎穿刺"],
                keywords=["腰椎穿刺", "腰穿", "检查", "危险", "并发症"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="病理检查是查什么的？",
                reference_answer="病理检查通过显微镜观察病变组织的细胞形态和结构，是诊断肿瘤、炎症等疾病的金标准。",
                expected_intent="examination_query",
                expected_entities=["病理检查"],
                keywords=["病理", "检查", "肿瘤", "诊断", "金标准"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="肺功能检查需要注意什么？",
                reference_answer="肺功能检查需要配合医生口令进行吸气、呼气，检查前避免剧烈运动、吸烟、饮酒，气胸及严重心脑血管疾病患者慎做。",
                expected_intent="examination_query",
                expected_entities=["肺功能检查"],
                keywords=["肺功能", "检查", "注意", "呼吸", "气胸"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="核酸检测多久出结果？",
                reference_answer="核酸检测通常4-24小时出结果，具体时间根据检测机构和标本量有所不同，急诊检测可能更快。",
                expected_intent="examination_query",
                expected_entities=["核酸检测"],
                keywords=["核酸检测", "检查", "结果", "时间", "急诊"],
                category="examination",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="心脏彩超能查什么？",
                reference_answer="心脏彩超可检查心脏结构和功能，评估心室壁厚度、心脏瓣膜、心功能、射血分数等，用于诊断心脏病。",
                expected_intent="examination_query",
                expected_entities=["心脏彩超"],
                keywords=["心脏彩超", "检查", "心脏", "瓣膜", "功能"],
                category="examination",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="什么情况下需要做血管造影？",
                reference_answer="血管造影用于诊断血管狭窄、动脉瘤、血管畸形等疾病，常用于冠心病、脑血管病、外周血管病的检查。",
                expected_intent="examination_query",
                expected_entities=["血管造影"],
                keywords=["血管造影", "检查", "血管", "冠心病", "动脉瘤"],
                category="examination",
                difficulty="medium"
            ),
            # ──────────────────────────────────────────────
            # 新增：drug 药物查询补充 (10 cases)
            # ──────────────────────────────────────────────
            BenchmarkItem(
                question="奥美拉唑是治什么病的？",
                reference_answer="奥美拉唑是质子泵抑制剂，用于治疗胃溃疡、十二指肠溃疡、胃食管反流病等胃酸相关疾病。",
                expected_intent="drug_query",
                expected_entities=["奥美拉唑"],
                keywords=["奥美拉唑", "胃酸", "溃疡", "质子泵"],
                category="drug",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="阿莫西林属于哪类抗生素？",
                reference_answer="阿莫西林属于青霉素类广谱抗生素，用于治疗敏感菌引起的呼吸道感染、尿路感染、皮肤感染等。",
                expected_intent="drug_query",
                expected_entities=["阿莫西林"],
                keywords=["阿莫西林", "青霉素", "抗生素", "感染"],
                category="drug",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="氨氯地平有什么副作用？",
                reference_answer="氨氯地平常见副作用包括下肢水肿、头痛、面部潮红、心悸等，少数患者可出现头晕和乏力。",
                expected_intent="drug_query",
                expected_entities=["氨氯地平"],
                keywords=["氨氯地平", "副作用", "水肿", "头痛"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="氯吡格雷的作用是什么？",
                reference_answer="氯吡格雷是抗血小板药物，通过抑制血小板聚集预防动脉血栓形成，常用于冠心病和脑梗死的二级预防。",
                expected_intent="drug_query",
                expected_entities=["氯吡格雷"],
                keywords=["氯吡格雷", "抗血小板", "血栓", "预防"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="蒙脱石散怎么服用？",
                reference_answer="蒙脱石散用于治疗急慢性腹泻，需空腹服用，将药粉倒入半杯温水中搅匀后服下，与其他药物间隔1-2小时。",
                expected_intent="drug_query",
                expected_entities=["蒙脱石散"],
                keywords=["蒙脱石散", "服用", "腹泻", "空腹"],
                category="drug",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="辛伐他汀的用法用量？",
                reference_answer="辛伐他汀是降脂药，常用剂量为每晚10-40mg，建议晚间服用（胆固醇合成高峰在夜间），需定期监测肝功能和肌酸激酶。",
                expected_intent="drug_query",
                expected_entities=["辛伐他汀"],
                keywords=["辛伐他汀", "降脂", "晚间", "肝功能"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="头孢克洛和头孢克肟有什么区别？",
                reference_answer="头孢克洛属第二代头孢菌素，对革兰阳性菌作用较强；头孢克肟属第三代头孢菌素，对革兰阴性菌作用更强。两者抗菌谱不同，适应证也有差异。",
                expected_intent="drug_query",
                expected_entities=["头孢克洛", "头孢克肟"],
                keywords=["头孢", "区别", "第二代", "第三代", "抗菌谱"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="氯雷他定能长期服用吗？",
                reference_answer="氯雷他定是第二代抗组胺药，用于过敏性疾病。一般短期使用安全，长期服用需在医生指导下进行，注意监测肝功能。",
                expected_intent="drug_query",
                expected_entities=["氯雷他定"],
                keywords=["氯雷他定", "长期", "过敏", "抗组胺"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="甲硝唑能治什么感染？",
                reference_answer="甲硝唑对厌氧菌和部分原虫有效，用于治疗滴虫性阴道炎、阿米巴病、厌氧菌感染、幽门螺杆菌感染等。服药期间及停药后3天内禁止饮酒。",
                expected_intent="drug_query",
                expected_entities=["甲硝唑"],
                keywords=["甲硝唑", "厌氧菌", "滴虫", "饮酒"],
                category="drug",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="地塞米松是激素药吗？",
                reference_answer="地塞米松是糖皮质激素类药物，具有抗炎、抗过敏、免疫抑制等作用。短期使用较安全，长期使用需注意骨质疏松、血糖升高、感染风险增加等副作用。",
                expected_intent="drug_query",
                expected_entities=["地塞米松"],
                keywords=["地塞米松", "激素", "糖皮质激素", "副作用"],
                category="drug",
                difficulty="easy"
            ),
            # ──────────────────────────────────────────────
            # 新增：treatment 治疗查询补充 (8 cases)
            # ──────────────────────────────────────────────
            BenchmarkItem(
                question="冠心病的治疗方法有哪些？",
                reference_answer="冠心病治疗包括药物治疗（抗血小板、他汀、β受体阻滞剂等）、介入治疗（支架植入）和冠脉搭桥手术，根据病情严重程度选择。",
                expected_intent="treatment_query",
                expected_entities=["冠心病"],
                keywords=["冠心病", "治疗", "支架", "药物", "手术"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="胃溃疡怎么治疗？",
                reference_answer="胃溃疡治疗包括抑酸药物（质子泵抑制剂）、胃黏膜保护剂和根除幽门螺杆菌治疗，疗程通常6-8周。",
                expected_intent="treatment_query",
                expected_entities=["胃溃疡"],
                keywords=["胃溃疡", "治疗", "抑酸", "幽门螺杆菌"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="甲亢的治疗方案有哪些？",
                reference_answer="甲亢治疗包括抗甲状腺药物（甲巯咪唑、丙硫氧嘧啶）、放射性碘治疗和手术切除甲状腺，根据病情和患者情况选择。",
                expected_intent="treatment_query",
                expected_entities=["甲亢"],
                keywords=["甲亢", "治疗", "抗甲状腺", "放射性碘", "手术"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="腰椎间盘突出怎么治疗？",
                reference_answer="腰椎间盘突出首选保守治疗（卧床休息、牵引、理疗、药物），保守治疗无效或出现马尾综合征时需手术治疗。",
                expected_intent="treatment_query",
                expected_entities=["腰椎间盘突出"],
                keywords=["腰椎间盘突出", "治疗", "保守", "手术", "牵引"],
                category="treatment",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="慢性肾衰竭怎么治疗？",
                reference_answer="慢性肾衰竭治疗包括控制原发病、延缓肾功能恶化（ACEI/ARB）、纠正并发症（贫血、骨病），终末期需透析或肾移植。",
                expected_intent="treatment_query",
                expected_entities=["慢性肾衰竭"],
                keywords=["肾衰竭", "治疗", "透析", "移植", "延缓"],
                category="treatment",
                difficulty="hard"
            ),
            BenchmarkItem(
                question="白内障需要手术吗？",
                reference_answer="白内障是晶状体混浊导致的视力下降，手术是唯一有效治疗方法。当视力下降影响日常生活时即可考虑手术，无需等到完全看不见。",
                expected_intent="treatment_query",
                expected_entities=["白内障"],
                keywords=["白内障", "手术", "晶状体", "视力"],
                category="treatment",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="房颤怎么治疗？",
                reference_answer="房颤治疗包括心率控制（β受体阻滞剂、地高辛）、节律控制（胺碘酮复律）、抗凝预防脑卒中（华法林或新型口服抗凝药），必要时可行导管消融术。",
                expected_intent="treatment_query",
                expected_entities=["房颤"],
                keywords=["房颤", "治疗", "抗凝", "复律", "消融"],
                category="treatment",
                difficulty="hard"
            ),
            BenchmarkItem(
                question="肩周炎怎么治疗和康复？",
                reference_answer="肩周炎治疗包括止痛药物、局部封闭注射、理疗和功能锻炼。康复关键是坚持肩关节活动度训练（爬墙运动、钟摆运动等），避免长期不动导致关节僵硬。",
                expected_intent="treatment_query",
                expected_entities=["肩周炎"],
                keywords=["肩周炎", "治疗", "康复", "锻炼", "理疗"],
                category="treatment",
                difficulty="medium"
            ),
            # ──────────────────────────────────────────────
            # 新增：prevention 预防查询补充 (6 cases)
            # ──────────────────────────────────────────────
            BenchmarkItem(
                question="如何预防糖尿病？",
                reference_answer="预防糖尿病需合理饮食（减少精制糖和高脂食物）、规律运动（每周150分钟中等强度）、控制体重、定期检测血糖，尤其有家族史者更需注意。",
                expected_intent="prevention_query",
                expected_entities=["糖尿病"],
                keywords=["糖尿病", "预防", "饮食", "运动", "血糖"],
                category="prevention",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="怎样预防骨质疏松？",
                reference_answer="预防骨质疏松需补充钙和维生素D、适度负重运动、避免吸烟和过量饮酒、定期检测骨密度，绝经后女性更需关注。",
                expected_intent="prevention_query",
                expected_entities=["骨质疏松"],
                keywords=["骨质疏松", "预防", "钙", "维生素D", "运动"],
                category="prevention",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="如何预防肝癌？",
                reference_answer="预防肝癌需接种乙肝疫苗、积极治疗慢性乙肝/丙肝、戒酒、避免食用霉变食物（含黄曲霉毒素）、定期体检筛查。",
                expected_intent="prevention_query",
                expected_entities=["肝癌"],
                keywords=["肝癌", "预防", "乙肝", "戒酒", "黄曲霉毒素"],
                category="prevention",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="怎样预防流感传播？",
                reference_answer="预防流感需每年接种流感疫苗、勤洗手、咳嗽打喷嚏遮住口鼻、避免去人群密集场所、保持室内通风。",
                expected_intent="prevention_query",
                expected_entities=["流感"],
                keywords=["流感", "预防", "疫苗", "洗手", "通风"],
                category="prevention",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="如何预防脑卒中？",
                reference_answer="预防脑卒中需控制高血压、糖尿病、高血脂等危险因素，戒烟限酒，规律运动，保持健康体重，有心房颤动者需规范抗凝治疗。",
                expected_intent="prevention_query",
                expected_entities=["脑卒中"],
                keywords=["脑卒中", "预防", "高血压", "抗凝", "戒烟"],
                category="prevention",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="如何预防深静脉血栓？",
                reference_answer="预防深静脉血栓需避免久坐不动、术后早期下床活动、必要时使用弹力袜或抗凝药物、多饮水、长途旅行中定时活动下肢。",
                expected_intent="prevention_query",
                expected_entities=["深静脉血栓"],
                keywords=["深静脉血栓", "预防", "活动", "弹力袜", "抗凝"],
                category="prevention",
                difficulty="medium"
            ),
            # ──────────────────────────────────────────────
            # 新增：health_advice 健康建议补充 (6 cases)
            # ──────────────────────────────────────────────
            BenchmarkItem(
                question="高血压患者日常饮食注意什么？",
                reference_answer="高血压患者应低盐饮食（每日盐摄入<6g）、多吃蔬果、控制体重、限制饮酒、避免高脂高胆固醇食物。",
                expected_intent="health_advice",
                expected_entities=["高血压"],
                keywords=["高血压", "饮食", "低盐", "蔬果", "控制体重"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="糖尿病患者能吃水果吗？",
                reference_answer="糖尿病患者在血糖控制良好时可适量食用低糖水果（如苹果、梨、柚子），应在两餐之间食用，避免高糖水果（如荔枝、龙眼），并计入每日总热量。",
                expected_intent="health_advice",
                expected_entities=["糖尿病"],
                keywords=["糖尿病", "水果", "血糖", "低糖", "热量"],
                category="health_advice",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="久坐办公室如何保护腰椎？",
                reference_answer="久坐应保持正确坐姿（腰部有支撑、双脚平放）、每45分钟起身活动、加强腰背肌锻炼（如小燕飞）、避免长时间弯腰。",
                expected_intent="health_advice",
                expected_entities=["腰椎"],
                keywords=["腰椎", "坐姿", "锻炼", "久坐", "腰背肌"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="备孕期间需要补充什么营养？",
                reference_answer="备孕期间女性应每日补充0.4-0.8mg叶酸（预防神经管缺陷），保持均衡营养，戒烟戒酒，男性也应注意戒烟戒酒和补充锌。",
                expected_intent="health_advice",
                expected_entities=["备孕"],
                keywords=["备孕", "叶酸", "营养", "戒烟", "均衡"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="经常熬夜对身体有什么危害？",
                reference_answer="长期熬夜可导致免疫力下降、内分泌紊乱、心血管疾病风险增加、认知功能下降、情绪障碍、肥胖等，建议保持规律作息。",
                expected_intent="health_advice",
                expected_entities=["熬夜"],
                keywords=["熬夜", "危害", "免疫", "心血管", "作息"],
                category="health_advice",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="如何科学减肥？",
                reference_answer="科学减肥需控制总热量摄入（减少高脂高糖食物）、增加运动量（有氧+力量训练）、每周减重0.5-1kg为宜、避免极端节食、保证充足睡眠。",
                expected_intent="health_advice",
                expected_entities=["减肥"],
                keywords=["减肥", "热量", "运动", "科学", "节食"],
                category="health_advice",
                difficulty="easy"
            ),
            # ──────────────────────────────────────────────
            # 新增：disease 疾病查询补充 (8 cases)
            # ──────────────────────────────────────────────
            BenchmarkItem(
                question="系统性红斑狼疮是什么病？",
                reference_answer="系统性红斑狼疮是一种自身免疫性疾病，免疫系统攻击自身组织，可累及皮肤、关节、肾脏、血液等多个器官系统，好发于育龄女性。",
                expected_intent="disease_query",
                expected_entities=["系统性红斑狼疮"],
                keywords=["红斑狼疮", "自身免疫", "全身", "女性"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="慢性阻塞性肺疾病的病因是什么？",
                reference_answer="慢阻肺主要病因包括长期吸烟（最常见）、空气污染、职业粉尘和化学物质暴露、儿童期反复呼吸道感染等。",
                expected_intent="disease_query",
                expected_entities=["慢性阻塞性肺疾病"],
                keywords=["慢阻肺", "病因", "吸烟", "空气污染"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="肝硬化能治好吗？",
                reference_answer="肝硬化是肝脏不可逆的纤维化改变，无法完全治愈，但可通过病因治疗（如抗病毒、戒酒）延缓进展，代偿期可长期稳定。",
                expected_intent="disease_query",
                expected_entities=["肝硬化"],
                keywords=["肝硬化", "不可逆", "延缓", "抗病毒"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="甲状腺功能减退有什么表现？",
                reference_answer="甲减表现为乏力、怕冷、体重增加、皮肤干燥、便秘、心率减慢、月经紊乱等，严重者可出现黏液性水肿。",
                expected_intent="disease_query",
                expected_entities=["甲状腺功能减退"],
                keywords=["甲减", "乏力", "怕冷", "体重", "黏液性水肿"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="阑尾炎的典型症状是什么？",
                reference_answer="阑尾炎典型症状为转移性右下腹痛（先上腹或脐周痛，后转移至右下腹），伴恶心呕吐、发热，右下腹麦氏点压痛。",
                expected_intent="disease_query",
                expected_entities=["阑尾炎"],
                keywords=["阑尾炎", "转移性腹痛", "右下腹", "麦氏点"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="带状疱疹会传染吗？",
                reference_answer="带状疱疹本身不直接传染，但水疱液中的水痘-带状疱疹病毒可致未免疫者感染水痘。避免接触水疱液，尤其孕妇和儿童。",
                expected_intent="disease_query",
                expected_entities=["带状疱疹"],
                keywords=["带状疱疹", "传染", "水痘", "水疱"],
                category="disease",
                difficulty="easy"
            ),
            BenchmarkItem(
                question="慢性肾炎会发展成尿毒症吗？",
                reference_answer="慢性肾炎如不及时规范治疗，部分患者可逐渐进展为慢性肾衰竭和尿毒症。早期诊断和规范治疗可延缓肾功能恶化。",
                expected_intent="disease_query",
                expected_entities=["慢性肾炎", "尿毒症"],
                keywords=["慢性肾炎", "尿毒症", "进展", "延缓"],
                category="disease",
                difficulty="medium"
            ),
            BenchmarkItem(
                question="胃食管反流病怎么引起的？",
                reference_answer="胃食管反流病由下食管括约肌功能障碍导致胃内容物反流入食管，诱因包括肥胖、暴饮暴食、卧位进食、吸烟饮酒等。",
                expected_intent="disease_query",
                expected_entities=["胃食管反流"],
                keywords=["胃食管反流", "括约肌", "反流", "诱因"],
                category="disease",
                difficulty="medium"
            ),
        ]
        
        return dataset