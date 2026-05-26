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
        ]
        
        return dataset