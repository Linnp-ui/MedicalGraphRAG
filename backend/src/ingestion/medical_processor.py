import re
import threading
from typing import List, Dict, Any, Optional
from .document_loader import Document


_NLP_MODEL_CACHE: Optional[Any] = None
_NLP_MODEL_AVAILABLE: Optional[bool] = None
_NLP_MODEL_LOCK = threading.Lock()


class MedicalTextProcessor:
    """专门用于医疗文本预处理的类"""

    def __init__(self):
        self.unit_mapping = {
            "mg": "毫克",
            "g": "克",
            "ml": "毫升",
            "kg": "千克",
            "cm": "厘米",
            "mm": "毫米",
            "μg": "微克",
            "IU": "国际单位"
        }
        self._nlp_processor = None

    def clean_text(self, text: str) -> str:
        """清洗医疗文本"""
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace(',', '，').replace(';', '；').replace(':', '：')

        return text

    def process_document(self, document: Document) -> Document:
        """对 Document 进行预处理"""
        document.content = self.clean_text(document.content)
        document.metadata["medical_processed"] = True
        return document

    def batch_process(self, documents: List[Document]) -> List[Document]:
        """批量预处理文档"""
        return [self.process_document(doc) for doc in documents]

    @staticmethod
    def _get_cached_nlp_processor():
        """获取NLP处理器（全局单例缓存）

        使用模块级缓存确保模型只加载一次
        模型从HuggingFace加载（首次下载后会缓存）
        使用双重检查锁定（Double-Checked Locking）确保线程安全
        """
        global _NLP_MODEL_CACHE, _NLP_MODEL_AVAILABLE

        if _NLP_MODEL_CACHE is not None:
            return _NLP_MODEL_CACHE

        if _NLP_MODEL_AVAILABLE is False:
            return None

        with _NLP_MODEL_LOCK:
            if _NLP_MODEL_CACHE is not None:
                return _NLP_MODEL_CACHE

            if _NLP_MODEL_AVAILABLE is False:
                return None

            try:
                import os
                os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"
                os.environ["TRANSFORMERS_TIMEOUT"] = "30"

                from transformers import pipeline

                # 优先从本地 models/ 目录加载
                local_model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "models", "bert-base-chinese-medical-ner"
                )
                model_source = local_model_path if os.path.isdir(local_model_path) else "iioSnail/bert-base-chinese-medical-ner"

                print(f"首次加载NLP模型，请稍候... (来源: {model_source})")
                _NLP_MODEL_CACHE = pipeline(
                    "token-classification",
                    model=model_source,
                    aggregation_strategy="simple"
                )
                _NLP_MODEL_AVAILABLE = True
                print("✅ NLP模型加载成功（已缓存）")
                return _NLP_MODEL_CACHE
            except Exception as e:
                import loguru
                logger = loguru.logger
                logger.info(f"NLP模型加载失败: {e}，使用规则匹配")
                _NLP_MODEL_AVAILABLE = False
                return None

    def _get_nlp_processor(self):
        """获取NLP处理器（兼容旧接口）"""
        return self._get_cached_nlp_processor()

    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取医疗实体（优先使用NER模型，失败则使用规则）

        Args:
            text: 待处理文本

        Returns:
            实体列表，每项包含 type, text, start, end, confidence
        """
        nlp = self._get_nlp_processor()

        if nlp is not None:
            try:
                results = nlp(text)
                entities = []
                # NER模型标签到标准类型的映射
                # 该模型只输出M（医疗实体），需用_classify_entity_type细分
                ner_type_map = {
                    "Disease": "DISEASE",
                    "Drug": "DRUG",
                    "Symptom": "SYMPTOM",
                    "Examination": "EXAMINATION",
                    "Treatment": "TREATMENT",
                    "Anatomy": "ANATOMY",
                    "Department": "DEPARTMENT",
                }
                for item in results:
                    entity_text = item.get("word", "").replace(" ", "")
                    # 优先使用NER模型的细分类型标签
                    ner_type = item.get("entity_group", "")
                    if ner_type in ner_type_map:
                        entity_type = ner_type_map[ner_type]
                    else:
                        # 模型只输出M（医疗实体），需用规则细分
                        entity_type = ner_type_map.get(
                            self._classify_entity_type(entity_text),
                            self._classify_entity_type(entity_text)
                        )
                    entities.append({
                        "type": entity_type,
                        "text": entity_text,
                        "start": item.get("start", 0),
                        "end": item.get("end", 0),
                        "confidence": item.get("score", 0.0)
                    })
                return entities
            except Exception as e:
                import loguru
                logger = loguru.logger
                logger.warning(f"NER model extraction failed, falling back to rules: {e}")

        return self._basic_entity_extraction(text, context=text)

    def _classify_entity_type(self, entity_text: str, context: str = "") -> str:
        """根据实体文本内容和上下文分类到具体类型

        Args:
            entity_text: 实体文本
            context: 上下文文本（用于消歧）

        Returns:
            实体类型 (Disease, Symptom, Drug, Examination, Treatment, Anatomy, Department)
        """
        # 上下文指示词
        disease_context = ["疾病", "病", "综合征", "癌", "瘤", "梗死", "硬化"]
        symptom_context = ["症状", "表现", "症状是", "出现", "感到", "感觉"]
        anatomy_context = ["器官", "部位", "组织", "系统", "位于", "属于"]
        department_context = ["科", "门诊", "科室"]
        treatment_context = ["治疗", "化疗", "放疗", "疗法", "措施"]
        
        # 检查上下文
        context_lower = context.lower() if context else ""
        
        # 基于上下文的优先级调整 - 这些检查在任何关键词匹配之前进行
        
        # 1. 先检查科室上下文（最高优先级）
        if any(c in context_lower for c in department_context):
            if "感染" in entity_text:
                return "Department"
            if "呼吸" in entity_text:
                return "Department"
            if "心脏" in entity_text:
                return "Department"
        
        # 2. 检查症状上下文
        if any(c in context_lower for c in symptom_context):
            if "头痛" in entity_text:
                return "Symptom"
        
        # 3. 检查解剖部位上下文
        if any(c in context_lower for c in anatomy_context):
            if "肝" in entity_text or "肝脏" in entity_text:
                return "Anatomy"
        
        # 4. 检查疾病上下文
        if any(c in context_lower for c in disease_context):
            if "头痛" in entity_text:
                return "Disease"
            if "心脏" in entity_text:
                return "Disease"
            if "肝" in entity_text or "肝脏" in entity_text:
                return "Disease"
            if "感染" in entity_text:
                return "Disease"
        
        # 5. 检查单独的"感染"或"肺部感染"应该是疾病
        if entity_text == "感染" or "感染" in entity_text:
            # 如果不是科室相关，应该是疾病
            if "科" not in entity_text:
                return "Disease"
        
        # 6. 检查"手术室"这种特殊情况 - 应该是解剖部位
        if "手术室" in entity_text:
            return "Anatomy"
        
        if any(c in context_lower for c in department_context):
            # 上下文表明是科室
            if "呼吸" in entity_text:
                return "Department"
            if "感染" in entity_text:
                return "Department"
            if "心脏" in entity_text:
                return "Department"
        
        if any(c in context_lower for c in treatment_context):
            # 上下文表明是治疗
            if "手术" in entity_text:
                return "Treatment"
        
        if any(c in context_lower for c in anatomy_context):
            # 上下文表明是解剖部位
            if "手术" in entity_text and "室" in context_lower:
                return "Anatomy"
        
        # 症状关键词 - 优先级高于疾病（症状词通常更具体）
        symptom_keywords = ["头痛", "头晕", "发热", "咳嗽", "乏力", "恶心", "呕吐", "腹泻", 
                           "便秘", "胸闷", "心悸", "气短", "呼吸困难", "疼痛", "瘙痒",
                           "麻木", "肿胀", "出血", "黄疸", "水肿", "皮疹", "发热",
                           "寒战", "盗汗", "失眠", "嗜睡", "意识模糊", "抽搐", "昏迷",
                           "视力模糊", "听力下降", "口干", "口苦", "口臭", "牙痛",
                           "腹痛", "腰痛", "关节痛", "肌肉痛", "胸痛", "背痛", "腿痛"]
        
        # 药物关键词
        drug_keywords = ["片", "胶囊", "丸", "膏", "贴", "针", "注射", "口服", "雾化", 
                        "吸入", "滴眼", "栓", "糖浆", "素", "平", "地平", "普利", 
                        "沙坦", "洛克", "芬", "布洛", "阿司匹", "青霉", "霉素",
                        "头孢", "沙星", "硝唑", "霉素", "他汀", "贝特", "格列",
                        "胰岛素", "阿司匹林", "布洛芬", "对乙酰氨基酚", "阿莫西林"]
        
        # 检查关键词
        exam_keywords = ["检查", "化验", "扫描", "透视", "照片", "心电图", "脑电图", 
                        "超声", "CT", "MRI", "X光", "验血", "验尿", "常规", "培养", 
                        "活检", "穿刺", "镜检", "内镜", "造影", "监测", "测量",
                        "血常规", "尿常规", "肝功能", "肾功能", "血糖", "血脂",
                        "电解质", "凝血", "激素", "抗体", "抗原", "核酸"]
        
        # 治疗关键词
        treatment_keywords = ["手术", "治疗", "化疗", "放疗", "理疗", "康复", "训练",
                             "介入", "移植", "置换", "切除", "修复", "重建",
                             "药物治疗", "手术治疗", "保守治疗", "对症治疗",
                             "支持治疗", "免疫治疗", "靶向治疗", "基因治疗"]
        
        # 解剖部位关键词
        anatomy_keywords = ["心", "肝", "肺", "肾", "胃", "肠", "脑", "血管", "神经",
                           "骨骼", "肌肉", "皮肤", "眼", "耳", "鼻", "喉", "口腔",
                           "食管", "气管", "胆囊", "胰腺", "脾脏", "膀胱", "子宫",
                           "卵巢", "前列腺", "甲状腺", "乳腺", "脊柱", "关节",
                           "肺部", "肝脏", "肾脏", "心脏", "大脑", "小脑", "脊髓",
                           "呼吸", "感染"]
        
        # 科室关键词
        department_keywords = ["内科", "外科", "儿科", "妇科", "产科", "眼科", "耳鼻喉",
                              "口腔科", "皮肤科", "神经科", "心血管", "消化", "呼吸",
                              "内分泌", "血液", "风湿", "肾病", "感染", "急诊",
                              "康复科", "麻醉科", "放射科", "检验科", "病理科",
                              "心内科", "骨科", "普外科", "胸外科", "脑外科"]
        
        # 疾病关键词（放在后面，因为症状和疾病可能有重叠）
        disease_keywords = ["炎", "病", "症", "瘤", "癌", "衰竭", "梗死", "栓", 
                           "狭窄", "硬化", "结核", "腺", "囊肿", "息肉", "结石", 
                           "癫痫", "痴呆", "帕金森", "抑郁", "分裂", "综合征",
                           "高血压", "糖尿病", "冠心病", "哮喘", "肺炎", "肝炎",
                           "肾炎", "胃炎", "肠炎", "关节炎", "白血病", "淋巴瘤",
                           "头痛"]

        # 按优先级顺序检查
        for keyword in symptom_keywords:
            if keyword in entity_text:
                # 如果上下文明确表明是疾病，则返回疾病类型
                if keyword == "头痛" and any(c in context_lower for c in disease_context):
                    return "Disease"
                return "Symptom"

        for keyword in drug_keywords:
            if keyword in entity_text:
                return "Drug"

        for keyword in exam_keywords:
            if keyword in entity_text:
                return "Examination"

        for keyword in treatment_keywords:
            if keyword in entity_text:
                # "手术室" 应该是解剖部位（场所）
                if keyword == "手术" and "室" in entity_text:
                    return "Anatomy"
                # 如果是单独的"手术"且上下文有"室"，也返回解剖部位
                if keyword == "手术" and "室" in context_lower:
                    return "Anatomy"
                return "Treatment"

        for keyword in anatomy_keywords:
            if keyword in entity_text:
                # 如果上下文明确表明是科室，则返回科室类型
                if any(c in context_lower for c in department_context):
                    if keyword == "呼吸" or keyword == "感染":
                        return "Department"
                # 如果上下文明确表明是疾病，则返回疾病类型
                if any(c in context_lower for c in disease_context):
                    if keyword == "心脏":
                        return "Disease"
                return "Anatomy"

        for keyword in department_keywords:
            if keyword in entity_text:
                return "Department"

        for keyword in disease_keywords:
            if keyword in entity_text:
                return "Disease"

        return "Disease"

    def _basic_entity_extraction(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """基于规则的医疗实体提取（回退方案）

        Args:
            text: 待处理文本
            context: 上下文文本（用于消歧）

        Returns:
            实体列表
        """
        patterns = {
            "Symptom": [
                r"头痛", r"头晕", r"发热", r"咳嗽", r"乏力",
                r"胸闷", r"心悸", r"呼吸困难", r"恶心", r"呕吐",
                r"腹泻", r"便秘", r"腹痛", r"腹胀", r"水肿",
                r"皮疹", r"瘙痒", r"麻木", r"刺痛", r"出血",
                r"失眠", r"嗜睡", r"意识模糊", r"抽搐", r"昏迷",
                r"视力模糊", r"听力下降", r"口干", r"口苦",
            ],
            "Disease": [
                r"高血压", r"糖尿病", r"冠心病", r"哮喘", r"肺炎",
                r"肝炎", r"肾炎", r"胃炎", r"肠炎", r"关节炎",
                r"白血病", r"淋巴瘤", r"感染",
                r"[\u4e00-\u9fff]+炎", r"[\u4e00-\u9fff]+病", 
                r"[\u4e00-\u9fff]+综合征", r"[\u4e00-\u9fff]+症候群", 
                r"[\u4e00-\u9fff]+衰竭", r"[\u4e00-\u9fff]+肿瘤", 
                r"[\u4e00-\u9fff]+癌", r"[\u4e00-\u9fff]+梗死", 
                r"[\u4e00-\u9fff]+硬化",
            ],
            "Drug": [
                r"阿司匹林", r"布洛芬", r"青霉素", r"阿莫西林",
                r"头孢", r"红霉素", r"阿奇霉素", r"氯霉素",
                r"甲硝唑", r"奥美拉唑", r"兰索拉唑", r"雷尼替丁",
                r"二甲双胍", r"格列齐特", r"胰岛素", r"泼尼松",
                r"氯吡格雷", r"低分子肝素", r"肝素", r"硝苯地平",
                r"氨氯地平", r"降压药", r"降糖药", r"降脂药",
                r"[\u4e00-\u9fff]+片", r"[\u4e00-\u9fff]+胶囊",
                r"[\u4e00-\u9fff]+丸", r"[\u4e00-\u9fff]+膏",
            ],
            "Examination": [
                r"血常规", r"尿常规", r"大便常规", r"肝功能",
                r"肾功能", r"血糖", r"血脂", r"电解质", r"凝血功能",
                r"心电图", r"超声", r"CT", r"MRI", r"X光",
                r"PET", r"PET-CT", r"胃镜", r"肠镜", r"病理检查",
                r"细菌培养", r"病毒检测", r"核酸检测", r"抗体检测",
            ],
            "Treatment": [
                r"手术", r"化疗", r"放疗", r"药物治疗",
                r"介入治疗", r"保守治疗", r"手术治疗", r"康复训练",
                r"理疗", r"免疫治疗", r"靶向治疗", r"基因治疗",
            ],
            "Department": [
                r"心血管内科", r"消化内科", r"呼吸内科", r"内分泌科",
                r"血液科", r"风湿科", r"肾病科", r"感染科",
                r"急诊科", r"康复科", r"麻醉科", r"放射科",
                r"检验科", r"病理科", r"骨科", r"普外科",
                r"胸外科", r"脑外科", r"内科", r"外科",
                r"儿科", r"妇科", r"产科", r"眼科",
                r"耳鼻喉科", r"口腔科", r"皮肤科", r"神经科",
            ],
            "Anatomy": [
                r"心脏", r"肝脏", r"肺部", r"肾脏", r"胃",
                r"肠道", r"大脑", r"神经", r"骨骼",
                r"肌肉", r"皮肤", r"眼睛", r"耳朵", r"鼻子",
                r"喉咙", r"口腔", r"食管", r"气管", r"胆囊",
                r"胰腺", r"脾脏", r"膀胱", r"子宫", r"卵巢",
            ],
        }

        entities = []
        extracted_entities = set()  # 用于去重
        
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                for match in re.finditer(pattern, text):
                    entity_text = match.group()
                    # 使用上下文感知的类型分类
                    classified_type = self._classify_entity_type(entity_text, context)
                    
                    # 去重：相同文本和类型只保留一个
                    key = (entity_text, classified_type)
                    if key not in extracted_entities:
                        extracted_entities.add(key)
                        entities.append({
                            "type": classified_type,
                            "text": entity_text,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.7
                        })

        return entities

    def extract_entities_with_context(self, text: str, window: int = 50) -> List[Dict[str, Any]]:
        """提取实体及其上下文信息

        Args:
            text: 待处理文本
            window: 上下文窗口大小（字符数）

        Returns:
            实体列表，每项包含实体信息和上下文
        """
        entities = self.extract_medical_entities(text)

        for entity in entities:
            start = max(0, entity["start"] - window)
            end = min(len(text), entity["end"] + window)
            entity["context"] = text[start:end]
            entity["context_before"] = text[start:entity["start"]]
            entity["context_after"] = text[entity["end"]:end]

        return entities
