import re
from typing import List, Dict, Any, Optional
from .document_loader import Document


_NLP_MODEL_CACHE: Optional[Any] = None
_NLP_MODEL_AVAILABLE: Optional[bool] = None


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
        """
        global _NLP_MODEL_CACHE, _NLP_MODEL_AVAILABLE

        if _NLP_MODEL_CACHE is not None:
            return _NLP_MODEL_CACHE

        if _NLP_MODEL_AVAILABLE is False:
            return None

        try:
            import os
            os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"
            os.environ["TRANSFORMERS_TIMEOUT"] = "30"

            from transformers import pipeline

            print("首次加载NLP模型，请稍候...")
            _NLP_MODEL_CACHE = pipeline(
                "token-classification",
                model="iioSnail/bert-base-chinese-medical-ner",
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
                for item in results:
                    entity_text = item.get("word", "").replace(" ", "")
                    entity_type = self._classify_entity_type(entity_text)
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

        return self._basic_entity_extraction(text)

    def _classify_entity_type(self, entity_text: str) -> str:
        """根据实体文本内容分类到具体类型

        Args:
            entity_text: 实体文本

        Returns:
            实体类型 (DISEASE, SYMPTOM, DRUG, EXAMINATION, TREATMENT)
        """
        disease_keywords = ["炎", "病", "症", "瘤", "癌", "衰竭", "梗死", "栓", "狭窄", "硬化", "结核", "腺", "囊肿", "息肉", "结石", "癫痫", "痴呆", "帕金森", "抑郁", "分裂", "焦", "血糖", "血压", "血脂", "蛋白", "霉素", "病史"]
        symptom_keywords = ["痛", "晕", "热", "咳", "喘", "呕", "泻", "秘", "胀", "麻", "痒", "肿", "红", "汗", "悸", "闷", "乏", "恶", "抖", "颤", "抽", "僵", "盲", "聋", "哑", "昏", "迷", "悸", "闷", "慌", "困", "倦", "酸", "软", "硬", "紧", "塞", "阻", "灼", "刺"]
        drug_keywords = ["片", "胶囊", "丸", "膏", "贴", "针", "注射", "口服", "雾化", "吸入", "滴眼", "栓", "糖浆", "素", "平", "地平", "普利", "沙坦", "洛克", "芬", "布洛", "阿司匹", "青霉", "霉"]
        exam_keywords = ["检查", "化验", "扫描", "透视", "照片", "心电图", "脑电图", "超声", "CT", "MRI", "X光", "验血", "验尿", "常规", "培养", "活检", "穿刺", "镜检"]

        for keyword in disease_keywords:
            if keyword in entity_text:
                return "DISEASE"

        for keyword in symptom_keywords:
            if keyword in entity_text:
                return "SYMPTOM"

        for keyword in drug_keywords:
            if keyword in entity_text:
                return "DRUG"

        for keyword in exam_keywords:
            if keyword in entity_text:
                return "EXAMINATION"

        return "DISEASE"

    def _basic_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """基于规则的医疗实体提取（回退方案）

        Args:
            text: 待处理文本

        Returns:
            实体列表
        """
        patterns = {
            "DISEASE": [
                r"[\u4e00-\u9fff]+炎", r"[\u4e00-\u9fff]+病",
                r"[\u4e00-\u9fff]+综合征", r"[\u4e00-\u9fff]+症",
                r"[\u4e00-\u9fff]+衰竭", r"[\u4e00-\u9fff]+肿瘤",
            ],
            "SYMPTOM": [
                r"头痛", r"头晕", r"发热", r"咳嗽", r"乏力",
                r"胸闷", r"心悸", r"呼吸困难", r"恶心", r"呕吐",
                r"腹泻", r"便秘", r"腹痛", r"腹胀", r"水肿",
                r"皮疹", r"瘙痒", r"麻木", r"刺痛", r"出血",
            ],
            "DRUG": [
                r"阿司匹林", r"布洛芬", r"青霉素", r"阿莫西林",
                r"头孢", r"红霉素", r"阿奇霉素", r"氯霉素",
                r"甲硝唑", r"奥美拉唑", r"兰索拉唑", r"雷尼替丁",
                r"二甲双胍", r"格列齐特", r"胰岛素", r"泼尼松",
                r"氯吡格雷", r"低分子肝素", r"肝素", r"硝苯地平",
                r"氨氯地平", r"降压药", r"降糖药", r"降脂药",
            ],
            "EXAMINATION": [
                r"血常规", r"尿常规", r"大便常规", r"肝功能",
                r"肾功能", r"血糖", r"血脂", r"电解质", r"凝血功能",
                r"心电图", r"超声", r"CT", r"MRI", r"X光",
            ],
            "TREATMENT": [
                r"手术", r"化疗", r"放疗", r"药物治疗",
                r"介入治疗", r"保守治疗", r"手术治疗",
            ]
        }

        entities = []
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                for match in re.finditer(pattern, text):
                    entities.append({
                        "type": entity_type,
                        "text": match.group(),
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
