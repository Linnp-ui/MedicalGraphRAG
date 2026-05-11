import re
from typing import List, Dict, Any
from .document_loader import Document

class MedicalTextProcessor:
    """专门用于医疗文本预处理的类"""

    def __init__(self):
        # 常见医疗单位标准化映射
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

    def clean_text(self, text: str) -> str:
        """清洗医疗文本"""
        if not text:
            return ""
            
        # 1. 基础清洗：去除多余空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 2. 标准化符号（如将英文逗号改为中文逗号，如果大部分是中文文本的话）
        # 考虑到本项目主要是中文医疗图谱，这里进行一些中文适配
        text = text.replace(',', '，').replace(';', '；').replace(':', '：')
        
        # 3. 简单的单位标准化（可选）
        # for eng, chn in self.unit_mapping.items():
        #     text = re.sub(rf'(\d+)\s*{eng}', rf'\1{chn}', text, flags=re.IGNORECASE)
            
        return text

    def process_document(self, document: Document) -> Document:
        """对 Document 进行预处理"""
        document.content = self.clean_text(document.content)
        # 可以在 metadata 中标记已进行医疗预处理
        document.metadata["medical_processed"] = True
        return document

    def batch_process(self, documents: List[Document]) -> List[Document]:
        """批量预处理文档"""
        return [self.process_document(doc) for doc in documents]
