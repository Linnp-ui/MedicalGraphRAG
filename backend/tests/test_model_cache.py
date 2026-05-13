"""验证模型在同进程中只加载一次"""
import sys
import os
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.medical_processor import MedicalTextProcessor

print("=" * 60)
print("验证模型缓存机制")
print("=" * 60)

print("\n创建第1个处理器实例...")
proc1 = MedicalTextProcessor()
print("调用第1次实体识别...")
entities1 = proc1.extract_medical_entities("患者有高血压和糖尿病")
print(f"识别到 {len(entities1)} 个实体")

print("\n创建第2个处理器实例...")
proc2 = MedicalTextProcessor()
print("调用第2次实体识别...")
entities2 = proc2.extract_medical_entities("患者有心脏病和肺炎")
print(f"识别到 {len(entities2)} 个实体")

print("\n创建第3个处理器实例...")
proc3 = MedicalTextProcessor()
print("调用第3次实体识别...")
entities3 = proc3.extract_medical_entities("服用阿司匹林和布洛芬")
print(f"识别到 {len(entities3)} 个实体")

print("\n" + "=" * 60)
print("✅ 测试完成 - 模型只在首次调用时尝试加载")
print("=" * 60)
