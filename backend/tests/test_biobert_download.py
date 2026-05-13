"""测试下载和使用中文医疗NER模型"""
import sys
import os
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.medical_processor import MedicalTextProcessor

def test_chinese_medical_ner():
    print("=" * 60)
    print("中文医疗NER模型测试")
    print("=" * 60)

    processor = MedicalTextProcessor()

    print("\n1. 加载模型...")
    nlp = processor._get_nlp_processor()

    if nlp is not None:
        print("✅ 模型加载成功!")
    else:
        print("❌ 模型加载失败")
        return

    print("\n2. 测试实体识别...")
    test_text = "患者因胸痛入院，既往有高血压病史10年，糖尿病史5年。"
    entities = processor.extract_medical_entities(test_text)

    print(f"\n输入文本: {test_text}")
    print(f"\n识别到的实体 ({len(entities)}个):")
    for entity in entities:
        print(f"  [{entity['type']:10}] {entity['text']} (置信度: {entity['confidence']:.2f})")

    disease_count = len([e for e in entities if e['type'] == 'DISEASE'])
    symptom_count = len([e for e in entities if e['type'] == 'SYMPTOM'])
    drug_count = len([e for e in entities if e['type'] == 'DRUG'])

    print(f"\n实体统计: 疾病={disease_count}, 症状={symptom_count}, 药物={drug_count}")

    if disease_count >= 2 and symptom_count >= 1:
        print("\n✅ 测试通过!")
    else:
        print("\n⚠️ 测试结果可进一步优化")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_chinese_medical_ner()