#!/usr/bin/env python
"""测试多类型实体支持功能"""

from src.ingestion.medical_processor import MedicalTextProcessor

def test_entity_extraction():
    processor = MedicalTextProcessor()

    test_text = """患者因高血压和糖尿病入院，伴有头痛、头晕症状。
医生开具了硝苯地平缓释片和二甲双胍进行治疗。
建议进行血常规、血糖、血脂检查，必要时做CT扫描。
患者有冠心病家族史，目前在心血管内科接受治疗。
治疗方案包括药物治疗和康复训练。"""

    print('=' * 60)
    print('测试文本:')
    print(test_text)
    print('=' * 60)

    entities = processor.extract_medical_entities(test_text)

    print(f'\n提取到 {len(entities)} 个实体:')
    print('-' * 60)

    for i, entity in enumerate(entities, 1):
        print(f'{i}. 类型: {entity["type"]:15} 实体: {entity["text"]:20} 置信度: {entity["confidence"]:.2f}')

    print('-' * 60)

    # 统计各类实体数量
    type_counts = {}
    for entity in entities:
        etype = entity['type']
        type_counts[etype] = type_counts.get(etype, 0) + 1

    print('\n实体类型分布:')
    for etype, count in type_counts.items():
        print(f'  {etype}: {count} 个')

def test_7_entity_types():
    """测试7种医疗实体类型"""
    from src.core.medical_schema import MedicalEntityType
    
    processor = MedicalTextProcessor()
    
    test_cases = [
        ("高血压是一种常见疾病", MedicalEntityType.DISEASE),
        ("头痛是常见症状", MedicalEntityType.SYMPTOM),
        ("阿司匹林是常用药物", MedicalEntityType.DRUG),
        ("血常规检查很重要", MedicalEntityType.EXAMINATION),
        ("手术治疗效果好", MedicalEntityType.TREATMENT),
        ("肺部是重要器官", MedicalEntityType.ANATOMY),
        ("心血管内科负责诊治", MedicalEntityType.DEPARTMENT),
    ]
    
    print('\n' + '=' * 60)
    print('测试7种医疗实体类型:')
    print('=' * 60)
    
    for text, expected_type in test_cases:
        entities = processor.extract_medical_entities(text)
        if entities:
            detected_type = entities[0]['type']
            status = '✓' if detected_type == expected_type.value else '✗'
            print(f'{status} 文本: "{text}"')
            print(f'    期望类型: {expected_type.value}, 检测类型: {detected_type}')
        else:
            print(f'✗ 文本: "{text}" - 未检测到实体')

if __name__ == '__main__':
    test_entity_extraction()
    test_7_entity_types()