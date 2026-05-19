#!/usr/bin/env python
"""测试同一实体在不同上下文中标注为不同标签"""

from src.ingestion.medical_processor import MedicalTextProcessor

def test_contextual_entity_disambiguation():
    """测试同一实体在不同上下文的类型识别"""
    processor = MedicalTextProcessor()
    
    test_cases = [
        # 同一个词在不同上下文
        ("头痛是常见症状", "头痛", "Symptom"),
        ("头痛是一种疾病", "头痛", "Disease"),
        ("头痛患者需要检查", "头痛", "Symptom"),
        
        # "心脏" 的不同上下文
        ("心脏是重要器官", "心脏", "Anatomy"),
        ("心脏病是严重疾病", "心脏病", "Disease"),
        ("心脏内科负责诊治", "心脏内科", "Department"),
        
        # "高血压" 的不同上下文  
        ("高血压是常见疾病", "高血压", "Disease"),
        ("高血压患者需定期检查", "高血压", "Disease"),
        
        # "手术" 的不同上下文
        ("手术治疗效果好", "手术", "Treatment"),
        ("手术室在三楼", "手术室", "Anatomy"),
        
        # "呼吸" 的不同上下文
        ("呼吸困难是症状", "呼吸困难", "Symptom"),
        ("呼吸内科诊治肺病", "呼吸内科", "Department"),
        
        # "感染" 的不同上下文
        ("感染是常见问题", "感染", "Disease"),
        ("感染科负责诊治", "感染科", "Department"),
        
        # 更多消歧测试
        ("肺部感染是疾病", "肺部感染", "Disease"),
        ("肺部是重要器官", "肺部", "Anatomy"),
        ("肝病是严重疾病", "肝病", "Disease"),
        ("肝脏是代谢器官", "肝脏", "Anatomy"),
    ]
    
    print("=" * 70)
    print("测试同一实体在不同上下文中标注为不同标签")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for text, target_entity, expected_type in test_cases:
        entities = processor.extract_medical_entities(text)
        
        # 查找目标实体（精确匹配）
        found_entities = [e for e in entities if e["text"] == target_entity]
        
        # 如果没有精确匹配，尝试包含匹配
        if not found_entities:
            found_entities = [e for e in entities if target_entity in e["text"] or e["text"] in target_entity]
        
        if found_entities:
            detected_type = found_entities[0]["type"]
            status = "✓" if detected_type == expected_type else "✗"
            if status == "✓":
                passed += 1
            else:
                failed += 1
            
            print(f"{status} 文本: \"{text}\"")
            print(f"    实体: \"{target_entity}\"")
            print(f"    期望类型: {expected_type}, 检测类型: {detected_type}")
        else:
            print(f"✗ 文本: \"{text}\" - 未找到实体 \"{target_entity}\"")
            failed += 1
        print()
    
    print("-" * 70)
    print(f"测试结果: {passed} 个通过, {failed} 个失败")
    print("-" * 70)

def test_document_ingestion():
    """测试文档摄入时的实体消歧"""
    processor = MedicalTextProcessor()
    
    document = """
    患者因头痛入院，头痛是常见症状。
    医生诊断为高血压，高血压是一种慢性疾病。
    患者有心脏病家族史，心脏是重要的器官。
    建议去心血管内科就诊，内科医生会进行检查。
    """
    
    print("\n" + "=" * 70)
    print("测试文档摄入 - 复杂文本中的实体识别")
    print("=" * 70)
    print("文档内容:")
    print(document)
    
    entities = processor.extract_entities_with_context(document)
    
    print(f"\n提取到 {len(entities)} 个实体:")
    print("-" * 70)
    
    entity_summary = {}
    for i, entity in enumerate(entities, 1):
        etype = entity["type"]
        if etype not in entity_summary:
            entity_summary[etype] = set()
        entity_summary[etype].add(entity["text"])
        
        print(f"{i}. 类型: {entity['type']:15} 实体: {entity['text']:15} 上下文: \"{entity['context'][:30]}...\"")
    
    print("-" * 70)
    print("\n实体类型汇总:")
    for etype, entities_set in entity_summary.items():
        print(f"  {etype}: {', '.join(sorted(entities_set))}")

if __name__ == '__main__':
    test_contextual_entity_disambiguation()
    test_document_ingestion()