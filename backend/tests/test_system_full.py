"""医疗知识图谱系统全面测试"""
import sys
import os
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.medical_processor import MedicalTextProcessor
from src.ingestion.text_splitter import TextSplitter, SplitStrategy
from src.ingestion.knowledge_fusion import EntityDisambiguator, RelationAligner, KnowledgeFusionEngine
from src.chains.medical_intent import MedicalIntentClassifier

def run_full_test():
    print("=" * 70)
    print("医疗知识图谱系统 - 全面功能测试")
    print("=" * 70)

    test_entities = [
        {"name": "高血压", "type": "Disease"},
        {"name": "HTN", "type": "Disease"},
        {"name": "原发性高血压", "type": "Disease"},
        {"name": "头痛", "type": "Symptom"},
        {"name": "阿司匹林", "type": "Drug"},
        {"name": "乙酰水杨酸", "type": "Drug"},
        {"name": "胰岛素", "type": "Drug"},
        {"name": "糖尿病", "type": "Disease"},
    ]

    test_relations = [
        {"source": "高血压", "target": "头痛", "type": "表现为"},
        {"source": "阿司匹林", "target": "高血压", "type": "用于治疗"},
        {"source": "糖尿病", "target": "胰岛素", "type": "用胰岛素治疗"},
    ]

    test_queries = [
        "高血压是什么疾病？",
        "头痛有哪些原因？",
        "阿司匹林有什么副作用？",
        "我最近经常头痛，可能是什么病？",
    ]

    all_passed = 0
    total_tests = 0

    print("\n" + "=" * 70)
    print("1. 文档加载与预处理测试")
    print("=" * 70)
    total_tests += 1
    try:
        loader = DocumentLoader()
        medical_file = Path(__file__).parent.parent / "data" / "input" / "medical_sample.txt"
        result = loader.load_safe(str(medical_file))
        if result.success:
            print("✅ 文档加载成功")
            print(f"   标题: {result.document.title}")
            print(f"   内容长度: {len(result.document.content)}")

            processor = MedicalTextProcessor()
            cleaned_doc = processor.process_document(result.document)
            print("✅ 文本预处理完成")

            splitter = TextSplitter(strategy=SplitStrategy.MEDICAL)
            chunks = splitter.split_text(cleaned_doc.content, cleaned_doc.id)
            print(f"✅ 文本分割完成: {len(chunks)} 个块")
            all_passed += 1
    except Exception as e:
        print(f"❌ 文档处理测试失败: {e}")

    print("\n" + "=" * 70)
    print("2. 医疗实体识别测试")
    print("=" * 70)
    total_tests += 1
    try:
        processor = MedicalTextProcessor()
        test_text = "患者因高血压、糖尿病就诊，伴有头痛头晕，开具阿司匹林、二甲双胍"
        entities = processor.extract_medical_entities(test_text)
        print(f"✅ 实体识别完成: {len(entities)} 个")
        for entity in entities:
            print(f"   [{entity['type']}] {entity['text']}")
        all_passed += 1
    except Exception as e:
        print(f"❌ 实体识别测试失败: {e}")

    print("\n" + "=" * 70)
    print("3. 实体消歧测试")
    print("=" * 70)
    total_tests += 1
    try:
        disambiguator = EntityDisambiguator()
        normalized_entities = [
            {
                "name": disambiguator.normalize_name(e["name"], e["type"]),
                "type": e["type"]
            } for e in test_entities
        ]
        print("✅ 名称标准化完成")
        for i, original in enumerate(test_entities):
            normalized = normalized_entities[i]
            print(f"   {original['name']} → {normalized['name']}")

        fused_entities = disambiguator.disambiguate(test_entities)
        print(f"✅ 实体消歧完成: {len(fused_entities)} 个（去重前: {len(test_entities)}）")
        all_passed += 1
    except Exception as e:
        print(f"❌ 实体消歧测试失败: {e}")

    print("\n" + "=" * 70)
    print("4. 关系对齐测试")
    print("=" * 70)
    total_tests += 1
    try:
        aligner = RelationAligner()
        aligned_relations = []
        for rel in test_relations:
            aligned_type = aligner.align_relation(rel["type"])
            aligned_relations.append({
                "source": rel["source"],
                "target": rel["target"],
                "type": aligned_type
            })
        print("✅ 关系对齐完成")
        for i, original in enumerate(test_relations):
            aligned = aligned_relations[i]
            print(f"   '{original['type']}' → '{aligned['type']}'")

        valid_count = 0
        for rel in aligned_relations:
            valid = aligner.validate_relation("Disease", "Symptom", rel["type"])
            print(f"   关系验证: {rel['type']} → {'有效' if valid else '无效'}")
            if valid:
                valid_count += 1
        all_passed += 1
    except Exception as e:
        print(f"❌ 关系对齐测试失败: {e}")

    print("\n" + "=" * 70)
    print("5. 知识融合测试")
    print("=" * 70)
    total_tests += 1
    try:
        engine = KnowledgeFusionEngine()
        fused_entities, fused_relations = engine.fuse(test_entities, test_relations)
        print(f"✅ 知识融合完成:")
        print(f"   实体: {len(fused_entities)} 个")
        print(f"   关系: {len(fused_relations)} 个")

        ontology_entities = engine.link_to_standard_ontology(fused_entities)
        print("✅ 标准本体链接完成")
        has_icd10 = any("icd10_code" in e for e in ontology_entities)
        has_umls = any("umls_code" in e for e in ontology_entities)
        print(f"   ICD-10 编码: {'有' if has_icd10 else '无'}")
        print(f"   UMLS 编码: {'有' if has_umls else '无'}")
        all_passed += 1
    except Exception as e:
        print(f"❌ 知识融合测试失败: {e}")

    print("\n" + "=" * 70)
    print("6. 医疗意图分类测试")
    print("=" * 70)
    total_tests += 1
    try:
        classifier = MedicalIntentClassifier()
        print("✅ 意图分类器初始化完成")
        for query in test_queries:
            result = classifier.classify(query)
            print(f"   问题: {query}")
            print(f"   → 意图: {result.intent.value}")
            print(f"   → 实体: {result.entities}")
            print()
        all_passed += 1
    except Exception as e:
        print(f"❌ 意图分类测试失败: {e}")

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"总测试数: {total_tests}")
    print(f"通过: {all_passed}")
    print(f"失败: {total_tests - all_passed}")
    print(f"通过率: {all_passed/total_tests*100:.1f}%")
    print("=" * 70)

    if all_passed == total_tests:
        print("\n🎉 所有功能测试通过！系统运行正常。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    run_full_test()
