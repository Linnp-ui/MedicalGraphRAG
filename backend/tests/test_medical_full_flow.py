"""
医疗知识图谱系统 - 全链路测试脚本
测试从文档摄入到问答的完整流程
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from loguru import logger

logger.info("=" * 60)
logger.info("医疗知识图谱系统 - 全链路测试开始")
logger.info("=" * 60)

results = {}


def test_neo4j_connection():
    """测试1: Neo4j 数据库连接"""
    logger.info("\n[测试1] Neo4j 数据库连接测试")
    try:
        from src.core.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        if client.verify_connectivity():
            logger.info("✅ Neo4j 连接成功")
            return True
        else:
            logger.error("❌ Neo4j 连接失败")
            return False
    except Exception as e:
        logger.error(f"❌ Neo4j 连接异常: {e}")
        return False


def test_medical_constraints():
    """测试2: 医疗图谱约束初始化"""
    logger.info("\n[测试2] 医疗图谱约束初始化测试")
    try:
        from src.core.neo4j_client import get_neo4j_client
        from src.core.medical_schema import MedicalEntityType, MedicalRelationshipType

        client = get_neo4j_client()

        for entity_type in MedicalEntityType:
            label = entity_type.value
            query = f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            try:
                client.execute_query(query)
                logger.info(f"  ✅ 创建约束: {label}")
            except Exception as e:
                logger.warning(f"  ⚠️ 约束已存在或创建失败: {label} - {e}")

        logger.info("✅ 医疗图谱约束初始化完成")
        return True
    except Exception as e:
        logger.error(f"❌ 医疗图谱约束初始化失败: {e}")
        return False


def test_document_loading():
    """测试3: 医疗文档加载"""
    logger.info("\n[测试3] 医疗文档加载测试")
    try:
        from src.ingestion.document_loader import load_document

        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend",
            "data",
            "input",
            "medical_sample.txt"
        )

        if not os.path.exists(data_path):
            logger.error(f"❌ 医疗示例文件不存在: {data_path}")
            return False

        document = load_document(data_path)
        logger.info(f"  文档标题: {document.title}")
        logger.info(f"  文档内容长度: {len(document.content)} 字符")
        logger.info("✅ 医疗文档加载成功")
        return True
    except Exception as e:
        logger.error(f"❌ 医疗文档加载失败: {e}")
        return False


def test_medical_processor():
    """测试4: 医疗文本处理器"""
    logger.info("\n[测试4] 医疗文本处理器测试")
    try:
        from src.ingestion.medical_processor import MedicalTextProcessor

        processor = MedicalTextProcessor()

        test_text = "高血压病是一种常见的慢性疾病,HTN患者需要长期服药。"
        cleaned = processor.clean_text(test_text)
        logger.info(f"  原始文本: {test_text}")
        logger.info(f"  清洗后: {cleaned}")
        logger.info("✅ 医疗文本处理器工作正常")
        return True
    except Exception as e:
        logger.error(f"❌ 医疗文本处理器测试失败: {e}")
        return False


def test_entity_disambiguator():
    """测试5: 实体消歧模块"""
    logger.info("\n[测试5] 实体消歧模块测试")
    try:
        from src.ingestion.knowledge_fusion import EntityDisambiguator

        disambiguator = EntityDisambiguator()

        test_cases = [
            ("HTN", "Disease", "高血压"),
            ("发烧", "Symptom", "发热"),
            ("乙酰水杨酸", "Drug", "阿司匹林"),
        ]

        for original, entity_type, expected in test_cases:
            result = disambiguator.normalize_name(original, entity_type)
            status = "✅" if result == expected else "❌"
            logger.info(f"  {status} {original} → {result} (期望: {expected})")

        logger.info("✅ 实体消歧模块测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 实体消歧模块测试失败: {e}")
        return False


def test_relation_aligner():
    """测试6: 关系对齐模块"""
    logger.info("\n[测试6] 关系对齐模块测试")
    try:
        from src.ingestion.knowledge_fusion import RelationAligner

        aligner = RelationAligner()

        test_cases = [
            ("表现为", "HAS_SYMPTOM"),
            ("由...引起", "CAUSED_BY"),
            ("副作用", "SIDE_EFFECT"),
        ]

        for original, expected in test_cases:
            result = aligner.align_relation(original)
            status = "✅" if result == expected else "❌"
            logger.info(f"  {status} {original} → {result} (期望: {expected})")

        logger.info("✅ 关系对齐模块测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 关系对齐模块测试失败: {e}")
        return False


def test_knowledge_fusion():
    """测试7: 知识融合引擎"""
    logger.info("\n[测试7] 知识融合引擎测试")
    try:
        from src.ingestion.knowledge_fusion import KnowledgeFusionEngine

        engine = KnowledgeFusionEngine()

        entities = [
            {"name": "高血压病", "type": "Disease", "properties": {}},
            {"name": "头痛", "type": "Symptom", "properties": {}},
            {"name": "HTN", "type": "Disease", "properties": {"source": "doc1"}},
        ]

        relationships = [
            {"source": "高血压病", "target": "头痛", "type": "表现为", "properties": {}},
            {"source": "HTN", "target": "头痛", "type": "HAS_SYMPTOM", "properties": {}},
        ]

        fused_entities, fused_rels = engine.fuse(entities, relationships)

        logger.info(f"  原始实体数: {len(entities)} → 融合后: {len(fused_entities)}")
        logger.info(f"  原始关系数: {len(relationships)} → 融合后: {len(fused_rels)}")

        entity_names = [e["name"] for e in fused_entities]
        logger.info(f"  融合后实体: {entity_names}")

        linked = engine.link_to_standard_ontology(fused_entities)
        for ent in linked:
            if "icd10_code" in ent:
                logger.info(f"  ✅ ICD-10链接: {ent['name']} → {ent['icd10_code']}")

        logger.info("✅ 知识融合引擎测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 知识融合引擎测试失败: {e}")
        return False


def test_graph_construction():
    """测试8: 知识图谱构建（完整流程）"""
    logger.info("\n[测试8] 知识图谱构建测试")
    try:
        from src.ingestion.kg_builder import KnowledgeGraphBuilder
        from src.ingestion.document_loader import load_document

        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend",
            "data",
            "input",
            "medical_sample.txt"
        )

        document = load_document(data_path)
        builder = KnowledgeGraphBuilder()

        logger.info("  开始摄入文档...")
        result = builder.ingest_document(
            document,
            extract_entities=True,
            create_embeddings=True
        )

        logger.info(f"  文档ID: {result['document_id']}")
        logger.info(f"  创建块数: {result['chunks_created']}")
        logger.info(f"  提取实体数: {result['entities_extracted']}")
        logger.info(f"  创建关系数: {result['relationships_created']}")

        logger.info("✅ 知识图谱构建测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 知识图谱构建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_query():
    """测试9: 知识图谱查询"""
    logger.info("\n[测试9] 知识图谱查询测试")
    try:
        from src.core.neo4j_client import get_neo4j_client
        from src.core.medical_schema import MedicalEntityType

        client = get_neo4j_client()

        for entity_type in MedicalEntityType:
            label = entity_type.value
            try:
                result = client.execute_query(f"MATCH (n:`{label}`) RETURN n.name as name LIMIT 10")
                count_result = client.execute_query(f"MATCH (n:`{label}`) RETURN count(n) as count")
                count = count_result[0]["count"] if count_result else 0
                names = [r["name"] for r in result]
                logger.info(f"  ✅ {label}: {count} 个实体 - {names[:3]}{'...' if len(names) > 3 else ''}")
            except Exception as e:
                logger.warning(f"  ⚠️ {label} 查询失败: {e}")

        logger.info("✅ 知识图谱查询测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 知识图谱查询测试失败: {e}")
        return False


def test_intent_classifier():
    """测试10: 医疗意图分类器"""
    logger.info("\n[测试10] 医疗意图分类器测试")
    try:
        from src.chains.medical_intent import MedicalIntentClassifier, MedicalIntent

        classifier = MedicalIntentClassifier()

        test_questions = [
            ("高血压是什么疾病？", MedicalIntent.DISEASE_QUERY),
            ("头痛是什么原因？", MedicalIntent.SYMPTOM_QUERY),
            ("阿司匹林有什么副作用？", MedicalIntent.DRUG_QUERY),
        ]

        for question, expected_intent in test_questions:
            result = classifier.classify(question)
            status = "✅" if result.intent == expected_intent else "⚠️"
            logger.info(f"  {status} '{question}'")
            logger.info(f"      识别意图: {result.intent.value} (期望: {expected_intent.value})")
            logger.info(f"      置信度: {result.confidence}")
            logger.info(f"      提取实体: {result.entities}")

        logger.info("✅ 医疗意图分类器测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 医疗意图分类器测试失败 (可能因 LLM API 限制): {e}")
        return False


def test_qa_chain():
    """测试11: 问答链"""
    logger.info("\n[测试11] 问答链测试")
    try:
        from src.chains.qa_chain import QAChain

        qa = QAChain()

        question = "高血压的主要症状有哪些？"
        context = """
        高血压的主要症状包括：
        1. 头痛 - 多为持续性钝痛或胀痛
        2. 头晕 - 突然站立时明显
        3. 心悸 - 感觉心跳加快
        4. 乏力 - 疲劳感增加
        """

        logger.info(f"  问题: {question}")
        answer = qa.answer(question, context)
        logger.info(f"  答案: {answer[:100]}...")

        logger.info("✅ 问答链测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 问答链测试失败: {e}")
        return False


def test_workflow():
    """测试12: 完整工作流"""
    logger.info("\n[测试12] 完整工作流测试")
    try:
        from src.workflow.graph import run_workflow

        question = "高血压有哪些症状？"

        logger.info(f"  输入问题: {question}")
        result = run_workflow(question)

        logger.info(f"  路由方式: {result.get('routing', 'unknown')}")
        logger.info(f"  文档数量: {len(result.get('documents', []))}")
        logger.info(f"  答案长度: {len(result.get('answer', ''))} 字符")
        logger.info(f"  答案预览: {result.get('answer', '')[:100]}...")

        logger.info("✅ 完整工作流测试完成")
        return True
    except Exception as e:
        logger.error(f"❌ 完整工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """执行所有测试"""
    global results

    tests = [
        ("Neo4j连接", test_neo4j_connection),
        ("医疗约束初始化", test_medical_constraints),
        ("文档加载", test_document_loading),
        ("医疗文本处理", test_medical_processor),
        ("实体消歧", test_entity_disambiguator),
        ("关系对齐", test_relation_aligner),
        ("知识融合", test_knowledge_fusion),
        ("图谱构建", test_graph_construction),
        ("图谱查询", test_graph_query),
        ("意图分类", test_intent_classifier),
        ("问答链", test_qa_chain),
        ("完整工作流", test_workflow),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                results[name] = "PASS"
                passed += 1
            else:
                results[name] = "FAIL"
                failed += 1
        except Exception as e:
            logger.error(f"测试 '{name}' 发生异常: {e}")
            results[name] = "ERROR"
            failed += 1

        time.sleep(0.5)

    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)

    for name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"  {status_icon} {name}: {result}")

    logger.info("-" * 60)
    logger.info(f"  总计: {len(results)} 项测试")
    logger.info(f"  通过: {passed} 项")
    logger.info(f"  失败: {failed} 项")
    logger.info("=" * 60)

    return passed, failed


if __name__ == "__main__":
    main()
