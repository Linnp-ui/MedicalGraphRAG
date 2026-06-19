#!/usr/bin/env python3
"""测试新功能：社区检测、分层摘要和DRIFT Search"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_community_detection():
    """测试社区检测功能"""
    print("\n" + "="*60)
    print("测试1: 社区检测")
    print("="*60)
    
    try:
        from src.core.community_detector import get_community_detector
        
        detector = get_community_detector()
        communities = detector.get_communities()
        
        print(f"检测到社区数量: {len(communities)}")
        
        top_communities = detector.get_top_communities(top_n=3)
        print("\n最大的3个社区:")
        for comm_id, count in top_communities:
            members = detector.get_community_members(comm_id)
            print(f"  社区 {comm_id}: {count} 个成员")
            print(f"    成员示例: {', '.join(members[:5])}")
        
        if communities:
            first_comm_id = list(communities.keys())[0]
            centrality = detector.compute_community_centrality(first_comm_id)
            print(f"\n社区 {first_comm_id} 的中心性分析:")
            for entity, scores in list(centrality.items())[:3]:
                print(f"  {entity}: betweenness={scores['betweenness']:.3f}, degree={scores['degree']:.3f}")
        
        print("✅ 社区检测测试通过")
        return True
    except Exception as e:
        print(f"❌ 社区检测测试失败: {e}")
        logger.exception(e)
        return False


def test_summary_generator():
    """测试分层摘要生成功能"""
    print("\n" + "="*60)
    print("测试2: 分层摘要生成")
    print("="*60)
    
    try:
        from src.core.summary_generator import get_summary_generator
        
        generator = get_summary_generator()
        
        global_summary = generator.generate_global_summary()
        print("全局摘要:")
        print(global_summary[:500] + "..." if len(global_summary) > 500 else global_summary)
        
        detector = __import__('src.core.community_detector', fromlist=['get_community_detector'])
        communities = detector.get_community_detector().get_communities()
        
        if communities:
            first_comm_id = list(communities.keys())[0]
            comm_summary = generator.generate_community_summary(first_comm_id)
            print(f"\n社区 {first_comm_id} 摘要:")
            print(comm_summary[:500] + "..." if len(comm_summary) > 500 else comm_summary)
        
        print("✅ 分层摘要测试通过")
        return True
    except Exception as e:
        print(f"❌ 分层摘要测试失败: {e}")
        logger.exception(e)
        return False


def test_drift_search():
    """测试DRIFT Search功能"""
    print("\n" + "="*60)
    print("测试3: DRIFT Search")
    print("="*60)
    
    try:
        from src.retrieval.drift_search import drift_search, explain_drift_strategy
        
        test_queries = [
            "总结一下高血压相关的所有信息",
            "阿司匹林有哪些副作用",
            "介绍一下糖尿病的治疗方法",
        ]
        
        for query in test_queries:
            explanation = explain_drift_strategy(query)
            print(f"\n查询: '{query}'")
            print(f"  检测意图: {explanation['detected_intent']}")
            print(f"  策略: {explanation['strategy']}")
            print(f"  解释: {explanation['explanation']}")
            
            results = drift_search(query)
            print(f"  结果类型: {results.get('search_type')}")
        
        print("\n✅ DRIFT Search测试通过")
        return True
    except Exception as e:
        print(f"❌ DRIFT Search测试失败: {e}")
        logger.exception(e)
        return False


def test_llm_factory():
    """测试LLM Provider工厂"""
    print("\n" + "="*60)
    print("测试4: LLM Provider工厂")
    print("="*60)
    
    try:
        from src.core.providers.llm_provider import LLMFactory, get_llm_provider
        
        providers = LLMFactory.get_available_providers()
        print(f"可用的LLM提供者: {providers}")
        
        mock_config = {"type": "mock", "api_key": "test"}
        mock_provider = LLMFactory.create(mock_config)
        response = mock_provider.generate("Hello World")
        print(f"Mock响应: {response}")
        
        embedding = mock_provider.embed("Test text")
        print(f"Mock嵌入维度: {len(embedding)}")
        
        print("✅ LLM工厂测试通过")
        return True
    except Exception as e:
        print(f"❌ LLM工厂测试失败: {e}")
        logger.exception(e)
        return False


def test_vector_factory():
    """测试向量存储Provider工厂"""
    print("\n" + "="*60)
    print("测试5: 向量存储Provider工厂")
    print("="*60)
    
    try:
        from src.core.providers.vector_provider import VectorFactory
        
        providers = VectorFactory.get_available_providers()
        print(f"可用的向量存储提供者: {providers}")
        
        mock_config = {"type": "mock"}
        mock_provider = VectorFactory.create(mock_config)
        
        mock_provider.create_index("test_index", 384)
        print("创建索引成功")
        
        vectors = [[0.1]*384, [0.2]*384]
        ids = ["doc1", "doc2"]
        mock_provider.add_vectors("test_index", vectors, ids)
        print("添加向量成功")
        
        results = mock_provider.search("test_index", [0.1]*384, 10)
        print(f"搜索结果数量: {len(results)}")
        
        print("✅ 向量工厂测试通过")
        return True
    except Exception as e:
        print(f"❌ 向量工厂测试失败: {e}")
        logger.exception(e)
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("测试新功能模块")
    print("="*60)
    
    results = []
    
    results.append(test_community_detection())
    results.append(test_summary_generator())
    results.append(test_drift_search())
    results.append(test_llm_factory())
    results.append(test_vector_factory())
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if all(results):
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print("\n⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())