import pytest
import networkx as nx
from src.core.leiden_detector import LeidenCommunityDetector
from src.core.hierarchical_communities import HierarchicalCommunityManager
from src.core.llm_cache import LLMCache


class TestPhase1Integration:
    """Phase 1 集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程：Leiden 检测 → 分层管理 → LLM 缓存"""
        detector = LeidenCommunityDetector(resolution=0.1)
        manager = HierarchicalCommunityManager(levels=3)
        cache = LLMCache()
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partition = detector.detect_communities(str_graph)
        assert len(partition) == 34
        
        modularity = detector.compute_modularity(str_graph, partition)
        assert modularity > 0.3
        
        manager.build_hierarchy(str_graph)
        stats = manager.get_stats()
        
        assert stats["status"] == "built"
        assert stats["levels"] == 3
        
        call_count = [0]
        def generate():
            call_count[0] += 1
            return "test result"
        
        result1 = cache.get_or_generate("prompt", generate, model="test")
        result2 = cache.get_or_generate("prompt", generate, model="test")
        
        assert result1 == result2
        assert call_count[0] == 1
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
    
    def test_leiden_with_manager(self):
        """测试 Leiden 检测器与分层管理器集成"""
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        manager.build_hierarchy(str_graph)
        
        for level in range(3):
            communities = manager.get_communities_by_level(level)
            assert len(communities) > 0
            
            for comm_id, members in communities.items():
                assert len(members) > 0
    
    def test_cache_with_different_models(self):
        """测试不同模型的缓存隔离"""
        cache = LLMCache()
        
        call_count = {"model_a": 0, "model_b": 0}
        
        def generate_a():
            call_count["model_a"] += 1
            return "result_a"
        
        def generate_b():
            call_count["model_b"] += 1
            return "result_b"
        
        result_a1 = cache.get_or_generate("prompt", generate_a, model="model_a")
        result_a2 = cache.get_or_generate("prompt", generate_a, model="model_a")
        result_b1 = cache.get_or_generate("prompt", generate_b, model="model_b")
        
        assert result_a1 == result_a2 == "result_a"
        assert result_b1 == "result_b"
        assert call_count["model_a"] == 1
        assert call_count["model_b"] == 1
    
    def test_community_summary_caching(self):
        """测试社区摘要缓存"""
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        first_comm_id = list(communities.keys())[0]
        
        summary1 = manager.get_community_summary(level=0, community_id=first_comm_id)
        summary2 = manager.get_community_summary(level=0, community_id=first_comm_id)
        
        assert summary1 == summary2
        assert "成员数量" in summary1
