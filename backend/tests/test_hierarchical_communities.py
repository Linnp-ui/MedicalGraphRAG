import pytest
import networkx as nx
from src.core.hierarchical_communities import HierarchicalCommunityManager


class TestHierarchicalCommunityManager:

    def test_init(self):
        manager = HierarchicalCommunityManager(levels=3)
        assert manager.levels == 3
        assert manager._partitions == []

    def test_build_hierarchy(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        manager.build_hierarchy(str_graph)
        
        assert len(manager._partitions) == 3
        assert all(len(p) == 34 for p in manager._partitions)

    def test_get_community_at_level(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        comm_id = manager.get_community_at_level("0", level=0)
        assert isinstance(comm_id, int)
        
        comm_id_1 = manager.get_community_at_level("0", level=1)
        assert isinstance(comm_id_1, int)

    def test_get_communities_by_level(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        
        assert isinstance(communities, dict)
        assert len(communities) > 0
        assert all(isinstance(members, list) for members in communities.values())

    def test_get_community_members(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        first_comm_id = list(communities.keys())[0]
        
        members = manager.get_community_members(level=0, community_id=first_comm_id)
        
        assert isinstance(members, list)
        assert len(members) > 0

    def test_get_community_summary(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        first_comm_id = list(communities.keys())[0]
        
        summary = manager.get_community_summary(level=0, community_id=first_comm_id)
        
        assert isinstance(summary, str)
        assert "社区" in summary
        assert "层级" in summary

    def test_community_embedding_cache(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        first_comm_id = list(communities.keys())[0]
        
        embedding = manager.get_community_embedding(level=0, community_id=first_comm_id)
        assert embedding is None
        
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        manager.set_community_embedding(level=0, community_id=first_comm_id, embedding=test_embedding)
        
        cached = manager.get_community_embedding(level=0, community_id=first_comm_id)
        assert cached == test_embedding

    def test_get_stats(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        stats = manager.get_stats()
        assert stats["status"] == "not_built"
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        stats = manager.get_stats()
        assert stats["status"] == "built"
        assert stats["levels"] == 3
        assert len(stats["communities_per_level"]) == 3

    def test_invalid_level_raises_error(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        with pytest.raises(ValueError, match="Level must be between"):
            manager.get_community_at_level("0", level=5)
        
        with pytest.raises(ValueError, match="Level must be between"):
            manager.get_community_at_level("0", level=-1)

    def test_not_built_raises_error(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        with pytest.raises(ValueError, match="Hierarchy not built"):
            manager.get_community_at_level("0", level=0)
        
        with pytest.raises(ValueError, match="Hierarchy not built"):
            manager.get_communities_by_level(level=0)

    def test_find_relevant_communities(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=1)
        for comm_id in list(communities.keys())[:2]:
            embedding = [0.1 * comm_id] * 10
            manager.set_community_embedding(level=1, community_id=comm_id, embedding=embedding)
        
        query_embedding = [0.1] * 10
        results = manager.find_relevant_communities(query_embedding, level=1, top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
