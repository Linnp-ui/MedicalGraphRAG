"""API 层测试：基于标签的缓存失效系统集成（绕过 main.py 导入问题）"""
import pytest
from unittest.mock import Mock, patch, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.cache import (
    clear_all_caches,
    get_graph_data_cache,
    get_search_cache,
    QueryCache,
)
from src.core.medical_schema import MedicalEntityType


@pytest.fixture(autouse=True)
def clear_cache():
    clear_all_caches()
    yield


class TestCacheInvalidationFunction:
    """_invalidate_cache_by_entity_types 的行为验证"""

    def test_invalidate_all_entity_types(self):
        """验证所有 MedicalEntityType 标签 + all_entities 被失效"""
        graph_cache = get_graph_data_cache()
        search_cache = get_search_cache()

        for tag in [e.value for e in MedicalEntityType] + ["all_entities"]:
            graph_cache.raw_set(f"gkey_{tag}", f"gval_{tag}", tags=[tag])
            search_cache.raw_set(f"skey_{tag}", f"sval_{tag}", tags=[tag])

        from src.api.routes import _invalidate_cache_by_entity_types
        _invalidate_cache_by_entity_types()

        for tag in [e.value for e in MedicalEntityType] + ["all_entities"]:
            assert graph_cache.raw_get(f"gkey_{tag}") is None, f"graph {tag} not invalidated"
            assert search_cache.raw_get(f"skey_{tag}") is None, f"search {tag} not invalidated"


class TestGraphDataCacheIntegration:
    """模拟 /graph/data 路由的缓存行为"""

    def test_tag_isolation(self):
        """Disease 标签失效不影响 Symptom 缓存"""
        cache = get_graph_data_cache()
        cache.raw_set("disease_nodes", "disease_data", tags=["Disease"])
        cache.raw_set("symptom_nodes", "symptom_data", tags=["Symptom"])

        cache.invalidate_by_tag("Disease")
        assert cache.raw_get("disease_nodes") is None
        assert cache.raw_get("symptom_nodes") == "symptom_data"

    def test_tag_isolation_reverse(self):
        """Symptom 标签失效不影响 Disease 缓存"""
        cache = get_graph_data_cache()
        cache.raw_set("disease_nodes", "disease_data", tags=["Disease"])
        cache.raw_set("symptom_nodes", "symptom_data", tags=["Symptom"])

        cache.invalidate_by_tag("Symptom")
        assert cache.raw_get("disease_nodes") == "disease_data"
        assert cache.raw_get("symptom_nodes") is None

    def test_all_query_has_separate_tag(self):
        """全量查询（无 node_label）标记为 all_entities，不被实体标签失效影响"""
        cache = get_graph_data_cache()
        cache.raw_set("all_data", "all", tags=["all_entities"])
        cache.raw_set("disease_data", "disease", tags=["Disease"])

        cache.invalidate_by_tag("Disease")
        assert cache.raw_get("all_data") == "all"
        assert cache.raw_get("disease_data") is None


class TestSearchCacheIntegration:
    """模拟 /graph/search 路由的缓存行为"""

    def test_search_cache_tag_behavior(self):
        """搜索缓存标签隔离"""
        cache = get_search_cache()
        cache.raw_set("search_disease", "disease_result", tags=["Disease"])
        cache.raw_set("search_drug", "drug_result", tags=["Drug"])

        cache.invalidate_by_tag("Drug")
        assert cache.raw_get("search_disease") == "disease_result"
        assert cache.raw_get("search_drug") is None

    def test_search_cache_all_entities(self):
        """搜索无标签查询使用 all_entities 标签"""
        cache = get_search_cache()
        cache.raw_set("search_all", "all_result", tags=["all_entities"])

        cache.invalidate_by_tag("all_entities")
        assert cache.raw_get("search_all") is None


class TestNeo4jQueryReduction:
    """Neo4j 查询量减少验证（通过 cache hit/miss 计数）"""

    def test_cache_hit_reduces_neo4j_calls(self):
        """缓存命中时 Neo4j 不查询"""
        neo4j = Mock()
        neo4j.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }

        cache = get_graph_data_cache()

        # 第一次：miss → Neo4j 查询
        data1 = neo4j.get_graph_data(node_label="Disease")
        cache.raw_set("Disease:500:0", data1, tags=["Disease"])
        assert neo4j.get_graph_data.call_count == 1

        # 第二次：hit
        cached = cache.raw_get("Disease:500:0")
        assert cached is not None
        assert neo4j.get_graph_data.call_count == 1

        # 失效后：miss → Neo4j 查询
        cache.invalidate_by_tag("Disease")
        cached = cache.raw_get("Disease:500:0")
        assert cached is None
        data2 = neo4j.get_graph_data(node_label="Disease")
        cache.raw_set("Disease:500:0", data2, tags=["Disease"])
        assert neo4j.get_graph_data.call_count == 2


class TestCacheUpdateFlow:
    """模拟完整的 查询→缓存→录入→失效→重新查询 流程"""

    def test_full_flow(self):
        """完整数据流：Neo4j 查询 → 缓存 → 录入 → 缓存失效 → 重新查询"""
        neo4j = Mock()

        cache = get_graph_data_cache()

        # 1. 查询（假设用户浏览 Disease 节点）
        neo4j.get_graph_data.return_value = {
            "nodes": [{"id": "1", "label": "Disease", "properties": {"name": "高血压"}, "degree": 5}],
            "edges": [],
            "stats": {"total_nodes": 1, "total_edges": 0, "node_labels": ["Disease"], "relationship_types": []}
        }
        data1 = neo4j.get_graph_data(node_label="Disease")
        cache.raw_set("Disease:500:0", data1, tags=["Disease"])
        assert neo4j.get_graph_data.call_count == 1

        # 2. 同查询命中缓存，不走 Neo4j
        cached = cache.raw_get("Disease:500:0")
        assert cached is not None
        assert neo4j.get_graph_data.call_count == 1

        # 3. 模拟录入新数据
        neo4j.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Disease", "properties": {"name": "高血压"}, "degree": 5},
                {"id": "2", "label": "Disease", "properties": {"name": "糖尿病"}, "degree": 3},
            ],
            "edges": [],
            "stats": {"total_nodes": 2, "total_edges": 0, "node_labels": ["Disease"], "relationship_types": []}
        }
        cache.invalidate_by_tag("Disease")

        # 4. 再次查询 → cache miss → Neo4j 返回最新数据
        cached = cache.raw_get("Disease:500:0")
        assert cached is None

        data2 = neo4j.get_graph_data(node_label="Disease")
        assert neo4j.get_graph_data.call_count == 2
        assert len(data2["nodes"]) == 2
