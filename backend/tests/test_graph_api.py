"""
图谱可视化接口全面测试套件

包含：
- 正常用例测试
- 边界条件测试
- 异常场景测试
- 特殊字符和安全测试
- 并发测试
- 查询结果图谱测试
- 响应格式测试
"""
import pytest
import threading
import time
from unittest.mock import Mock, patch

from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestHealthCheck:
    """健康检查测试"""

    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "neo4j_connected" in data

    def test_liveness_check(self):
        """测试存活检查接口"""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestNormalCases:
    """正常用例测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_basic(self, mock_get_client):
        """测试基本图谱数据获取"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}, "degree": 5}
            ],
            "edges": [],
            "stats": {"total_nodes": 1, "total_edges": 0, "node_labels": ["Person"], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_with_pagination(self, mock_get_client):
        """测试分页参数"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?limit=100&offset=50")
        assert response.status_code == 200
        mock_client.get_graph_data.assert_called_once()

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_with_node_label(self, mock_get_client):
        """测试按节点类型筛选"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [{"id": "1", "label": "Person", "properties": {}, "degree": 1}],
            "edges": [],
            "stats": {"total_nodes": 1, "total_edges": 0, "node_labels": ["Person"], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?node_label=Person")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_search_nodes_basic(self, mock_get_client):
        """测试基本节点搜索"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}}
        ]
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?query=张三")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_detail_basic(self, mock_get_client):
        """测试获取节点详情"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = {
            "node": {"id": "1", "label": "Person", "properties": {"name": "张三"}},
            "relationships": {"incoming": [], "outgoing": []},
            "neighbors": []
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1")
        assert response.status_code == 200
        assert "node" in response.json()

    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_neighbors_basic(self, mock_get_client):
        """测试获取节点邻居"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [{"id": "2", "label": "Person", "properties": {}}],
            "edges": [],
            "center_node": "1"
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1/neighbors")
        assert response.status_code == 200


class TestBoundaryConditions:
    """边界条件测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_large_limit_value(self, mock_get_client):
        """测试过大的limit值"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?limit=10000")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_zero_limit_value(self, mock_get_client):
        """测试limit为0"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?limit=0")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_negative_offset(self, mock_get_client):
        """测试负数offset"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?offset=-1")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_empty_search_query(self, mock_get_client):
        """测试空搜索查询"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?query=")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    @patch('src.api.routes.get_neo4j_client')
    def test_very_long_search_query(self, mock_get_client):
        """测试超长搜索查询"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        long_query = "a" * 1000
        response = client.get(f"/api/v1/graph/search?query={long_query}")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_max_depth_value(self, mock_get_client):
        """测试最大深度值"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [],
            "edges": [],
            "center_node": "1"
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1/neighbors?depth=3")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_large_node_id(self, mock_get_client):
        """测试大数值节点ID"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = {
            "node": {"id": "999999999999", "label": "Person", "properties": {}},
            "relationships": {"incoming": [], "outgoing": []},
            "neighbors": []
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/999999999999")
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_large_data_response(self, mock_get_client):
        """测试大数据量响应"""
        nodes = [{"id": str(i), "label": "Person", "properties": {"name": f"用户{i}"}, "degree": i} for i in range(1000)]
        edges = [{"id": f"e{i}", "from": str(i), "to": str((i+1) % 1000), "type": "KNOWS", "properties": {}} for i in range(500)]

        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": nodes,
            "edges": edges,
            "stats": {"total_nodes": 1000, "total_edges": 500, "node_labels": ["Person"], "relationship_types": ["KNOWS"]}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?limit=1000")
        assert response.status_code == 200
        assert len(response.json()["nodes"]) == 1000


class TestExceptionScenarios:
    """异常场景测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_database_error(self, mock_get_client):
        """测试数据库错误"""
        mock_client = Mock()
        mock_client.get_graph_data.side_effect = Exception("Database connection failed")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data")
        assert response.status_code == 500

    @patch('src.api.routes.get_neo4j_client')
    def test_node_not_found(self, mock_get_client):
        """测试节点不存在"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = None
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/999999")
        assert response.status_code == 404

    @patch('src.api.routes.get_neo4j_client')
    def test_invalid_node_id_format(self, mock_get_client):
        """测试无效的节点ID格式"""
        mock_client = Mock()
        mock_client.get_node_detail.side_effect = ValueError("Invalid ID format")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/invalid_id")
        assert response.status_code == 500

    @patch('src.api.routes.get_neo4j_client')
    def test_search_error(self, mock_get_client):
        """测试搜索错误"""
        mock_client = Mock()
        mock_client.search_nodes.side_effect = Exception("Search failed")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?query=test")
        assert response.status_code == 500

    @patch('src.api.routes.get_neo4j_client')
    def test_neighbors_error(self, mock_get_client):
        """测试邻居查询错误"""
        mock_client = Mock()
        mock_client.get_node_neighbors.side_effect = Exception("Query failed")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1/neighbors")
        assert response.status_code == 500

    @patch('src.api.routes.get_neo4j_client')
    def test_timeout_simulation(self, mock_get_client):
        """测试超时模拟"""
        def slow_query(*args, **kwargs):
            time.sleep(0.1)
            return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}}

        mock_client = Mock()
        mock_client.get_graph_data.side_effect = slow_query
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data")
        assert response.status_code == 200


class TestSpecialCharactersAndSecurity:
    """特殊字符和安全测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_special_characters_in_search(self, mock_get_client):
        """测试搜索特殊字符"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        special_chars = ["<script>", "'; DROP TABLE--", "&amp;", "\"quote\"", "'single'"]
        for char in special_chars:
            response = client.get(f"/api/v1/graph/search?query={char}")
            assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_unicode_search_query(self, mock_get_client):
        """测试Unicode搜索查询"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}}
        ]
        mock_get_client.return_value = mock_client

        unicode_queries = ["张三", "日本語", "한국어", "العربية", "🎉🎊"]
        for query in unicode_queries:
            response = client.get(f"/api/v1/graph/search?query={query}")
            assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_sql_injection_attempt(self, mock_get_client):
        """测试SQL注入尝试"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        injection_attempts = [
            "'; DROP TABLE nodes;--",
            "1' OR '1'='1",
            "admin'--",
            "1; SELECT * FROM users"
        ]
        for attempt in injection_attempts:
            response = client.get(f"/api/v1/graph/search?query={attempt}")
            assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_cypher_injection_attempt(self, mock_get_client):
        """测试Cypher注入尝试"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        cypher_injection = [
            "MATCH (n) RETURN n",
            "CREATE (n:Hacked)",
            "DELETE (n)"
        ]
        for injection in cypher_injection:
            response = client.get(f"/api/v1/graph/search?query={injection}")
            assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_xss_attempt(self, mock_get_client):
        """测试XSS攻击尝试"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>"
        ]
        for payload in xss_payloads:
            response = client.get(f"/api/v1/graph/search?query={payload}")
            assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_path_traversal_attempt(self, mock_get_client):
        """测试路径遍历尝试"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [], "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?node_label=../../../etc/passwd")
        assert response.status_code == 200


class TestConcurrentRequests:
    """并发测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_concurrent_graph_data_requests(self, mock_get_client):
        """测试并发图谱数据请求"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client

        results = []
        errors = []

        def make_request():
            try:
                response = client.get("/api/v1/graph/data")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert all(status == 200 for status in results)

    @patch('src.api.routes.get_neo4j_client')
    def test_concurrent_search_requests(self, mock_get_client):
        """测试并发搜索请求"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = []
        mock_get_client.return_value = mock_client

        results = []
        errors = []

        def make_request(i):
            try:
                response = client.get(f"/api/v1/graph/search?query=test{i}")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert all(status == 200 for status in results)


class TestQueryResultGraph:
    """查询结果图谱测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_query_result_graph_basic(self, mock_get_client):
        """测试基本查询结果图谱"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [{"id": "1", "label": "Person", "properties": {}}],
            "edges": [],
            "center_node": "1"
        }
        mock_client.execute_query.return_value = [{
            "total_nodes": 10,
            "total_edges": 5,
            "node_labels": ["Person"],
            "relationship_types": ["KNOWS"]
        }]
        mock_get_client.return_value = mock_client

        response = client.post(
            "/api/v1/graph/query-result",
            json={
                "query": "张三认识谁？",
                "node_ids": ["1", "2"],
                "max_depth": 2
            }
        )
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_query_result_graph_empty_nodes(self, mock_get_client):
        """测试空节点列表"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [],
            "edges": [],
            "center_node": "1"
        }
        mock_client.execute_query.return_value = [{
            "total_nodes": 0,
            "total_edges": 0,
            "node_labels": [],
            "relationship_types": []
        }]
        mock_get_client.return_value = mock_client

        response = client.post(
            "/api/v1/graph/query-result",
            json={
                "query": "测试查询",
                "node_ids": [],
                "max_depth": 2
            }
        )
        assert response.status_code == 200

    @patch('src.api.routes.get_neo4j_client')
    def test_query_result_graph_max_depth_boundary(self, mock_get_client):
        """测试最大深度边界"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [],
            "edges": [],
            "center_node": "1"
        }
        mock_client.execute_query.return_value = [{
            "total_nodes": 0,
            "total_edges": 0,
            "node_labels": [],
            "relationship_types": []
        }]
        mock_get_client.return_value = mock_client

        for depth in [1, 2, 3]:
            response = client.post(
                "/api/v1/graph/query-result",
                json={
                    "query": "测试",
                    "node_ids": ["1"],
                    "max_depth": depth
                }
            )
            assert response.status_code == 200


class TestResponseFormat:
    """响应格式测试"""

    @patch('src.api.routes.get_neo4j_client')
    def test_graph_data_response_structure(self, mock_get_client):
        """测试图谱数据响应结构"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [{"id": "1", "label": "Person", "properties": {}, "degree": 1}],
            "edges": [{"id": "e1", "from": "1", "to": "2", "type": "KNOWS", "properties": {}}],
            "stats": {"total_nodes": 1, "total_edges": 1, "node_labels": ["Person"], "relationship_types": ["KNOWS"]}
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data")
        data = response.json()

        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert isinstance(data["stats"], dict)

    @patch('src.api.routes.get_neo4j_client')
    def test_search_response_structure(self, mock_get_client):
        """测试搜索响应结构"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}}
        ]
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?query=张")
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)

    @patch('src.api.routes.get_neo4j_client')
    def test_node_detail_response_structure(self, mock_get_client):
        """测试节点详情响应结构"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = {
            "node": {"id": "1", "label": "Person", "properties": {}},
            "relationships": {"incoming": [], "outgoing": []},
            "neighbors": []
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1")
        data = response.json()

        assert "node" in data
        assert "relationships" in data
        assert "neighbors" in data
        assert "incoming" in data["relationships"]
        assert "outgoing" in data["relationships"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
