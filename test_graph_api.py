"""
测试图谱可视化接口 - 模拟测试
"""
import httpx
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestGraphVisualizationAPI:
    """图谱可视化API测试"""

    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "neo4j_connected" in data
        print(f"✅ 健康检查: {data['status']}")

    def test_liveness_check(self):
        """测试存活检查接口"""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        print(f"✅ 存活检查: {data['status']}")

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data(self, mock_get_client):
        """测试获取图谱数据接口"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}, "degree": 5},
                {"id": "2", "label": "Person", "properties": {"name": "李四"}, "degree": 3}
            ],
            "edges": [
                {"id": "e1", "from": "1", "to": "2", "type": "KNOWS", "properties": {}}
            ],
            "stats": {
                "total_nodes": 100,
                "total_edges": 200,
                "node_labels": ["Person", "Organization"],
                "relationship_types": ["KNOWS", "WORKS_FOR"]
            }
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        print(f"✅ 获取图谱数据: {len(data['nodes'])} 个节点, {len(data['edges'])} 条边")

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_with_node_label(self, mock_get_client):
        """测试按节点类型获取图谱数据"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}, "degree": 5}
            ],
            "edges": [],
            "stats": {
                "total_nodes": 1,
                "total_edges": 0,
                "node_labels": ["Person"],
                "relationship_types": []
            }
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?node_label=Person")
        assert response.status_code == 200
        data = response.json()
        assert all(node["label"] == "Person" for node in data["nodes"])
        print(f"✅ 按类型筛选: Person 类型 {len(data['nodes'])} 个节点")

    @patch('src.api.routes.get_neo4j_client')
    def test_search_nodes(self, mock_get_client):
        """测试搜索节点接口"""
        mock_client = Mock()
        mock_client.search_nodes.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}},
            {"id": "2", "label": "Person", "properties": {"name": "张四"}}
        ]
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?query=张")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        print(f"✅ 搜索节点: 找到 {data['total']} 个结果")

    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_detail(self, mock_get_client):
        """测试获取节点详情接口"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = {
            "node": {"id": "1", "label": "Person", "properties": {"name": "张三"}},
            "relationships": {
                "incoming": [],
                "outgoing": [{"to_node": "2", "type": "KNOWS", "properties": {}}]
            },
            "neighbors": [{"id": "2", "label": "Person", "properties": {"name": "李四"}}]
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1")
        assert response.status_code == 200
        data = response.json()
        assert "node" in data
        assert "relationships" in data
        assert "neighbors" in data
        print(f"✅ 节点详情: {data['node']['properties']['name']}")

    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_neighbors(self, mock_get_client):
        """测试获取节点邻居接口"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}},
                {"id": "2", "label": "Person", "properties": {"name": "李四"}}
            ],
            "edges": [
                {"id": "e1", "from": "1", "to": "2", "type": "KNOWS", "properties": {}}
            ],
            "center_node": "1"
        }
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1/neighbors?depth=2")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "center_node" in data
        print(f"✅ 节点邻居: {len(data['nodes'])} 个节点, {len(data['edges'])} 条边")

    @patch('src.api.routes.get_neo4j_client')
    def test_get_query_result_graph(self, mock_get_client):
        """测试获取查询结果图谱接口"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [{"id": "1", "label": "Person", "properties": {}}],
            "edges": [],
            "center_node": "1"
        }
        mock_client.execute_query.return_value = [{
            "total_nodes": 100,
            "total_edges": 200,
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
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        print(f"✅ 查询结果图谱: {len(data['nodes'])} 个节点")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
