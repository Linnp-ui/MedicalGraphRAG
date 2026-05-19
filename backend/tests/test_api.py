"""
GraphRAG API测试套件

测试所有API端点的功能、边界条件和异常场景
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import app
from src.core.cache import clear_all_caches


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    clear_all_caches()
    yield


class TestHealthAPI:
    """健康检查API测试"""
    
    def test_health_check_success(self):
        """测试健康检查成功响应"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "neo4j_connected" in data
        assert "version" in data
    
    def test_health_check_structure(self):
        """测试健康检查响应结构"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["neo4j_connected"], bool)
        assert isinstance(data["version"], str)


class TestGraphAPI:
    """图谱API测试"""
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_success(self, mock_get_client):
        """测试获取图谱数据成功"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}, "degree": 5}
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
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_with_limit(self, mock_get_client):
        """测试带限制的图谱数据获取"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes": 0,
                "total_edges": 0,
                "node_labels": [],
                "relationship_types": []
            }
        }
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/data?limit=100")
        assert response.status_code == 200
        mock_client.get_graph_data.assert_called_once()
        _, kwargs = mock_client.get_graph_data.call_args
        assert kwargs["limit"] == 100
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_with_node_label(self, mock_get_client):
        """测试按节点类型筛选"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [
                {"id": "1", "label": "Person", "properties": {"name": "张三"}, "degree": 3}
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
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_error(self, mock_get_client):
        """测试获取图谱数据失败"""
        mock_client = Mock()
        mock_client.get_graph_data.side_effect = Exception("Database error")
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/data")
        assert response.status_code == 500

    @patch('src.api.routes.get_neo4j_client')
    def test_get_graph_data_invalid_node_label(self, mock_get_client):
        """测试非法节点标签被拒绝"""
        mock_client = Mock()
        mock_client.get_graph_data.side_effect = ValueError("Invalid node_label")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/data?node_label=Person;MATCH")
        assert response.status_code == 400
    
    @patch('src.api.routes.get_neo4j_client')
    def test_search_nodes_success(self, mock_get_client):
        """测试搜索节点成功"""
        mock_client = Mock()
        mock_client.search_nodes_with_score.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}, "score": 1.0}
        ]
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/search?query=张三")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 1
    
    @patch('src.api.routes.get_neo4j_client')
    def test_search_nodes_empty_query(self, mock_get_client):
        """测试空查询字符串"""
        mock_client = Mock()
        mock_client.search_nodes_with_score.return_value = []
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/search?query=")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
    
    @patch('src.api.routes.get_neo4j_client')
    def test_search_nodes_with_limit(self, mock_get_client):
        """测试搜索结果限制"""
        mock_client = Mock()
        mock_client.search_nodes_with_score.return_value = [
            {"id": str(i), "label": "Person", "properties": {"name": f"用户{i}"}, "score": 1.0}
            for i in range(10)
        ]
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/search?query=用户&limit=10")
        assert response.status_code == 200
        _, kwargs = mock_client.search_nodes_with_score.call_args
        assert kwargs["limit"] == 10
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_detail_success(self, mock_get_client):
        """测试获取节点详情成功"""
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
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_detail_not_found(self, mock_get_client):
        """测试节点不存在"""
        mock_client = Mock()
        mock_client.get_node_detail.return_value = None
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/node/999999")
        assert response.status_code == 404
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_neighbors_success(self, mock_get_client):
        """测试获取节点邻居成功"""
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
        
        response = client.get("/api/v1/graph/node/1/neighbors")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "center_node" in data
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_neighbors_with_depth(self, mock_get_client):
        """测试指定深度的邻居查询"""
        mock_client = Mock()
        mock_client.get_node_neighbors.return_value = {
            "nodes": [],
            "edges": [],
            "center_node": "1"
        }
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/node/1/neighbors?depth=2")
        assert response.status_code == 200
        _, kwargs = mock_client.get_node_neighbors.call_args
        assert kwargs["depth"] == 2

    @patch('src.api.routes.get_neo4j_client')
    def test_get_node_neighbors_invalid_relationship_type(self, mock_get_client):
        """测试非法关系类型被拒绝"""
        mock_client = Mock()
        mock_client.get_node_neighbors.side_effect = ValueError("Invalid relationship_type")
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/node/1/neighbors?relationship_type=KNOWS;DELETE")
        assert response.status_code == 400
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_query_result_graph_success(self, mock_get_client):
        """测试获取查询结果图谱成功"""
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


class TestIngestAPI:
    """文档摄取API测试"""
    
    @patch('src.api.routes.KnowledgeGraphBuilder')
    @patch('src.ingestion.document_loader.load_document')
    @patch('src.api.routes.get_neo4j_client')
    def test_ingest_file_success(self, mock_get_client, mock_load_doc, mock_builder_class):
        """测试文件摄取成功"""
        mock_client = Mock()
        mock_client.execute_query.return_value = []
        mock_client.invalidate_schema_cache = Mock()
        mock_get_client.return_value = mock_client
        
        mock_load_doc.return_value = {"content": "test content", "metadata": {}}
        
        mock_builder = Mock()
        mock_builder.ingest_document.return_value = {"status": "success", "nodes_created": 1}
        mock_builder.get_stats.return_value = {"nodes_created": 1}
        mock_builder_class.return_value = mock_builder
        
        response = client.post(
            "/api/v1/ingest",
            json={
                "file_path": "test.txt",
                "extract_entities": True,
                "create_embeddings": True
            }
        )
        assert response.status_code == 200
    
    @patch('src.api.routes.KnowledgeGraphBuilder')
    @patch('src.api.routes.load_documents_from_directory')
    @patch('src.api.routes.get_neo4j_client')
    def test_ingest_directory_success(self, mock_get_client, mock_load_dir, mock_builder_class):
        """测试目录摄取成功"""
        mock_client = Mock()
        mock_client.execute_query.return_value = []
        mock_client.invalidate_schema_cache = Mock()
        mock_get_client.return_value = mock_client
        
        mock_load_dir.return_value = [
            {"content": "doc1", "metadata": {}},
            {"content": "doc2", "metadata": {}}
        ]
        
        mock_builder = Mock()
        mock_builder.ingest_document.return_value = {"status": "success", "nodes_created": 1}
        mock_builder.get_stats.return_value = {"nodes_created": 2}
        mock_builder_class.return_value = mock_builder
        
        response = client.post(
            "/api/v1/ingest",
            json={
                "directory": "./data",
                "extract_entities": True,
                "create_embeddings": False
            }
        )
        assert response.status_code == 200


class TestQueryAPI:
    """问答API测试"""
    
    @patch('src.api.routes.run_workflow')
    @patch('src.api.routes.get_neo4j_client')
    def test_query_success(self, mock_get_client, mock_run_workflow):
        """测试问答成功"""
        mock_client = Mock()
        mock_client.execute_query.return_value = []
        mock_get_client.return_value = mock_client
        
        mock_run_workflow.return_value = {
            "question": "张三是谁？",
            "answer": "测试答案",
            "routing": "drift",
            "documents": [{"content": "测试内容", "similarity": 0.9}]
        }
        
        response = client.post(
            "/api/v1/query",
            json={"question": "张三是谁？"}
        )
        assert response.status_code == 200


class TestSchemaAPI:
    """Schema API测试"""
    
    @patch('src.api.routes.get_neo4j_client')
    def test_get_schema_success(self, mock_get_client):
        """测试获取Schema成功"""
        mock_client = Mock()
        mock_client.get_schema.return_value = "Node properties:\n  Person: name, age"
        mock_client.execute_query.side_effect = [
            [{"labels": ["Person", "Organization"]}],
            [{"types": ["KNOWS", "WORKS_FOR"]}]
        ]
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/schema")
        assert response.status_code == 200
        data = response.json()
        assert "schema" in data
        assert "node_labels" in data
        assert "relationship_types" in data


class TestMetricsAPI:
    """Metrics API测试"""
    
    @patch('src.api.routes.get_metrics')
    def test_get_metrics_success(self, mock_get_metrics):
        """测试获取指标成功"""
        mock_metrics = Mock()
        mock_metrics.get_metrics.return_value = {
            "requests_total": {"total": 100},
            "errors_total": {"total": 5},
            "requests_duration": {"avg": [50.0]},
            "neo4j_pool": {"active": 10}
        }
        mock_get_metrics.return_value = mock_metrics
        
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "errors_total" in data


class TestEdgeCases:
    """边界条件和异常场景测试"""
    
    @patch('src.api.routes.get_neo4j_client')
    def test_large_limit_value(self, mock_get_client):
        """测试过大的limit值被参数校验拒绝"""
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/data?limit=10000")
        assert response.status_code == 422
    
    @patch('src.api.routes.get_neo4j_client')
    def test_invalid_node_id_format(self, mock_get_client):
        """测试无效的节点ID格式"""
        response = client.get("/api/v1/graph/node/invalid_id")
        assert response.status_code == 400
    
    @patch('src.api.routes.get_neo4j_client')
    def test_special_characters_in_search(self, mock_get_client):
        """测试搜索特殊字符"""
        mock_client = Mock()
        mock_client.search_nodes_with_score.return_value = []
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/search?query=<script>alert('xss')</script>")
        assert response.status_code == 200

    def test_query_result_graph_invalid_node_ids(self):
        """测试查询结果图谱接口拒绝非法节点ID"""
        response = client.post(
            "/api/v1/graph/query-result",
            json={
                "query": "张三认识谁？",
                "node_ids": ["1", "invalid"],
                "max_depth": 2
            }
        )
        assert response.status_code == 422
    
    @patch('src.api.routes.get_neo4j_client')
    def test_unicode_search_query(self, mock_get_client):
        """测试Unicode搜索查询"""
        mock_client = Mock()
        mock_client.search_nodes_with_score.return_value = [
            {"id": "1", "label": "Person", "properties": {"name": "张三"}, "score": 1.0}
        ]
        mock_get_client.return_value = mock_client
        
        response = client.get("/api/v1/graph/search?query=张三")
        assert response.status_code == 200
    
    @patch('src.api.routes.get_neo4j_client')
    def test_concurrent_requests(self, mock_get_client):
        """测试并发请求处理"""
        import threading
        import time
        
        mock_client = Mock()
        mock_client.get_graph_data.return_value = {
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "node_labels": [], "relationship_types": []}
        }
        mock_get_client.return_value = mock_client
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/graph/data")
            results.append(response.status_code)
        
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
