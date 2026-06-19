"""Tests for src.retrieval.graph_retriever module.

Pure unit tests with all external dependencies mocked:
- Neo4jClient.execute_query, EmbeddingClient.embed_text, cache
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.retrieval.graph_retriever import GraphRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_retriever():
    """Create a GraphRetriever with a mocked Neo4jClient."""
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    with patch("src.retrieval.graph_retriever.cached", side_effect=lambda c: (lambda f: f)):
        gr = GraphRetriever(neo4j_client=mock_client)
    # Override _get_embedding_client to return our mock
    gr._get_embedding_client = MagicMock(return_value=mock_embedding)
    return gr, mock_client, mock_embedding


# ===================================================================
# TestGraphRetrieverSearch
# ===================================================================

class TestGraphRetrieverSearch:

    def test_search_with_query_type(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [{"name": "高血压", "type": "Disease"}]
        # Mock _get_queries to return a predefined query
        gr._queries = {"queries": {"find_entity": "MATCH (e:Entity) RETURN e"}}
        result = gr.search("query", query_type="find_entity")
        assert len(result) == 1
        assert result[0]["name"] == "高血压"
        mock_client.execute_query.assert_called_once()

    def test_search_direct_query(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [{"name": "糖尿病"}]
        result = gr.search("MATCH (e:Entity) RETURN e")
        assert len(result) == 1
        mock_client.execute_query.assert_called_once()

    def test_search_query_type_not_found(self):
        gr, mock_client, _ = _make_graph_retriever()
        gr._queries = {"queries": {}}
        result = gr.search("query", query_type="nonexistent")
        assert result == []
        mock_client.execute_query.assert_not_called()


# ===================================================================
# TestFindEntities
# ===================================================================

class TestFindEntities:

    def test_find_by_name(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"name": "高血压", "type": "Disease", "properties": None}
        ]
        result = gr.find_entities(entity_name="高血压")
        assert len(result) == 1
        assert result[0]["name"] == "高血压"
        # Verify the query contains CONTAINS
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "CONTAINS" in query_text

    def test_find_by_type(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"name": "高血压", "type": "Disease", "properties": None}
        ]
        result = gr.find_entities(entity_type="Disease")
        assert len(result) == 1
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "e.type" in query_text

    def test_find_by_name_and_type(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"name": "高血压", "type": "Disease", "properties": None}
        ]
        result = gr.find_entities(entity_name="高血压", entity_type="Disease")
        assert len(result) == 1
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "CONTAINS" in query_text
        assert "e.type" in query_text

    def test_find_no_filters(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"name": "高血压", "type": "Disease", "properties": None},
            {"name": "糖尿病", "type": "Disease", "properties": None},
        ]
        result = gr.find_entities()
        assert len(result) == 2
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        # No WHERE clause when no filters
        assert "WHERE" not in query_text


# ===================================================================
# TestFindRelationships
# ===================================================================

class TestFindRelationships:

    def test_find_relationships_success(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {
                "source": "高血压",
                "relationship_type": "TREATED_BY",
                "target": "硝苯地平",
                "target_type": "Drug",
            }
        ]
        result = gr.find_relationships("高血压", depth=2, limit=5)
        assert len(result) == 1
        assert result[0]["source"] == "高血压"
        assert result[0]["relationship_type"] == "TREATED_BY"


# ===================================================================
# TestFindPaths
# ===================================================================

class TestFindPaths:

    def test_find_paths_success(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"path": "...", "path_length": 2}
        ]
        result = gr.find_paths("高血压", "心肌梗死", max_depth=3)
        assert len(result) == 1
        assert result[0]["path_length"] == 2


# ===================================================================
# TestGetEntityCount
# ===================================================================

class TestGetEntityCount:

    def test_entity_count(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"entity_type": "Disease", "count": 50},
            {"entity_type": "Drug", "count": 30},
        ]
        result = gr.get_entity_count()
        assert result == {"Disease": 50, "Drug": 30}


# ===================================================================
# TestGetDocumentChunks
# ===================================================================

class TestGetDocumentChunks:

    def test_get_chunks(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"chunk_id": "c1", "content": "text1", "index": 0},
            {"chunk_id": "c2", "content": "text2", "index": 1},
        ]
        result = gr.get_document_chunks("doc123")
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"


# ===================================================================
# TestGetChunkParent
# ===================================================================

class TestGetChunkParent:

    def test_parent_found(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"document_id": "doc1", "title": "Test Doc", "source": "test", "properties": None}
        ]
        result = gr.get_chunk_parent("chunk1")
        assert result is not None
        assert result["document_id"] == "doc1"

    def test_parent_not_found(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = []
        result = gr.get_chunk_parent("nonexistent")
        assert result is None


# ===================================================================
# TestFindChunksByEntity
# ===================================================================

class TestFindChunksByEntity:

    def test_find_chunks(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"chunk_id": "c1", "content": "text", "index": 0, "document_id": "d1", "document_title": "Doc1"},
        ]
        result = gr.find_chunks_by_entity("高血压")
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"


# ===================================================================
# TestMultiHopSearch
# ===================================================================

class TestMultiHopSearch:

    def test_multi_hop_success(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {
                "entity_name": "心肌梗死",
                "entity_type": "Disease",
                "properties": None,
                "hop_distance": 2,
                "relation_types": ["CAUSES"],
                "path_nodes": [{"name": "高血压", "type": "Disease"}],
                "frequency": 3,
            }
        ]
        result = gr.multi_hop_search("高血压", hop_count=2)
        assert result["start_entity"] == "高血压"
        assert result["hop_count"] == 2
        assert len(result["entities"]) == 1
        assert result["entities"][0]["entity_name"] == "心肌梗死"

    def test_multi_hop_failure_returns_empty(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.side_effect = Exception("Neo4j error")
        result = gr.multi_hop_search("高血压", hop_count=2)
        assert result["start_entity"] == "高血压"
        assert result["entities"] == []


# ===================================================================
# TestFindRelatedEntities
# ===================================================================

class TestFindRelatedEntities:

    def test_find_related_success(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.return_value = [
            {"rel_type": "TREATED_BY", "entity_name": "硝苯地平", "entity_type": "Drug", "properties": None},
            {"rel_type": "HAS_SYMPTOM", "entity_name": "头痛", "entity_type": "Symptom", "properties": None},
        ]
        result = gr.find_related_entities("高血压", depth=1)
        assert result["entity_name"] == "高血压"
        assert "TREATED_BY" in result["related_by_relation"]
        assert "HAS_SYMPTOM" in result["related_by_relation"]
        assert result["total_count"] == 2

    def test_find_related_failure_returns_empty(self):
        gr, mock_client, _ = _make_graph_retriever()
        mock_client.execute_query.side_effect = Exception("Neo4j error")
        result = gr.find_related_entities("高血压", depth=1)
        assert result["entity_name"] == "高血压"
        assert result["related_by_relation"] == {}
        assert result["total_count"] == 0
