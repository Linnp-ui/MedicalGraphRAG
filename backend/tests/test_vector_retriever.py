"""Tests for src.retrieval.vector_retriever module.

Pure unit tests with all external dependencies mocked:
- Neo4jClient, EmbeddingClient, cache
"""

import pytest
from unittest.mock import patch, MagicMock

from src.retrieval.vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vector_retriever():
    """Create a VectorRetriever with mocked Neo4jClient and EmbeddingClient."""
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    with patch("src.retrieval.vector_retriever.cached", side_effect=lambda c: (lambda f: f)):
        vr = VectorRetriever(neo4j_client=mock_client, embedding_client=mock_embedding)
    return vr, mock_client, mock_embedding


# ===================================================================
# TestVectorRetrieverSearch
# ===================================================================

class TestVectorRetrieverSearch:

    def test_search_vector_index_success(self):
        vr, mock_client, mock_embedding = _make_vector_retriever()
        mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_client.execute_query.return_value = [
            {"chunk_id": "c1", "content": "text1", "document_id": "d1", "index": 0, "similarity": 0.95}
        ]
        result = vr.search("高血压的治疗")
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"
        mock_embedding.embed_text.assert_called_once_with("高血压的治疗")

    def test_search_fallback_to_cosine(self):
        vr, mock_client, mock_embedding = _make_vector_retriever()
        mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]
        # First call (vector index) raises, second call (cosine scan) succeeds
        mock_client.execute_query.side_effect = [
            Exception("vector index not available"),
            [{"chunk_id": "c2", "content": "text2", "document_id": "d2", "index": 1, "similarity": 0.8}],
        ]
        result = vr.search("糖尿病的预防方法")
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c2"

    def test_search_fallback_to_text(self):
        vr, mock_client, mock_embedding = _make_vector_retriever()
        mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]
        # Both vector index and cosine scan fail
        mock_client.execute_query.side_effect = [
            Exception("vector index not available"),
            Exception("cosine scan failed"),
            [{"chunk_id": "c3", "content": "text3", "document_id": "d3", "index": 2, "similarity": 1.0}],
        ]
        result = vr.search("感冒的症状表现")
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c3"

    def test_search_with_filters(self):
        vr, mock_client, mock_embedding = _make_vector_retriever()
        mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_client.execute_query.return_value = [
            {"chunk_id": "c1", "content": "text1", "document_id": "d1", "index": 0, "similarity": 0.9}
        ]
        result = vr.search("高血压", filters={"document_id": "doc1"})
        assert len(result) == 1
        # Verify filters were passed through
        call_args = mock_client.execute_query.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert "document_id" in params


# ===================================================================
# TestVectorIndexSearch
# ===================================================================

class TestVectorIndexSearch:

    def test_builds_correct_query(self):
        vr, mock_client, _ = _make_vector_retriever()
        mock_client.execute_query.return_value = []
        embedding = [0.1, 0.2, 0.3]
        vr._vector_index_search(mock_client, embedding, top_k=5, filters=None)
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "db.index.vector.queryNodes" in query_text
        params = call_args[0][1]
        assert params["index_name"] == "chunk_vector_index"
        assert params["top_k"] == 5
        assert params["embedding"] == embedding

    def test_applies_filters(self):
        vr, mock_client, _ = _make_vector_retriever()
        mock_client.execute_query.return_value = []
        embedding = [0.1, 0.2, 0.3]
        filters = {"document_id": "doc1"}
        vr._vector_index_search(mock_client, embedding, top_k=5, filters=filters)
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "WHERE" in query_text
        params = call_args[0][1]
        assert params["document_id"] == "doc1"


# ===================================================================
# TestCosineScanSearch
# ===================================================================

class TestCosineScanSearch:

    def test_builds_correct_query(self):
        vr, mock_client, _ = _make_vector_retriever()
        mock_client.execute_query.return_value = []
        embedding = [0.1, 0.2, 0.3]
        vr._cosine_scan_search(mock_client, embedding, top_k=5, filters=None)
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "vector.similarity.cosine" in query_text
        assert "MATCH (c:Chunk)" in query_text
        params = call_args[0][1]
        assert params["embedding"] == embedding
        assert params["top_k"] == 5


# ===================================================================
# TestFallbackTextSearch
# ===================================================================

class TestFallbackTextSearch:

    def test_builds_correct_query(self):
        vr, mock_client, _ = _make_vector_retriever()
        mock_client.execute_query.return_value = []
        vr._fallback_text_search("高血压", top_k=5, filters=None)
        call_args = mock_client.execute_query.call_args
        query_text = call_args[0][0]
        assert "CONTAINS" in query_text
        assert "MATCH (c:Chunk)" in query_text
        params = call_args[0][1]
        assert params["query"] == "高血压"
        assert params["top_k"] == 5
