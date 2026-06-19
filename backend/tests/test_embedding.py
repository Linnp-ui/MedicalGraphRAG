"""Tests for src.ingestion.embedding"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from src.ingestion.embedding import EmbeddingClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_settings(**overrides):
    """Create a mock settings object with sensible defaults."""
    defaults = {
        "embedding_model": "text-embedding-v3",
        "embedding_dimensions": 1024,
        "embedding_api_key": "test-key",
        "embedding_base_url": "https://api.example.com/v1",
        "dashscope_api_key": "dash-key",
        "dashscope_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# TestEmbeddingClient
# ---------------------------------------------------------------------------
class TestEmbeddingClient:

    def test_init_with_defaults(self):
        settings = _make_settings()
        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient()
            assert client.model == "text-embedding-v3"
            assert client.dimensions == 1024
            assert client._client is None

    def test_init_with_custom_params(self):
        settings = _make_settings()
        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient(model="custom-model", dimensions=512)
            assert client.model == "custom-model"
            assert client.dimensions == 512

    def test_embed_text_success(self):
        settings = _make_settings()
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create.return_value = mock_response

        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient()
            client._client = mock_openai  # bypass lazy init

            result = client.embed_text("hello world")
            assert result == [0.1, 0.2, 0.3]
            mock_openai.embeddings.create.assert_called_once_with(
                model=client.model,
                input="hello world",
                dimensions=client.dimensions,
            )

    def test_embed_texts_success(self):
        settings = _make_settings()
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_openai.embeddings.create.return_value = mock_response

        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient()
            client._client = mock_openai

            result = client.embed_texts(["text1", "text2"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]

    def test_embed_texts_empty_list(self):
        settings = _make_settings()
        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient()
            result = client.embed_texts([])
            assert result == []

    def test_get_embedding_dimension(self):
        settings = _make_settings()
        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient(dimensions=768)
            assert client.get_embedding_dimension() == 768

    def test_get_client_lazy_init(self):
        settings = _make_settings()
        with patch("src.ingestion.embedding.get_settings", return_value=settings):
            client = EmbeddingClient()

        assert client._client is None

        with patch("src.ingestion.embedding.get_settings", return_value=settings), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_openai_instance = MagicMock()
            mock_openai_cls.return_value = mock_openai_instance

            result = client._get_client()

            assert result is mock_openai_instance
            assert client._client is mock_openai_instance
            mock_openai_cls.assert_called_once()
