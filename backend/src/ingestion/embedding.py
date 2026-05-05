from typing import List, Optional

from loguru import logger

from ..core.config import get_settings


class EmbeddingClient:
    """OpenAI embedding client"""

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI

                settings = get_settings()
                self._client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url or "https://api.openai.com/v1",
                )
            except ImportError:
                logger.error("openai package not installed")
                raise ImportError("openai is required for embeddings")
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions


# Singleton instance
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Get singleton embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def embed_text(text: str) -> List[float]:
    """Convenience function to get text embedding"""
    client = get_embedding_client()
    return client.embed_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convenience function to get multiple text embeddings"""
    client = get_embedding_client()
    return client.embed_texts(texts)
