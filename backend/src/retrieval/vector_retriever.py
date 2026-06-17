from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.neo4j_client import Neo4jClient
from ..core.cache import cached, get_query_cache
from ..ingestion.embedding import EmbeddingClient, get_embedding_client


class VectorRetriever:
    """Vector similarity search in Neo4j"""

    # Name of the vector index created during ingestion
    VECTOR_INDEX_NAME = "chunk_vector_index"

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.neo4j_client = neo4j_client
        self.embedding_client = embedding_client

    def _get_neo4j_client(self) -> Neo4jClient:
        if self.neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client

            self.neo4j_client = get_neo4j_client()
        return self.neo4j_client

    def _get_embedding_client(self) -> EmbeddingClient:
        if self.embedding_client is None:
            self.embedding_client = get_embedding_client()
        return self.embedding_client

    @cached(get_query_cache)
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using the vector index (Neo4j 5.x / 2026.x)"""
        client = self._get_neo4j_client()

        # Get query embedding
        embedding = self._get_embedding_client().embed_text(query)

        # Use Neo4j 5.x vector index query API for best performance
        try:
            results = self._vector_index_search(client, embedding, top_k, filters)
            logger.info(f"Vector index search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Vector index search failed ({e}), falling back to cosine scan")
            try:
                return self._cosine_scan_search(client, embedding, top_k, filters)
            except Exception as e2:
                logger.warning(f"Cosine scan failed ({e2}), falling back to text search")
                return self._fallback_text_search(query, top_k, filters)

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using the vector index (async)"""
        client = self._get_neo4j_client()

        # Get query embedding
        embedding = self._get_embedding_client().embed_text(query)

        # Use Neo4j 5.x vector index query API for best performance
        try:
            results = await self._vector_index_search_async(client, embedding, top_k, filters)
            logger.info(f"Vector index search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Vector index search failed ({e}), falling back to cosine scan")
            try:
                return await self._cosine_scan_search_async(client, embedding, top_k, filters)
            except Exception as e2:
                logger.warning(f"Cosine scan failed ({e2}), falling back to text search")
                return self._fallback_text_search(query, top_k, filters)

    async def _vector_index_search_async(
        self,
        client: Neo4jClient,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use Neo4j vector index for ANN search (async)"""
        base_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node AS c, score AS similarity
        """
        params: Dict[str, Any] = {
            "index_name": self.VECTOR_INDEX_NAME,
            "top_k": top_k,
            "embedding": embedding,
        }

        if filters:
            filter_parts = [f"c.{k} = ${k}" for k in filters.keys()]
            base_query += "WHERE " + " AND ".join(filter_parts) + "\n"
            params.update(filters)

        base_query += """
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               similarity
        """
        return await client.execute_query_async(base_query, params)

    async def _cosine_scan_search_async(
        self,
        client: Neo4jClient,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Full-scan cosine similarity (async fallback)"""
        filter_clause = ""
        params: Dict[str, Any] = {"embedding": embedding, "top_k": top_k}
        if filters:
            filter_parts = [f"c.{k} = ${k}" for k in filters.keys()]
            filter_clause = "AND " + " AND ".join(filter_parts)
            params.update(filters)

        query_cypher = f"""
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL {filter_clause}
        WITH c, vector.similarity.cosine(c.embedding, $embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               similarity
        """
        return await client.execute_query_async(query_cypher, params)

    def _vector_index_search(
        self,
        client: Neo4jClient,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use Neo4j vector index for ANN search (fastest, Neo4j 5.x+)"""
        # Neo4j 5.x vector index query: db.index.vector.queryNodes
        base_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node AS c, score AS similarity
        """
        params: Dict[str, Any] = {
            "index_name": self.VECTOR_INDEX_NAME,
            "top_k": top_k,
            "embedding": embedding,
        }

        # Apply post-filter if needed (vector index doesn't support pre-filters)
        if filters:
            filter_parts = [f"c.{k} = ${k}" for k in filters.keys()]
            base_query += "WHERE " + " AND ".join(filter_parts) + "\n"
            params.update(filters)

        base_query += """
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               similarity
        """
        return client.execute_query(base_query, params)

    def _cosine_scan_search(
        self,
        client: Neo4jClient,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Full-scan cosine similarity (fallback when vector index not available)"""
        filter_clause = ""
        params: Dict[str, Any] = {"embedding": embedding, "top_k": top_k}
        if filters:
            filter_parts = [f"c.{k} = ${k}" for k in filters.keys()]
            filter_clause = "AND " + " AND ".join(filter_parts)
            params.update(filters)

        query_cypher = f"""
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL {filter_clause}
        WITH c, vector.similarity.cosine(c.embedding, $embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               similarity
        """
        return client.execute_query(query_cypher, params)

    def _fallback_text_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback to text-based search when no embeddings exist"""
        client = self._get_neo4j_client()

        params: Dict[str, Any] = {"query": query, "top_k": top_k}
        filter_clause = ""
        if filters:
            filter_parts = [f"c.{k} = ${k}" for k in filters.keys()]
            filter_clause = "AND " + " AND ".join(filter_parts)
            params.update(filters)

        query_cypher = f"""
        MATCH (c:Chunk)
        WHERE c.content CONTAINS $query {filter_clause}
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               1.0 AS similarity
        LIMIT $top_k
        """

        return client.execute_query(query_cypher, params)


def search_vectors(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for vector search"""
    retriever = VectorRetriever()
    return retriever.search(query, top_k, filters)
