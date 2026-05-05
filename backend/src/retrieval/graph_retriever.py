from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.neo4j_client import Neo4jClient
from ..core.cache import cached, get_query_cache
from ..core.config import load_cypher_queries
from ..ingestion.embedding import EmbeddingClient, get_embedding_client


class GraphRetriever:
    """Graph-based retrieval using Cypher queries"""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j_client = neo4j_client
        self._queries = None

    def _get_neo4j_client(self) -> Neo4jClient:
        if self.neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client

            self.neo4j_client = get_neo4j_client()
        return self.neo4j_client

    def _get_queries(self) -> Dict[str, Any]:
        if self._queries is None:
            self._queries = load_cypher_queries()
        return self._queries

    @cached(get_query_cache)
    def search(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        query_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a graph query"""
        client = self._get_neo4j_client()

        # If query type is specified, use predefined query
        if query_type:
            queries = self._get_queries()
            cypher = queries.get("queries", {}).get(query_type, "")
            if not cypher:
                logger.warning(f"Query type '{query_type}' not found")
                return []

            return client.execute_query(cypher, params or {})

        # Otherwise, try to execute the query directly
        try:
            return client.execute_query(query, params or {})
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []

    @cached(get_query_cache)
    def find_entities(
        self,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find entities in the graph"""
        client = self._get_neo4j_client()

        query = "MATCH (e:Entity)"
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if entity_name:
            conditions.append("e.name CONTAINS $entity_name")
            params["entity_name"] = entity_name

        if entity_type:
            conditions.append("e.type = $entity_type")
            params["entity_type"] = entity_type

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += """
        RETURN e.name as name, e.type as type, e.properties as properties
        LIMIT $limit
        """

        return client.execute_query(query, params)

    def _get_embedding_client(self) -> EmbeddingClient:
        return get_embedding_client()

    @cached(get_query_cache)
    def find_entities_by_embedding(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find entities using embedding similarity"""
        client = self._get_neo4j_client()

        try:
            embedding = self._get_embedding_client().embed_text(query)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return self.find_entities(entity_name=query, limit=limit)

        cypher = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
        WHERE score > 0.7
        RETURN e.name as name, e.type as type, e.properties as properties, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            return client.execute_query(cypher, {"embedding": embedding, "limit": limit})
        except Exception as e:
            logger.warning(f"Entity embedding search failed: {e}, falling back to name search")
            return self.find_entities(entity_name=query, limit=limit)

    @cached(get_query_cache)
    def find_relationships(
        self,
        entity_name: str,
        depth: int = 1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relationships for an entity"""
        client = self._get_neo4j_client()

        query = f"""
        MATCH (e:Entity {{name: $entity_name}})-[r*1..{depth}]-(related)
        RETURN e.name as source,
               type(r[0]) as relationship_type,
               related.name as target,
               related.type as target_type
        LIMIT $limit
        """

        return client.execute_query(query, {"entity_name": entity_name, "limit": limit})

    @cached(get_query_cache)
    def find_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find paths between two entities"""
        client = self._get_neo4j_client()

        query = f"""
        MATCH path = (start:Entity {{name: $start_entity}})-[*1..{max_depth}]-(end:Entity {{name: $end_entity}})
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT 5
        """

        return client.execute_query(
            query,
            {"start_entity": start_entity, "end_entity": end_entity},
        )

    def get_entity_count(self) -> Dict[str, int]:
        """Get count of entities by type"""
        client = self._get_neo4j_client()

        query = """
        MATCH (e:Entity)
        RETURN labels(e)[0] as entity_type, count(e) as count
        ORDER BY count DESC
        """

        results = client.execute_query(query)
        return {r["entity_type"]: r["count"] for r in results}

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get chunks for a document"""
        client = self._get_neo4j_client()

        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id as chunk_id, c.content as content, c.index as index
        ORDER BY c.index
        LIMIT $limit
        """

        return client.execute_query(query, {"document_id": document_id, "limit": limit})


def find_entities(
    entity_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Convenience function to find entities"""
    retriever = GraphRetriever()
    return retriever.find_entities(entity_name, entity_type, limit)


def find_relationships(
    entity_name: str,
    depth: int = 1,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Convenience function to find relationships"""
    retriever = GraphRetriever()
    return retriever.find_relationships(entity_name, depth, limit)
