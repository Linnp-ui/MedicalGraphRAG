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

    def get_chunk_parent(
        self,
        chunk_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get parent document of a chunk"""
        client = self._get_neo4j_client()

        query = """
        MATCH (c:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(d:Document)
        RETURN d.id as document_id, d.title as title, d.source as source, d.properties as properties
        """

        results = client.execute_query(query, {"chunk_id": chunk_id})
        return results[0] if results else None

    def get_chunk_context(
        self,
        chunk_id: str,
        context_chunks: int = 2,
    ) -> Dict[str, Any]:
        """Get chunk with surrounding context chunks"""
        client = self._get_neo4j_client()

        query = """
        MATCH (c:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(d:Document)
        WITH d, c.index as target_index
        MATCH (d)-[:HAS_CHUNK]->(context_chunk:Chunk)
        WHERE context_chunk.index >= target_index - $context_chunks
          AND context_chunk.index <= target_index + $context_chunks
        RETURN context_chunk.id as chunk_id,
               context_chunk.content as content,
               context_chunk.index as index,
               context_chunk.index = target_index as is_target,
               d.id as document_id,
               d.title as document_title
        ORDER BY context_chunk.index
        """

        results = client.execute_query(
            query,
            {"chunk_id": chunk_id, "context_chunks": context_chunks}
        )

        if not results:
            return None

        return {
            "chunk_id": chunk_id,
            "document_id": results[0]["document_id"],
            "document_title": results[0]["document_title"],
            "context": [
                {
                    "chunk_id": r["chunk_id"],
                    "content": r["content"],
                    "index": r["index"],
                    "is_target": r["is_target"],
                }
                for r in results
            ],
        }

    def reconstruct_document(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct full document from chunks"""
        client = self._get_neo4j_client()

        query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, c
        ORDER BY c.index
        RETURN d.id as document_id,
               d.title as title,
               d.source as source,
               d.properties as properties,
               collect(c.content) as chunk_contents,
               collect(c.index) as chunk_indices
        """

        results = client.execute_query(query, {"document_id": document_id})

        if not results:
            return None

        doc = results[0]
        chunk_contents = doc.get("chunk_contents", [])
        chunk_indices = doc.get("chunk_indices", [])

        sorted_chunks = sorted(
            zip(chunk_indices, chunk_contents),
            key=lambda x: x[0]
        )

        full_content = "\n\n".join([content for idx, content in sorted_chunks if content])

        return {
            "document_id": doc["document_id"],
            "title": doc["title"],
            "source": doc["source"],
            "properties": doc["properties"],
            "full_content": full_content,
            "chunk_count": len(sorted_chunks),
        }

    def find_chunks_by_entity(
        self,
        entity_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find chunks containing a specific entity"""
        client = self._get_neo4j_client()

        query = """
        MATCH (e:Entity {name: $entity_name})<-[:CONTAINS_ENTITY]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        RETURN c.id as chunk_id,
               c.content as content,
               c.index as index,
               d.id as document_id,
               d.title as document_title
        ORDER BY d.title, c.index
        LIMIT $limit
        """

        return client.execute_query(query, {"entity_name": entity_name, "limit": limit})

    def multi_hop_search(
        self,
        start_entity: str,
        hop_count: int = 2,
        relation_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """多跳搜索：从起始实体出发，通过多层关系找到相关实体

        Args:
            start_entity: 起始实体名称
            hop_count: 跳跃次数（1=直接邻居，2=邻居的邻居）
            relation_types: 要跟随的关系类型列表（None表示所有关系）
            entity_types: 要返回的实体类型列表（None表示所有类型）
            limit: 返回的实体数量限制

        Returns:
            包含路径信息和实体的字典
        """
        client = self._get_neo4j_client()

        rel_pattern = ""
        if relation_types:
            rel_filter = "|".join([f":`{r}`" for r in relation_types])
            rel_pattern = f"[r:{rel_filter}*1..{hop_count}]"
        else:
            rel_pattern = f"[r*1..{hop_count}]"

        entity_type_filter = ""
        if entity_types:
            type_labels = "|".join([f"`{t}`" for t in entity_types])
            entity_type_filter = f"WHERE exists(any(labels(n)) IN [{type_labels}])"

        query = f"""
        MATCH path = (start:Entity {{name: $start_entity}})-{rel_pattern}-(n)
        {entity_type_filter}
        WITH n, path,
             length(path) as hop_distance,
             relationships(path) as rels
        WITH n,
             hop_distance,
             [r IN rels | type(r)] as relation_types,
             [node IN nodes(path) | {{name: node.name, type: head(labels(node))}}] as path_nodes
        RETURN n.name as entity_name,
               head(labels(n)) as entity_type,
               n.properties as properties,
               hop_distance,
               relation_types,
               path_nodes,
               count(*) as frequency
        ORDER BY frequency DESC, hop_distance ASC
        LIMIT $limit
        """

        try:
            results = client.execute_query(query, {
                "start_entity": start_entity,
                "limit": limit
            })
            return {
                "start_entity": start_entity,
                "hop_count": hop_count,
                "entities": results,
            }
        except Exception as e:
            logger.warning(f"Multi-hop search failed: {e}")
            return {"start_entity": start_entity, "hop_count": hop_count, "entities": []}

    def find_related_entities(
        self,
        entity_name: str,
        depth: int = 1,
        relation_filter: Optional[List[str]] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """查找与给定实体相关的所有实体（支持关系过滤）

        Args:
            entity_name: 实体名称
            depth: 搜索深度
            relation_filter: 关系类型白名单（如 ["TREATED_BY", "HAS_SYMPTOM"]）
            limit: 返回数量限制

        Returns:
            按关系类型分组的实体列表
        """
        client = self._get_neo4j_client()

        rel_pattern = f"[*1..{depth}]"
        if relation_filter:
            rel_filter = "|".join([f":`{r}`" for r in relation_filter])
            rel_pattern = f"[r:{rel_filter}*1..{depth}]"

        query = f"""
        MATCH (start:Entity {{name: $entity_name}})-{rel_pattern}-(related)
        WHERE related <> start
        WITH related, type(relationships(path)[0]) as rel_type
        RETURN rel_type,
               related.name as entity_name,
               head(labels(related)) as entity_type,
               related.properties as properties
        ORDER BY rel_type
        LIMIT $limit
        """

        try:
            results = client.execute_query(query, {
                "entity_name": entity_name,
                "limit": limit
            })

            grouped = {}
            for r in results:
                rel_type = r.get("rel_type", "RELATED")
                if rel_type not in grouped:
                    grouped[rel_type] = []
                grouped[rel_type].append({
                    "entity_name": r.get("entity_name"),
                    "entity_type": r.get("entity_type"),
                    "properties": r.get("properties"),
                })

            return {
                "entity_name": entity_name,
                "depth": depth,
                "related_by_relation": grouped,
                "total_count": len(results),
            }
        except Exception as e:
            logger.warning(f"Find related entities failed: {e}")
            return {"entity_name": entity_name, "related_by_relation": {}, "total_count": 0}


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
