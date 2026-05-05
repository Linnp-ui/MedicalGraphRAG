from typing import Any, Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from ..core.cache import cached, get_query_cache


class HybridRetriever:
    """Hybrid retrieval combining vector and graph search"""

    def __init__(
        self,
        alpha: float = 0.5,
        vector_top_k: int = 5,
        graph_top_k: int = 10,
        enable_parallel: bool = True,
        enable_entity_embedding: bool = True,
    ):
        """
        Initialize hybrid retriever

        Args:
            alpha: Weight for vector search (1-alpha for graph search)
            vector_top_k: Number of results from vector search
            graph_top_k: Number of results from graph search
            enable_parallel: Whether to parallelize vector and graph search
            enable_entity_embedding: Whether to use embedding-based entity matching
        """
        self.alpha = alpha
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.enable_parallel = enable_parallel
        self.enable_entity_embedding = enable_entity_embedding
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self._executor = ThreadPoolExecutor(max_workers=2)

    @cached(get_query_cache)
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """执行混合搜索，结合向量搜索和图谱搜索结果

        Args:
            query: 搜索查询
            filters: 可选的过滤条件
            alpha: 可选的 alpha 权重覆盖（未提供时自动计算）

        Returns:
            包含向量搜索结果、图谱搜索结果和组合结果的字典
        """
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = self._compute_dynamic_alpha(query)

        vector_results = []
        graph_results = []

        if self.enable_parallel:
            vector_results, graph_results = self._parallel_search(query, filters)
        else:
            vector_results = self._vector_search(query, filters)
            graph_results = self._graph_search(query, filters)

        combined_results = self._combine_results(vector_results, graph_results)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_results": combined_results,
            "query": query,
            "alpha_used": self.alpha,
        }

    def _compute_dynamic_alpha(self, query: str) -> float:
        """根据查询特征动态计算 alpha 权重

        Args:
            query: 搜索查询

        Returns:
            计算得到的 alpha 权重值
        """
        entity_indicators = ["谁", "什么实体", "关系", "连接", "包含", "节点", "实体"]
        numeric_indicators = ["多少", "数量", "统计", "排名", "前", "最高", "最低"]

        query_lower = query.lower()

        if any(ind in query_lower for ind in entity_indicators):
            return 0.3  # 实体相关查询，增加图谱搜索权重
        elif any(ind in query_lower for ind in numeric_indicators):
            return 0.7  # 数值相关查询，增加向量搜索权重

        return 0.5  # 默认平衡权重

    def _parallel_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """并行执行向量搜索和图谱搜索

        Args:
            query: 搜索查询
            filters: 可选的过滤条件

        Returns:
            向量搜索结果和图谱搜索结果的元组
        """
        vector_future = self._executor.submit(self._vector_search, query, filters)
        graph_future = self._executor.submit(self._graph_search, query, filters)

        vector_results = vector_future.result()
        graph_results = graph_future.result()

        return vector_results, graph_results

    def _vector_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """获取向量搜索结果

        Args:
            query: 搜索查询
            filters: 可选的过滤条件

        Returns:
            向量搜索结果列表
        """
        try:
            return self.vector_retriever.search(
                query=query,
                top_k=self.vector_top_k,
                filters=filters,
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _graph_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """获取基于实体嵌入的图谱搜索结果

        Args:
            query: 搜索查询
            filters: 可选的过滤条件

        Returns:
            图谱搜索结果列表
        """
        try:
            if self.enable_entity_embedding:
                entities = self.graph_retriever.find_entities_by_embedding(
                    query=query,
                    limit=self.graph_top_k,
                )
            else:
                entities = self.graph_retriever.find_entities(
                    entity_name=query,
                    limit=self.graph_top_k,
                )

            graph_results = []
            for entity in entities:
                chunks = self._get_chunks_for_entity(entity["name"])
                for chunk in chunks:
                    chunk["entity_score"] = entity.get("score", 1.0)
                    chunk["entity_type"] = entity.get("type")
                graph_results.extend(chunks)

            return graph_results
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return []

    def _get_chunks_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取包含特定实体的文本块

        Args:
            entity_name: 实体名称

        Returns:
            包含该实体的文本块列表
        """
        client = self.graph_retriever._get_neo4j_client()

        query = """
        MATCH (e:Entity {name: $entity_name})<-[:CONTAINS_ENTITY]-(c:Chunk)
        RETURN c.id as chunk_id,
               c.content as content,
               c.document_id as document_id,
               c.index as index
        LIMIT 5
        """

        try:
            return client.execute_query(query, {"entity_name": entity_name})
        except Exception as e:
            logger.warning(f"Failed to get chunks for entity: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """组合并排序来自两个来源的结果

        Args:
            vector_results: 向量搜索结果
            graph_results: 图谱搜索结果

        Returns:
            按综合得分排序的组合结果列表
        """
        seen_chunks = {}
        seen_docs = {}
        combined = []

        vector_max_score = max((r.get("similarity", 0) for r in vector_results), default=1.0)

        for result in vector_results:
            chunk_id = result.get("chunk_id")
            doc_id = result.get("document_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = True
                norm_score = (
                    result.get("similarity", 0) / vector_max_score if vector_max_score > 0 else 0
                )
                combined.append(
                    {
                        **result,
                        "vector_score": norm_score,
                        "graph_score": 0,
                        "combined_score": norm_score * self.alpha,
                    }
                )
                if doc_id:
                    seen_docs[doc_id] = len(combined) - 1

        for result in graph_results:
            chunk_id = result.get("chunk_id")
            doc_id = result.get("document_id")
            entity_score = result.get("entity_score", 1.0)

            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = True
                combined.append(
                    {
                        **result,
                        "vector_score": 0,
                        "graph_score": entity_score,
                        "combined_score": entity_score * (1 - self.alpha),
                    }
                )
                if doc_id:
                    seen_docs[doc_id] = len(combined) - 1
            elif chunk_id in seen_chunks and doc_id:
                for i, item in enumerate(combined):
                    if item.get("chunk_id") == chunk_id:
                        item["graph_score"] = max(item.get("graph_score", 0), entity_score)
                        item["combined_score"] += entity_score * (1 - self.alpha)
                        break

        combined.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return combined[: self.vector_top_k + self.graph_top_k]


def hybrid_search(
    query: str,
    alpha: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function for hybrid search"""
    retriever = HybridRetriever(alpha=alpha)
    return retriever.search(query, filters)
