from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

from loguru import logger

from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from ..core.cache import cached, get_query_cache

if TYPE_CHECKING:
    from ..chains.medical_intent import MedicalIntent, IntentResult

INTENT_ALPHA_MAP: Dict[str, float] = {
    "disease_query": 0.6,
    "drug_query": 0.4,
    "drug_interaction": 0.3,
    "symptom_query": 0.7,
    "diagnosis_assist": 0.5,
    "treatment_query": 0.5,
    "examination_query": 0.7,
    "prevention_query": 0.5,
    "health_advice": 0.5,
    "medical_knowledge": 0.6,
    "unknown": 0.5,
}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"CrossEncoder model loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model: {e}")
                self._model = None

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not results:
            return results

        self._load_model()
        if self._model is None:
            return results

        pairs = [(query, r.get("content", "") or r.get("text", "") or "") for r in results]
        try:
            scores = self._model.predict(pairs)
            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.warning(f"CrossEncoder reranking failed: {e}")
            return results


class IntentAwareAlpha:
    @staticmethod
    def detect_intent_from_query(query: str) -> str:
        from ..chains.medical_intent import MedicalIntentClassifier
        classifier = MedicalIntentClassifier()
        try:
            result = classifier.classify(query)
            intent_str = result.intent.value if hasattr(result.intent, 'value') else str(result.intent)
            return intent_str
        except Exception:
            return "unknown"

    @staticmethod
    def get_alpha(intent: Optional[str]) -> float:
        if intent and intent in INTENT_ALPHA_MAP:
            return INTENT_ALPHA_MAP[intent]
        return 0.5


class HybridRetriever:
    def __init__(
        self,
        alpha: float = 0.5,
        vector_top_k: int = 5,
        graph_top_k: int = 10,
        enable_parallel: bool = True,
        enable_entity_embedding: bool = True,
        enable_cross_encoder: bool = True,
        enable_dynamic_alpha: bool = True,
        rerank_top_k: int = 10,
    ):
        self.alpha = alpha
        self.fixed_alpha = alpha
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.enable_parallel = enable_parallel
        self.enable_entity_embedding = enable_entity_embedding
        self.enable_cross_encoder = enable_cross_encoder
        self.enable_dynamic_alpha = enable_dynamic_alpha
        self.rerank_top_k = rerank_top_k
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._closed = False
        if self.enable_cross_encoder:
            self._reranker = CrossEncoderReranker()
        else:
            self._reranker = None

    def close(self, wait: bool = True):
        if not self._closed:
            self._executor.shutdown(wait=wait)
            self._closed = True
            logger.info("HybridRetriever executor closed")

    def __del__(self):
        self.close(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _resolve_intent(self, query: str) -> str:
        intent = IntentAwareAlpha.detect_intent_from_query(query)
        return intent

    def _resolve_alpha(self, query: str, intent: Optional[str]) -> float:
        if intent:
            return IntentAwareAlpha.get_alpha(intent)
        if self.enable_dynamic_alpha:
            detected = self._resolve_intent(query)
            return IntentAwareAlpha.get_alpha(detected)
        return self.fixed_alpha

    @cached(get_query_cache)
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        vector_results = []
        graph_results = []

        self.alpha = self._resolve_alpha(query, intent)

        if self.enable_parallel:
            vector_results, graph_results = self._parallel_search(query, filters)
        else:
            vector_results = self._vector_search(query, filters)
            graph_results = self._graph_search(query, filters)

        combined_results = self._combine_results(vector_results, graph_results)

        if self.enable_cross_encoder and self._reranker and combined_results:
            combined_results = self._reranker.rerank(query, combined_results, top_k=self.rerank_top_k)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_results": combined_results,
            "query": query,
            "alpha_used": self.alpha,
            "intent_used": intent or (self._resolve_intent(query) if self.enable_dynamic_alpha else None),
            "reranker_enabled": self.enable_cross_encoder,
        }

    def _parallel_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    intent: Optional[str] = None,
    enable_cross_encoder: bool = True,
    enable_dynamic_alpha: bool = True,
) -> Dict[str, Any]:
    retriever = HybridRetriever(
        alpha=alpha,
        enable_cross_encoder=enable_cross_encoder,
        enable_dynamic_alpha=enable_dynamic_alpha,
    )
    return retriever.search(query, filters, intent=intent)
