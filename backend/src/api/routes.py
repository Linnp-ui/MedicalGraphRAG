import os
import time
import threading
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from loguru import logger
import tempfile

from .schemas import (
    QuestionRequest,
    QuestionResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    SchemaResponse,
    MetricsResponse,
    GraphDataResponse,
    GraphSearchResponse,
    NodeDetailResponse,
    QueryResultRequest,
    ChunkContextResponse,
    DocumentReconstructionResponse,
    ChunkParentResponse,
    EntityChunksResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    IngestionStatsResponse,
    UpdateDocumentRequest,
    BatchUpdateRequest,
    UpdateResultResponse,
    DocumentVersionResponse,
)
from ..core.neo4j_client import get_neo4j_client
from ..core.config import get_settings
from ..core.cache import get_graph_data_cache, get_search_cache
from ..core.hierarchical_communities import get_hierarchical_manager
from ..core.llm_cache import get_llm_cache
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.graph_retriever import GraphRetriever
from ..retrieval.drift_search import DRIFTSearch, drift_search, explain_drift_strategy
from ..core.community_detector import CommunityDetector, get_community_detector
from ..core.summary_generator import SummaryGenerator, get_summary_generator
from ..workflow.graph import run_workflow
from ..ingestion.document_loader import load_documents_from_directory
from ..ingestion.kg_builder import KnowledgeGraphBuilder
from ..core.metrics import get_metrics, get_metrics_middleware
from ..utils.logger import get_request_id

router = APIRouter()
_metrics_middleware = get_metrics_middleware()

_start_time = time.time()


def _get_uptime() -> float:
    return time.time() - _start_time


def _get_memory_mb() -> float:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def _parse_node_id(node_id: str) -> int:
    if not node_id.isdigit():
        raise HTTPException(status_code=400, detail="node_id must be a numeric string")
    return int(node_id)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    logger.info(f"Health check called, NEO4J_URI from settings: {settings.neo4j_uri}")

    neo4j_connected = False
    neo4j_pool = None
    try:
        client = get_neo4j_client()
        logger.info(f"Got Neo4j client: {client}")
        logger.info(f"Client settings URI: {client.settings.neo4j_uri}")
        neo4j_connected = client.verify_connectivity()
        logger.info(f"Connectivity verified: {neo4j_connected}")
        if neo4j_connected:
            result = client.execute_query("RETURN 1 as n")
            neo4j_pool = {"connected": 1}
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")

    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        neo4j_connected=neo4j_connected,
        version="1.0.0",
        uptime_seconds=_get_uptime(),
        memory_usage_mb=_get_memory_mb(),
        neo4j_pool=neo4j_pool,
    )


@router.get("/health/live")
async def liveness_check():
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check():
    try:
        client = get_neo4j_client()
        client.verify_connectivity()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")
    return {"status": "ready"}


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics_endpoint():
    metrics = get_metrics()
    all_metrics = metrics.get_metrics()

    return MetricsResponse(
        requests_total=all_metrics.get("requests_total", {}),
        errors_total=all_metrics.get("errors_total", {}),
        request_duration_ms=all_metrics.get("requests_duration", {}),
        neo4j_pool=all_metrics.get("neo4j_pool", {}),
    )


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = get_metrics()
    prometheus_text = metrics.export_prometheus()
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=prometheus_text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/schema", response_model=SchemaResponse)
async def get_schema():
    neo4j_client = get_neo4j_client()

    try:
        schema = neo4j_client.get_schema()

        labels_result = neo4j_client.execute_query(
            "CALL db.labels() YIELD label RETURN collect(label) as labels"
        )
        node_labels = labels_result[0].get("labels", []) if labels_result else []

        rels_result = neo4j_client.execute_query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        )
        relationship_types = rels_result[0].get("types", []) if rels_result else []

        return SchemaResponse(
            schema_text=schema,
            node_labels=node_labels,
            relationship_types=relationship_types,
        )
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=QuestionResponse)
async def search(request: QuestionRequest):
    request_id = get_request_id() or "unknown"
    logger.info(f"[{request_id}] Received search query: {request.question}")

    try:
        result = run_workflow(request.question, history=request.history)

        sources = []
        for doc in result.get("documents", [])[:3]:
            sources.append(
                {
                    "content": doc.get("content", "")[:200] + "...",
                    "score": doc.get("similarity", doc.get("score", 0)),
                }
            )

        return QuestionResponse(
            question=result["question"],
            answer=result.get("answer", "No answer generated"),
            routing=result.get("routing", "drift"),
            documents_count=len(result.get("documents", [])),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QuestionResponse)
async def query(request: QuestionRequest):
    request_id = get_request_id() or "unknown"
    logger.info(f"[{request_id}] Received query: {request.question}")

    try:
        result = run_workflow(request.question, history=request.history)

        sources = []
        for doc in result.get("documents", [])[:3]:
            sources.append(
                {
                    "content": doc.get("content", "")[:200] + "...",
                    "score": doc.get("similarity", doc.get("score", 0)),
                }
            )

        return QuestionResponse(
            question=result["question"],
            answer=result.get("answer", "No answer generated"),
            routing=result.get("routing", "drift"),
            documents_count=len(result.get("documents", [])),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/async", response_model=QuestionResponse)
async def search_async(request: QuestionRequest):
    """Async search endpoint using non-blocking LLM calls"""
    request_id = get_request_id() or "unknown"
    logger.info(f"[{request_id}] Received async search query: {request.question}")

    try:
        from ..workflow.graph import run_workflow_async

        result = await run_workflow_async(request.question, history=request.history)

        sources = []
        for doc in result.get("documents", [])[:3]:
            sources.append(
                {
                    "content": doc.get("content", "")[:200] + "...",
                    "score": doc.get("similarity", doc.get("score", 0)),
                }
            )

        return QuestionResponse(
            question=result["question"],
            answer=result.get("answer", "No answer generated"),
            routing=result.get("routing", "drift"),
            documents_count=len(result.get("documents", [])),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Async search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/async", response_model=QuestionResponse)
async def query_async(request: QuestionRequest):
    """Async query endpoint using non-blocking LLM calls"""
    request_id = get_request_id() or "unknown"
    logger.info(f"[{request_id}] Received async query: {request.question}")

    try:
        from ..workflow.graph import run_workflow_async

        result = await run_workflow_async(request.question, history=request.history)

        sources = []
        for doc in result.get("documents", [])[:3]:
            sources.append(
                {
                    "content": doc.get("content", "")[:200] + "...",
                    "score": doc.get("similarity", doc.get("score", 0)),
                }
            )

        return QuestionResponse(
            question=result["question"],
            answer=result.get("answer", "No answer generated"),
            routing=result.get("routing", "drift"),
            documents_count=len(result.get("documents", [])),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Async query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    logger.info("Received ingest request")

    try:
        builder = KnowledgeGraphBuilder()
        results = []

        if request.file_path:
            from ..ingestion.document_loader import load_document

            doc = load_document(request.file_path)
            result = builder.ingest_document(
                doc,
                extract_entities=request.extract_entities,
                create_embeddings=request.create_embeddings,
            )
            results.append(result)
        elif request.directory:
            docs = load_documents_from_directory(request.directory)
            for doc in docs:
                result = builder.ingest_document(
                    doc,
                    extract_entities=request.extract_entities,
                    create_embeddings=request.create_embeddings,
                )
                results.append(result)
        else:
            raise HTTPException(
                status_code=400, detail="Either file_path or directory must be provided"
            )

        with _kg_builder_stats_lock:
            for key, value in builder.get_stats().items():
                if key in _kg_builder_stats:
                    _kg_builder_stats[key] += value

        return IngestResponse(
            status="success",
            documents_processed=len(results),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            get_neo4j_client().invalidate_schema_cache()
        except Exception:
            pass


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    extract_entities: bool = Form(True),
    create_embeddings: bool = Form(True),
):
    logger.info(f"Received file upload: {file.filename}")

    suffix = os.path.splitext(file.filename or "")[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from ..ingestion.document_loader import load_document

        doc = load_document(tmp_path)
        builder = KnowledgeGraphBuilder()
        result = builder.ingest_document(
            doc,
            extract_entities=extract_entities,
            create_embeddings=create_embeddings,
        )

        with _kg_builder_stats_lock:
            for key, value in builder.get_stats().items():
                if key in _kg_builder_stats:
                    _kg_builder_stats[key] += value

        return IngestResponse(
            status="success",
            documents_processed=1,
            results=[result],
        )
    except Exception as e:
        logger.error(f"Upload and ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        try:
            get_neo4j_client().invalidate_schema_cache()
        except Exception:
            pass


@router.post("/retrieval/vector")
async def vector_search(query: str, top_k: int = 5):
    try:
        retriever = VectorRetriever()
        results = retriever.search(query, top_k=top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/graph")
async def graph_search(query: str):
    try:
        client = get_neo4j_client()
        schema = client.get_schema()

        from ..chains.cypher_gen import CypherGenerator

        generator = CypherGenerator()
        cypher = generator.generate(query, schema)

        results = client.execute_query(cypher) if cypher else []

        return {"cypher": cypher, "results": results}
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    try:
        searcher = DRIFTSearch()
        results = searcher.hybrid_search(request.query, alpha=request.alpha)
        return HybridSearchResponse(**results)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/data", response_model=GraphDataResponse)
async def get_graph_data(
    node_label: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """获取图谱数据"""
    cache_key = f"{node_label or 'all'}:{limit}:{offset}"
    cache = get_graph_data_cache()
    
    cached_result = cache.cache.get(cache_key)
    if cached_result:
        logger.debug(f"Cache hit for graph data: {cache_key}")
        return GraphDataResponse(**cached_result)
    
    try:
        client = get_neo4j_client()
        data = client.get_graph_data(
            node_label=node_label,
            limit=limit,
            offset=offset,
        )
        cache.cache.set(cache_key, data, ttl=300)
        return GraphDataResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/search", response_model=GraphSearchResponse)
async def search_nodes(
    query: str = Query(..., min_length=0, max_length=2000),
    node_label: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=20, ge=1, le=100),
    fuzzy: bool = Query(default=False, description="Enable fuzzy search"),
    fuzzy_mode: str = Query(default="contains", description="Fuzzy mode: contains, prefix, suffix, regex"),
):
    """搜索节点

    Args:
        query: 搜索文本
        node_label: 节点标签过滤
        limit: 返回数量限制
        fuzzy: 是否启用模糊搜索
        fuzzy_mode: 模糊搜索模式
            - contains: 包含匹配（默认）
            - prefix: 前缀匹配
            - suffix: 后缀匹配
            - regex: 正则表达式匹配
    """
    cache_key = f"{query}:{node_label or 'all'}:{limit}:{fuzzy}:{fuzzy_mode}"
    cache = get_search_cache()
    
    cached_result = cache.cache.get(cache_key)
    if cached_result:
        logger.debug(f"Cache hit for search: {cache_key}")
        return GraphSearchResponse(**cached_result)
    
    try:
        client = get_neo4j_client()
        
        if fuzzy:
            results = client.fuzzy_search_nodes(
                search_text=query,
                node_label=node_label,
                limit=limit,
                fuzzy_mode=fuzzy_mode,
            )
        else:
            results = client.search_nodes_with_score(
                search_text=query,
                node_label=node_label,
                limit=limit,
            )
        
        response_data = {
            "results": results,
            "total": len(results),
        }
        cache.cache.set(cache_key, response_data, ttl=600)
        return GraphSearchResponse(**response_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to search nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/node/{node_id}", response_model=NodeDetailResponse)
async def get_node_detail(node_id: str):
    """获取节点详情"""
    try:
        _parse_node_id(node_id)
        client = get_neo4j_client()
        data = client.get_node_detail(node_id)
        if not data:
            raise HTTPException(status_code=404, detail="Node not found")
        return NodeDetailResponse(**data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/node/{node_id}/neighbors")
async def get_node_neighbors(
    node_id: str,
    depth: int = Query(default=1, ge=1, le=3),
    relationship_type: str | None = Query(default=None, max_length=128),
):
    """获取节点邻居"""
    try:
        _parse_node_id(node_id)
        client = get_neo4j_client()
        data = client.get_node_neighbors(
            node_id=node_id,
            depth=depth,
            relationship_type=relationship_type,
        )
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get node neighbors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/query-result")
async def get_query_result_graph(request: QueryResultRequest):
    """获取查询结果图谱"""
    try:
        client = get_neo4j_client()
        
        all_nodes = {}
        all_edges = {}
        
        for node_id in request.node_ids:
            neighbor_data = client.get_node_neighbors(
                node_id=node_id,
                depth=request.max_depth
            )
            
            for node in neighbor_data.get('nodes', []):
                all_nodes[node['id']] = node
            
            for edge in neighbor_data.get('edges', []):
                all_edges[edge['id']] = edge
        
        stats_query = """
        MATCH (n)
        WITH count(n) as total_nodes
        MATCH ()-[r]->()
        WITH total_nodes, count(r) as total_edges
        CALL db.labels() YIELD label
        WITH total_nodes, total_edges, collect(label) as node_labels
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN total_nodes, total_edges, node_labels, collect(relationshipType) as relationship_types
        """
        
        stats_result = client.execute_query(stats_query)
        stats_data = stats_result[0] if stats_result else {}
        
        return {
            'nodes': list(all_nodes.values()),
            'edges': list(all_edges.values()),
            'stats': {
                'total_nodes': stats_data.get('total_nodes', 0),
                'total_edges': stats_data.get('total_edges', 0),
                'node_labels': stats_data.get('node_labels', []),
                'relationship_types': stats_data.get('relationship_types', [])
            }
        }
    except Exception as e:
        logger.error(f"Failed to get query result graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk/{chunk_id}/parent", response_model=ChunkParentResponse)
async def get_chunk_parent(chunk_id: str):
    """获取 chunk 所属的父文档"""
    try:
        retriever = GraphRetriever()
        result = retriever.get_chunk_parent(chunk_id)
        if not result:
            raise HTTPException(status_code=404, detail="Chunk or parent document not found")
        return ChunkParentResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk parent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk/{chunk_id}/context", response_model=ChunkContextResponse)
async def get_chunk_context(
    chunk_id: str,
    context_chunks: int = Query(default=2, ge=0, le=10),
):
    """获取 chunk 及其上下文"""
    try:
        retriever = GraphRetriever()
        result = retriever.get_chunk_context(chunk_id, context_chunks=context_chunks)
        if not result:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return ChunkContextResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{document_id}/reconstruct", response_model=DocumentReconstructionResponse)
async def reconstruct_document(document_id: str):
    """从 chunks 重构完整文档"""
    try:
        retriever = GraphRetriever()
        result = retriever.reconstruct_document(document_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentReconstructionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reconstruct document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_name}/chunks", response_model=EntityChunksResponse)
async def find_chunks_by_entity(
    entity_name: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """查找包含指定实体的所有 chunks"""
    try:
        retriever = GraphRetriever()
        chunks = retriever.find_chunks_by_entity(entity_name, limit=limit)
        return EntityChunksResponse(
            entity_name=entity_name,
            chunks=chunks,
        )
    except Exception as e:
        logger.error(f"Failed to find chunks by entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_name}/multi-hop")
async def multi_hop_search(
    entity_name: str,
    hop_count: int = Query(default=2, ge=1, le=3, description="跳跃次数"),
    relation_types: str = Query(default=None, description="逗号分隔的关系类型，如 TREATED_BY,HAS_SYMPTOM"),
    entity_types: str = Query(default=None, description="逗号分隔的实体类型，如 Disease,Drug"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """多跳搜索：从起始实体出发，通过多层关系找到相关实体

    例如：查询"高血压"的2跳邻居
    - 1跳：高血压 -> 硝苯地平 (TREATED_BY)
    - 2跳：硝苯地平 -> 头痛 (SIDE_EFFECT)
    """
    try:
        retriever = GraphRetriever()

        rel_types = None
        if relation_types:
            rel_types = [r.strip() for r in relation_types.split(",")]

        ent_types = None
        if entity_types:
            ent_types = [t.strip() for t in entity_types.split(",")]

        results = retriever.multi_hop_search(
            start_entity=entity_name,
            hop_count=hop_count,
            relation_types=rel_types,
            entity_types=ent_types,
            limit=limit,
        )
        return results
    except Exception as e:
        logger.error(f"Multi-hop search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_name}/related")
async def find_related_entities(
    entity_name: str,
    depth: int = Query(default=1, ge=1, le=3, description="搜索深度"),
    relation_filter: str = Query(default=None, description="逗号分隔的关系类型，如 TREATED_BY,HAS_SYMPTOM"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """查找与给定实体相关的所有实体（按关系类型分组）

    例如：查找"高血压"相关的所有实体
    - TREATED_BY: [硝苯地平, 氨氯地平]
    - HAS_SYMPTOM: [头痛, 头晕]
    """
    try:
        retriever = GraphRetriever()

        rel_filter = None
        if relation_filter:
            rel_filter = [r.strip() for r in relation_filter.split(",")]

        results = retriever.find_related_entities(
            entity_name=entity_name,
            depth=depth,
            relation_filter=rel_filter,
            limit=limit,
        )
        return results
    except Exception as e:
        logger.error(f"Find related entities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """获取文档的所有 chunks"""
    try:
        retriever = GraphRetriever()
        chunks = retriever.get_document_chunks(document_id, limit=limit)
        return {
            "document_id": document_id,
            "chunks": chunks,
        }
    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


_kg_builder_stats = {
    "documents_skipped": 0,
    "chunks_skipped": 0,
    "entities_skipped": 0,
    "embeddings_cached_hits": 0,
}
_kg_builder_stats_lock = threading.Lock()


@router.get("/ingestion/stats", response_model=IngestionStatsResponse)
async def get_ingestion_stats():
    """获取摄入统计信息，包括缓存命中情况"""
    with _kg_builder_stats_lock:
        return IngestionStatsResponse(**_kg_builder_stats)


@router.post("/ingestion/stats/reset")
async def reset_ingestion_stats():
    """重置摄入统计计数器"""
    global _kg_builder_stats
    with _kg_builder_stats_lock:
        _kg_builder_stats = {
            "documents_skipped": 0,
            "chunks_skipped": 0,
            "entities_skipped": 0,
            "embeddings_cached_hits": 0,
        }
    return {"status": "reset", "message": "Ingestion stats reset successfully"}


@router.post("/retrieval/drift")
async def drift_search_endpoint(
    query: str,
    strategy: str = Query(default=None, description="检索策略: global, local, hybrid"),
):
    """DRIFT搜索 - 根据查询意图自动选择检索策略"""
    try:
        results = drift_search(query, strategy=strategy)
        return results
    except Exception as e:
        logger.error(f"DRIFT search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/drift/explain")
async def explain_drift_endpoint(query: str):
    """解释DRIFT搜索策略选择"""
    try:
        explanation = explain_drift_strategy(query)
        return explanation
    except Exception as e:
        logger.error(f"DRIFT explain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/detect")
async def detect_communities():
    """检测知识图谱中的社区结构"""
    try:
        detector = get_community_detector()
        communities = detector.get_communities()
        top_communities = detector.get_top_communities(top_n=5)
        
        return {
            "communities": communities,
            "community_count": len(communities),
            "top_communities": top_communities,
        }
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/{community_id}/members")
async def get_community_members(community_id: int):
    """获取指定社区的成员列表"""
    try:
        detector = get_community_detector()
        members = detector.get_community_members(community_id)
        centrality = detector.compute_community_centrality(community_id)
        
        return {
            "community_id": community_id,
            "members": members,
            "member_count": len(members),
            "centrality": centrality,
        }
    except Exception as e:
        logger.error(f"Get community members failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_name}/community")
async def get_entity_community(entity_name: str):
    """获取实体所属的社区"""
    try:
        detector = get_community_detector()
        community_id = detector.get_entity_community(entity_name)
        
        if community_id is None:
            return {
                "entity_name": entity_name,
                "community_id": None,
                "message": "Entity not found in any community",
            }
        
        members = detector.get_community_members(community_id)
        
        return {
            "entity_name": entity_name,
            "community_id": community_id,
            "community_member_count": len(members),
            "community_members": members[:10],
        }
    except Exception as e:
        logger.error(f"Get entity community failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/entity/{entity_name}")
async def get_entity_summary(entity_name: str):
    """获取实体摘要"""
    try:
        generator = get_summary_generator()
        summary = generator.generate_entity_summary(entity_name)
        return {"entity_name": entity_name, "summary": summary}
    except Exception as e:
        logger.error(f"Get entity summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/community/{community_id}")
async def get_community_summary(community_id: int, level: int = Query(default=0)):
    """获取社区摘要"""
    try:
        generator = get_summary_generator()
        summary = generator.generate_community_summary(community_id, level)
        return {"community_id": community_id, "level": level, "summary": summary}
    except Exception as e:
        logger.error(f"Get community summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/global")
async def get_global_summary():
    """获取全局摘要"""
    try:
        generator = get_summary_generator()
        summary = generator.generate_global_summary()
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Get global summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary/query")
async def summarize_query_context(query: str):
    """根据查询生成相关摘要上下文"""
    try:
        generator = get_summary_generator()
        context = generator.summarize_query_context(query)
        return context
    except Exception as e:
        logger.error(f"Summarize query context failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 增量更新 API ────────────────────────────────────────────────────────────────


@router.put("/documents/{document_id}", response_model=UpdateResultResponse)
async def update_document(
    document_id: str,
    request: UpdateDocumentRequest
):
    """增量更新文档
    
    Args:
        document_id: 要更新的文档ID
        request: 更新请求体，包含新内容和更新策略
    
    Returns:
        更新结果，包含更新的实体数、关系数和向量数
    """
    logger.info(f"收到增量更新请求: {document_id}")
    
    try:
        from ..ingestion.incremental_updater import create_incremental_updater
        
        updater = create_incremental_updater(strategy=request.strategy)
        result = updater.update_document(document_id, request.content)
        
        return UpdateResultResponse(
            document_id=result.document_id,
            success=result.success,
            message=result.message,
            updated_entities=result.updated_entities,
            updated_relations=result.updated_relations,
            updated_vectors=result.updated_vectors
        )
    except Exception as e:
        logger.error(f"增量更新失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/documents/batch")
async def batch_update_documents(request: BatchUpdateRequest):
    """批量增量更新文档"""
    logger.info(f"收到批量更新请求，共 {len(request.documents)} 个文档")
    
    try:
        from ..ingestion.incremental_updater import create_incremental_updater
        
        updater = create_incremental_updater(strategy=request.strategy)
        results = updater.batch_update(request.documents)
        
        return {
            "results": [
                {
                    "document_id": r.document_id,
                    "success": r.success,
                    "message": r.message,
                    "updated_entities": r.updated_entities,
                    "updated_relations": r.updated_relations,
                    "updated_vectors": r.updated_vectors
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"批量更新失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", response_model=UpdateResultResponse)
async def delete_document(document_id: str):
    """删除文档及其相关数据"""
    logger.info(f"收到删除请求: {document_id}")
    
    try:
        from ..ingestion.incremental_updater import create_incremental_updater
        
        updater = create_incremental_updater()
        result = updater.delete_document(document_id)
        
        return UpdateResultResponse(
            document_id=result.document_id,
            success=result.success,
            message=result.message
        )
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/version", response_model=DocumentVersionResponse)
async def get_document_version(document_id: str):
    """获取文档版本信息"""
    try:
        from ..ingestion.incremental_updater import create_incremental_updater
        
        updater = create_incremental_updater()
        version_info = updater.get_document_version(document_id)
        
        if version_info:
            return DocumentVersionResponse(**version_info)
        else:
            return DocumentVersionResponse(document_id=document_id)
    except Exception as e:
        logger.error(f"获取版本信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 分层社区管理 API ────────────────────────────────────────────────────────────


@router.get("/community/stats")
async def get_community_stats():
    """获取分层社区统计信息"""
    manager = get_hierarchical_manager()
    return manager.get_stats()


@router.get("/community/level/{level}")
async def get_communities_at_level(level: int):
    """获取指定层级的所有社区"""
    manager = get_hierarchical_manager()
    try:
        communities = manager.get_communities_by_level(level)
        return {
            "level": level,
            "communities": {
                k: {"member_count": len(v), "members": v[:10]}
                for k, v in communities.items()
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/community/{level}/{community_id}")
async def get_community_detail(level: int, community_id: int):
    """获取社区详情"""
    manager = get_hierarchical_manager()
    try:
        members = manager.get_community_members(level, community_id)
        summary = manager.get_community_summary(level, community_id)
        
        return {
            "level": level,
            "community_id": community_id,
            "member_count": len(members),
            "members": members,
            "summary": summary
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── LLM 缓存管理 API ──────────────────────────────────────────────────────────────


@router.get("/cache/stats")
async def get_cache_stats():
    """获取 LLM 缓存统计"""
    cache = get_llm_cache()
    stats = cache.get_stats()
    stats["hit_rate"] = cache.get_hit_rate()
    return stats


@router.post("/cache/clear")
async def clear_cache():
    """清空 LLM 缓存"""
    cache = get_llm_cache()
    cache.clear()
    return {"status": "cleared"}
