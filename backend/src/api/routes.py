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
)
from ..core.neo4j_client import get_neo4j_client
from ..core.config import get_settings
from ..core.cache import get_graph_data_cache, get_search_cache
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.graph_retriever import GraphRetriever
from ..retrieval.hybrid import HybridRetriever
from ..workflow.graph import run_workflow
from ..ingestion.document_loader import load_documents_from_directory
from ..ingestion.kg_builder import KnowledgeGraphBuilder
from ..core.metrics import get_metrics
from ..utils.logger import get_request_id

router = APIRouter()

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

    neo4j_connected = False
    neo4j_pool = None
    try:
        client = get_neo4j_client()
        neo4j_connected = client.verify_connectivity()
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
                    "score": doc.get("similarity", 0),
                }
            )

        return QuestionResponse(
            question=result["question"],
            answer=result.get("answer", "No answer generated"),
            routing=result.get("routing", "unknown"),
            documents_count=len(result.get("documents", [])),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Query failed: {e}")
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


@router.post("/retrieval/hybrid")
async def hybrid_search(query: str, alpha: float = 0.5):
    try:
        retriever = HybridRetriever(alpha=alpha)
        results = retriever.search(query)
        return results
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
):
    """搜索节点"""
    cache_key = f"{query}:{node_label or 'all'}:{limit}"
    cache = get_search_cache()
    
    cached_result = cache.cache.get(cache_key)
    if cached_result:
        logger.debug(f"Cache hit for search: {cache_key}")
        return GraphSearchResponse(**cached_result)
    
    try:
        client = get_neo4j_client()
        results = client.search_nodes(
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
