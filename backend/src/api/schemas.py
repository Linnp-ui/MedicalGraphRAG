import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
import re


class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=2000)
    use_hybrid: bool = Field(default=True, description="Use hybrid retrieval")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history",
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = v.strip()
        v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
        return v


class QuestionResponse(BaseModel):
    question: str
    answer: str
    routing: str
    documents_count: int = 0
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to file to ingest")
    directory: Optional[str] = Field(None, description="Path to directory to ingest")
    extract_entities: bool = Field(default=True, description="Extract entities")
    create_embeddings: bool = Field(default=True, description="Create embeddings")

    @field_validator("file_path", "directory")
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            normalized = os.path.normpath(v)
            if normalized.startswith("..") or normalized.startswith("/") or ".." in normalized.split(os.sep):
                raise ValueError("Invalid path")
        return v


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    neo4j_pool: Optional[Dict[str, int]] = None


class SchemaResponse(BaseModel):
    schema_text: str = Field(..., serialization_alias="schema")
    node_labels: List[str]
    relationship_types: List[str]


class DurationStats(BaseModel):
    count: int
    sum: float
    avg: float


class MetricsResponse(BaseModel):
    requests_total: Dict[str, int]
    errors_total: Dict[str, int]
    request_duration_ms: Dict[str, DurationStats]
    neo4j_pool: Dict[str, int]


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]
    degree: Optional[int] = None


class GraphEdge(BaseModel):
    id: str
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    type: str
    properties: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    node_labels: List[str]
    relationship_types: List[str]


class GraphDataResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats


class GraphSearchResult(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]
    score: Optional[float] = None


class GraphSearchResponse(BaseModel):
    results: List[GraphSearchResult]
    total: int


class NodeRelationships(BaseModel):
    incoming: List[Dict[str, Any]]
    outgoing: List[Dict[str, Any]]


class NodeDetailResponse(BaseModel):
    node: GraphNode
    relationships: NodeRelationships
    neighbors: List[GraphNode]


class QueryResultRequest(BaseModel):
    query: str
    node_ids: List[str]
    max_depth: int = Field(default=2, ge=1, le=3)

    @field_validator("node_ids")
    @classmethod
    def validate_node_ids(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("node_ids cannot be empty")
        if any(not node_id.isdigit() for node_id in value):
            raise ValueError("node_ids must contain only numeric strings")
        return value


class ChunkContextItem(BaseModel):
    chunk_id: str
    content: str
    index: int
    is_target: bool


class ChunkContextResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    context: List[ChunkContextItem]


class DocumentReconstructionResponse(BaseModel):
    document_id: str
    title: str
    source: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    full_content: str
    chunk_count: int


class ChunkParentResponse(BaseModel):
    document_id: str
    title: str
    source: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class EntityChunkItem(BaseModel):
    chunk_id: str
    content: str
    index: int
    document_id: str
    document_title: str


class EntityChunksResponse(BaseModel):
    entity_name: str
    chunks: List[EntityChunkItem]


class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=2000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector search (0=graph only, 1=vector only)")


class HybridSearchResponse(BaseModel):
    query: str
    alpha_used: float
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    combined_results: List[Dict[str, Any]]


class IngestionStatsResponse(BaseModel):
    documents_skipped: int = Field(description="Number of duplicate documents skipped")
    chunks_skipped: int = Field(description="Number of chunks with cached embeddings skipped")
    entities_skipped: int = Field(description="Number of entities with cached embeddings skipped")
    embeddings_cached_hits: int = Field(description="Total number of embedding cache hits")


class UpdateDocumentRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to update")
    content: str = Field(..., description="New document content")
    strategy: str = Field(default="incremental", description="Update strategy: incremental, full_rebuild, lazy_update")


class BatchUpdateRequest(BaseModel):
    documents: List[Dict[str, str]] = Field(..., description="List of documents to update")
    strategy: str = Field(default="incremental", description="Update strategy")


class UpdateResultResponse(BaseModel):
    document_id: str
    success: bool
    message: str = ""
    updated_entities: int = 0
    updated_relations: int = 0
    updated_vectors: int = 0


class DocumentVersionResponse(BaseModel):
    document_id: str
    version: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: Optional[str] = None


class ErrorLogReportRequest(BaseModel):
    error_type: str = Field(..., description="Error type (e.g., TypeError, NetworkError)")
    message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    severity: str = Field(default="error", description="Severity: debug, info, warning, error, critical")
    url: Optional[str] = Field(None, description="URL where error occurred")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    extra: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ErrorLogResponse(BaseModel):
    error_id: str
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    source: str
    severity: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float
    count: int
    first_seen: float
    last_seen: float


class ErrorLogListResponse(BaseModel):
    errors: List[ErrorLogResponse]
    total: int


class ErrorStatsResponse(BaseModel):
    total_errors: int
    unique_errors: int
    errors_by_type: Dict[str, int]
    errors_by_source: Dict[str, int]
    errors_by_severity: Dict[str, int]


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=2000)
    answer: str = Field(..., description="System answer")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    corrected_answer: Optional[str] = Field(None, description="User corrected answer")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="A/B test metadata")


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str


class ABTestConfig(BaseModel):
    variant: str = Field(..., description="A or B")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_cross_encoder: bool = Field(default=True)
    enable_dynamic_alpha: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=False)


class ABTestStats(BaseModel):
    total_feedback: int
    avg_rating_a: float
    avg_rating_b: float
    count_a: int
    count_b: int
    improvement_pct: Optional[float] = None
