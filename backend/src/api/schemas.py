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
            if ".." in v or v.startswith("/"):
                raise ValueError("Invalid path")
            v = v.strip()
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


class MetricsResponse(BaseModel):
    requests_total: Dict[str, int]
    errors_total: Dict[str, int]
    request_duration_ms: Dict[str, List[float]]
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
