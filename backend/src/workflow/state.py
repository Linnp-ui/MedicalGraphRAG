from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph workflow.

    Attributes:
        question: The original user question
        documents: Retrieved documents/chunks
        entities: Extracted entities
        cypher_query: Generated Cypher query
        graph_result: Result from graph query
        answer: Final answer to the user
        routing: Decision about which retrieval path was used (global/local/hybrid)
        subqueries: Decomposed sub-queries
        context: Additional context for query generation
        error: Error message if any
    """

    question: str
    documents: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    cypher_query: str
    graph_result: List[Dict[str, Any]]
    answer: str
    routing: str
    subqueries: List[str]
    context: str
    history: List[Dict[str, str]]
    error: str | None
