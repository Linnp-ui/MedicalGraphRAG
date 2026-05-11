from typing import Union, List, Dict

from langgraph.graph import END, StateGraph
from loguru import logger

from .state import GraphState
from .router import route_question, decompose_question
from .nodes import (
    retrieve_vector,
    retrieve_graph,
    retrieve_hybrid,
    generate_answer,
    handle_error,
)
from ..core.circuit_breaker import get_circuit_breaker


def should_degrade(state: GraphState) -> bool:
    """Check if service should degrade based on circuit breaker states"""
    llm_qa_breaker = get_circuit_breaker("llm_qa")
    llm_cypher_breaker = get_circuit_breaker("llm_cypher")
    
    return llm_qa_breaker.state.name == "OPEN" or llm_cypher_breaker.state.name == "OPEN"


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("decompose", decompose_question)
    workflow.add_node("vector_retrieval", retrieve_vector)
    workflow.add_node("graph_retrieval", retrieve_graph)
    workflow.add_node("hybrid_retrieval", retrieve_hybrid)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("error", handle_error)

    # Set entry point
    workflow.set_entry_point("decompose")

    # Add conditional routing from decompose
    workflow.add_conditional_edges(
        "decompose",
        route_question,
        {
            "vector": "vector_retrieval",
            "graph": "graph_retrieval",
            "hybrid": "hybrid_retrieval",
        },
    )

    # Vector retrieval -> generate
    workflow.add_edge("vector_retrieval", "generate")

    # Graph retrieval -> generate
    workflow.add_edge("graph_retrieval", "generate")

    # Hybrid -> generate
    workflow.add_edge("hybrid_retrieval", "generate")

    # Generate -> END
    workflow.add_edge("generate", END)

    # Error -> END
    workflow.add_edge("error", END)

    return workflow.compile()


# Singleton workflow
_workflow = None


def get_workflow() -> StateGraph:
    """Get the compiled workflow"""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
        logger.info("LangGraph workflow compiled")
    return _workflow


def run_workflow(question: str, history: List[Dict[str, str]] | None = None) -> dict:
    """Run the workflow with a question"""
    workflow = get_workflow()

    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "entities": [],
        "cypher_query": "",
        "graph_result": [],
        "answer": "",
        "routing": "",
        "subqueries": [],
        "context": "",
        "history": history or [],
        "error": None,
    }

    try:
        if should_degrade(initial_state):
            logger.warning("Service in degraded mode, using simplified response")
            return _degraded_response(question, initial_state)
        
        result = workflow.invoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        initial_state["error"] = str(e)
        return _degraded_response(question, initial_state)


def _degraded_response(question: str, state: GraphState) -> dict:
    """Generate a degraded response when service is unavailable"""
    state["answer"] = (
        "抱歉，当前服务处于高负载状态，暂时无法提供完整回答。\n"
        "请稍后重试，或者尝试简化您的问题。\n\n"
        "如果问题持续存在，请联系系统管理员。"
    )
    state["routing"] = "degraded"
    return state
