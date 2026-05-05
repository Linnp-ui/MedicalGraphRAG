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
        result = workflow.invoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        initial_state["error"] = str(e)
        return initial_state
