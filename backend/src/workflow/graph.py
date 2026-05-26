from typing import Union, List, Dict

from langgraph.graph import END, StateGraph
from loguru import logger

from .state import GraphState
from .nodes import (
    retrieve_drift,
    generate_answer,
    agenerate_answer,
    handle_error,
)
from ..core.circuit_breaker import get_circuit_breaker
from ..retrieval.drift_search import DRIFTSearch
from ..utils.process_monitor import track_process, get_structured_logger

_structured_logger = get_structured_logger("workflow")


def should_degrade(state: GraphState) -> bool:
    llm_qa_breaker = get_circuit_breaker("llm_qa")
    llm_cypher_breaker = get_circuit_breaker("llm_cypher")

    return llm_qa_breaker.is_open() or llm_cypher_breaker.is_open()


def create_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("drift_retrieval", retrieve_drift)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("error", handle_error)

    workflow.set_entry_point("drift_retrieval")

    workflow.add_edge("drift_retrieval", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("error", END)

    return workflow.compile()


def create_async_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("drift_retrieval", retrieve_drift)
    workflow.add_node("generate", agenerate_answer)
    workflow.add_node("error", handle_error)

    workflow.set_entry_point("drift_retrieval")

    workflow.add_edge("drift_retrieval", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("error", END)

    return workflow.compile()


_workflow = None
_async_workflow = None


def get_workflow() -> StateGraph:
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
        logger.info("LangGraph workflow compiled")
    return _workflow


def get_async_workflow() -> StateGraph:
    global _async_workflow
    if _async_workflow is None:
        _async_workflow = create_async_workflow()
        logger.info("Async LangGraph workflow compiled")
    return _async_workflow


@track_process("workflow.run")
def run_workflow(question: str, history: List[Dict[str, str]] | None = None) -> dict:
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

    _structured_logger.info(
        "workflow_started",
        question_length=len(question),
        has_history=bool(history),
    )

    try:
        if should_degrade(initial_state):
            logger.warning("Service in degraded mode, using simplified response")
            _structured_logger.warning(
                "workflow_degraded",
                reason="circuit_breaker_open",
            )
            return _degraded_response(question, initial_state)

        result = workflow.invoke(initial_state)
        
        _structured_logger.info(
            "workflow_completed",
            routing=result.get("routing", "unknown"),
            documents_count=len(result.get("documents", [])),
            answer_length=len(result.get("answer", "")),
        )
        
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        _structured_logger.error(
            "workflow_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        initial_state["error"] = str(e)
        return _degraded_response(question, initial_state)


@track_process("workflow.run_async")
async def run_workflow_async(question: str, history: List[Dict[str, str]] | None = None) -> dict:
    """Run the workflow with async LLM calls for non-blocking execution"""
    workflow = get_async_workflow()

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

    _structured_logger.info(
        "async_workflow_started",
        question_length=len(question),
        has_history=bool(history),
    )

    try:
        if should_degrade(initial_state):
            logger.warning("Service in degraded mode, using simplified response")
            _structured_logger.warning(
                "async_workflow_degraded",
                reason="circuit_breaker_open",
            )
            return _degraded_response(question, initial_state)

        result = await workflow.ainvoke(initial_state)
        
        _structured_logger.info(
            "async_workflow_completed",
            routing=result.get("routing", "unknown"),
            documents_count=len(result.get("documents", [])),
            answer_length=len(result.get("answer", "")),
        )
        
        return result
    except Exception as e:
        logger.error(f"Async workflow execution failed: {e}")
        _structured_logger.error(
            "async_workflow_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        initial_state["error"] = str(e)
        return _degraded_response(question, initial_state)


def _degraded_response(question: str, state: GraphState) -> dict:
    """生成降级响应，尝试从缓存获取相关信息"""
    fallback_answer = (
        "抱歉，当前服务处于高负载状态，暂时无法提供完整回答。\n"
        "请稍后重试，或者尝试简化您的问题。\n\n"
        "如果问题持续存在，请联系系统管理员。"
    )

    cached_info = _get_cached_info_for_degraded(question)
    if cached_info:
        entities_str = "\n".join([f"- {entity.get('name', '')} ({entity.get('type', '')})" 
                                 for entity in cached_info.get("entities", [])[:5]])
        documents_str = "\n".join([f"- {doc.get('content', '')[:100]}..." 
                                  for doc in cached_info.get("documents", [])[:3]])
        
        state["answer"] = f"当前服务负载较高，以下是与您问题相关的信息摘要：\n\n"
        if entities_str:
            state["answer"] += f"**相关实体：**\n{entities_str}\n\n"
        if documents_str:
            state["answer"] += f"**相关文档片段：**\n{documents_str}\n\n"
        state["answer"] += "如需更详细的回答，请稍后重试。"
    else:
        state["answer"] = fallback_answer
    
    state["routing"] = "degraded"
    state["documents"] = cached_info.get("documents", []) if cached_info else []
    state["entities"] = cached_info.get("entities", []) if cached_info else []
    return state


def _get_cached_info_for_degraded(question: str) -> Dict[str, List[Dict]]:
    """尝试从缓存获取降级响应所需的信息"""
    try:
        searcher = DRIFTSearch()
        intent = searcher._classify_query_intent(question)
        
        if intent == "local" or intent == "hybrid":
            result = searcher.local_search(question, top_k=5)
            entities = []
            for item in result.get("results", []):
                entities.append({
                    "name": item.get("entity"),
                    "type": item.get("type"),
                    "score": item.get("score")
                })
            return {"entities": entities, "documents": []}
        else:
            return {"entities": [], "documents": []}
    except Exception as e:
        logger.warning(f"Failed to get cached info for degraded response: {e}")
        return None
