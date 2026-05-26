from typing import Any, Dict, List

from loguru import logger

from ..retrieval.drift_search import DRIFTSearch
from ..chains.qa_chain import QAChain, AsyncQAChain
from .state import GraphState


def retrieve_drift(state: GraphState) -> GraphState:
    logger.info("=== DRIFT Retrieval ===")
    question = state["question"]

    drift_searcher = DRIFTSearch()
    results = drift_searcher.search(question)

    search_type = results.get("search_type", "unknown")
    state["routing"] = search_type

    if search_type == "global":
        global_summary = results.get("global_summary", "")
        community_summaries = results.get("community_summaries", {})
        
        context_parts = []
        if global_summary:
            context_parts.append(f"全局摘要: {global_summary}")
        for comm_id, summary in community_summaries.items():
            context_parts.append(f"社区{comm_id}摘要: {summary}")
        
        state["context"] = "\n\n".join(context_parts)
        state["documents"] = []
        state["graph_result"] = []
        
    elif search_type == "local":
        local_results = results.get("results", [])
        
        documents = []
        graph_result = []
        
        for item in local_results:
            entity_name = item.get("entity")
            entity_type = item.get("type")
            summary = item.get("summary", "")
            relationships = item.get("relationships", [])
            
            if summary:
                documents.append({
                    "content": f"实体【{entity_name}】({entity_type}): {summary}",
                    "entity": entity_name,
                    "score": item.get("score", 1.0)
                })
            
            for rel in relationships:
                graph_result.append({
                    "source": entity_name,
                    "relationship": rel.get("relationship", ""),
                    "target": rel.get("target", "")
                })
        
        state["documents"] = documents
        state["graph_result"] = graph_result
        
    elif search_type == "hybrid":
        state["documents"] = results.get("combined_results", [])
        state["graph_result"] = results.get("graph_results", [])

    logger.info(f"DRIFT search ({search_type}) returned {len(state['documents'])} document results")

    return state


async def agenerate_answer(state: GraphState) -> GraphState:
    """Async version of generate_answer using non-blocking LLM calls"""
    logger.info("=== Generate Answer (Async) ===")
    question = state["question"]

    graph_context = _format_graph_context(
        state.get("graph_result", []),
        state.get("cypher_query", ""),
    )

    doc_context = _format_document_context(state.get("documents", []))

    context_parts = []
    if graph_context:
        context_parts.append(f"[图谱数据]\n{graph_context}")
    if doc_context:
        context_parts.append(f"[文档内容]\n{doc_context}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."

    qa_chain = AsyncQAChain()
    history = state.get("history", [])
    answer = await qa_chain.aanswer(question, context, history=history)

    state["answer"] = answer

    return state


def generate_answer(state: GraphState) -> GraphState:
    logger.info("=== Generate Answer ===")
    question = state["question"]

    graph_context = _format_graph_context(
        state.get("graph_result", []),
        state.get("cypher_query", ""),
    )

    doc_context = _format_document_context(state.get("documents", []))

    context_parts = []
    if graph_context:
        context_parts.append(f"[图谱数据]\n{graph_context}")
    if doc_context:
        context_parts.append(f"[文档内容]\n{doc_context}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."

    qa_chain = QAChain()
    history = state.get("history", [])
    answer = qa_chain.answer(question, context, history=history)

    state["answer"] = answer

    return state


def _format_graph_context(graph_results: list, cypher_query: str) -> str:
    if not graph_results:
        return ""

    lines = []

    if cypher_query:
        lines.append(f"执行的查询: {cypher_query.strip()}")

    for i, row in enumerate(graph_results[:10]):
        if isinstance(row, dict):
            parts = []
            for key, value in row.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, (int, float, bool)):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    parts.append(f"{key}: {', '.join(str(v) for v in value[:5])}")
                elif value is not None:
                    parts.append(f"{key}: {value}")
            if parts:
                lines.append(f"  {i + 1}. {', '.join(parts)}")
        else:
            lines.append(f"  {i + 1}. {row}")

    if not lines:
        return ""

    return "\n".join(lines)


def _format_document_context(documents: list) -> str:
    parts = []
    for i, doc in enumerate(documents[:5]):
        content = doc.get("content", "").strip()
        if content:
            parts.append(f"片段{i + 1}: {content}")
    return "\n\n".join(parts)


def handle_error(state: GraphState) -> GraphState:
    logger.error(f"Error: {state.get('error', 'Unknown error')}")
    state["answer"] = "An error occurred while processing your request."
    return state
