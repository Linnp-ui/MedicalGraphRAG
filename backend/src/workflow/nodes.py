from typing import Any, Dict, List

from loguru import logger

from ..core.neo4j_client import Neo4jClient
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.graph_retriever import GraphRetriever
from ..retrieval.hybrid import HybridRetriever
from ..chains.cypher_gen import CypherGenerator
from ..chains.qa_chain import QAChain
from .state import GraphState
from .router import route_question


def retrieve_vector(state: GraphState) -> GraphState:
    """使用向量搜索检索文档

    Args:
        state: 工作流状态

    Returns:
        更新后的工作流状态，包含向量搜索结果
    """
    logger.info("=== Vector Retrieval ===")
    question = state["question"]

    retriever = VectorRetriever()
    results = retriever.search(question, top_k=5)

    state["documents"] = results
    logger.info(f"Vector search returned {len(results)} results")

    return state


def retrieve_graph(state: GraphState) -> GraphState:
    """使用图谱查询进行检索，包含验证、重试和回退逻辑

    Args:
        state: 工作流状态

    Returns:
        更新后的工作流状态，包含图谱查询结果和执行的Cypher查询
    """
    logger.info("=== Graph Retrieval ===")
    question = state["question"]

    client = Neo4jClient()
    schema = client.get_schema()
    generator = CypherGenerator()

    cypher = ""
    results: List[Dict[str, Any]] = []

    # 步骤1: 生成并验证Cypher（带重试）
    for attempt in range(2):
        cypher = generator.generate(question, schema)
        if not cypher:
            logger.warning(f"Cypher generation attempt {attempt + 1} returned empty")
            continue

        # 使用EXPLAIN验证语法
        if _validate_cypher(client, cypher):
            break
        else:
            logger.warning(f"Cypher syntax invalid, retrying... (attempt {attempt + 1})")
            cypher = ""

    # 步骤2: 执行Cypher
    if cypher:
        try:
            results = client.execute_query(cypher)
            logger.info(f"Graph query returned {len(results)} results")
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            results = []

    # 步骤3: 如果没有结果，移除LIMIT重试
    if cypher and not results:
        relaxed_cypher = _relax_cypher(cypher)
        if relaxed_cypher != cypher:
            logger.info("No results, retrying with relaxed query")
            try:
                results = client.execute_query(relaxed_cypher)
                if results:
                    cypher = relaxed_cypher
                    logger.info(f"Relaxed query returned {len(results)} results")
            except Exception:
                pass

    # 步骤4: 如果仍然没有结果，回退到向量搜索
    if not results:
        logger.info("Graph query returned no results, falling back to vector search")
        retriever = VectorRetriever()
        doc_results = retriever.search(question, top_k=5)
        state["documents"].extend(doc_results)
        logger.info(f"Vector fallback returned {len(doc_results)} results")

    state["cypher_query"] = cypher
    state["graph_result"] = results

    return state


def _validate_cypher(client: Neo4jClient, cypher: str) -> bool:
    """使用EXPLAIN验证Cypher语法

    Args:
        client: Neo4j客户端
        cypher: 要验证的Cypher查询

    Returns:
        如果Cypher语法有效则返回True，否则返回False
    """
    try:
        explain_query = f"EXPLAIN {cypher}"
        client.execute_query(explain_query)
        return True
    except Exception as e:
        logger.error(f"Cypher validation failed: {e}")
        return False


def _relax_cypher(cypher: str) -> str:
    """从Cypher中移除LIMIT子句以扩大结果范围

    Args:
        cypher: 要处理的Cypher查询

    Returns:
        移除了LIMIT子句的Cypher查询
    """
    import re

    # 移除LIMIT子句
    relaxed = re.sub(r"\s+LIMIT\s+\d+", "", cypher, flags=re.IGNORECASE)
    return relaxed.strip()


def retrieve_hybrid(state: GraphState) -> GraphState:
    """使用混合方法进行检索

    Args:
        state: 工作流状态

    Returns:
        更新后的工作流状态，包含混合检索结果
    """
    logger.info("=== Hybrid Retrieval ===")
    question = state["question"]

    retriever = HybridRetriever(alpha=0.5)
    results = retriever.search(question)

    state["documents"] = results.get("combined_results", [])
    state["graph_result"] = results.get("graph_results", [])

    logger.info(f"Hybrid search returned {len(state['documents'])} combined results")

    return state


def generate_answer(state: GraphState) -> GraphState:
    """使用检索到的上下文生成最终答案

    Args:
        state: 工作流状态

    Returns:
        更新后的工作流状态，包含生成的答案
    """
    logger.info("=== Generate Answer ===")
    question = state["question"]

    # 首先从图谱结果构建上下文（结构化数据）
    graph_context = _format_graph_context(
        state.get("graph_result", []),
        state.get("cypher_query", ""),
    )

    # 从文档搜索结果构建上下文
    doc_context = _format_document_context(state.get("documents", []))

    # 组合上下文：图谱上下文优先
    context_parts = []
    if graph_context:
        context_parts.append(f"[图谱数据]\n{graph_context}")
    if doc_context:
        context_parts.append(f"[文档内容]\n{doc_context}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."

    # 生成答案
    qa_chain = QAChain()
    history = state.get("history", [])
    answer = qa_chain.answer(question, context, history=history)

    state["answer"] = answer

    return state


def _format_graph_context(graph_results: list, cypher_query: str) -> str:
    """将图谱查询结果格式化为LLM可读取的文本

    Args:
        graph_results: 图谱查询结果
        cypher_query: 执行的Cypher查询

    Returns:
        格式化后的文本
    """
    if not graph_results:
        return ""

    lines = []

    # 如果有Cypher查询，显示生成的Cypher
    if cypher_query:
        lines.append(f"执行的查询: {cypher_query.strip()}")

    # 格式化每个结果行
    for i, row in enumerate(graph_results[:10]):  # 限制为10行
        if isinstance(row, dict):
            parts = []
            for key, value in row.items():
                # 跳过内部Neo4j键
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
    """将文档搜索结果格式化为LLM可读取的文本

    Args:
        documents: 文档搜索结果

    Returns:
        格式化后的文本
    """
    parts = []
    for i, doc in enumerate(documents[:5]):  # 限制为5个文档
        content = doc.get("content", "").strip()
        if content:
            parts.append(f"片段{i + 1}: {content}")
    return "\n\n".join(parts)


def decompose_query(state: GraphState) -> GraphState:
    """将复杂问题分解为子查询

    Args:
        state: 工作流状态

    Returns:
        更新后的工作流状态，包含分解后的子查询
    """
    logger.info("=== Query Decomposition ===")
    # 这是一个简化版本
    # 在生产环境中，使用LLM进行分解
    question = state["question"]

    # 简单分解 - 仅存储原始问题
    state["subqueries"] = [question]

    return state


def handle_error(state: GraphState) -> GraphState:
    """处理工作流中的错误

    Args:
        state: 工作流状态，包含错误信息

    Returns:
        更新后的工作流状态，包含错误处理后的答案
    """
    logger.error(f"Error: {state.get('error', 'Unknown error')}")
    state["answer"] = "An error occurred while processing your request."
    return state
