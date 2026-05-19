from typing import List

from loguru import logger

from ..retrieval.drift_search import DRIFTSearch
from .state import GraphState


def route_question(state: GraphState) -> GraphState:
    """Route question using DRIFTSearch intent classification"""
    question = state["question"]
    searcher = DRIFTSearch()
    intent = searcher._classify_query_intent(question)
    state["routing"] = intent
    logger.info(f"Routed question '{question}' to: {intent}")
    return state


def decompose_question(state: GraphState) -> GraphState:
    """Decompose complex question into sub-queries"""
    question = state["question"]
    # Simple heuristic decomposition - split by conjunctions
    conjunctions = ["和", "与", "以及", "还有", "并且"]
    subqueries = [question]
    
    for conj in conjunctions:
        if conj in question:
            parts = [p.strip() for p in question.split(conj) if p.strip()]
            if len(parts) > 1:
                subqueries = parts
                break
    
    state["subqueries"] = subqueries
    logger.info(f"Decomposed '{question}' into {subqueries}")
    return state
