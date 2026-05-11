from typing import List, Literal

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field
from loguru import logger

from ..core.config import get_settings, load_prompts
from .state import GraphState


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vector", "graph", "hybrid"] = Field(
        ...,
        description="Route to vector search, graph query, or hybrid",
    )


class DecomposeQuery(BaseModel):
    """Decompose a complex question into sub-queries"""

    subqueries: List[str] = Field(
        ...,
        description="List of sub-questions to answer the original question",
    )


class QuestionRouter:
    """Route user questions to appropriate retrieval method"""

    def __init__(self):
        self.settings = get_settings()
        self._chain: RunnableSerializable | None = None
        self._prompts: dict | None = None

    def _get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.dashscope_model,
            temperature=0,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def _get_prompts(self) -> dict:
        if self._prompts is None:
            self._prompts = load_prompts()
        return self._prompts

    def _get_chain(self) -> RunnableSerializable:
        if self._chain is None:
            prompts = self._get_prompts()
            router_prompt = prompts.get("prompts", {}).get("router", {})

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", router_prompt.get("system", "")),
                    ("human", router_prompt.get("human", "")),
                ]
            )

            self._chain = prompt | self._get_llm().with_structured_output(RouteQuery)

        return self._chain

    def _get_decompose_chain(self) -> RunnableSerializable:
        prompts = self._get_prompts()
        decomposer_prompt = prompts.get("prompts", {}).get("decomposer", {})

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", decomposer_prompt.get("system", "")),
                ("human", decomposer_prompt.get("human", "")),
            ]
        )

        return prompt | self._get_llm().with_structured_output(DecomposeQuery)

    def route(self, question: str) -> str:
        """Route question to vector, graph, or hybrid"""
        try:
            result = self._get_chain().invoke({"question": question})
            datasource = result.datasource
            logger.info(f"Routed question to: {datasource}")
            return datasource
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return "hybrid"

    def decompose(self, question: str) -> List[str]:
        """Decompose question into sub-queries"""
        try:
            result = self._get_decompose_chain().invoke({"question": question})
            subqueries = result.subqueries
            logger.info(f"Decomposed into {len(subqueries)} subqueries")
            return subqueries
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, using original question")
            return [question]  # Default to hybrid


# ─── Module-level singleton ───────────────────────────────────────────────────
# Reused across all requests to avoid rebuilding the LLM chain on every call.
_router_instance: QuestionRouter | None = None


def _get_router() -> QuestionRouter:
    """Get (or lazily create) the singleton QuestionRouter."""
    global _router_instance
    if _router_instance is None:
        _router_instance = QuestionRouter()
        logger.info("QuestionRouter singleton created")
    return _router_instance


def route_question(state: GraphState) -> Literal["vector", "graph", "hybrid"]:
    """LangGraph node for routing questions"""
    question = state["question"]
    router = _get_router()
    datasource = router.route(question)
    state["routing"] = datasource
    return datasource


def decompose_question(state: GraphState) -> GraphState:
    """LangGraph node for decomposing complex questions"""
    question = state["question"]
    router = _get_router()
    subqueries = router.decompose(question)
    state["subqueries"] = subqueries
    logger.info(f"Decomposed '{question}' into {subqueries}")
    return state
