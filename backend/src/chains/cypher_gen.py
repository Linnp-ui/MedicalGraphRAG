from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ..core.config import get_settings, load_prompts
from ..core.circuit_breaker import get_circuit_breaker

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2


class CypherGenerator:
    """Generate Cypher queries from natural language using LLM"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.settings = get_settings()
        self.llm = llm
        self._prompts = None

    def _get_llm(self) -> ChatOpenAI:
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=self.settings.dashscope_model,
                temperature=0,
                api_key=self.settings.dashscope_api_key,
                base_url=self.settings.dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                request_timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        return self.llm

    def _get_prompts(self) -> Dict[str, Any]:
        if self._prompts is None:
            self._prompts = load_prompts()
        return self._prompts

    def generate(
        self,
        question: str,
        schema: str,
        examples: Optional[List[str]] = None,
    ) -> str:
        """Generate Cypher query from question"""
        prompts = self._get_prompts()
        cypher_prompt = prompts.get("prompts", {}).get("cypher_generator", {})

        examples_str = ""
        if examples:
            examples_str = "\n\n".join([f"Question: {ex}" for ex in examples])

        template = cypher_prompt.get("system", "").format(
            schema=schema,
            examples=examples_str or "No examples available",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", cypher_prompt.get("human", "").format(question=question)),
            ]
        )

        chain = prompt | self._get_llm() | StrOutputParser()

        circuit_breaker = get_circuit_breaker("llm_cypher")
        
        try:
            if circuit_breaker.state.name == "OPEN":
                logger.warning("Circuit breaker OPEN for cypher generation, returning empty")
                return ""
            
            cypher = chain.invoke({"question": question})
            circuit_breaker.record_success()
            logger.info(f"Generated Cypher: {cypher[:100]}...")
            return cypher.strip()
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"Cypher generation failed: {e}")
            return ""

    def generate_with_context(
        self,
        question: str,
        schema: str,
        context: str,
    ) -> str:
        """Generate Cypher query with additional context"""
        prompts = self._get_prompts()
        cypher_prompt = prompts.get("prompts", {}).get("cypher_generator", {})

        template = cypher_prompt.get("system", "").format(
            schema=schema,
            examples="",
        )
        template += f"\n\nContext from vector search:\n{context}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", cypher_prompt.get("human", "").format(question=question)),
            ]
        )

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            cypher = chain.invoke({"question": question})
            return cypher.strip()
        except Exception as e:
            logger.error(f"Cypher generation with context failed: {e}")
            return ""


def generate_cypher(
    question: str,
    schema: str,
    examples: Optional[List[str]] = None,
) -> str:
    """Convenience function to generate Cypher"""
    generator = CypherGenerator()
    return generator.generate(question, schema, examples)
