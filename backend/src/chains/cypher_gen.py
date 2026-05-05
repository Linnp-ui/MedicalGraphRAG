from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ..core.config import get_settings, load_prompts


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
                model=self.settings.openai_model,
                temperature=0,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url or "https://api.openai.com/v1",
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

        # Build examples string
        examples_str = ""
        if examples:
            examples_str = "\n\n".join([f"Question: {ex}" for ex in examples])

        # Create prompt template
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

        # Generate
        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            cypher = chain.invoke({"question": question})
            logger.info(f"Generated Cypher: {cypher[:100]}...")
            return cypher.strip()
        except Exception as e:
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
