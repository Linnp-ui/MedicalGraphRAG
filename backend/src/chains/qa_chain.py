from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ..core.config import get_settings, load_prompts
from ..core.circuit_breaker import get_circuit_breaker, CircuitOpenError

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2


def _compress_history(
    history: List[Dict[str, str]], max_recent: int = 4, max_summary_len: int = 50
) -> List[Dict[str, str]]:
    """Compress conversation history: summarize old messages, keep recent ones intact.

    Args:
        history: Full message list [{'role': 'user'|'bot', 'content': '...'}]
        max_recent: Number of recent messages to keep intact (default 4 = 2 turns)
        max_summary_len: Max chars per message in summary

    Returns:
        Compressed history list with summary + recent messages
    """
    if len(history) <= max_recent:
        return history

    # Split into old (to summarize) and recent (keep intact)
    old_messages = history[:-max_recent]
    recent_messages = history[-max_recent:]

    # Build summary from old messages (pair user questions with bot answers)
    summary_parts = []
    i = 0
    while i < len(old_messages):
        msg = old_messages[i]
        if msg.get("role") == "user":
            question = msg.get("content", "")[:max_summary_len]
            answer = ""
            # Look for the next bot message
            if i + 1 < len(old_messages) and old_messages[i + 1].get("role") == "bot":
                answer = old_messages[i + 1].get("content", "")[:max_summary_len]
                i += 2
            else:
                i += 1
            if question:
                summary_parts.append(f"问: {question}")
                if answer:
                    summary_parts.append(f"答: {answer}")
        else:
            i += 1

    if not summary_parts:
        return recent_messages

    summary_text = "；".join(summary_parts)
    summary_msg = {"role": "bot", "content": f"[对话摘要] {summary_text}"}

    return [summary_msg] + recent_messages


class QAChain:
    """Question answering chain using LLM"""

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
                request_timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        return self.llm

    def _get_prompts(self) -> Dict[str, Any]:
        if self._prompts is None:
            self._prompts = load_prompts()
        return self._prompts

    def answer(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Answer question with context and optional conversation history"""
        prompts = self._get_prompts()
        qa_prompt = prompts.get("prompts", {}).get("qa", {})

        system_template = qa_prompt.get("system", "")
        human_template = qa_prompt.get("human", "")

        messages = [("system", system_template)]

        if history:
            compressed = _compress_history(history, max_recent=4, max_summary_len=50)
            for msg in compressed:
                role = "human" if msg.get("role") == "user" else "ai"
                messages.append((role, msg.get("content", "")))

        messages.append(("human", human_template))

        prompt = ChatPromptTemplate.from_messages(messages)

        chain = prompt | self._get_llm() | StrOutputParser()

        circuit_breaker = get_circuit_breaker("llm_qa")
        
        try:
            if circuit_breaker.state.name == "OPEN":
                logger.warning("Circuit breaker OPEN, returning fallback response")
                return self._get_fallback_response(question, context)
            
            answer = chain.invoke(
                {
                    "question": question,
                    "context": context,
                }
            )
            circuit_breaker.record_success()
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"QA chain failed: {e}")
            return self._get_fallback_response(question, context)

    def _get_fallback_response(self, question: str, context: str) -> str:
        """Generate a fallback response when LLM is unavailable"""
        if context and len(context) > 100:
            return f"基于检索到的信息，我找到了相关内容，但当前无法生成完整回答。请参考以下信息：\n\n{context[:500]}..."
        return "抱歉，当前服务繁忙，请稍后重试。"

    def answer_with_sources(
        self,
        question: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Answer question with sources"""
        answer = self.answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": sources or [],
        }


def answer_question(
    question: str,
    context: str,
) -> str:
    """Convenience function to answer a question"""
    chain = QAChain()
    return chain.answer(question, context)
