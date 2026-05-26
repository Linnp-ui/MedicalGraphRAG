from typing import Any, Dict, List, Optional
import asyncio
import hashlib
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ..core.config import get_settings, load_prompts
from ..core.circuit_breaker import get_circuit_breaker, CircuitOpenError
from ..core.cache import cached, get_query_cache
from ..core.metrics import get_metrics

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2
QA_CACHE_SIZE = 1000


def _compress_history(
    history: List[Dict[str, str]], max_recent: int = 4, max_summary_len: int = 50
) -> List[Dict[str, str]]:
    if len(history) <= max_recent:
        return history

    old_messages = history[:-max_recent]
    recent_messages = history[-max_recent:]

    summary_parts = []
    i = 0
    while i < len(old_messages):
        msg = old_messages[i]
        if msg.get("role") == "user":
            question = msg.get("content", "")[:max_summary_len]
            answer = ""
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


class AsyncQAChain:
    """Async question answering chain using LLM with non-blocking calls"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.settings = get_settings()
        self.llm = llm
        self._prompts = None
        self._async_llm = None

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

    def _get_async_llm(self) -> ChatOpenAI:
        if self._async_llm is None:
            self._async_llm = ChatOpenAI(
                model=self.settings.dashscope_model,
                temperature=0,
                api_key=self.settings.dashscope_api_key,
                base_url=self.settings.dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                request_timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        return self._async_llm

    def _get_prompts(self) -> Dict[str, Any]:
        if self._prompts is None:
            self._prompts = load_prompts()
        return self._prompts

    async def aanswer(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Async answer question with context and optional conversation history"""
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
        chain = prompt | self._get_async_llm() | StrOutputParser()

        circuit_breaker = get_circuit_breaker("llm_qa")
        metrics = get_metrics()

        try:
            if circuit_breaker.is_open():
                logger.warning("Circuit breaker OPEN, returning fallback response")
                metrics.increment("llm_fallback_total", {"reason": "circuit_open"})
                return self._get_fallback_response(question, context)

            start_time = time.perf_counter()
            answer = await chain.ainvoke(
                {
                    "question": question,
                    "context": context,
                }
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.observe("llm_call_duration_ms", duration_ms)
            metrics.increment("llm_calls_total", {"status": "success"})

            circuit_breaker.record_success()
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            circuit_breaker.record_failure()
            metrics.increment("llm_calls_total", {"status": "error"})
            logger.error(f"Async QA chain failed: {e}")
            return self._get_fallback_response(question, context)

    def _get_fallback_response(self, question: str, context: str) -> str:
        if context and len(context) > 100:
            return f"基于检索到的信息，我找到了相关内容，但当前无法生成完整回答。请参考以下信息：\n\n{context[:500]}..."
        return "抱歉，当前服务繁忙，请稍后重试。"


class QAChain:
    """Question answering chain using LLM (sync wrapper for backward compatibility)"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.async_chain = AsyncQAChain(llm=llm)

    def _get_prompts(self) -> Dict[str, Any]:
        return self.async_chain._get_prompts()

    @cached(get_query_cache)
    def answer(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Answer question with context and optional conversation history"""
        return self._run_sync(question, context, history)

    def _run_sync(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
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
        chain = prompt | self.async_chain._get_llm() | StrOutputParser()

        circuit_breaker = get_circuit_breaker("llm_qa")

        try:
            if circuit_breaker.is_open():
                logger.warning("Circuit breaker OPEN, returning fallback response")
                return self.async_chain._get_fallback_response(question, context)

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
            return self.async_chain._get_fallback_response(question, context)

    def answer_with_sources(
        self,
        question: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
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
    chain = QAChain()
    return chain.answer(question, context)
