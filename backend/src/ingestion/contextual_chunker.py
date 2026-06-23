"""上下文感知切分模块

实现两个关键技术：
1. Contextual Retrieval (Anthropic) — 用 LLM 为每个 chunk 生成上下文摘要
2. Late Chunking (Jina AI) — 先编码全文再 pool 为 chunk 向量（接口预留）

评估指标（对比普通切分 vs 上下文增强切分）：
  - retrieval_recall 提升 > 5%
  - boundary 质量不变或提升
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from loguru import logger

from .text_splitter import TextChunk, TextSplitter


CONTEXTUAL_PROMPT_TEMPLATE = """你是一个文档分析助手。给定一个文档和其中的一个段落，请用一句简洁的话概括该段落在文档中的上下文位置和核心主题。

<document>
{document_text}
</document>

<chunk>
{chunk_text}
</chunk>

请只用一句话概括（不超过50字），不要多余的解释："""


@dataclass
class ContextConfig:
    enabled: bool = False
    llm_model: str = "qwen-flash"
    context_max_chars: int = 200
    prompt_template: str = CONTEXTUAL_PROMPT_TEMPLATE


class ContextualChunkEnricher:
    """上下文增强切分器 — 用 LLM 为 chunk 添加上下文摘要（Anthropic 方案）"""

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        llm_call: Optional[Callable] = None,
    ):
        self.config = config or ContextConfig()
        self.llm_call = llm_call

    def enrich_chunk(
        self,
        chunk: TextChunk,
        document_text: str,
    ) -> TextChunk:
        if not self.config.enabled or not self.llm_call:
            return chunk

        prompt = self.config.prompt_template.format(
            document_text=document_text[:min(len(document_text), 8000)],
            chunk_text=chunk.content[:500],
        )

        try:
            context = self.llm_call(prompt, model=self.config.llm_model)
            context = context.strip().strip('"').strip("'")[:self.config.context_max_chars]

            chunk.content = f"[上下文] {context}\n{chunk.content}"
            chunk.metadata["contextualized"] = True
            chunk.metadata["context_summary"] = context
        except Exception as e:
            logger.warning(f"Contextual enrichment failed for chunk {chunk.id}: {e}")
            chunk.metadata["contextualized"] = False

        return chunk

    def enrich_batch(
        self,
        chunks: List[TextChunk],
        document_text: str,
    ) -> List[TextChunk]:
        return [self.enrich_chunk(c, document_text) for c in chunks]


class LateChunkingAdapter:
    """延迟分块适配器 — 嵌入阶段利用全文上下文（Jina AI 方案）

    使用方式：
      1. embedder.encode(full_text)  →  得到全文档 token 嵌入
      2. 根据 chunk 边界划分 token 序列
      3. 对每个 chunk 的 token 做 mean pooling

    当前状态：接口预留，待嵌入层集成。
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def encode_chunks(
        self,
        chunks: List[TextChunk],
        full_text: str,
        encode_func: Callable,
    ) -> List[List[float]]:
        """编码 chunks（带全文上下文）

        Args:
            chunks: 文本块列表
            full_text: 完整文档文本
            encode_func: 嵌入函数，输入文本，输出向量列表
                        对长上下文模型应 encode(full_text) 后按 token 位置切分

        Returns:
            每个 chunk 的嵌入向量
        """
        if not self.enabled or not chunks:
            if not chunks:
                return []
            return [encode_func(c.content) for c in chunks]

        try:
            full_embedding = encode_func(full_text)
            chunk_count = len(chunks)
            total_chars = len(full_text)

            embeddings = []
            for i, chunk in enumerate(chunks):
                start_ratio = sum(len(c.content) for c in chunks[:i]) / max(total_chars, 1)
                end_ratio = start_ratio + len(chunk.content) / max(total_chars, 1)

                if isinstance(full_embedding, list) and full_embedding and isinstance(full_embedding[0], (int, float)):
                    dim = len(full_embedding)
                    start_idx = int(dim * start_ratio)
                    end_idx = int(dim * end_ratio)
                    pooled = self._mean_pool(full_embedding[start_idx:end_idx]) if end_idx > start_idx else full_embedding
                    embeddings.append(pooled)
                else:
                    embeddings.append(encode_func(chunk.content))

            return embeddings

        except Exception as e:
            logger.warning(f"Late chunking failed: {e}, falling back to per-chunk encoding")
            try:
                return [encode_func(c.content) for c in chunks]
            except Exception as fallback_e:
                logger.warning(f"Fallback encoding also failed: {fallback_e}")
                empty_dim = 1536
                return [[0.0] * empty_dim for _ in chunks]

    def _mean_pool(self, vectors: List[float]) -> List[float]:
        if not vectors:
            return []
        if isinstance(vectors[0], list):
            return [sum(dim) / len(vectors) for dim in zip(*vectors)]
        return vectors


class ContextualSplitter:
    """集成上下文感知的切分流

    流程：
      1. 用 TextSplitter 做基础切分
      2. 可选 ContextualChunkEnricher 做上下文增强
      3. 可选 LateChunkingAdapter 做延迟分块编码
    """

    def __init__(
        self,
        base_splitter: TextSplitter,
        contextual_config: Optional[ContextConfig] = None,
        llm_call: Optional[Callable] = None,
        late_chunking_enabled: bool = False,
    ):
        self.base_splitter = base_splitter
        self.enricher = ContextualChunkEnricher(contextual_config, llm_call)
        self.late_adapter = LateChunkingAdapter(late_chunking_enabled)

    def split(self, text: str, document_id: str = "") -> List[TextChunk]:
        chunks = self.base_splitter.split_text(text, document_id)
        chunks = self.enricher.enrich_batch(chunks, text)
        return chunks

    def encode_embeddings(
        self,
        chunks: List[TextChunk],
        full_text: str,
        encode_func: Callable,
    ) -> List[List[float]]:
        return self.late_adapter.encode_chunks(chunks, full_text, encode_func)


def create_contextual_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 75,
    strategy: str = "recursive",
    contextual_enabled: bool = False,
    llm_call: Optional[Callable] = None,
    late_chunking: bool = False,
    **kwargs,
) -> ContextualSplitter:
    from .text_splitter import SplitStrategy

    base = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=SplitStrategy(strategy),
        soft_max=kwargs.get("soft_max", int(chunk_size * 0.75)),
        separators=kwargs.get("separators"),
    )

    ctx_config = None
    if contextual_enabled:
        ctx_config = ContextConfig(
            enabled=True,
            llm_model=kwargs.get("contextual_llm_model", "qwen-flash"),
        )

    return ContextualSplitter(
        base_splitter=base,
        contextual_config=ctx_config,
        llm_call=llm_call,
        late_chunking_enabled=late_chunking,
    )
