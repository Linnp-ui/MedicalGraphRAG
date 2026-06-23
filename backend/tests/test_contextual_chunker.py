"""Tests for contextual chunking module"""
from src.ingestion.text_splitter import TextSplitter, SplitStrategy, TextChunk
from src.ingestion.contextual_chunker import (
    ContextualChunkEnricher,
    ContextConfig,
    LateChunkingAdapter,
    ContextualSplitter,
    create_contextual_splitter,
)


class TestContextualChunkEnricher:

    def test_disabled_by_default(self):
        chunk = TextChunk(id="c1", content="test", document_id="d1", index=0, metadata={})
        enricher = ContextualChunkEnricher()
        result = enricher.enrich_chunk(chunk, "full document text")
        assert result.content == "test"
        assert "contextualized" not in result.metadata

    def test_enabled_no_llm_noop(self):
        chunk = TextChunk(id="c1", content="test", document_id="d1", index=0, metadata={})
        config = ContextConfig(enabled=True)
        enricher = ContextualChunkEnricher(config)
        result = enricher.enrich_chunk(chunk, "full document text")
        assert result.content == "test"

    def test_enabled_with_llm(self):
        def fake_llm(prompt, model=None):
            return "这是关于测试的段落。"
        chunk = TextChunk(id="c1", content="test content", document_id="d1", index=0, metadata={})
        config = ContextConfig(enabled=True)
        enricher = ContextualChunkEnricher(config, llm_call=fake_llm)
        result = enricher.enrich_chunk(chunk, "full document text")
        assert "上下文" in result.content
        assert result.metadata.get("contextualized") is True

    def test_batch_enrich(self):
        def fake_llm(prompt, model=None):
            return "上下文摘要。"
        chunks = [
            TextChunk(id="c1", content="a", document_id="d1", index=0, metadata={}),
            TextChunk(id="c2", content="b", document_id="d1", index=1, metadata={}),
        ]
        config = ContextConfig(enabled=True)
        enricher = ContextualChunkEnricher(config, llm_call=fake_llm)
        results = enricher.enrich_batch(chunks, "full document text")
        assert len(results) == 2
        assert all(r.metadata.get("contextualized") for r in results)


class TestLateChunkingAdapter:

    def test_disabled_fallback(self):
        adapter = LateChunkingAdapter(enabled=False)
        chunks = [TextChunk(id="c1", content="hello", document_id="d1", index=0, metadata={})]
        results = adapter.encode_chunks(chunks, "full text", lambda t: [len(t)])
        assert len(results) == 1

    def test_empty_chunks(self):
        adapter = LateChunkingAdapter(enabled=True)
        results = adapter.encode_chunks([], "full text", lambda t: [0.1])
        assert results == []

    def test_enabled_fallback_on_error(self):
        def broken_encode(text):
            raise ValueError("mock failure")
        adapter = LateChunkingAdapter(enabled=True)
        chunks = [TextChunk(id="c1", content="hello", document_id="d1", index=0, metadata={})]
        results = adapter.encode_chunks(chunks, "full text", broken_encode)
        assert len(results) == 1


class TestContextualSplitter:

    def test_basic_split(self):
        splitter = ContextualSplitter(
            base_splitter=TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=50),
        )
        chunks = splitter.split("测试文本。", "doc1")
        assert len(chunks) >= 1

    def test_split_with_enrichment(self):
        def fake_llm(prompt, model=None):
            return "摘要。"
        config = ContextConfig(enabled=True)
        splitter = ContextualSplitter(
            base_splitter=TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=50),
            contextual_config=config,
            llm_call=fake_llm,
        )
        chunks = splitter.split("测试文本。\n\n更多内容。", "doc1")
        assert len(chunks) >= 1

    def test_late_chunking_encode(self):
        splitter = ContextualSplitter(
            base_splitter=TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=50),
            late_chunking_enabled=True,
        )
        chunks = [TextChunk(id="c1", content="hello", document_id="d1", index=0, metadata={})]
        results = splitter.encode_embeddings(chunks, "hello world", lambda t: [0.5, 0.5])
        assert len(results) == 1


class TestCreateContextualSplitter:

    def test_create_default(self):
        splitter = create_contextual_splitter()
        assert splitter.enricher.config.enabled is False
        assert splitter.late_adapter.enabled is False

    def test_create_with_contextual(self):
        splitter = create_contextual_splitter(
            contextual_enabled=True,
            llm_call=lambda p, m=None: "ctx",
        )
        assert splitter.enricher.config.enabled is True

    def test_create_with_late_chunking(self):
        splitter = create_contextual_splitter(late_chunking=True)
        assert splitter.late_adapter.enabled is True
