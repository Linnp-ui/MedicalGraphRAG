"""Tests for text splitter strategies"""
import pytest
from src.ingestion.text_splitter import (
    TextSplitter,
    SplitStrategy,
    select_chunking_strategy,
    _is_medical_report,
    _has_markdown_structure,
    _has_clear_structure,
    _has_multi_level_structure,
)


class TestTextSplitterDefaults:
    """Test default chunking parameters"""

    def test_default_chunk_size(self):
        splitter = TextSplitter()
        assert splitter.chunk_size == 512

    def test_default_chunk_overlap(self):
        splitter = TextSplitter()
        assert splitter.chunk_overlap == 75

    def test_default_strategy(self):
        splitter = TextSplitter()
        assert splitter.strategy == SplitStrategy.HYBRID


class TestSplitStrategies:
    """Test different splitting strategies"""

    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        splitter = TextSplitter(strategy=SplitStrategy.PARAGRAPH, chunk_size=50)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_sentence_split(self):
        text = "First sentence. Second sentence. Third sentence."
        splitter = TextSplitter(strategy=SplitStrategy.SENTENCE, chunk_size=50)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_markdown_split(self):
        text = "# Header 1\n\nContent 1\n\n## Header 2\n\nContent 2"
        splitter = TextSplitter(strategy=SplitStrategy.MARKDOWN, chunk_size=100)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_hybrid_split(self):
        text = "# Header\n\nSome content here.\n\nMore content below."
        splitter = TextSplitter(strategy=SplitStrategy.HYBRID, chunk_size=100)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1


class TestSemanticSplitting:
    """Test semantic splitting strategy"""

    @pytest.mark.skip(reason="Requires sentence-transformers model download")
    def test_semantic_split_basic(self):
        text = "机器学习是人工智能的一个分支。它使用算法来从数据中学习。深度学习是机器学习的一种方法。神经网络是深度学习的基础。"
        splitter = TextSplitter(strategy=SplitStrategy.SEMANTIC, chunk_size=100)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    @pytest.mark.skip(reason="Requires sentence-transformers model download")
    def test_semantic_split_metadata(self):
        text = "Test sentence one. Test sentence two."
        splitter = TextSplitter(strategy=SplitStrategy.SEMANTIC, chunk_size=100)
        chunks = splitter.split_text(text, "doc1")
        if chunks:
            assert chunks[0].metadata["source"] == "semantic"


class TestHierarchicalSplitting:
    """Test hierarchical splitting strategy"""

    def test_hierarchical_split_basic(self):
        text = "Paragraph one with some content. Paragraph two with more content. Paragraph three with additional content. Paragraph four with even more content to test the hierarchical splitting."
        splitter = TextSplitter(strategy=SplitStrategy.HIERARCHICAL, chunk_size=256)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_hierarchical_split_metadata(self):
        text = "Test paragraph one. Test paragraph two."
        splitter = TextSplitter(strategy=SplitStrategy.HIERARCHICAL, chunk_size=256)
        chunks = splitter.split_text(text, "doc1")
        if chunks:
            assert "hierarchical" in chunks[0].metadata["source"]


class TestStrategySelector:
    """Test automatic strategy selection"""

    def test_select_medical_strategy(self):
        text = "病史摘要：患者男性，45岁。主诉：头痛一周。体格检查：血压正常。"
        strategy = select_chunking_strategy(text, domain="medical")
        assert strategy == SplitStrategy.MEDICAL

    def test_select_markdown_strategy(self):
        text = "# Introduction\n\nSome content here.\n\n## Details\n\nMore content here.\n\n### Subsection\n\nEven more content."
        strategy = select_chunking_strategy(text)
        assert strategy == SplitStrategy.MARKDOWN

    def test_select_hybrid_strategy(self):
        text = "Simple text without any special structure or medical content."
        strategy = select_chunking_strategy(text)
        assert strategy == SplitStrategy.HYBRID


class TestDocumentDetectors:
    """Test document type detection helpers"""

    def test_is_medical_report_true(self):
        text = "病史摘要：患者信息。诊断：高血压。"
        assert _is_medical_report(text) is True

    def test_is_medical_report_false(self):
        text = "This is a regular text without medical content."
        assert _is_medical_report(text) is False

    def test_has_markdown_structure_true(self):
        text = "# H1\n\n## H2\n\n### H3\n\nContent"
        assert _has_markdown_structure(text) is True

    def test_has_markdown_structure_false(self):
        text = "No headers here."
        assert _has_markdown_structure(text) is False

    def test_has_clear_structure_true(self):
        text = "第一章 引言\n\n内容在这里。"
        assert _has_clear_structure(text) is True

    def test_has_clear_structure_false(self):
        text = "No structured content here."
        assert _has_clear_structure(text) is False

    def test_has_multi_level_structure_true(self):
        text = "# Main\n\n## Section\n\n### Subsection"
        assert _has_multi_level_structure(text) is True

    def test_has_multi_level_structure_false(self):
        text = "# Only one level"
        assert _has_multi_level_structure(text) is False


class TestChunkMetadata:
    """Test chunk metadata"""

    def test_chunk_contains_strategy_metadata(self):
        text = "Test content."
        splitter = TextSplitter(strategy=SplitStrategy.PARAGRAPH)
        chunks = splitter.split_text(text, "doc1")
        assert chunks[0].metadata["strategy"] == "paragraph"

    def test_chunk_contains_source_metadata(self):
        text = "Test content."
        splitter = TextSplitter(strategy=SplitStrategy.SENTENCE)
        chunks = splitter.split_text(text, "doc1")
        assert "source" in chunks[0].metadata
