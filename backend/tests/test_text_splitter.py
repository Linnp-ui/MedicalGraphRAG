"""Tests for text splitter strategies and quality metrics"""
import pytest
from src.ingestion.text_splitter import (
    TextSplitter,
    SplitStrategy,
    SplitStats,
    select_chunking_strategy,
    _is_medical_report,
    _has_markdown_structure,
    _has_clear_structure,
    _has_multi_level_structure,
    CJK_SEPARATORS,
)

# ---------------------------------------------------------------------------
# 基础默认值
# ---------------------------------------------------------------------------

class TestTextSplitterDefaults:
    def test_default_chunk_size(self):
        splitter = TextSplitter()
        assert splitter.chunk_size == 512

    def test_default_chunk_overlap(self):
        splitter = TextSplitter()
        assert splitter.chunk_overlap == 75

    def test_default_strategy(self):
        splitter = TextSplitter()
        assert splitter.strategy == SplitStrategy.HYBRID

    def test_soft_max_default(self):
        splitter = TextSplitter(chunk_size=512)
        assert splitter.soft_max == 384

    def test_separators_default(self):
        splitter = TextSplitter()
        assert splitter.separators == CJK_SEPARATORS

    def test_hierarchical_sizes_default(self):
        splitter = TextSplitter()
        assert splitter.hierarchical_sizes == [1024, 512, 256]

    def test_custom_soft_max(self):
        splitter = TextSplitter(chunk_size=512, soft_max=200)
        assert splitter.soft_max == 200

    def test_custom_separators(self):
        splitter = TextSplitter(separators=["\n\n", "\n", " "])
        assert splitter.separators == ["\n\n", "\n", " "]

    def test_custom_hierarchical_sizes(self):
        splitter = TextSplitter(hierarchical_sizes=[2048, 1024])
        assert splitter.hierarchical_sizes == [2048, 1024]

# ---------------------------------------------------------------------------
# 递归切分
# ---------------------------------------------------------------------------

class TestRecursiveSplit:
    def test_recursive_basic(self):
        text = "第一段内容。第二段内容。第三段内容。"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=20, chunk_overlap=0, soft_max=15)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1
        assert all(c.content.strip() for c in chunks)

    def test_recursive_cjk_boundary(self):
        text = "机器学习是AI的分支。深度学习是ML的方法。神经网络是DL的基础。"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=30, chunk_overlap=0, soft_max=20)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.content

    def test_recursive_paragraphs(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=20, chunk_overlap=0, soft_max=10)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_recursive_with_overlap(self):
        text = "句子一。句子二。句子三。句子四。句子五。"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=30, chunk_overlap=10, soft_max=20)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_recursive_empty(self):
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks = splitter.split_text("", "doc1")
        assert len(chunks) == 0

    def test_recursive_single_short(self):
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks = splitter.split_text("短文本", "doc1")
        assert len(chunks) == 1

    def test_recursive_large_paragraph(self):
        text = "A" * 2000
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=100, chunk_overlap=10, soft_max=80)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 10

# ---------------------------------------------------------------------------
# 原有策略兼容
# ---------------------------------------------------------------------------

class TestSplitStrategies:
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

    def test_hybrid_fallback_to_recursive(self):
        text = "没有markdown结构的纯文本。只是普通句子。测试混合策略。"
        splitter = TextSplitter(strategy=SplitStrategy.HYBRID, chunk_size=50, soft_max=30)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

# ---------------------------------------------------------------------------
# 语义切分
# ---------------------------------------------------------------------------

class TestSemanticSplitting:
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

    def test_semantic_fallback(self):
        from unittest.mock import patch
        with patch("src.ingestion.text_splitter.TextSplitter._get_semantic_model", side_effect=ImportError("mock")):
            text = "没有模型的降级测试。"
            splitter = TextSplitter(strategy=SplitStrategy.SEMANTIC, chunk_size=100)
            chunks = splitter.split_text(text, "doc1")
            assert len(chunks) >= 1

# ---------------------------------------------------------------------------
# 层级切分
# ---------------------------------------------------------------------------

class TestHierarchicalSplitting:
    def test_hierarchical_split_basic(self):
        text = ("Paragraph one with some content. " * 10)
        splitter = TextSplitter(strategy=SplitStrategy.HIERARCHICAL, chunk_size=256)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_hierarchical_split_metadata(self):
        text = "Test paragraph one. Test paragraph two."
        splitter = TextSplitter(strategy=SplitStrategy.HIERARCHICAL, chunk_size=256)
        chunks = splitter.split_text(text, "doc1")
        if chunks:
            assert "hierarchical" in chunks[0].metadata["source"]

    def test_hierarchical_custom_sizes(self):
        text = "A" * 3000
        splitter = TextSplitter(
            strategy=SplitStrategy.HIERARCHICAL,
            hierarchical_sizes=[500, 250],
            chunk_size=256,
        )
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

# ---------------------------------------------------------------------------
# 医疗切分
# ---------------------------------------------------------------------------

class TestMedicalSplitting:
    def test_medical_basic(self):
        text = "【病史摘要】患者男性。\n【诊断】高血压。\n【治疗方案】药物治疗。"
        splitter = TextSplitter(strategy=SplitStrategy.MEDICAL, chunk_size=50, soft_max=30)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

    def test_medical_fallback(self):
        text = "没有医疗结构的普通文本。"
        splitter = TextSplitter(strategy=SplitStrategy.MEDICAL, chunk_size=50)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1

# ---------------------------------------------------------------------------
# 策略选择器
# ---------------------------------------------------------------------------

class TestStrategySelector:
    def test_select_medical_strategy(self):
        text = "病史摘要：患者男性，45岁。主诉：头痛一周。体格检查：血压正常。"
        strategy = select_chunking_strategy(text, domain="medical")
        assert strategy == SplitStrategy.MEDICAL

    def test_select_markdown_strategy(self):
        text = "# Introduction\n\nSome content here.\n\n## Details\n\nMore content here.\n\n### Subsection\n\nEven more content."
        strategy = select_chunking_strategy(text)
        assert strategy == SplitStrategy.MARKDOWN

    def test_select_recursive_as_default(self):
        text = "Simple text without any special structure or medical content."
        strategy = select_chunking_strategy(text)
        assert strategy == SplitStrategy.RECURSIVE

    def test_select_semantic_for_long_unstructured(self):
        text = "无结构文本。" * 2000
        assert len(text) > 5000
        strategy = select_chunking_strategy(text)
        assert strategy == SplitStrategy.SEMANTIC

# ---------------------------------------------------------------------------
# 文档检测器
# ---------------------------------------------------------------------------

class TestDocumentDetectors:
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

# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------

class TestChunkMetadata:
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

# ---------------------------------------------------------------------------
# SplitStats
# ---------------------------------------------------------------------------

class TestSplitStats:
    def test_split_text_with_stats(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks, stats = splitter.split_text_with_stats(text, "doc1")
        assert isinstance(stats, SplitStats)
        assert stats.num_chunks == len(chunks)
        assert stats.total_chars == sum(len(c.content) for c in chunks)
        assert stats.split_time_ms >= 0

    def test_split_stats_empty(self):
        splitter = TextSplitter()
        chunks, stats = splitter.split_text_with_stats("", "doc1")
        assert stats.num_chunks == 0

# ---------------------------------------------------------------------------
# 边界条件
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        for strategy in SplitStrategy:
            if strategy == SplitStrategy.SEMANTIC:
                continue
            splitter = TextSplitter(strategy=strategy)
            chunks = splitter.split_text("", "doc1")
            assert len(chunks) == 0, f"Strategy {strategy} failed on empty text"

    def test_single_char(self):
        splitter = TextSplitter()
        chunks = splitter.split_text("A", "doc1")
        assert len(chunks) == 1

    def test_single_line(self):
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks = splitter.split_text("这是一行测试文本。", "doc1")
        assert len(chunks) == 1

    def test_only_newlines(self):
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks = splitter.split_text("\n\n\n\n", "doc1")
        assert len(chunks) == 0

    def test_long_single_word(self):
        text = "A" * 10000
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE, chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 50

    def test_code_preservation(self):
        text = "```python\nprint('hello')\n```"
        splitter = TextSplitter(strategy=SplitStrategy.RECURSIVE)
        chunks = splitter.split_text(text, "doc1")
        assert len(chunks) >= 1
