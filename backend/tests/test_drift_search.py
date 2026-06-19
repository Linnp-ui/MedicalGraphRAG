"""Tests for src.retrieval.drift_search module.

Pure unit tests with all external dependencies mocked:
- VectorRetriever, GraphRetriever, MedicalQueryExpander
- community_detector, summary_generator, cache, process_monitor
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from src.retrieval.drift_search import DRIFTSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_drift_search(**overrides):
    """Create a DRIFTSearch instance with all external deps mocked."""
    with patch("src.retrieval.drift_search.VectorRetriever"), \
         patch("src.retrieval.drift_search.GraphRetriever"), \
         patch("src.retrieval.drift_search.MedicalQueryExpander"), \
         patch("src.retrieval.drift_search.get_community_detector"), \
         patch("src.retrieval.drift_search.get_summary_generator"), \
         patch("src.retrieval.drift_search.cached", side_effect=lambda c: (lambda f: f)), \
         patch("src.retrieval.drift_search.track_process", side_effect=lambda n: (lambda f: f)):
        ds = DRIFTSearch(**overrides)
    return ds


# ===================================================================
# TestClassifyQueryIntent
# ===================================================================

class TestClassifyQueryIntent:
    """Test DRIFTSearch._classify_query_intent."""

    def test_global_indicators(self):
        ds = _make_drift_search()
        assert ds._classify_query_intent("请总结高血压的主要症状") == "global"
        assert ds._classify_query_intent("概述糖尿病的并发症") == "global"
        assert ds._classify_query_intent("全面介绍冠心病的概况") == "global"

    def test_local_indicators(self):
        ds = _make_drift_search()
        assert ds._classify_query_intent("高血压是什么") == "local"
        assert ds._classify_query_intent("谁发明了青霉素") == "local"
        assert ds._classify_query_intent("详细描述心肌梗死的症状") == "local"

    def test_hybrid_when_ambiguous(self):
        ds = _make_drift_search()
        # No global or local indicators present
        assert ds._classify_query_intent("高血压") == "hybrid"

    def test_numeric_indicators(self):
        ds = _make_drift_search()
        # Numeric indicators alone should yield "local"
        assert ds._classify_query_intent("高血压有多少种分类") == "local"
        assert ds._classify_query_intent("糖尿病患者数量统计") == "local"


# ===================================================================
# TestComputeDynamicAlpha
# ===================================================================

class TestComputeDynamicAlpha:
    """Test DRIFTSearch._compute_dynamic_alpha."""

    def test_global_alpha(self):
        ds = _make_drift_search()
        alpha = ds._compute_dynamic_alpha("总结高血压的治疗方案")
        assert alpha == 0.2

    def test_local_alpha(self):
        ds = _make_drift_search()
        alpha = ds._compute_dynamic_alpha("高血压是什么")
        assert alpha == 0.7

    def test_hybrid_alpha(self):
        ds = _make_drift_search()
        alpha = ds._compute_dynamic_alpha("高血压")
        assert alpha == 0.5


# ===================================================================
# TestCombineResults
# ===================================================================

class TestCombineResults:
    """Test DRIFTSearch._combine_results."""

    def test_combine_vector_and_graph(self):
        ds = _make_drift_search()
        vector_results = [
            {"chunk_id": "c1", "similarity": 0.9, "content": "vec content 1"},
            {"chunk_id": "c2", "similarity": 0.7, "content": "vec content 2"},
        ]
        graph_results = [
            {"name": "高血压", "type": "Disease", "score": 0.8},
            {"name": "糖尿病", "type": "Disease", "score": 0.6},
        ]
        combined = ds._combine_results("高血压", vector_results, graph_results, alpha=0.5)
        assert len(combined) == 4
        sources = {r["source"] for r in combined}
        assert "vector" in sources
        assert "graph" in sources

    def test_deduplication(self):
        ds = _make_drift_search()
        # chunk_id and entity name overlap – should be deduplicated
        vector_results = [
            {"chunk_id": "高血压", "similarity": 0.9, "content": "vec"},
        ]
        graph_results = [
            {"name": "高血压", "type": "Disease", "score": 0.8},
        ]
        combined = ds._combine_results("高血压", vector_results, graph_results, alpha=0.5)
        # "高血压" appears as chunk_id in vector and as name in graph
        # They are different keys (chunk_id vs name), so both should remain
        # But if chunk_id == entity name, the seen set deduplicates
        assert len(combined) == 1

    def test_empty_results(self):
        ds = _make_drift_search()
        combined = ds._combine_results("query", [], [], alpha=0.5)
        assert combined == []

    def test_sorting_by_combined_score(self):
        ds = _make_drift_search()
        vector_results = [
            {"chunk_id": "c1", "similarity": 0.5, "content": "low"},
            {"chunk_id": "c2", "similarity": 1.0, "content": "high"},
        ]
        graph_results = [
            {"name": "entity1", "type": "Disease", "score": 0.9},
        ]
        combined = ds._combine_results("query", vector_results, graph_results, alpha=0.5)
        # Results should be sorted by combined_score descending
        scores = [r["combined_score"] for r in combined]
        assert scores == sorted(scores, reverse=True)


# ===================================================================
# TestExplainStrategy
# ===================================================================

class TestExplainStrategy:
    """Test DRIFTSearch.explain_strategy."""

    def test_explain_returns_correct_structure(self):
        ds = _make_drift_search()
        explanation = ds.explain_strategy("总结高血压的治疗方案")
        assert "query" in explanation
        assert "detected_intent" in explanation
        assert "alpha_weight" in explanation
        assert "strategy" in explanation
        assert "explanation" in explanation
        assert explanation["query"] == "总结高血压的治疗方案"
        assert explanation["detected_intent"] == "global"
        assert explanation["alpha_weight"] == 0.2
        assert isinstance(explanation["explanation"], str)
        assert len(explanation["explanation"]) > 0


# ===================================================================
# TestClassifyFineIntent
# ===================================================================

class TestClassifyFineIntent:
    """Test DRIFTSearch._classify_fine_intent."""

    def test_drug_query(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("阿司匹林的用法") == "drug_query"

    def test_drug_interaction(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("阿司匹林与华法林的相互作用") == "drug_interaction"

    def test_diagnosis(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("高血压的诊断方法") == "diagnosis_assist"

    def test_treatment(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("高血压治疗方案") == "treatment_query"

    def test_symptom(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("高血压的症状表现") == "symptom_query"

    def test_general(self):
        ds = _make_drift_search()
        assert ds._classify_fine_intent("高血压") == "general"
