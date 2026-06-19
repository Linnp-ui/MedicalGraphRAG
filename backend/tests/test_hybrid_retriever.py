"""Tests for src.retrieval.hybrid module.

Pure unit tests with all external dependencies mocked:
- VectorRetriever, GraphRetriever, CrossEncoder, cache
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from src.retrieval.hybrid import (
    CrossEncoderReranker,
    IntentAwareAlpha,
    HybridRetriever,
    INTENT_ALPHA_MAP,
)


# ===================================================================
# TestCrossEncoderReranker
# ===================================================================

class TestCrossEncoderReranker:

    def test_rerank_empty_results(self):
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_model_load_failure(self):
        reranker = CrossEncoderReranker()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Simulate _load_model failing by making import raise
            reranker._model = None
            with patch.object(reranker, "_load_model") as mock_load:
                mock_load.side_effect = lambda: setattr(reranker, "_model", None)
                results = [{"content": "some text"}]
                # _load_model sets _model to None on failure
                reranker._load_model()
                output = reranker.rerank("query", results)
                # When model is None, returns original results unchanged
                assert output == results

    def test_rerank_success(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.6]
        reranker._model = mock_model

        results = [
            {"content": "low score text"},
            {"content": "high score text"},
            {"content": "mid score text"},
        ]
        output = reranker.rerank("query", results, top_k=3)
        assert len(output) == 3
        # Should be sorted by rerank_score descending
        assert output[0]["rerank_score"] == 0.9
        assert output[1]["rerank_score"] == 0.6
        assert output[2]["rerank_score"] == 0.3


# ===================================================================
# TestIntentAwareAlpha
# ===================================================================

class TestIntentAwareAlpha:

    def test_get_alpha_known_intent(self):
        assert IntentAwareAlpha.get_alpha("disease_query") == 0.6
        assert IntentAwareAlpha.get_alpha("drug_query") == 0.4
        assert IntentAwareAlpha.get_alpha("symptom_query") == 0.7

    def test_get_alpha_unknown_intent(self):
        assert IntentAwareAlpha.get_alpha("nonexistent_intent") == 0.5

    def test_get_alpha_none_intent(self):
        assert IntentAwareAlpha.get_alpha(None) == 0.5


# ===================================================================
# TestHybridRetriever
# ===================================================================

class TestHybridRetriever:

    def _make_retriever(self, **overrides):
        with patch("src.retrieval.hybrid.VectorRetriever"), \
             patch("src.retrieval.hybrid.GraphRetriever"), \
             patch("src.retrieval.hybrid.CrossEncoderReranker"), \
             patch("src.retrieval.hybrid.cached", side_effect=lambda c: (lambda f: f)):
            return HybridRetriever(**overrides)

    def test_init_defaults(self):
        hr = self._make_retriever()
        assert hr.alpha == 0.5
        assert hr.vector_top_k == 5
        assert hr.graph_top_k == 10
        assert hr.enable_parallel is True
        assert hr.enable_cross_encoder is True
        assert hr._closed is False

    def test_combine_results(self):
        hr = self._make_retriever()
        hr.alpha = 0.5
        vector_results = [
            {"chunk_id": "c1", "similarity": 0.9, "document_id": "d1"},
            {"chunk_id": "c2", "similarity": 0.7, "document_id": "d2"},
        ]
        graph_results = [
            {"chunk_id": "c3", "entity_score": 0.8, "document_id": "d3"},
        ]
        combined = hr._combine_results(vector_results, graph_results)
        assert len(combined) == 3
        # All should have combined_score
        for item in combined:
            assert "combined_score" in item

    def test_combine_results_with_overlap(self):
        hr = self._make_retriever()
        hr.alpha = 0.5
        vector_results = [
            {"chunk_id": "c1", "similarity": 0.9, "document_id": "d1"},
        ]
        graph_results = [
            {"chunk_id": "c1", "entity_score": 0.8, "document_id": "d1"},
        ]
        combined = hr._combine_results(vector_results, graph_results)
        # c1 appears in both – should be deduplicated but graph_score boosted
        assert len(combined) == 1
        assert combined[0]["graph_score"] == 0.8
        # combined_score should include both vector and graph contributions
        assert combined[0]["vector_score"] > 0
        assert combined[0]["graph_score"] > 0

    def test_close(self):
        hr = self._make_retriever()
        assert hr._closed is False
        hr.close()
        assert hr._closed is True


# ===================================================================
# TestINTENT_ALPHA_MAP
# ===================================================================

class TestINTENT_ALPHA_MAP:

    def test_all_intents_have_alpha_values(self):
        expected_intents = [
            "disease_query", "drug_query", "drug_interaction",
            "symptom_query", "diagnosis_assist", "treatment_query",
            "examination_query", "prevention_query", "health_advice",
            "medical_knowledge", "unknown",
        ]
        for intent in expected_intents:
            assert intent in INTENT_ALPHA_MAP, f"Missing intent: {intent}"
            assert 0.0 <= INTENT_ALPHA_MAP[intent] <= 1.0, f"Alpha out of range for {intent}"
