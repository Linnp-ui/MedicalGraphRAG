"""Tests for src.api.schemas (Pydantic model validation)"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    QuestionRequest,
    IngestRequest,
    QueryResultRequest,
    HybridSearchRequest,
    FeedbackRequest,
)


# ---------------------------------------------------------------------------
# TestQuestionRequest
# ---------------------------------------------------------------------------
class TestQuestionRequest:

    def test_valid(self):
        req = QuestionRequest(question="高血压的症状是什么？")
        assert req.question == "高血压的症状是什么？"

    def test_sanitize_strips_whitespace(self):
        req = QuestionRequest(question="  高血压  ")
        assert req.question == "高血压"

    def test_sanitize_removes_control_chars(self):
        req = QuestionRequest(question="高血压\x00\x01症状")
        assert req.question == "高血压症状"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRequest(question="")

    def test_too_long_question_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRequest(question="x" * 2001)


# ---------------------------------------------------------------------------
# TestIngestRequest
# ---------------------------------------------------------------------------
class TestIngestRequest:

    def test_valid_file_path(self):
        req = IngestRequest(file_path="/data/medical.pdf")
        assert req.file_path == "/data/medical.pdf"

    def test_valid_directory(self):
        req = IngestRequest(directory="/data/documents")
        assert req.directory == "/data/documents"

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError):
            IngestRequest(file_path="../../../etc/passwd")

    def test_none_paths_allowed(self):
        req = IngestRequest()
        assert req.file_path is None
        assert req.directory is None


# ---------------------------------------------------------------------------
# TestQueryResultRequest
# ---------------------------------------------------------------------------
class TestQueryResultRequest:

    def test_valid_node_ids(self):
        req = QueryResultRequest(query="test", node_ids=["1", "2", "3"])
        assert req.node_ids == ["1", "2", "3"]

    def test_empty_node_ids_rejected(self):
        with pytest.raises(ValidationError):
            QueryResultRequest(query="test", node_ids=[])

    def test_non_numeric_node_ids_rejected(self):
        with pytest.raises(ValidationError):
            QueryResultRequest(query="test", node_ids=["abc"])


# ---------------------------------------------------------------------------
# TestHybridSearchRequest
# ---------------------------------------------------------------------------
class TestHybridSearchRequest:

    def test_valid_alpha(self):
        req = HybridSearchRequest(query="test", alpha=0.5)
        assert req.alpha == 0.5

    def test_alpha_below_range_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", alpha=-0.1)

    def test_alpha_above_range_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", alpha=1.1)


# ---------------------------------------------------------------------------
# TestFeedbackRequest
# ---------------------------------------------------------------------------
class TestFeedbackRequest:

    def test_valid_rating(self):
        req = FeedbackRequest(question="q", answer="a", rating=3)
        assert req.rating == 3

    def test_rating_below_range_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(question="q", answer="a", rating=0)

    def test_rating_above_range_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(question="q", answer="a", rating=6)
