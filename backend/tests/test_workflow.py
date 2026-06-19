"""Tests for src.workflow (router, nodes, state)"""

import pytest
from unittest.mock import patch, MagicMock

from src.workflow.router import route_question, decompose_question
from src.workflow.nodes import (
    generate_answer,
    _format_graph_context,
    _format_document_context,
    handle_error,
)
from src.workflow.state import GraphState


# ---------------------------------------------------------------------------
# TestRouteQuestion
# ---------------------------------------------------------------------------
class TestRouteQuestion:

    @patch("src.workflow.router.DRIFTSearch")
    def test_route_global_query(self, mock_drift_cls):
        mock_drift = MagicMock()
        mock_drift._classify_query_intent.return_value = "global"
        mock_drift_cls.return_value = mock_drift

        state: GraphState = {
            "question": "整体趋势是什么",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": None,
        }
        result = route_question(state)
        assert result["routing"] == "global"

    @patch("src.workflow.router.DRIFTSearch")
    def test_route_local_query(self, mock_drift_cls):
        mock_drift = MagicMock()
        mock_drift._classify_query_intent.return_value = "local"
        mock_drift_cls.return_value = mock_drift

        state: GraphState = {
            "question": "高血压的症状是什么",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": None,
        }
        result = route_question(state)
        assert result["routing"] == "local"


# ---------------------------------------------------------------------------
# TestDecomposeQuestion
# ---------------------------------------------------------------------------
class TestDecomposeQuestion:

    def test_decompose_with_conjunction(self):
        state: GraphState = {
            "question": "高血压和糖尿病的症状",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": None,
        }
        result = decompose_question(state)
        assert len(result["subqueries"]) == 2
        assert "高血压" in result["subqueries"][0]
        assert "糖尿病" in result["subqueries"][1]

    def test_decompose_no_conjunction(self):
        state: GraphState = {
            "question": "高血压的症状",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": None,
        }
        result = decompose_question(state)
        assert len(result["subqueries"]) == 1
        assert result["subqueries"][0] == "高血压的症状"


# ---------------------------------------------------------------------------
# TestFormatGraphContext
# ---------------------------------------------------------------------------
class TestFormatGraphContext:

    def test_format_with_results(self):
        results = [
            {"source": "高血压", "relationship": "HAS_SYMPTOM", "target": "头痛"},
        ]
        output = _format_graph_context(results, "")
        assert "高血压" in output
        assert "头痛" in output

    def test_format_empty(self):
        output = _format_graph_context([], "")
        assert output == ""

    def test_format_with_cypher_query(self):
        results = [{"name": "test"}]
        output = _format_graph_context(results, "MATCH (n) RETURN n")
        assert "MATCH (n) RETURN n" in output

    def test_format_skips_private_keys(self):
        results = [{"__internal": "hidden", "name": "visible"}]
        output = _format_graph_context(results, "")
        assert "hidden" not in output
        assert "visible" in output


# ---------------------------------------------------------------------------
# TestFormatDocumentContext
# ---------------------------------------------------------------------------
class TestFormatDocumentContext:

    def test_format_with_documents(self):
        docs = [
            {"content": "高血压是常见疾病"},
            {"content": "糖尿病需要长期管理"},
        ]
        output = _format_document_context(docs)
        assert "高血压" in output
        assert "糖尿病" in output

    def test_format_empty(self):
        output = _format_document_context([])
        assert output == ""


# ---------------------------------------------------------------------------
# TestHandleError
# ---------------------------------------------------------------------------
class TestHandleError:

    def test_sets_error_answer(self):
        state: GraphState = {
            "question": "test",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": "Something went wrong",
        }
        result = handle_error(state)
        assert "error" in result["answer"].lower() or "Error" in result["answer"]


# ---------------------------------------------------------------------------
# TestGraphState
# ---------------------------------------------------------------------------
class TestGraphState:

    def test_state_fields(self):
        expected_keys = {
            "question", "documents", "entities", "cypher_query",
            "graph_result", "answer", "routing", "subqueries",
            "context", "history", "error",
        }
        state: GraphState = {
            "question": "test",
            "documents": [],
            "entities": [],
            "cypher_query": "",
            "graph_result": [],
            "answer": "",
            "routing": "",
            "subqueries": [],
            "context": "",
            "history": [],
            "error": None,
        }
        assert set(state.keys()) == expected_keys
