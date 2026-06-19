"""Unit tests for error_collector module."""

import time
import hashlib
import pytest

from src.core.error_collector import (
    ErrorLog,
    ErrorLogCollector,
    get_error_collector,
    record_error,
)


# ---------------------------------------------------------------------------
# Helper: reset singleton before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure a fresh ErrorLogCollector for each test."""
    ErrorLogCollector._instance = None
    yield
    ErrorLogCollector._instance = None


# ---------------------------------------------------------------------------
# TestErrorLog
# ---------------------------------------------------------------------------

class TestErrorLog:
    def test_to_dict(self):
        now = time.time()
        log = ErrorLog(
            error_id="err_abc123",
            error_type="ValueError",
            message="bad value",
            stack_trace="traceback...",
            source="backend",
            severity="error",
            request_id="req_1",
            user_id="user_1",
            session_id="sess_1",
            url="/api/test",
            user_agent="pytest",
            extra={"key": "val"},
            timestamp=now,
            count=2,
            first_seen=now,
            last_seen=now,
        )
        d = log.to_dict()
        assert d["error_id"] == "err_abc123"
        assert d["error_type"] == "ValueError"
        assert d["message"] == "bad value"
        assert d["stack_trace"] == "traceback..."
        assert d["source"] == "backend"
        assert d["severity"] == "error"
        assert d["count"] == 2
        assert d["extra"] == {"key": "val"}


# ---------------------------------------------------------------------------
# TestErrorLogCollector
# ---------------------------------------------------------------------------

class TestErrorLogCollector:
    def test_record_error_returns_id(self):
        collector = ErrorLogCollector()
        error_id = collector.record_error("TypeError", "type mismatch")
        assert error_id.startswith("err_")

    def test_record_error_aggregates_same_fingerprint(self):
        collector = ErrorLogCollector()
        id1 = collector.record_error("TypeError", "type mismatch")
        id2 = collector.record_error("TypeError", "type mismatch")
        # Same fingerprint -> same error_id
        assert id1 == id2
        errors = collector.get_errors()
        assert len(errors) == 1
        assert errors[0]["count"] == 2

    def test_get_errors_no_filter(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a")
        collector.record_error("ValueError", "b")
        errors = collector.get_errors()
        assert len(errors) == 2

    def test_get_errors_filter_by_source(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a", source="backend")
        collector.record_error("TypeError", "b", source="frontend")
        errors = collector.get_errors(source="frontend")
        assert len(errors) == 1
        assert errors[0]["source"] == "frontend"

    def test_get_errors_filter_by_type(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a")
        collector.record_error("ValueError", "b")
        errors = collector.get_errors(error_type="ValueError")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ValueError"

    def test_get_errors_filter_by_severity(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a", severity="warning")
        collector.record_error("TypeError", "b", severity="critical")
        errors = collector.get_errors(severity="critical")
        assert len(errors) == 1
        assert errors[0]["severity"] == "critical"

    def test_get_errors_pagination(self):
        collector = ErrorLogCollector()
        for i in range(5):
            collector.record_error("TypeError", f"error_{i}")
        page1 = collector.get_errors(limit=2, offset=0)
        page2 = collector.get_errors(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should not overlap
        ids1 = {e["error_id"] for e in page1}
        ids2 = {e["error_id"] for e in page2}
        assert ids1.isdisjoint(ids2)

    def test_get_error_by_id_found(self):
        collector = ErrorLogCollector()
        error_id = collector.record_error("TypeError", "found me")
        result = collector.get_error_by_id(error_id)
        assert result is not None
        assert result["error_id"] == error_id

    def test_get_error_by_id_not_found(self):
        collector = ErrorLogCollector()
        result = collector.get_error_by_id("nonexistent_id")
        assert result is None

    def test_get_stats(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a", source="backend", severity="error")
        collector.record_error("ValueError", "b", source="frontend", severity="warning")
        stats = collector.get_stats()
        assert stats["total_errors"] == 2
        assert stats["unique_errors"] == 2
        assert stats["errors_by_type"]["TypeError"] == 1
        assert stats["errors_by_source"]["backend"] == 1
        assert stats["errors_by_severity"]["error"] == 1

    def test_get_top_errors(self):
        collector = ErrorLogCollector()
        # Record same error 5 times
        for _ in range(5):
            collector.record_error("TypeError", "frequent")
        # Record another error once
        collector.record_error("ValueError", "rare")
        top = collector.get_top_errors(limit=2)
        assert len(top) == 2
        assert top[0]["count"] == 5
        assert top[0]["error_type"] == "TypeError"
        assert top[1]["count"] == 1

    def test_clear_errors_all(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "a")
        collector.record_error("ValueError", "b")
        count = collector.clear_errors()
        assert count == 2
        assert collector.get_errors() == []

    def test_clear_errors_before_timestamp(self):
        collector = ErrorLogCollector()
        collector.record_error("TypeError", "old")
        collector.record_error("ValueError", "new")
        # Manually set last_seen to control timestamps precisely
        fingerprints = list(collector._errors.keys())
        collector._errors[fingerprints[0]].last_seen = 1000.0  # old
        collector._errors[fingerprints[1]].last_seen = 2000.0  # newer
        # Clear errors before 1500 — should remove only the first one
        cleared = collector.clear_errors(before=1500.0)
        assert cleared == 1
        remaining = collector.get_errors()
        assert len(remaining) == 1
        assert remaining[0]["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# TestConvenienceFunctions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_record_error_convenience(self):
        error_id = record_error("TypeError", "convenience test")
        assert error_id.startswith("err_")
        collector = get_error_collector()
        found = collector.get_error_by_id(error_id)
        assert found is not None
        assert found["error_type"] == "TypeError"
