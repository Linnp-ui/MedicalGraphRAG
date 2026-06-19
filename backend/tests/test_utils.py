"""Tests for src.utils.logger and src.utils.process_monitor"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from src.utils.logger import (
    get_request_id,
    set_request_id,
    RequestIdFilter,
)
from src.utils.process_monitor import (
    ProcessMonitor,
    track_process,
    track_process_context,
    StructuredLogger,
)


# ---------------------------------------------------------------------------
# TestRequestId
# ---------------------------------------------------------------------------
class TestRequestId:

    def test_get_set_request_id(self):
        set_request_id("req-123")
        assert get_request_id() == "req-123"
        # Clean up
        set_request_id(None)

    def test_default_none(self):
        set_request_id(None)
        assert get_request_id() is None


# ---------------------------------------------------------------------------
# TestRequestIdFilter
# ---------------------------------------------------------------------------
class TestRequestIdFilter:

    def test_filter_adds_request_id(self):
        filt = RequestIdFilter()
        record = {}
        set_request_id("req-456")
        result = filt(record)
        assert result is True
        assert record["request_id"] == "req-456"
        set_request_id(None)

    def test_filter_default_dash(self):
        filt = RequestIdFilter()
        record = {}
        set_request_id(None)
        filt(record)
        assert record["request_id"] == "-"


# ---------------------------------------------------------------------------
# TestProcessMonitor
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_monitor():
    """Reset singleton before each test."""
    ProcessMonitor._instance = None
    yield
    ProcessMonitor._instance = None


class TestProcessMonitor:

    def test_start_process(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("test_process")
        assert isinstance(pid, int)
        assert len(monitor.get_active_processes()) == 1

    def test_end_process_success(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("test_process")
        time.sleep(0.01)
        metrics = monitor.end_process(pid, status="success")
        assert metrics is not None
        assert metrics.status == "success"
        assert metrics.duration_ms > 0
        assert len(monitor.get_active_processes()) == 0

    def test_end_process_error(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("test_process")
        metrics = monitor.end_process(pid, status="error", error="something failed")
        assert metrics is not None
        assert metrics.status == "error"
        assert metrics.error == "something failed"

    def test_end_process_unknown_id(self):
        monitor = ProcessMonitor()
        metrics = monitor.end_process(999999, status="success")
        assert metrics is None

    def test_get_stats(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("stats_test")
        monitor.end_process(pid, status="success")
        stats = monitor.get_stats()
        assert "stats_test" in stats
        assert stats["stats_test"]["count"] == 1
        assert stats["stats_test"]["success"] == 1

    def test_get_stats_by_name(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("named_test")
        monitor.end_process(pid, status="success")
        stats = monitor.get_stats("named_test")
        assert stats["count"] == 1
        assert "avg_duration_ms" in stats

    def test_get_active_processes(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("active_test", extra={"key": "value"})
        active = monitor.get_active_processes()
        assert len(active) == 1
        assert active[0]["process_name"] == "active_test"
        monitor.end_process(pid)

    def test_clear_stats(self):
        monitor = ProcessMonitor()
        pid = monitor.start_process("clear_test")
        monitor.end_process(pid)
        monitor.clear_stats()
        stats = monitor.get_stats()
        assert stats == {}


# ---------------------------------------------------------------------------
# TestTrackProcess
# ---------------------------------------------------------------------------
class TestTrackProcess:

    @track_process("my_func")
    def _success_func(self):
        return 42

    @track_process("error_func")
    def _error_func(self):
        raise ValueError("boom")

    def test_decorator_success(self):
        result = self._success_func()
        assert result == 42
        monitor = ProcessMonitor()
        stats = monitor.get_stats("my_func")
        assert stats["success"] == 1

    def test_decorator_error_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            self._error_func()
        monitor = ProcessMonitor()
        stats = monitor.get_stats("error_func")
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# TestTrackProcessContext
# ---------------------------------------------------------------------------
class TestTrackProcessContext:

    def test_context_success(self):
        with track_process_context("ctx_test"):
            pass
        monitor = ProcessMonitor()
        stats = monitor.get_stats("ctx_test")
        assert stats["success"] == 1

    def test_context_error_propagates(self):
        with pytest.raises(RuntimeError, match="ctx error"):
            with track_process_context("ctx_error_test"):
                raise RuntimeError("ctx error")
        monitor = ProcessMonitor()
        stats = monitor.get_stats("ctx_error_test")
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# TestStructuredLogger
# ---------------------------------------------------------------------------
class TestStructuredLogger:

    @patch("src.utils.process_monitor.logger")
    def test_info_log(self, mock_logger):
        slog = StructuredLogger("test_module")
        slog.info("test_event", key1="val1")
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "[test_module] test_event" in call_args[0][0]

    @patch("src.utils.process_monitor.logger")
    def test_error_log(self, mock_logger):
        slog = StructuredLogger("test_module")
        slog.error("error_event", code=500)
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "[test_module] error_event" in call_args[0][0]

    @patch("src.utils.process_monitor.logger")
    def test_log_api_request(self, mock_logger):
        slog = StructuredLogger("api")
        slog.log_api_request("GET", "/health", 200, 12.5)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        msg = call_args[0][0]
        assert "api_request" in msg
        assert "GET" in msg
        assert "/health" in msg

    @patch("src.utils.process_monitor.logger")
    def test_log_workflow_step(self, mock_logger):
        slog = StructuredLogger("workflow")
        slog.log_workflow_step("ingestion", "split", "completed")
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        msg = call_args[0][0]
        assert "workflow_step" in msg
        assert "ingestion" in msg
