"""Unit tests for metrics module."""

import pytest

from src.core.metrics import (
    MetricValue,
    MetricsCollector,
    track_request,
    track_duration,
    MetricsMiddleware,
)
import src.core.metrics as metrics_module


# ---------------------------------------------------------------------------
# Helper: reset singleton before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_metrics_singleton():
    """Ensure a fresh MetricsCollector for each test."""
    MetricsCollector._instance = None
    metrics_module._metrics = MetricsCollector()
    yield
    MetricsCollector._instance = None


# ---------------------------------------------------------------------------
# TestMetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_increment(self):
        mc = MetricsCollector()
        mc.increment("http_requests_total")
        assert mc._counters["http_requests_total"][""] == 1.0

    def test_increment_with_labels(self):
        mc = MetricsCollector()
        mc.increment("http_requests_total", {"method": "GET", "path": "/api"})
        key = mc._labels_to_key({"method": "GET", "path": "/api"})
        assert mc._counters["http_requests_total"][key] == 1.0

    def test_gauge(self):
        mc = MetricsCollector()
        mc.gauge("temperature", 23.5)
        assert mc._gauges["temperature"][""] == 23.5

    def test_observe(self):
        mc = MetricsCollector()
        mc.observe("latency_ms", 100.0)
        mc.observe("latency_ms", 200.0)
        assert len(mc._histograms["latency_ms"]) == 2
        assert mc._histograms["latency_ms"][0].value == 100.0
        assert mc._histograms["latency_ms"][1].value == 200.0

    def test_get_metrics(self):
        mc = MetricsCollector()
        mc.increment("http_requests_total", {"method": "GET"})
        mc.observe("http_request_duration_ms", 150.0)
        result = mc.get_metrics()
        assert "requests_total" in result
        assert "requests_duration" in result

    def test_export_prometheus_format(self):
        mc = MetricsCollector()
        mc.increment("http_requests_total", {"method": "GET"})
        mc.gauge("cpu_usage", 0.75)
        mc.observe("latency_ms", 50.0)
        output = mc.export_prometheus()
        assert "# TYPE" in output
        assert "medicalgraph_" in output

    def test_labels_to_key(self):
        mc = MetricsCollector()
        assert mc._labels_to_key({}) == ""
        key = mc._labels_to_key({"method": "GET", "path": "/api"})
        assert 'method="GET"' in key
        assert 'path="/api"' in key

    def test_parse_key_labels(self):
        mc = MetricsCollector()
        assert mc._parse_key_labels("") == {}
        labels = mc._parse_key_labels('method="GET",path="/api"')
        assert labels == {"method": "GET", "path": "/api"}

    def test_prometheus_name(self):
        mc = MetricsCollector()
        assert mc._to_prometheus_name("http requests-total.rate") == "http_requests_total_rate"

    def test_histogram_buckets(self):
        mc = MetricsCollector()
        values = [3, 7, 15, 60, 200]
        buckets = mc._compute_histogram_buckets(values)
        # bucket le=5 -> values <=5: [3] -> count=1
        assert buckets[0] == ("5", 1)
        # bucket le=10 -> values <=10: [3,7] -> count=2
        assert buckets[1] == ("10", 2)
        # bucket le=25 -> values <=25: [3,7,15] -> count=3
        assert buckets[2] == ("25", 3)

    def test_reset(self):
        mc = MetricsCollector()
        mc.increment("test_counter")
        mc.gauge("test_gauge", 1.0)
        mc.observe("test_hist", 10.0)
        mc.reset()
        assert mc._counters == {}
        assert mc._gauges == {}
        assert mc._histograms == {}


# ---------------------------------------------------------------------------
# TestTrackRequest
# ---------------------------------------------------------------------------

class TestTrackRequest:
    def test_success(self):
        mc = MetricsCollector()

        @track_request("test_requests")
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"
        key = mc._labels_to_key({"status": "success"})
        assert mc._counters["test_requests"][key] == 1.0

    def test_error_propagates(self):
        mc = MetricsCollector()

        @track_request("test_requests")
        def my_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            my_func()

        key = mc._labels_to_key({"status": "error"})
        assert mc._counters["test_requests"][key] == 1.0


# ---------------------------------------------------------------------------
# TestTrackDuration
# ---------------------------------------------------------------------------

class TestTrackDuration:
    def test_records_duration(self):
        mc = MetricsCollector()
        with track_duration("test_duration_ms"):
            pass  # minimal work
        assert "test_duration_ms" in mc._histograms
        assert len(mc._histograms["test_duration_ms"]) == 1


# ---------------------------------------------------------------------------
# TestMetricsMiddleware
# ---------------------------------------------------------------------------

class TestMetricsMiddleware:
    def test_record_request(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_request("GET", "/api/query", 200, 50.0)
        key = mc._labels_to_key({"method": "GET", "path": "/api/query", "status": "200"})
        assert mc._counters["http_requests_total"][key] == 1.0
        assert "http_request_duration_ms" in mc._histograms

    def test_record_request_error(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_request("POST", "/api/fail", 500, 200.0)
        # Should also increment http_errors_total
        err_key = mc._labels_to_key({"method": "POST", "path": "/api/fail", "status_code": "500"})
        assert mc._counters["http_errors_total"][err_key] == 1.0

    def test_record_llm_call(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_llm_call(300.0, "success", "gpt-4")
        key = mc._labels_to_key({"status": "success", "model": "gpt-4"})
        assert mc._counters["llm_calls_total"][key] == 1.0
        assert "llm_call_duration_ms" in mc._histograms

    def test_record_cache_access(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_cache_access(hit=True, cache_type="query")
        key = mc._labels_to_key({"cache": "query", "status": "hit"})
        assert mc._counters["cache_access_total"][key] == 1.0

    def test_record_neo4j_query(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_neo4j_query(120.0, "cypher")
        key = mc._labels_to_key({"query_type": "cypher"})
        assert mc._counters["neo4j_queries_total"][key] == 1.0
        assert "neo4j_query_duration_ms" in mc._histograms

    def test_record_circuit_breaker(self):
        mc = MetricsCollector()
        mw = MetricsMiddleware()
        mw.record_circuit_breaker("my_service", "OPEN")
        key = mc._labels_to_key({"name": "my_service"})
        assert mc._gauges["circuit_breaker_state"][key] == 1

        mw.record_circuit_breaker("my_service", "CLOSED")
        assert mc._gauges["circuit_breaker_state"][key] == 0
