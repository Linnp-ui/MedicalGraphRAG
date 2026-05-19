"""Prometheus-compatible metrics collector.

Provides comprehensive metrics for monitoring:
- Request rate, latency, error rate per endpoint
- LLM call metrics (duration, success/failure, fallback rate)
- Cache hit/miss rates
- Neo4j query performance
- Circuit breaker state
- System resource usage (memory, uptime)

Usage:
    from ..core.metrics import get_metrics
    metrics = get_metrics()
    metrics.increment("http_requests_total", {"method": "GET", "path": "/api/v1/query"})
    metrics.observe("http_request_duration_ms", 150.5, {"path": "/api/v1/query"})
    prometheus_text = metrics.export_prometheus()
"""

from functools import wraps
import time
import threading
from typing import Callable, Optional, Dict, List, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class MetricValue:
    """Single metric observation"""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._counters: Dict[str, Dict[str, float]] = {}
                    cls._instance._gauges: Dict[str, Dict[str, float]] = {}
                    cls._instance._histograms: Dict[str, List[MetricValue]] = {}
                    cls._instance._lock = threading.Lock()
        return cls._instance

    def increment(self, metric: str, labels: Optional[dict] = None, value: float = 1.0) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._labels_to_key(labels or {})
            if metric not in self._counters:
                self._counters[metric] = {}
            if key not in self._counters[metric]:
                self._counters[metric][key] = 0.0
            self._counters[metric][key] += value

    def gauge(self, metric: str, value: float, labels: Optional[dict] = None) -> None:
        """Set a gauge metric"""
        with self._lock:
            key = self._labels_to_key(labels or {})
            if metric not in self._gauges:
                self._gauges[metric] = {}
            self._gauges[metric][key] = value

    def observe(self, metric: str, value: float, labels: Optional[dict] = None) -> None:
        """Record an observation for histogram metric"""
        with self._lock:
            if metric not in self._histograms:
                self._histograms[metric] = []
            self._histograms[metric].append(
                MetricValue(value=value, labels=labels or {})
            )
            # Keep last 10000 observations per metric to limit memory
            if len(self._histograms[metric]) > 10000:
                self._histograms[metric] = self._histograms[metric][-10000:]

    def _labels_to_key(self, labels: dict) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def _parse_key_labels(self, key: str) -> Dict[str, str]:
        if not key:
            return {}
        labels = {}
        for part in key.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip('"')
        return labels

    def get_metrics(self) -> dict:
        """Get all metrics as dict (for backward compatibility)"""
        with self._lock:
            result = {
                "requests_total": dict(self._counters.get("http_requests_total", {})),
                "errors_total": dict(self._counters.get("http_errors_total", {})),
                "requests_duration": {},
                "neo4j_pool": {},
            }

            for metric, observations in self._histograms.items():
                if observations:
                    values = [o.value for o in observations]
                    result["requests_duration"][metric] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                    }

            return result

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format"""
        lines = []
        lines.append("# HELP medicalgraph_http_requests_total Total HTTP requests")
        lines.append("# TYPE medicalgraph_http_requests_total counter")

        with self._lock:
            for metric, values in self._counters.items():
                prom_name = self._to_prometheus_name(metric)
                prom_type = "counter"
                lines.append(f"# HELP medicalgraph_{prom_name} {metric}")
                lines.append(f"# TYPE medicalgraph_{prom_name} counter")
                for labels, value in values.items():
                    label_str = f"{{{labels}}}" if labels else ""
                    lines.append(f"medicalgraph_{prom_name}{label_str} {value}")

            for metric, values in self._gauges.items():
                prom_name = self._to_prometheus_name(metric)
                lines.append(f"# HELP medicalgraph_{prom_name} {metric}")
                lines.append(f"# TYPE medicalgraph_{prom_name} gauge")
                for labels, value in values.items():
                    label_str = f"{{{labels}}}" if labels else ""
                    lines.append(f"medicalgraph_{prom_name}{label_str} {value}")

            for metric, observations in self._histograms.items():
                if not observations:
                    continue
                prom_name = self._to_prometheus_name(metric)
                values = [o.value for o in observations]

                lines.append(f"# HELP medicalgraph_{prom_name}_total {metric} sum")
                lines.append(f"# TYPE medicalgraph_{prom_name}_total counter")
                lines.append(f"medicalgraph_{prom_name}_total {sum(values):.3f}")

                lines.append(f"# HELP medicalgraph_{prom_name}_count {metric} count")
                lines.append(f"# TYPE medicalgraph_{prom_name}_count counter")
                lines.append(f"medicalgraph_{prom_name}_count {len(values)}")

                lines.append(f"# HELP medicalgraph_{prom_name} {metric} histogram")
                lines.append(f"# TYPE medicalgraph_{prom_name} histogram")

                buckets = self._compute_histogram_buckets(values)
                cumulative = 0
                for boundary, count in buckets:
                    cumulative += count
                    lines.append(
                        f"medicalgraph_{prom_name}_bucket{{le=\"{boundary}\"}} {cumulative}"
                    )
                lines.append(f"medicalgraph_{prom_name}_bucket{{le=\"+Inf\"}} {len(values)}")

        return "\n".join(lines)

    def _to_prometheus_name(self, name: str) -> str:
        return name.replace(" ", "_").replace("-", "_").replace(".", "_")

    def _compute_histogram_buckets(self, values: List[float]) -> List[tuple]:
        boundaries = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000]
        buckets = []
        for boundary in boundaries:
            count = sum(1 for v in values if v <= boundary)
            buckets.append((str(boundary), count))
        return buckets

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics


def track_request(metric_name: str = "http_requests_total"):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                _metrics.increment(metric_name, {"status": "success"})
                return result
            except Exception as e:
                _metrics.increment(metric_name, {"status": "error"})
                raise
            finally:
                duration = (time.perf_counter() - start) * 1000
                _metrics.observe("http_request_duration_ms", duration)

        return wrapper

    return decorator


@contextmanager
def track_duration(metric: str, labels: Optional[dict] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000
        _metrics.observe(metric, duration, labels)


class MetricsMiddleware:
    """FastAPI middleware for automatic HTTP metrics collection"""

    def __init__(self):
        self._metrics = get_metrics()

    def record_request(self, method: str, path: str, status_code: int, duration_ms: float):
        labels = {"method": method, "path": path, "status": str(status_code)}
        self._metrics.increment("http_requests_total", labels)

        if status_code >= 400:
            self._metrics.increment("http_errors_total", {"method": method, "path": path, "status_code": str(status_code)})

        self._metrics.observe("http_request_duration_ms", duration_ms, {"method": method, "path": path})

    def record_llm_call(self, duration_ms: float, status: str, model: str = ""):
        labels = {"status": status}
        if model:
            labels["model"] = model
        self._metrics.increment("llm_calls_total", labels)
        self._metrics.observe("llm_call_duration_ms", duration_ms, labels)

    def record_cache_access(self, hit: bool, cache_type: str = "query"):
        status = "hit" if hit else "miss"
        self._metrics.increment("cache_access_total", {"cache": cache_type, "status": status})

    def record_neo4j_query(self, duration_ms: float, query_type: str = "unknown"):
        self._metrics.observe("neo4j_query_duration_ms", duration_ms, {"query_type": query_type})
        self._metrics.increment("neo4j_queries_total", {"query_type": query_type})

    def record_circuit_breaker(self, name: str, state: str):
        self._metrics.gauge("circuit_breaker_state", 1 if state == "OPEN" else 0, {"name": name})


_metrics_middleware = MetricsMiddleware()


def get_metrics_middleware() -> MetricsMiddleware:
    return _metrics_middleware


__all__ = [
    "MetricsCollector",
    "get_metrics",
    "track_request",
    "track_duration",
    "MetricsMiddleware",
    "get_metrics_middleware",
]
