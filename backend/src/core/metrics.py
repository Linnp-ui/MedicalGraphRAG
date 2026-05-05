from functools import wraps
import time
import threading
from typing import Callable, Optional
from contextlib import contextmanager

from loguru import logger


class MetricsCollector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._metrics = {
                        "requests_total": {},
                        "requests_duration": {},
                        "errors_total": {},
                        "neo4j_pool": {"acquired": 0, "available": 0, "total": 0},
                    }
                    cls._instance._lock = threading.Lock()
        return cls._instance

    def increment(self, metric: str, labels: Optional[dict] = None) -> None:
        with self._lock:
            key = self._labels_to_key(labels or {})
            if metric not in self._metrics:
                self._metrics[metric] = {}
            if key not in self._metrics[metric]:
                self._metrics[metric][key] = 0
            self._metrics[metric][key] += 1

    def observe(self, metric: str, value: float, labels: Optional[dict] = None) -> None:
        with self._lock:
            key = self._labels_to_key(labels or {})
            if metric not in self._metrics:
                self._metrics[metric] = {}
            if key not in self._metrics[metric]:
                self._metrics[metric][key] = []
            if not isinstance(self._metrics[metric][key], list):
                self._metrics[metric][key] = []
            self._metrics[metric][key].append(value)

    def gauge(self, metric: str, value: float, labels: Optional[dict] = None) -> None:
        with self._lock:
            key = self._labels_to_key(labels or {})
            if metric not in self._metrics:
                self._metrics[metric] = {}
            self._metrics[metric][key] = value

    def _labels_to_key(self, labels: dict) -> str:
        if not labels:
            return "default"
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def get_metrics(self) -> dict:
        with self._lock:
            return dict(self._metrics)

    def reset(self) -> None:
        with self._lock:
            self._metrics = {
                "requests_total": {},
                "requests_duration": {},
                "errors_total": {},
                "neo4j_pool": {"acquired": 0, "available": 0, "total": 0},
            }

    def export_prometheus(self) -> str:
        lines = []
        with self._lock:
            for metric, values in self._metrics.items():
                if isinstance(values, dict):
                    for labels, value in values.items():
                        if isinstance(value, list):
                            avg = sum(value) / len(value) if value else 0
                            lines.append(f"{metric}{{{labels}}} {avg}")
                        else:
                            lines.append(f"{metric}{{{labels}}} {value}")
                elif isinstance(values, (int, float)):
                    lines.append(f"{metric} {values}")
        return "\n".join(lines)


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics


def track_request(metric_name: str = "requests_total"):
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
                _metrics.observe("request_duration_ms", duration)

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


__all__ = [
    "MetricsCollector",
    "get_metrics",
    "track_request",
    "track_duration",
]
