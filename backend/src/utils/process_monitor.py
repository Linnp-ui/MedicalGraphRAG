"""Process monitoring and structured logging for critical workflows.

Provides decorators and utilities for tracking execution time, status,
and performance metrics of key processes.
"""

import time
import functools
import threading
from typing import Optional, Dict, Any, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager

from loguru import logger

T = TypeVar("T")


@dataclass
class ProcessMetrics:
    """Metrics for a single process execution."""
    process_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_name": self.process_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "extra": self.extra,
        }


class ProcessMonitor:
    """Monitor for tracking process execution and performance.

    Features:
    - Track process execution time
    - Aggregate statistics by process name
    - Thread-safe operations
    - Integration with error collector
    """

    _instance: Optional["ProcessMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._active_processes: Dict[int, ProcessMetrics] = {}
        self._completed_processes: list = []
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_duration_ms": 0.0,
                "min_duration_ms": float("inf"),
                "max_duration_ms": 0.0,
                "errors": 0,
                "success": 0,
            }
        )
        self._process_lock = threading.Lock()
        self._max_completed = 1000

    def start_process(
        self,
        process_name: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Start tracking a process.

        Args:
            process_name: Name of the process
            extra: Additional context

        Returns:
            Process ID for tracking
        """
        process_id = threading.get_ident() + int(time.time() * 1000000)

        metrics = ProcessMetrics(
            process_name=process_name,
            start_time=time.time(),
            extra=extra or {},
        )

        with self._process_lock:
            self._active_processes[process_id] = metrics

        logger.debug(
            f"[ProcessMonitor] Started: {process_name}",
            extra={"process_id": process_id, **(extra or {})}
        )

        return process_id

    def end_process(
        self,
        process_id: int,
        status: str = "success",
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProcessMetrics]:
        """End tracking a process.

        Args:
            process_id: Process ID from start_process
            status: Final status (success, error, timeout)
            error: Error message if failed
            extra: Additional context

        Returns:
            Process metrics if found
        """
        with self._process_lock:
            metrics = self._active_processes.pop(process_id, None)

        if not metrics:
            logger.warning(f"[ProcessMonitor] Unknown process ID: {process_id}")
            return None

        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.status = status
        metrics.error = error
        if extra:
            metrics.extra.update(extra)

        with self._process_lock:
            stats = self._stats[metrics.process_name]
            stats["count"] += 1
            stats["total_duration_ms"] += metrics.duration_ms
            stats["min_duration_ms"] = min(stats["min_duration_ms"], metrics.duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], metrics.duration_ms)

            if status == "success":
                stats["success"] += 1
            else:
                stats["errors"] += 1

            self._completed_processes.append(metrics)
            if len(self._completed_processes) > self._max_completed:
                self._completed_processes.pop(0)

        log_level = "error" if status == "error" else "info"
        getattr(logger, log_level)(
            f"[ProcessMonitor] Completed: {metrics.process_name} | "
            f"Status: {status} | Duration: {metrics.duration_ms:.2f}ms",
            extra={"process_id": process_id, "duration_ms": metrics.duration_ms}
        )

        return metrics

    def get_stats(self, process_name: Optional[str] = None) -> Dict[str, Any]:
        """Get process statistics.

        Args:
            process_name: Filter by process name, or all if None

        Returns:
            Statistics dictionary
        """
        with self._process_lock:
            if process_name:
                stats = self._stats.get(process_name, {})
                if stats:
                    stats = dict(stats)
                    if stats["count"] > 0:
                        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                    return stats
                return {}

            result = {}
            for name, stats in self._stats.items():
                stats_copy = dict(stats)
                if stats_copy["count"] > 0:
                    stats_copy["avg_duration_ms"] = stats_copy["total_duration_ms"] / stats_copy["count"]
                result[name] = stats_copy
            return result

    def get_active_processes(self) -> list:
        """Get list of currently active processes."""
        with self._process_lock:
            return [m.to_dict() for m in self._active_processes.values()]

    def clear_stats(self) -> None:
        """Clear all statistics."""
        with self._process_lock:
            self._stats.clear()
            self._completed_processes.clear()


def get_process_monitor() -> ProcessMonitor:
    """Get the singleton process monitor instance."""
    return ProcessMonitor()


def track_process(process_name: Optional[str] = None):
    """Decorator to track process execution.

    Usage:
        @track_process("my_process")
        def my_function():
            ...

        @track_process()
        def another_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = process_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            monitor = get_process_monitor()
            process_id = monitor.start_process(name)

            try:
                result = func(*args, **kwargs)
                monitor.end_process(process_id, status="success")
                return result
            except Exception as e:
                monitor.end_process(
                    process_id,
                    status="error",
                    error=str(e),
                    extra={"exception_type": type(e).__name__}
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            monitor = get_process_monitor()
            process_id = monitor.start_process(name)

            try:
                result = await func(*args, **kwargs)
                monitor.end_process(process_id, status="success")
                return result
            except Exception as e:
                monitor.end_process(
                    process_id,
                    status="error",
                    error=str(e),
                    extra={"exception_type": type(e).__name__}
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def track_process_context(process_name: str, **extra):
    """Context manager for tracking process execution.

    Usage:
        with track_process_context("data_processing", items=100):
            process_data()
    """
    monitor = get_process_monitor()
    process_id = monitor.start_process(process_name, extra=extra)

    try:
        yield process_id
        monitor.end_process(process_id, status="success")
    except Exception as e:
        monitor.end_process(
            process_id,
            status="error",
            error=str(e),
            extra={"exception_type": type(e).__name__}
        )
        raise


class StructuredLogger:
    """Structured logger for consistent log formatting across processes.

    Provides methods for logging key events with structured context.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name

    def _log(self, level: str, event: str, **kwargs):
        """Internal logging method."""
        extra = {
            "module": self.module_name,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        message = f"[{self.module_name}] {event}"
        if kwargs:
            context_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message += f" | {context_str}"

        getattr(logger, level)(message, extra=extra)

    def debug(self, event: str, **kwargs):
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs):
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs):
        self._log("critical", event, **kwargs)

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        """Log API request with standard format."""
        self.info(
            "api_request",
            method=method,
            path=path,
            status=status_code,
            duration_ms=f"{duration_ms:.2f}",
            **kwargs
        )

    def log_workflow_step(
        self,
        workflow: str,
        step: str,
        status: str,
        **kwargs
    ):
        """Log workflow step execution."""
        self.info(
            "workflow_step",
            workflow=workflow,
            step=step,
            status=status,
            **kwargs
        )

    def log_data_operation(
        self,
        operation: str,
        entity: str,
        count: int,
        duration_ms: Optional[float] = None,
        **kwargs
    ):
        """Log data operation (ingestion, retrieval, etc.)."""
        extra = {
            "operation": operation,
            "entity": entity,
            "count": count,
            **kwargs
        }
        if duration_ms is not None:
            extra["duration_ms"] = f"{duration_ms:.2f}"

        self.info("data_operation", **extra)

    def log_cache_event(
        self,
        cache_name: str,
        event: str,
        key: Optional[str] = None,
        **kwargs
    ):
        """Log cache events (hit, miss, set, evict)."""
        extra = {
            "cache": cache_name,
            "event": event,
            **kwargs
        }
        if key:
            extra["key"] = key[:50] + "..." if len(key) > 50 else key

        self.debug("cache_event", **extra)

    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        **kwargs
    ):
        """Log error with full context."""
        self.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
            **kwargs
        )


def get_structured_logger(module_name: str) -> StructuredLogger:
    """Get a structured logger for a module."""
    return StructuredLogger(module_name)
