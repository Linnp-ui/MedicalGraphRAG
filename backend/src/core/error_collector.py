"""Error log collector for aggregating and analyzing errors.

Provides centralized error collection, aggregation, and analysis capabilities
for both backend and frontend errors.
"""

import time
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from loguru import logger


@dataclass
class ErrorLog:
    """Represents a single error log entry."""
    error_id: str
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    source: str = "backend"
    severity: str = "error"
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorLogCollector:
    """Collects, aggregates, and analyzes error logs.

    Features:
    - Error aggregation by fingerprint (type + message hash)
    - In-memory storage with size limits
    - Error frequency tracking
    - Severity classification
    - Thread-safe operations
    """

    _instance: Optional["ErrorLogCollector"] = None
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

        self._errors: Dict[str, ErrorLog] = {}
        self._error_lock = threading.Lock()
        self._max_errors = 1000
        self._retention_hours = 24
        self._stats = {
            "total_errors": 0,
            "errors_by_type": defaultdict(int),
            "errors_by_source": defaultdict(int),
            "errors_by_severity": defaultdict(int),
        }

    def _generate_fingerprint(self, error_type: str, message: str) -> str:
        """Generate a unique fingerprint for error aggregation."""
        content = f"{error_type}:{message}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        import uuid
        return f"err_{uuid.uuid4().hex[:12]}"

    def record_error(
        self,
        error_type: str,
        message: str,
        stack_trace: Optional[str] = None,
        source: str = "backend",
        severity: str = "error",
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        url: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an error and return its ID.

        Args:
            error_type: Type of error (e.g., "TypeError", "NetworkError")
            message: Error message
            stack_trace: Stack trace if available
            source: Source of error ("backend" or "frontend")
            severity: Severity level ("debug", "info", "warning", "error", "critical")
            request_id: Associated request ID
            user_id: User ID if available
            session_id: Session ID if available
            url: URL where error occurred
            user_agent: User agent string
            extra: Additional context

        Returns:
            Error ID
        """
        fingerprint = self._generate_fingerprint(error_type, message)

        with self._error_lock:
            now = time.time()

            if fingerprint in self._errors:
                existing = self._errors[fingerprint]
                existing.count += 1
                existing.last_seen = now
                if stack_trace and not existing.stack_trace:
                    existing.stack_trace = stack_trace
                if extra:
                    existing.extra.update(extra)
                error_id = existing.error_id
            else:
                error_id = self._generate_error_id()
                error_log = ErrorLog(
                    error_id=error_id,
                    error_type=error_type,
                    message=message,
                    stack_trace=stack_trace,
                    source=source,
                    severity=severity,
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    url=url,
                    user_agent=user_agent,
                    extra=extra or {},
                    timestamp=now,
                    count=1,
                    first_seen=now,
                    last_seen=now,
                )
                self._errors[fingerprint] = error_log

            self._stats["total_errors"] += 1
            self._stats["errors_by_type"][error_type] += 1
            self._stats["errors_by_source"][source] += 1
            self._stats["errors_by_severity"][severity] += 1

            if len(self._errors) > self._max_errors:
                self._evict_old_errors()

        log_method = getattr(logger, severity, logger.error)
        log_method(
            f"[ErrorCollector] {source} error: [{error_type}] {message}",
            extra={
                "error_id": error_id,
                "fingerprint": fingerprint,
                "request_id": request_id,
                "user_id": user_id,
            }
        )

        return error_id

    def _evict_old_errors(self):
        """Remove old errors to stay within size limits."""
        cutoff = time.time() - (self._retention_hours * 3600)
        to_remove = [
            fp for fp, err in self._errors.items()
            if err.last_seen < cutoff
        ]
        for fp in to_remove:
            del self._errors[fp]

        if len(self._errors) > self._max_errors:
            sorted_errors = sorted(
                self._errors.items(),
                key=lambda x: x[1].last_seen
            )
            for fp, _ in sorted_errors[:len(self._errors) - self._max_errors]:
                del self._errors[fp]

    def get_errors(
        self,
        source: Optional[str] = None,
        error_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get filtered error logs.

        Args:
            source: Filter by source
            error_type: Filter by error type
            severity: Filter by severity
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of error logs
        """
        with self._error_lock:
            errors = list(self._errors.values())

        if source:
            errors = [e for e in errors if e.source == source]
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]
        if severity:
            errors = [e for e in errors if e.severity == severity]

        errors.sort(key=lambda e: e.last_seen, reverse=True)

        return [e.to_dict() for e in errors[offset:offset + limit]]

    def get_error_by_id(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific error by ID."""
        with self._error_lock:
            for error in self._errors.values():
                if error.error_id == error_id:
                    return error.to_dict()
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._error_lock:
            return {
                "total_errors": self._stats["total_errors"],
                "unique_errors": len(self._errors),
                "errors_by_type": dict(self._stats["errors_by_type"]),
                "errors_by_source": dict(self._stats["errors_by_source"]),
                "errors_by_severity": dict(self._stats["errors_by_severity"]),
            }

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top errors by frequency."""
        with self._error_lock:
            sorted_errors = sorted(
                self._errors.values(),
                key=lambda e: e.count,
                reverse=True
            )
        return [e.to_dict() for e in sorted_errors[:limit]]

    def clear_errors(self, before: Optional[float] = None) -> int:
        """Clear errors, optionally only those before a timestamp.

        Args:
            before: Unix timestamp, clear errors before this time

        Returns:
            Number of errors cleared
        """
        with self._error_lock:
            if before is None:
                count = len(self._errors)
                self._errors.clear()
                return count

            to_remove = [
                fp for fp, err in self._errors.items()
                if err.last_seen < before
            ]
            for fp in to_remove:
                del self._errors[fp]
            return len(to_remove)


def get_error_collector() -> ErrorLogCollector:
    """Get the singleton error collector instance."""
    return ErrorLogCollector()


def record_error(
    error_type: str,
    message: str,
    **kwargs
) -> str:
    """Convenience function to record an error."""
    return get_error_collector().record_error(
        error_type=error_type,
        message=message,
        **kwargs
    )
