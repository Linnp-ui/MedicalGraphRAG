import sys
import logging
import threading
from typing import Optional
from contextvars import ContextVar

from loguru import logger

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


class RequestIdFilter:
    def __init__(self):
        self._request_id = ""

    def __call__(self, record):
        record["request_id"] = get_request_id() or "-"
        return True


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    serialize: bool = False,
    log_dir: str = "logs",
) -> None:
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{request_id}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        filter=RequestIdFilter(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        f"{log_dir}/graphrag_{time}.log",
        level=level,
        format=format_string.replace("<cyan>", "").replace("</cyan>", ""),
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=serialize,
        filter=RequestIdFilter(),
    )

    logger.add(
        f"{log_dir}/warnings.log",
        level="WARNING",
        format=format_string.replace("<cyan>", "").replace("</cyan>", ""),
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=serialize,
        filter=RequestIdFilter(),
    )


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: Optional[str] = None,
) -> None:
    logger.info(
        f"{method} {path} | Status: {status_code} | Duration: {duration_ms:.2f}ms",
        request_id=request_id or get_request_id() or "-",
    )


def log_error(
    error: Exception,
    context: Optional[dict] = None,
    request_id: Optional[str] = None,
) -> None:
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}",
        request_id=request_id or get_request_id() or "-",
        extra=context or {},
    )


__all__ = [
    "logger",
    "get_request_id",
    "set_request_id",
    "setup_logging",
    "log_request",
    "log_error",
    "RequestIdFilter",
]
