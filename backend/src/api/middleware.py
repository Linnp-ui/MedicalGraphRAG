import time
from typing import Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from ..utils.logger import get_request_id
from ..core.metrics import get_metrics_middleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            request_id = get_request_id() or "unknown"
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.exception(f"[{request_id}] Unhandled exception: {error_type}: {error_message}")
            
            try:
                from ..core.error_collector import get_error_collector
                collector = get_error_collector()
                collector.record_error(
                    error_type=error_type,
                    message=error_message,
                    source="backend",
                    severity="error",
                    request_id=request_id,
                    extra={
                        "path": str(request.url.path),
                        "method": request.method,
                        "query": str(request.url.query) if request.url.query else None,
                    }
                )
            except Exception:
                pass
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": str(e),
                },
            )


class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        metrics = get_metrics_middleware()
        metrics.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        if duration_ms > 1000:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {duration_ms:.2f}ms"
            )

        return response


__all__ = ["ErrorHandlingMiddleware", "RequestTimingMiddleware"]
