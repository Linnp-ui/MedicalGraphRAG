"""Concurrency control middleware for request queueing and backpressure"""

import asyncio
import time
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from loguru import logger


class ConcurrencyLimiter:
    """Request queue with semaphore-based concurrency control"""
    
    def __init__(
        self,
        max_concurrent: int = 100,
        max_queue_size: int = 500,
        queue_timeout: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._active_requests = 0
        self._total_requests = 0
        self._rejected_requests = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire a slot for processing.
        
        Returns:
            True if acquired, False if queue is full
        """
        async with self._lock:
            self._total_requests += 1
            if self._queue.full():
                self._rejected_requests += 1
                return False
        
        try:
            # Wait for queue slot with timeout
            await asyncio.wait_for(self._queue.put(None), timeout=self.queue_timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._rejected_requests += 1
            return False
        
        # Now wait for semaphore (actual processing slot)
        await self._semaphore.acquire()
        
        async with self._lock:
            self._active_requests += 1
        
        return True
    
    async def release(self) -> None:
        """Release the processing slot"""
        self._semaphore.release()
        self._queue.get_nowait()
        self._queue.task_done()
        
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
    
    def get_stats(self) -> dict:
        """Get current concurrency stats"""
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self._active_requests,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "rejection_rate": (
                self._rejected_requests / self._total_requests 
                if self._total_requests > 0 else 0
            ),
        }


# Global concurrency limiter instance
_concurrency_limiter: Optional[ConcurrencyLimiter] = None


def get_concurrency_limiter() -> ConcurrencyLimiter:
    """Get or create global concurrency limiter"""
    global _concurrency_limiter
    if _concurrency_limiter is None:
        _concurrency_limiter = ConcurrencyLimiter(
            max_concurrent=100,
            max_queue_size=500,
            queue_timeout=30.0,
        )
    return _concurrency_limiter


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce concurrency limits and queue requests"""
    
    def __init__(self, app, max_concurrent: int = 100, max_queue_size: int = 500):
        super().__init__(app)
        self.limiter = ConcurrencyLimiter(
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
        )
    
    async def dispatch(self, request: Request, call_next):
        # Skip health checks and static files
        if request.url.path in ["/api/v1/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Try to acquire concurrency slot
        acquired = await self.limiter.acquire()
        
        if not acquired:
            stats = self.limiter.get_stats()
            logger.warning(
                f"Request rejected - queue full: {stats}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "Too many concurrent requests, please retry",
                    "retry_after": 5,
                    "queue_stats": stats,
                },
                headers={"Retry-After": "5"},
            )
        
        start_time = time.time()
        try:
            response = await call_next(request)
            return response
        finally:
            await self.limiter.release()
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Request processed in {duration_ms:.1f}ms")


async def get_concurrency_stats() -> dict:
    """Get current concurrency statistics"""
    limiter = get_concurrency_limiter()
    return limiter.get_stats()