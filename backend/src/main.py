import uuid
import time
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from .api.routes import router
from .api.middleware import ErrorHandlingMiddleware, RequestTimingMiddleware
from .api.middleware.concurrency import ConcurrencyMiddleware
from .core.config import get_settings
from .utils.logger import set_request_id, log_request


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        set_request_id(request_id)

        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        log_request(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 120, cleanup_interval: int = 300):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: dict = defaultdict(list)
        self._endpoint_limits = {
            "/api/v1/query": 30,
            "/api/v1/graph/search": 60,
            "/api/v1/graph/data": 60,
        }
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval

    def _cleanup_stale_keys(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = now - 60
        stale = [k for k, v in self._requests.items() if not v or max(v) < cutoff]
        for k in stale:
            del self._requests[k]

    async def dispatch(self, request: Request, call_next):
        self._cleanup_stale_keys()

        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        rpm = self.requests_per_minute
        for pattern, limit in self._endpoint_limits.items():
            if endpoint.startswith(pattern):
                rpm = limit
                break
        
        key = f"{client_ip}:{endpoint}"

        now = time.time()
        self._requests[key] = [t for t in self._requests[key] if now - t < 60]

        if len(self._requests[key]) >= rpm:
            logger.warning(f"Rate limit exceeded for {key}")
            return Response(
                content='{"error": "Rate limit exceeded", "retry_after": 60}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

        self._requests[key].append(now)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("GraphRAG API starting up...")

    try:
        from .core.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        if client.verify_connectivity():
            logger.info("Neo4j connection pool pre-warmed")
        else:
            logger.warning("Neo4j not reachable at startup — will retry on first request")
    except Exception as e:
        logger.warning(f"Neo4j pre-warm failed: {e}")

    try:
        from .workflow.graph import get_workflow

        get_workflow()
        logger.info("LangGraph workflow pre-compiled")
    except Exception as e:
        logger.warning(f"Workflow pre-compile failed: {e}")

    yield

    logger.info("GraphRAG API shutting down...")
    try:
        from .core.neo4j_client import get_neo4j_client

        get_neo4j_client().close()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="GraphRAG API",
        description="Knowledge Graph based RAG system with LangChain and Neo4j",
        version="1.0.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Read allowed origins from env, fall back to permissive for dev
    cors_origins = settings.cors_origins if hasattr(settings, 'cors_origins') and settings.cors_origins else ["*"]
    allow_creds = settings.cors_allow_credentials if hasattr(settings, 'cors_allow_credentials') else False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(ConcurrencyMiddleware, max_concurrent=100, max_queue_size=500)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(router, prefix="/api/v1", tags=["GraphRAG"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        workers=4,
        loop="uvloop",
        http="httptools",
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=30,
    )
