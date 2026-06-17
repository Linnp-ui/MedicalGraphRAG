import hashlib
import json
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Optional, Set, Tuple
from functools import wraps

from loguru import logger

from .config import get_settings


class CacheBackend(ABC):
    """Abstract cache backend interface"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600, tags: Optional[list[str]] = None):
        pass

    @abstractmethod
    def clear(self):
        pass

    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with the given tag"""
        pass


class LRUCache(CacheBackend):
    """Thread-safe LRU cache implementation with TTL support"""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self._lock:
            if key in self.cache:
                value, expires_at = self.cache[key]
                if time.time() > expires_at:
                    del self.cache[key]
                    return None

                self.cache.move_to_end(key)
                return value
            return None

    def set(self, key: str, value: Any, ttl: int = 3600, tags: Optional[list[str]] = None):
        """Set value in cache with TTL and optional tag indexing"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            expires_at = time.time() + ttl
            self.cache[key] = (value, expires_at)

            if tags:
                for tag in tags:
                    self._tag_index[tag].add(key)

            if len(self.cache) > self.max_size:
                evicted_key, _ = self.cache.popitem(last=False)
                for tag_set in self._tag_index.values():
                    tag_set.discard(evicted_key)

    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with the given tag"""
        with self._lock:
            keys = self._tag_index.pop(tag, set())
            for key in keys:
                self.cache.pop(key, None)
            if keys:
                logger.debug(f"LRU cache invalidated {len(keys)} entries for tag: {tag}")

    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self._tag_index.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key in self.cache:
                _, expires_at = self.cache[key]
                if time.time() <= expires_at:
                    return True
                else:
                    del self.cache[key]
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self.cache)


class RedisCache(CacheBackend):
    """Redis-based cache backend with serialization and tag index"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "graphrag",
        default_ttl: int = 3600,
    ):
        import redis

        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password if password else None,
            decode_responses=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
        self._available = self._check_connection()

    def _check_connection(self) -> bool:
        try:
            self._client.ping()
            logger.info("Redis connection established")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to in-memory cache: {e}")
            return False

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def _make_tag_key(self, tag: str) -> str:
        return f"{self.prefix}:tag:{tag}"

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None
        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600, tags: Optional[list[str]] = None):
        if not self._available:
            return
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            pipe = self._client.pipeline()
            pipe.setex(self._make_key(key), ttl, serialized)
            if tags:
                for tag in tags:
                    pipe.sadd(self._make_tag_key(tag), key)
            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def invalidate_by_tag(self, tag: str):
        if not self._available:
            return
        try:
            tag_key = self._make_tag_key(tag)
            keys = self._client.smembers(tag_key)
            if keys:
                full_keys = [self._make_key(k) for k in keys]
                full_keys.append(tag_key)
                self._client.delete(*full_keys)
                logger.info(f"Redis invalidated {len(keys)} entries for tag: {tag}")
        except Exception as e:
            logger.warning(f"Redis invalidate_by_tag failed: {e}")

    def clear(self):
        if not self._available:
            return
        try:
            keys = self._client.keys(f"{self.prefix}:*")
            if keys:
                self._client.delete(*keys)
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")


class CacheRouter:
    """Routes cache operations to Redis or in-memory backend based on availability"""

    def __init__(
        self,
        redis_cache: RedisCache,
        memory_cache: LRUCache,
    ):
        self._redis = redis_cache
        self._memory = memory_cache

    def get(self, key: str) -> Optional[Any]:
        if self._redis._available:
            value = self._redis.get(key)
            if value is not None:
                return value
        return self._memory.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600, tags: Optional[list[str]] = None):
        self._memory.set(key, value, ttl, tags=tags)
        if self._redis._available:
            self._redis.set(key, value, ttl, tags=tags)

    def invalidate_by_tag(self, tag: str):
        self._memory.invalidate_by_tag(tag)
        if self._redis._available:
            self._redis.invalidate_by_tag(tag)

    def clear(self):
        self._memory.clear()
        self._redis.clear()


class QueryCache:
    """Cache for query results"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        backend: Optional[CacheBackend] = None,
    ):
        settings = get_settings()
        if backend is not None:
            self.cache = backend
        elif settings.cache_backend == "redis":
            redis_cache = RedisCache(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                prefix=settings.redis_prefix,
                default_ttl=ttl,
            )
            memory_cache = LRUCache(max_size)
            self.cache = CacheRouter(redis_cache, memory_cache)
        else:
            self.cache = LRUCache(max_size)
        self.ttl = ttl

    def _generate_key(self, query: str, params: Optional[dict] = None) -> str:
        """Generate cache key from query and parameters"""
        key_data = {"query": query, "params": params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: Optional[dict] = None) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(query, params)
        return self.cache.get(key)

    def set(self, query: str, params: Optional[dict], value: Any, tags: Optional[list[str]] = None):
        """Cache a result with optional tag indexing"""
        key = self._generate_key(query, params)
        self.cache.set(key, value, ttl=self.ttl, tags=tags)
        logger.debug(f"Cached result for query key: {key}")

    def raw_get(self, key: str) -> Optional[Any]:
        """Get cached result by pre-computed key"""
        return self.cache.get(key)

    def raw_set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[list[str]] = None):
        """Cache a result by pre-computed key with optional tag indexing"""
        self.cache.set(key, value, ttl=ttl or self.ttl, tags=tags)

    def invalidate_by_tag(self, tag: str):
        """Invalidate all cached results with the given tag"""
        self.cache.invalidate_by_tag(tag)
        logger.info(f"Query cache invalidated for tag: {tag}")

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        logger.info("Query cache cleared")


def cached(cache_instance_factory: Callable[[], QueryCache]):
    """Decorator to cache function results using a cache instance factory"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            settings = get_settings()
            if not settings.cache_enabled:
                return func(*args, **kwargs)

            cache_instance = cache_instance_factory()

            # Generate key from function name, args and kwargs
            key_parts = {
                "func": func.__name__,
                "args": args[1:] if args and hasattr(args[0], 'search') else args,
                "kwargs": kwargs
            }
            key_str = json.dumps(key_parts, sort_keys=True, default=str)
            cache_key = f"{func.__name__}:{hashlib.md5(key_str.encode()).hexdigest()}"

            cached_result = cache_instance.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.cache.set(cache_key, result, ttl=cache_instance.ttl)
            return result

        return wrapper

    return decorator


# Singleton cache instances
_query_cache: Optional[QueryCache] = None
_graph_data_cache: Optional[QueryCache] = None
_search_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """Get singleton query cache"""
    global _query_cache
    settings = get_settings()
    if _query_cache is None:
        _query_cache = QueryCache(
            max_size=1000,
            ttl=settings.cache_ttl,
        )
    return _query_cache


def get_graph_data_cache() -> QueryCache:
    """Get singleton graph data cache for /graph/data endpoint"""
    global _graph_data_cache
    if _graph_data_cache is None:
        _graph_data_cache = QueryCache(
            max_size=100,
            ttl=300,
        )
    return _graph_data_cache


def get_search_cache() -> QueryCache:
    """Get singleton search cache for /graph/search endpoint"""
    global _search_cache
    if _search_cache is None:
        _search_cache = QueryCache(
            max_size=500,
            ttl=600,
        )
    return _search_cache


def clear_all_caches():
    """Clear all cache instances and reset singletons"""
    global _query_cache, _graph_data_cache, _search_cache
    if _query_cache:
        _query_cache.clear()
    if _graph_data_cache:
        _graph_data_cache.clear()
    if _search_cache:
        _search_cache.clear()
    _query_cache = None
    _graph_data_cache = None
    _search_cache = None
    logger.info("All caches cleared")
