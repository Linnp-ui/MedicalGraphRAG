import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps

from loguru import logger

from .config import get_settings


class LRUCache:
    """Simple LRU cache implementation with TTL support"""

    def __init__(self, max_size: int = 1000):
        # Store as {key: (value, expiration_time)}
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            value, expires_at = self.cache[key]
            # Check if expired
            if time.time() > expires_at:
                del self.cache[key]
                return None
                
            self.cache.move_to_end(key)
            return value
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if key in self.cache:
            self.cache.move_to_end(key)
        
        expires_at = time.time() + ttl
        self.cache[key] = (value, expires_at)
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear all cache"""
        self.cache.clear()

    def __contains__(self, key: str) -> bool:
        if key in self.cache:
            _, expires_at = self.cache[key]
            if time.time() <= expires_at:
                return True
            else:
                del self.cache[key]
        return False

    def __len__(self) -> int:
        return len(self.cache)


class QueryCache:
    """Cache for query results"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
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

    def set(self, query: str, params: Optional[dict], value: Any):
        """Cache a result"""
        key = self._generate_key(query, params)
        self.cache.set(key, value, ttl=self.ttl)
        logger.debug(f"Cached result for query key: {key}")

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
                "args": args[1:] if args and hasattr(args[0], 'search') else args, # Skip 'self' if method
                "kwargs": kwargs
            }
            # Simple string serialization for the key
            cache_key = f"{func.__name__}:{hashlib.md5(str(key_parts).encode()).hexdigest()}"
            
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
