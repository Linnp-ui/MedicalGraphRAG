import hashlib
import json
import threading
from typing import Any, Callable, Dict, Optional
from functools import wraps
from loguru import logger

from .cache import LRUCache, CacheBackend


class LLMCache:
    """LLM 调用缓存层"""
    
    def __init__(self, backend: Optional[CacheBackend] = None, max_size: int = 500):
        self.cache = backend or LRUCache(max_size=max_size)
        self._stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()
    
    def _make_key(
        self, 
        prompt: str, 
        model: str, 
        params: Dict[str, Any]
    ) -> str:
        """生成缓存键
        
        Args:
            prompt: 输入提示
            model: 模型标识
            params: 其他参数
            
        Returns:
            缓存键字符串
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            "params": _make_hashable(params)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm:{model}:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get_or_generate(
        self, 
        prompt: str,
        generate_fn: Callable[[], Any],
        model: str = "default",
        **params
    ) -> Any:
        """获取缓存或生成新结果
        
        Args:
            prompt: 输入提示
            generate_fn: 生成函数（缓存未命中时调用）
            model: 模型标识
            **params: 其他参数
            
        Returns:
            缓存或新生成的结果
        """
        cache_key = self._make_key(prompt, model, params)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            with self._lock:
                self._stats["hits"] += 1
            logger.debug(f"LLM cache hit for model {model}")
            return cached_result
        
        result = generate_fn()
        
        self.cache.set(cache_key, result)
        with self._lock:
            self._stats["misses"] += 1
        
        logger.debug(f"LLM cache miss for model {model}, cached result")
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计
        
        Returns:
            包含 hits 和 misses 的统计字典
        """
        with self._lock:
            return self._stats.copy()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        with self._lock:
            self._stats = {"hits": 0, "misses": 0}
        logger.info("LLM cache cleared")
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total == 0:
                return 0.0
            return self._stats["hits"] / total


def _make_hashable(obj: Any) -> Any:
    """将对象转换为可哈希的类型"""
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(item) for item in obj))
    else:
        return obj


_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """获取 LLM 缓存单例"""
    global _llm_cache
    if _llm_cache is None:
        from .config import get_settings
        settings = get_settings()
        _llm_cache = LLMCache(max_size=settings.llm_cache_max_size)
    return _llm_cache


def llm_cached(model: str = "default", **default_params):
    """LLM 调用缓存装饰器
    
    Usage:
        @llm_cached(model="qwen-max")
        def generate_summary(text: str) -> str:
            return llm.generate(f"总结: {text}")
    
    Args:
        model: 模型标识
        **default_params: 默认参数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .config import get_settings
            settings = get_settings()
            
            if not settings.llm_cache_enabled:
                return func(*args, **kwargs)
            
            cache = get_llm_cache()
            
            prompt_parts = [func.__name__, str(args), str(kwargs)]
            prompt = ":".join(prompt_parts)
            
            params = {**default_params, **kwargs}
            
            def generate():
                return func(*args, **kwargs)
            
            return cache.get_or_generate(
                prompt=prompt,
                generate_fn=generate,
                model=model,
                **params
            )
        
        return wrapper
    
    return decorator
