import pytest
from unittest.mock import Mock, patch
from src.core.llm_cache import LLMCache, llm_cached, _make_hashable


class TestLLMCache:
    
    def test_init(self):
        cache = LLMCache()
        assert cache is not None
    
    def test_get_or_generate_cache_miss(self):
        cache = LLMCache()
        call_count = [0]
        
        def generate_fn():
            call_count[0] += 1
            return "result"
        
        result = cache.get_or_generate(
            prompt="test prompt",
            generate_fn=generate_fn,
            model="test-model"
        )
        
        assert result == "result"
        assert call_count[0] == 1
    
    def test_get_or_generate_cache_hit(self):
        cache = LLMCache()
        call_count = [0]
        
        def generate_fn():
            call_count[0] += 1
            return "result"
        
        result1 = cache.get_or_generate(
            prompt="test prompt",
            generate_fn=generate_fn,
            model="test-model"
        )
        
        result2 = cache.get_or_generate(
            prompt="test prompt",
            generate_fn=generate_fn,
            model="test-model"
        )
        
        assert result1 == result2 == "result"
        assert call_count[0] == 1
    
    def test_get_stats(self):
        cache = LLMCache()
        
        cache.get_or_generate("prompt1", lambda: "r1", model="m1")
        cache.get_or_generate("prompt1", lambda: "r1", model="m1")
        cache.get_or_generate("prompt2", lambda: "r2", model="m1")
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 2
    
    def test_clear(self):
        cache = LLMCache()
        
        cache.get_or_generate("prompt", lambda: "result", model="m")
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_get_hit_rate(self):
        cache = LLMCache()
        
        assert cache.get_hit_rate() == 0.0
        
        cache.get_or_generate("p1", lambda: "r1", model="m")
        cache.get_or_generate("p1", lambda: "r1", model="m")
        
        assert cache.get_hit_rate() == 0.5
    
    def test_different_models_different_cache(self):
        cache = LLMCache()
        call_count = [0]
        
        def generate_fn():
            call_count[0] += 1
            return "result"
        
        cache.get_or_generate("prompt", generate_fn, model="model-a")
        cache.get_or_generate("prompt", generate_fn, model="model-b")
        
        assert call_count[0] == 2
    
    def test_different_params_different_cache(self):
        cache = LLMCache()
        call_count = [0]
        
        def generate_fn():
            call_count[0] += 1
            return "result"
        
        cache.get_or_generate("prompt", generate_fn, model="m", temperature=0.7)
        cache.get_or_generate("prompt", generate_fn, model="m", temperature=0.9)
        
        assert call_count[0] == 2


class TestMakeHashable:
    
    def test_hashable_dict(self):
        result = _make_hashable({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))
    
    def test_hashable_list(self):
        result = _make_hashable([1, 2, 3])
        assert result == (1, 2, 3)
    
    def test_hashable_nested(self):
        result = _make_hashable({"a": [1, 2], "b": {"c": 3}})
        assert result == (("a", (1, 2)), ("b", (("c", 3),)))
    
    def test_hashable_set(self):
        result = _make_hashable({3, 1, 2})
        assert result == (1, 2, 3)


class TestLLMCachedDecorator:
    
    def test_llm_cached_decorator_caches(self):
        call_count = [0]
        
        @llm_cached(model="test-model")
        def generate_text(prompt: str) -> str:
            call_count[0] += 1
            return f"generated: {prompt}"
        
        with patch("src.core.config.get_settings") as mock_settings:
            mock_settings.return_value.llm_cache_enabled = True
            mock_settings.return_value.llm_cache_max_size = 100
            
            import src.core.llm_cache as llm_cache_module
            llm_cache_module._llm_cache = None
            
            result1 = generate_text("hello")
            result2 = generate_text("hello")
            
            assert result1 == result2
            assert call_count[0] == 1
    
    def test_llm_cached_decorator_disabled(self):
        call_count = [0]
        
        @llm_cached(model="test-model")
        def generate_text(prompt: str) -> str:
            call_count[0] += 1
            return f"generated: {prompt}"
        
        with patch("src.core.config.get_settings") as mock_settings:
            mock_settings.return_value.llm_cache_enabled = False
            
            result1 = generate_text("hello")
            result2 = generate_text("hello")
            
            assert call_count[0] == 2
