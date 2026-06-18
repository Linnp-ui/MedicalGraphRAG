"""测试基于标签的缓存失效系统"""
import pytest
from src.core.cache import (
    LRUCache,
    CacheRouter,
    QueryCache,
    get_graph_data_cache,
    get_search_cache,
)


class TestLRUCacheTagInvalidation:

    def test_basic_tag_invalidation(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "v1", tags=["Disease"])
        cache.set("k2", "v2", tags=["Symptom"])
        cache.set("k3", "v3", tags=["Disease"])

        assert cache.get("k1") == "v1"
        assert cache.get("k2") == "v2"

        cache.invalidate_by_tag("Disease")

        assert cache.get("k1") is None
        assert cache.get("k3") is None
        assert cache.get("k2") == "v2"

    def test_invalidate_nonexistent_tag(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "v1", tags=["Disease"])
        cache.invalidate_by_tag("NonexistentTag")
        assert cache.get("k1") == "v1"

    def test_no_tags_passed(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "v1")
        cache.invalidate_by_tag("Disease")
        assert cache.get("k1") == "v1"

    def test_multi_tag_on_single_entry(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "v1", tags=["Disease", "Symptom"])

        cache.invalidate_by_tag("Disease")
        assert cache.get("k1") is None

        cache.set("k1", "v1", tags=["Disease", "Symptom"])
        cache.invalidate_by_tag("Symptom")
        assert cache.get("k1") is None

    def test_eviction_cleans_tag_index(self):
        cache = LRUCache(max_size=2)
        cache.set("a", "1", tags=["Disease"])
        cache.set("b", "2", tags=["Symptom"])
        cache.set("c", "3", tags=["Disease"])

        assert cache.get("a") is None

        cache.invalidate_by_tag("Disease")
        assert cache.get("c") is None
        assert cache.get("b") == "2"

    def test_clear_also_clears_tag_index(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "v1", tags=["Disease"])
        cache.clear()
        cache.invalidate_by_tag("Disease")
        assert cache.get("k1") is None

    def test_set_replaces_existing_key(self):
        cache = LRUCache(max_size=100)
        cache.set("k1", "old", tags=["Disease"])
        cache.invalidate_by_tag("Disease")
        assert cache.get("k1") is None

        cache.set("k1", "new", tags=["Disease"])
        assert cache.get("k1") == "new"

    def test_large_number_of_keys_same_tag(self):
        cache = LRUCache(max_size=1000)
        for i in range(100):
            cache.set(f"k{i}", f"v{i}", tags=["Disease"])
        cache.invalidate_by_tag("Disease")
        for i in range(100):
            assert cache.get(f"k{i}") is None


class TestQueryCacheTagInvalidation:

    def test_raw_set_with_tags(self):
        cache = QueryCache(max_size=100, ttl=60)
        cache.raw_set("mykey", "myvalue", tags=["Drug"])
        assert cache.raw_get("mykey") == "myvalue"

        cache.invalidate_by_tag("Drug")
        assert cache.raw_get("mykey") is None

    def test_set_with_tags(self):
        cache = QueryCache(max_size=100, ttl=60)
        cache.set("query1", {"p": 1}, "result1", tags=["Disease"])
        assert cache.get("query1", {"p": 1}) == "result1"

        cache.invalidate_by_tag("Disease")
        assert cache.get("query1", {"p": 1}) is None

    def test_raw_set_without_tags_ignored_by_invalidation(self):
        cache = QueryCache(max_size=100, ttl=60)
        cache.raw_set("mykey", "myvalue")
        assert cache.raw_get("mykey") == "myvalue"

        cache.invalidate_by_tag("Disease")
        assert cache.raw_get("mykey") == "myvalue"

    def test_mixed_tags(self):
        cache = QueryCache(max_size=100, ttl=60)
        cache.raw_set("key_a", "va", tags=["Disease"])
        cache.raw_set("key_b", "vb", tags=["Symptom"])
        cache.raw_set("key_c", "vc", tags=["Disease", "Symptom"])
        cache.raw_set("key_d", "vd")

        cache.invalidate_by_tag("Disease")
        assert cache.raw_get("key_a") is None
        assert cache.raw_get("key_c") is None
        assert cache.raw_get("key_b") == "vb"
        assert cache.raw_get("key_d") == "vd"

    def test_clear_resets_tag_index(self):
        cache = QueryCache(max_size=100, ttl=60)
        cache.raw_set("k1", "v1", tags=["Disease"])
        cache.clear()
        assert cache.raw_get("k1") is None
        cache.raw_set("k2", "v2", tags=["Disease"])
        cache.invalidate_by_tag("Disease")
        assert cache.raw_get("k2") is None


class TestCacheRouterIntegration:

    def test_router_propagates_tags_to_both_backends(self):
        from src.core.cache import RedisCache

        mock_redis = RedisCache(
            host="localhost", port=6379, db=0, prefix="test_router"
        )
        mock_lru = LRUCache(max_size=100)
        router = CacheRouter(mock_redis, mock_lru)

        router.set("rk1", "rv1", tags=["Disease"])
        assert mock_lru.get("rk1") == "rv1"
        router.invalidate_by_tag("Disease")
        assert mock_lru.get("rk1") is None

    def test_singleton_caches_exist(self):
        graph_cache = get_graph_data_cache()
        search_cache = get_search_cache()
        assert graph_cache is not None
        assert search_cache is not None
        assert isinstance(graph_cache, QueryCache)
        assert isinstance(search_cache, QueryCache)

    def test_graph_search_cache_share_invalidation(self):
        from src.core.cache import get_graph_data_cache, get_search_cache

        graph = get_graph_data_cache()
        search = get_search_cache()

        graph.raw_set("graph_key", "graph_val", tags=["Disease"])
        search.raw_set("search_key", "search_val", tags=["Symptom"])

        graph.invalidate_by_tag("Disease")
        assert graph.raw_get("graph_key") is None
        assert search.raw_get("search_key") == "search_val"

        search.invalidate_by_tag("Symptom")
        assert search.raw_get("search_key") is None
