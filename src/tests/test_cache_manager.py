# tests/test_cache_manager.py
import os
import time
import tempfile
from src.utils.cache_manager import CacheManager


def test_basic_set_get_clear():
    # Use tempfile directly to avoid potential pytest fixture issues
    with tempfile.TemporaryDirectory() as tmp_dir:
        d = os.path.join(tmp_dir, "cache")
        cm = CacheManager(
            cache_dir=d, size_limit_bytes=10_000, use_diskcache_if_available=False
        )

        k = ("a", 1)
        cm.set(k, {"x": 1})
        assert cm.exists(k)
        v = cm.get(k)
        assert isinstance(v, dict) and v["x"] == 1
        cm.delete(k)
        assert not cm.exists(k)
        cm.set(k, 123)
        assert cm.exists(k)
        cm.clear()
        assert not cm.exists(k)


def test_eviction_lru():
    with tempfile.TemporaryDirectory() as tmp_dir:
        d = os.path.join(tmp_dir, "cache_lru")
        # small limit to force eviction
        cm = CacheManager(
            cache_dir=d, size_limit_bytes=500, use_diskcache_if_available=False
        )
        # store several entries
        for i in range(10):
            key = ("k", i)
            cm.set(key, "x" * 200)  # 200 bytes-ish each -> should evict older ones
            time.sleep(0.01)  # ensure distinct atimes
        # total size should be <= limit
        stats = cm.stats()
        assert stats["curr_size"] <= 500


def test_cache_function_decorator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        d = os.path.join(tmp_dir, "cache_dec")
        cm = CacheManager(cache_dir=d, use_diskcache_if_available=False)

        @cm.cache_function(expire_seconds=None)
        def add(a, b):
            return a + b

        val1 = add(2, 3)
        val2 = add(2, 3)
        assert val1 == val2 == 5


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__]))
