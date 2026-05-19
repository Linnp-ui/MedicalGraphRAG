import redis
import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

from core.config import get_settings


def monitor_redis_cache():
    """监控 Redis 缓存使用情况"""
    settings = get_settings()

    print("=" * 60)
    print("Redis 缓存监控")
    print("=" * 60)
    print()

    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )

        info = r.info('memory')
        print(f"Redis 服务器: {settings.redis_host}:{settings.redis_port}")
        print(f"Redis 版本: {r.info('server')['redis_version']}")
        print()
        print("--- 内存使用 ---")
        print(f"已用内存: {info['used_memory_human']}")
        print(f"峰值内存: {info['used_memory_peak_human']}")
        print(f"内存碎片率: {info['mem_fragmentation_ratio']:.2f}")
        print()
        print("--- 缓存统计 ---")

        keys = r.keys(f"{settings.redis_prefix}:*")
        print(f"缓存键数量: {len(keys)}")
        print()

        if keys:
            print("--- 缓存键详情 ---")
            for i, key in enumerate(keys[:20], 1):
                ttl = r.ttl(key)
                value_size = len(r.get(key) or '')
                print(f"{i:2d}. {key}")
                print(f"    TTL: {ttl}秒, 大小: {value_size}字节")

            if len(keys) > 20:
                print(f"    ... 还有 {len(keys) - 20} 个键")

            print()

        print("--- 命中统计 ---")
        stats = r.info('stats')
        print(f"命中次数: {stats['keyspace_hits']}")
        print(f"未命中次数: {stats['keyspace_misses']}")
        total = stats['keyspace_hits'] + stats['keyspace_misses']
        if total > 0:
            hit_rate = stats['keyspace_hits'] / total * 100
            print(f"命中率: {hit_rate:.2f}%")
        print()

        print("--- 连接信息 ---")
        clients = r.info('clients')
        print(f"已连接客户端: {clients['connected_clients']}")
        print(f"阻塞客户端: {clients['blocked_clients']}")
        print()

        print("=" * 60)
        print("✅ Redis 缓存运行正常")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 错误: {e}")
        print()
        print("请确保 Redis 服务正在运行:")
        print("  redis-cli ping")
        print()
        print("如果 Redis 未运行，请启动:")
        print("  redis-server")


if __name__ == "__main__":
    monitor_redis_cache()
