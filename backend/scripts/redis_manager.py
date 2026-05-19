import redis
import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

from core.config import get_settings


def redis_management():
    """Redis 缓存管理工具"""
    settings = get_settings()

    print("=" * 60)
    print("Redis 缓存管理工具")
    print("=" * 60)
    print()

    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )

        print(f"Redis 服务器: {settings.redis_host}:{settings.redis_port}")
        print(f"键前缀: {settings.redis_prefix}")
        print()

        keys = r.keys(f"{settings.redis_prefix}:*")
        print(f"当前缓存键数量: {len(keys)}")
        print()

        print("请选择操作:")
        print("1. 查看所有缓存键")
        print("2. 清空所有缓存")
        print("3. 查看缓存统计")
        print("4. 性能测试")
        print("5. 退出")
        print()

        choice = input("请输入选项 (1-5): ").strip()

        if choice == '1':
            print("\n--- 所有缓存键 ---")
            if not keys:
                print("没有缓存键")
            else:
                for i, key in enumerate(keys, 1):
                    ttl = r.ttl(key)
                    value = r.get(key)
                    size = len(value) if value else 0
                    print(f"{i:3d}. {key}")
                    print(f"     TTL: {ttl}秒 | 大小: {size}字节")

        elif choice == '2':
            print("\n警告: 即将清空所有缓存!")
            confirm = input("确认清空? (yes/no): ").strip().lower()
            if confirm == 'yes':
                deleted = r.delete(*keys) if keys else 0
                print(f"✅ 已删除 {deleted} 个缓存键")
            else:
                print("取消操作")

        elif choice == '3':
            print("\n--- 缓存统计 ---")
            info = r.info()
            print(f"版本: {info['redis_version']}")
            print(f"运行时间: {info['uptime_in_days']} 天")
            print(f"已用内存: {info['used_memory_human']}")
            print(f"峰值内存: {info['used_memory_peak_human']}")
            print(f"总连接数: {info['total_connections_received']}")
            print(f"总命令数: {info['total_commands_processed']}")
            stats = info.get('stats', {})
            print(f"命中次数: {stats.get('keyspace_hits', 0)}")
            print(f"未命中次数: {stats.get('keyspace_misses', 0)}")
            print(f"命中率: {stats.get('keyspace_hits', 0) / max(stats.get('keyspace_hits', 0) + stats.get('keyspace_misses', 0), 1) * 100:.2f}%")

        elif choice == '4':
            print("\n--- 性能测试 ---")
            import time

            test_key = f"{settings.redis_prefix}:perf_test"
            test_data = {"test": "data", "numbers": list(range(100))}

            print("写入测试...")
            start = time.time()
            r.setex(test_key, 10, str(test_data))
            write_time = (time.time() - start) * 1000
            print(f"写入耗时: {write_time:.2f}ms")

            print("读取测试...")
            start = time.time()
            for _ in range(1000):
                r.get(test_key)
            read_time = (time.time() - start) * 1000
            print(f"1000次读取耗时: {read_time:.2f}ms")
            print(f"平均每次读取: {read_time/1000:.2f}ms")

            r.delete(test_key)
            print("✅ 性能测试完成")

        elif choice == '5':
            print("退出")

        else:
            print("无效选项")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保 Redis 服务正在运行")
        print("检查命令: redis-cli ping")


if __name__ == "__main__":
    redis_management()
