import sys
import os
from pathlib import Path

# 设置正确的工作目录
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

print('=== 应用内 Redis 验证 ===')
print()

# 1. 检查缓存初始化
print('1. 检查缓存初始化...')
from core.cache import get_query_cache, get_graph_data_cache, get_search_cache

query_cache = get_query_cache()
graph_cache = get_graph_data_cache()
search_cache = get_search_cache()

print(f'   QueryCache 类型: {type(query_cache.cache).__name__}')
print(f'   GraphCache 类型: {type(graph_cache.cache).__name__}')
print(f'   SearchCache 类型: {type(search_cache.cache).__name__}')
print()

# 2. 检查 CacheRouter 的 Redis 可用性
if hasattr(query_cache.cache, '_redis'):
    redis_available = query_cache.cache._redis._available
    memory_available = hasattr(query_cache.cache, '_memory')
    print(f'   Redis 可用: {redis_available}')
    print(f'   内存缓存可用: {memory_available}')
    print()

# 3. 测试写入 Redis
print('2. 测试写入 Redis...')
query_cache.set('test_query', {'param': 'value'}, {'result': 'data', 'count': 100})
print('   数据已写入')
print()

# 4. 测试从 Redis 读取
print('3. 测试从 Redis 读取...')
result = query_cache.get('test_query', {'param': 'value'})
print(f'   读取结果: {result}')
print()

# 5. 直接检查 Redis
print('4. 直接检查 Redis 中的缓存键...')
import redis

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
keys = r.keys('graphrag:*')
print(f'   Redis 中 graphrag:* 键数量: {len(keys)}')
if keys:
    print(f'   示例键:')
    for key in keys[:3]:
        print(f'     - {key}')
print()

# 6. 测试缓存命中性能
print('5. 测试缓存命中性能...')
import time

# 第一次访问（未命中）
start = time.time()
result1 = query_cache.get('nonexistent_key')
miss_time = (time.time() - start) * 1000
print(f'   未命中耗时: {miss_time:.2f}ms')

# 第二次访问（命中）
query_cache.set('benchmark_key', {}, {'large_data': list(range(1000))})
start = time.time()
result2 = query_cache.get('benchmark_key', {})
hit_time = (time.time() - start) * 1000
print(f'   命中耗时: {hit_time:.2f}ms')
print()

# 7. 清理测试数据
print('6. 清理测试数据...')
query_cache.clear()
keys_after = len(r.keys('graphrag:*'))
print(f'   清理后键数量: {keys_after}')
print()

print('=== ✅ 验证完成 ===')
