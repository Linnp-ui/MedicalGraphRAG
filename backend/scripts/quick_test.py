import sys
import os
from pathlib import Path
import time

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

import httpx

print('=' * 70)
print('GraphRAG 完整检索功能测试')
print('=' * 70)
print()

BASE_URL = 'http://localhost:8000/api/v1'

try:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 测试1: 健康检查
        print('[1] 健康检查...')
        resp = client.get('/health')
        data = resp.json()
        print(f'    状态: {data.get("status")}')
        print(f'    Neo4j: {data.get("neo4j_connected")}')
        print()

        # 测试2: 图谱数据
        print('[2] 图谱数据...')
        start = time.time()
        resp = client.get('/graph/data', params={'limit': 20})
        duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            print(f'    ✅ 成功')
            print(f'    节点数: {len(nodes)}')
            print(f'    边数: {len(edges)}')
            print(f'    耗时: {duration:.0f}ms')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试3: 图谱搜索
        print('[3] 图谱搜索...')
        start = time.time()
        resp = client.get('/graph/search', params={'query': '高血压', 'limit': 5})
        duration1 = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get('nodes', [])
            print(f'    ✅ 首次查询成功')
            print(f'    节点数: {len(nodes)}')
            print(f'    耗时: {duration1:.0f}ms')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试4: 图谱搜索缓存命中
        print('[4] 图谱搜索 (缓存命中)...')
        start = time.time()
        resp = client.get('/graph/search', params={'query': '高血压', 'limit': 5})
        duration2 = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get('nodes', [])
            print(f'    ✅ 缓存命中')
            print(f'    节点数: {len(nodes)}')
            print(f'    耗时: {duration2:.0f}ms')
            if duration1 > 0 and duration2 < duration1:
                speedup = (duration1 - duration2) / duration1 * 100
                print(f'    性能提升: {speedup:.1f}%')
        print()

        # 测试5: 向量检索 (POST + query params)
        print('[5] 向量检索...')
        start = time.time()
        resp = client.post('/retrieval/vector', params={'query': '高血压症状', 'top_k': 3})
        duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            print(f'    耗时: {duration:.0f}ms')
            if results:
                top = results[0]
                print(f'    最高分: {top.get("score", 0):.4f}')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
            print(f'    响应: {resp.text[:200]}')
        print()

        # 测试6: 混合检索
        print('[6] 混合检索...')
        start = time.time()
        resp = client.post('/retrieval/hybrid', json={
            'query': '高血压的治疗方法',
            'top_k': 5,
            'alpha': 0.5
        })
        duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            print(f'    耗时: {duration:.0f}ms')
            if results:
                top = results[0]
                print(f'    最高分: {top.get("score", 0):.4f}')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
            print(f'    响应: {resp.text[:200]}')
        print()

        # 测试7: Redis缓存验证
        print('[7] Redis缓存验证...')
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        keys = r.keys('graphrag:*')
        print(f'    Redis键数量: {len(keys)}')
        if keys:
            for key in keys[:5]:
                ttl = r.ttl(key)
                print(f'    - {key} (TTL: {ttl}s)')
        print()

        print('=' * 70)
        print('✅ 所有测试完成')
        print('=' * 70)

except Exception as e:
    print(f'❌ 错误: {e}')
    import traceback
    traceback.print_exc()
