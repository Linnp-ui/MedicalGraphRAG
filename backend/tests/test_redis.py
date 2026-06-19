import httpx
import time

print('=== 测试应用启动和 Redis 缓存 ===')
print()

print('等待应用启动...')
time.sleep(2)

BASE_URL = 'http://localhost:8000/api/v1'

with httpx.Client(base_url=BASE_URL) as client:
    print('1. 测试健康检查...')
    response = client.get('/health')
    print(f'   状态码: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'   响应: {data}')
    print()

    print('2. 测试图谱搜索（触发缓存）...')
    response = client.get('/graph/search', params={'query': 'test', 'limit': 10})
    print(f'   状态码: {response.status_code}')
    data = response.json()
    nodes = data.get('nodes', [])
    print(f'   响应节点数: {len(nodes)}')
    print()

    print('3. 再次相同查询（测试缓存命中）...')
    start = time.time()
    response = client.get('/graph/search', params={'query': 'test', 'limit': 10})
    duration = (time.time() - start) * 1000
    print(f'   状态码: {response.status_code}')
    print(f'   响应时间: {duration:.2f}ms')
    data = response.json()
    nodes = data.get('nodes', [])
    print(f'   响应节点数: {len(nodes)}')
    print()

    print('4. 测试混合检索...')
    payload = {
        'query': 'test query',
        'top_k': 5
    }
    response = client.post('/retrieval/hybrid', json=payload)
    print(f'   状态码: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f'   结果数: {len(results)}')
    print()

print('=== ✅ 测试完成 ===')
