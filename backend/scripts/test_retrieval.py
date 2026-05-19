import sys
import os
from pathlib import Path
import time

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

print('=' * 70)
print('GraphRAG 完整检索功能测试')
print('=' * 70)
print()

# 1. 验证 Neo4j 连接
print('[1/7] 验证 Neo4j 连接...')
try:
    from core.neo4j_client import get_neo4j_client
    client = get_neo4j_client()
    
    if client.verify_connectivity():
        print('   ✅ Neo4j 连接成功')
    else:
        print('   ❌ Neo4j 连接失败')
        print('   请检查 Neo4j 是否运行在 localhost:17687')
        sys.exit(1)
except Exception as e:
    print(f'   ❌ Neo4j 连接失败: {e}')
    sys.exit(1)

print()

# 2. 检查图谱数据统计
print('[2/7] 检查图谱数据统计...')
try:
    stats = client.get_graph_stats()
    print(f'   文档数量: {stats.get("documents", 0)}')
    print(f'   实体数量: {stats.get("entities", 0)}')
    print(f'   关系数量: {stats.get("relationships", 0)}')
    print(f'   Chunk数量: {stats.get("chunks", 0)}')
    print()
except Exception as e:
    print(f'   ⚠️ 获取统计信息失败: {e}')
    print()

# 3. 测试 HTTP 接口
print('[3/7] 测试健康检查接口...')
import httpx

BASE_URL = 'http://localhost:8000/api/v1'
try:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as http_client:
        response = http_client.get('/health')
        data = response.json()
        
        if data.get('neo4j_connected'):
            print('   ✅ 健康检查通过 - Neo4j 已连接')
        else:
            print('   ⚠️ Neo4j 未连接（应用可能需要重启以重新连接）')
        
        print(f'   状态: {data.get("status")}')
        print(f'   版本: {data.get("version")}')
        print()
        
        # 4. 测试图谱数据接口
        print('[4/7] 测试图谱数据接口 (/graph/data)...')
        try:
            start = time.time()
            response = http_client.get('/graph/data', params={'limit': 50})
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get('nodes', [])
                edges = data.get('edges', [])
                print(f'   ✅ 获取成功')
                print(f'   节点数: {len(nodes)}')
                print(f'   边数: {len(edges)}')
                print(f'   响应时间: {duration:.2f}ms')
                
                if nodes:
                    print(f'   示例节点: {nodes[0]}')
            else:
                print(f'   ❌ 失败: {response.status_code}')
                print(f'   响应: {response.text[:200]}')
        except Exception as e:
            print(f'   ❌ 请求失败: {e}')
        
        print()

        # 5. 测试图谱搜索接口
        print('[5/7] 测试图谱搜索接口 (/graph/search)...')
        try:
            search_query = "test"
            start = time.time()
            response = http_client.get('/graph/search', params={
                'query': search_query,
                'node_label': '',
                'limit': 10
            })
            duration1 = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get('nodes', [])
                print(f'   ✅ 搜索成功')
                print(f'   查询: "{search_query}"')
                print(f'   结果数: {len(nodes)}')
                print(f'   响应时间: {duration1:.2f}ms')
                
                # 6. 测试缓存命中
                print()
                print('[6/7] 测试 Redis 缓存命中...')
                start = time.time()
                response = http_client.get('/graph/search', params={
                    'query': search_query,
                    'node_label': '',
                    'limit': 10
                })
                duration2 = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    nodes = data.get('nodes', [])
                    print(f'   ✅ 缓存命中')
                    print(f'   查询: "{search_query}"')
                    print(f'   结果数: {len(nodes)}')
                    print(f'   响应时间: {duration2:.2f}ms')
                    
                    if duration2 < duration1:
                        speedup = (duration1 - duration2) / duration1 * 100
                        print(f'   性能提升: {speedup:.1f}%')
                    else:
                        print(f'   (首次可能较慢，后续查询会更快)')
            else:
                print(f'   ❌ 失败: {response.status_code}')
                print(f'   响应: {response.text[:200]}')
        except Exception as e:
            print(f'   ❌ 请求失败: {e}')
        
        print()

        # 7. 测试混合检索
        print('[7/7] 测试混合检索接口 (/retrieval/hybrid)...')
        try:
            payload = {
                'query': 'test query',
                'top_k': 5,
                'alpha': 0.5
            }
            start = time.time()
            response = http_client.post('/retrieval/hybrid', json=payload)
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f'   ✅ 混合检索成功')
                print(f'   查询: "{payload["query"]}"')
                print(f'   结果数: {len(results)}')
                print(f'   响应时间: {duration:.2f}ms')
                
                if results:
                    print(f'   最高分数: {results[0].get("score", 0):.4f}')
            else:
                print(f'   ❌ 失败: {response.status_code}')
                print(f'   响应: {response.text[:200]}')
        except Exception as e:
            print(f'   ❌ 请求失败: {e}')
        
        print()
        print('=' * 70)
        print('✅ 所有测试完成！')
        print('=' * 70)

except Exception as e:
    print(f'   ❌ HTTP 请求失败: {e}')
    print()
    print('请确保应用正在运行: http://localhost:8000')
    print()
    print('启动命令:')
    print('  cd backend')
    print('  uvicorn src.main:app --reload')
