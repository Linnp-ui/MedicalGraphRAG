import sys
import os
from pathlib import Path
import time

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / 'src'))
os.chdir(backend_dir)

import httpx

print('=' * 70)
print('GraphRAG 模糊搜索功能测试')
print('=' * 70)
print()

BASE_URL = 'http://localhost:8000/api/v1'

try:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 测试1: 标准搜索（带评分）
        print('[1] 标准搜索（带相似度评分）...')
        resp = client.get('/graph/search', params={'query': '高血压', 'limit': 5})
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            for i, r in enumerate(results[:3]):
                props = r.get('properties', {})
                name = props.get('name', props.get('title', 'N/A'))
                score = r.get('score', 0)
                print(f'    {i+1}. {name} (分数: {score})')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试2: 模糊搜索 - 包含匹配
        print('[2] 模糊搜索 - 包含匹配 (fuzzy=true)...')
        resp = client.get('/graph/search', params={
            'query': '血压',
            'limit': 5,
            'fuzzy': True,
            'fuzzy_mode': 'contains'
        })
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            for i, r in enumerate(results[:3]):
                props = r.get('properties', {})
                name = props.get('name', props.get('title', 'N/A'))
                matched = r.get('matched_property', '')[:50] if r.get('matched_property') else 'N/A'
                print(f'    {i+1}. {name}')
                print(f'       匹配: {matched}...')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试3: 模糊搜索 - 前缀匹配
        print('[3] 模糊搜索 - 前缀匹配 (fuzzy_mode=prefix)...')
        resp = client.get('/graph/search', params={
            'query': '高血',
            'limit': 5,
            'fuzzy': True,
            'fuzzy_mode': 'prefix'
        })
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            for i, r in enumerate(results[:3]):
                props = r.get('properties', {})
                name = props.get('name', props.get('title', 'N/A'))
                print(f'    {i+1}. {name}')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试4: 模糊搜索 - 正则表达式
        print('[4] 模糊搜索 - 正则表达式 (fuzzy_mode=regex)...')
        resp = client.get('/graph/search', params={
            'query': '高.*压',  # 匹配"高...压"
            'limit': 5,
            'fuzzy': True,
            'fuzzy_mode': 'regex'
        })
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            for i, r in enumerate(results[:3]):
                props = r.get('properties', {})
                name = props.get('name', props.get('title', 'N/A'))
                print(f'    {i+1}. {name}')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试5: 搜索药物相关实体
        print('[5] 搜索药物相关实体...')
        resp = client.get('/graph/search', params={
            'query': '阿司匹林',
            'limit': 5
        })
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f'    ✅ 成功')
            print(f'    结果数: {len(results)}')
            for i, r in enumerate(results[:3]):
                props = r.get('properties', {})
                name = props.get('name', props.get('title', 'N/A'))
                score = r.get('score', 0)
                print(f'    {i+1}. {name} (分数: {score})')
        else:
            print(f'    ❌ 失败: {resp.status_code}')
        print()

        # 测试6: 缓存验证
        print('[6] 缓存验证（重复查询）...')
        start = time.time()
        resp = client.get('/graph/search', params={'query': '高血压', 'limit': 5})
        duration1 = (time.time() - start) * 1000
        
        start = time.time()
        resp = client.get('/graph/search', params={'query': '高血压', 'limit': 5})
        duration2 = (time.time() - start) * 1000
        
        print(f'    首次查询: {duration1:.0f}ms')
        print(f'    缓存命中: {duration2:.0f}ms')
        if duration1 > 0 and duration2 < duration1:
            speedup = (duration1 - duration2) / duration1 * 100
            print(f'    性能提升: {speedup:.1f}%')
        print()

        print('=' * 70)
        print('✅ 所有测试完成')
        print('=' * 70)

except Exception as e:
    print(f'❌ 错误: {e}')
    import traceback
    traceback.print_exc()
