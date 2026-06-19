# Phase 1 优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现分层社区检测（Leiden 算法）和 LLM 缓存层，提升检索精度 30%，降低成本 40%。

**Architecture:** 使用 Leiden 算法替代 Louvain 实现 3 级分层社区检测，通过 HierarchicalCommunityManager 管理分层结构，LLMCache 为所有 LLM 调用提供缓存支持。

**Tech Stack:** Python, python-igraph, leidenalg, NetworkX, Neo4j, Redis

---

## 文件结构

### 新建文件

| 文件 | 职责 |
|------|------|
| `backend/src/core/leiden_detector.py` | Leiden 社区检测算法实现 |
| `backend/src/core/hierarchical_communities.py` | 分层社区管理器 |
| `backend/src/core/llm_cache.py` | LLM 调用缓存层 |
| `backend/tests/test_leiden_detector.py` | Leiden 检测器单元测试 |
| `backend/tests/test_hierarchical_communities.py` | 分层社区管理器单元测试 |
| `backend/tests/test_llm_cache.py` | LLM 缓存单元测试 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `backend/requirements.txt` | 添加 python-igraph, leidenalg 依赖 |
| `backend/src/core/config.py` | 添加社区检测和 LLM 缓存配置项 |
| `backend/src/core/community_detector.py` | 添加 Leiden 算法支持 |
| `backend/src/core/summary_generator.py` | 集成 LLM 缓存，支持多级摘要 |
| `backend/src/core/cache.py` | 扩展缓存模块支持 LLM 缓存 |

---

## Task 1: 安装依赖并配置

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/src/core/config.py`

- [ ] **Step 1: 添加依赖到 requirements.txt**

```txt
# Leiden 算法
python-igraph>=0.11.0
leidenalg>=0.10.0
```

- [ ] **Step 2: 安装依赖**

Run: `cd d:\code\project\GRAPHRAG\backend && pip install python-igraph leidenalg`
Expected: Successfully installed python-igraph and leidenalg

- [ ] **Step 3: 添加配置项到 config.py**

在 `Settings` 类中添加：

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    
    community_algorithm: str = "leiden"
    community_levels: int = 3
    community_resolution: float = 1.0
    community_min_size: int = 5
    
    llm_cache_enabled: bool = True
    llm_cache_ttl: int = 604800
    llm_cache_max_size: int = 1000
```

- [ ] **Step 4: 验证配置加载**

Run: `cd d:\code\project\GRAPHRAG\backend && python -c "from src.core.config import get_settings; s = get_settings(); print(s.community_algorithm, s.llm_cache_enabled)"`
Expected: `leiden True`

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/src/core/config.py
git commit -m "feat: 添加 Leiden 算法依赖和配置项"
```

---

## Task 2: 实现 Leiden 社区检测器

**Files:**
- Create: `backend/src/core/leiden_detector.py`
- Create: `backend/tests/test_leiden_detector.py`

- [ ] **Step 1: 编写 Leiden 检测器测试**

```python
import pytest
import networkx as nx
from src.core.leiden_detector import LeidenCommunityDetector


class TestLeidenCommunityDetector:
    
    def test_detect_communities_basic(self):
        detector = LeidenCommunityDetector()
        graph = nx.karate_club_graph()
        graph = nx.convert_node_labels_to_integers(graph)
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partition = detector.detect_communities(str_graph)
        
        assert isinstance(partition, dict)
        assert len(partition) == 34
        assert all(isinstance(v, int) for v in partition.values())
    
    def test_detect_hierarchical(self):
        detector = LeidenCommunityDetector()
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partitions = detector.detect_hierarchical(str_graph, levels=3)
        
        assert isinstance(partitions, list)
        assert len(partitions) == 3
        assert all(isinstance(p, dict) for p in partitions)
    
    def test_compute_modularity(self):
        detector = LeidenCommunityDetector()
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partition = detector.detect_communities(str_graph)
        modularity = detector.compute_modularity(str_graph, partition)
        
        assert isinstance(modularity, float)
        assert 0 < modularity < 1
    
    def test_resolution_parameter(self):
        detector_low = LeidenCommunityDetector(resolution=0.5)
        detector_high = LeidenCommunityDetector(resolution=2.0)
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partition_low = detector_low.detect_communities(str_graph)
        partition_high = detector_high.detect_communities(str_graph)
        
        num_communities_low = len(set(partition_low.values()))
        num_communities_high = len(set(partition_high.values()))
        
        assert num_communities_low <= num_communities_high
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_leiden_detector.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.core.leiden_detector'"

- [ ] **Step 3: 实现 LeidenCommunityDetector**

```python
import igraph as ig
import networkx as nx
from typing import Dict, List, Optional
from loguru import logger


class LeidenCommunityDetector:
    """基于 Leiden 算法的社区检测器"""
    
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
    
    def _nx_to_igraph(self, graph: nx.Graph) -> ig.Graph:
        """将 NetworkX 图转换为 igraph"""
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
        g = ig.Graph(n=len(graph.nodes()), edges=edges)
        g.vs["name"] = list(graph.nodes())
        return g
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """检测单层社区
        
        Args:
            graph: NetworkX 图对象
            
        Returns:
            节点到社区ID的映射
        """
        if len(graph.nodes()) == 0:
            return {}
        
        ig_graph = self._nx_to_igraph(graph)
        
        import leidenalg
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            resolution_parameter=self.resolution
        )
        
        node_to_community = {}
        for community_id, community in enumerate(partition):
            for node_idx in community:
                node_name = ig_graph.vs[node_idx]["name"]
                node_to_community[node_name] = community_id
        
        logger.info(f"Detected {len(set(node_to_community.values()))} communities")
        return node_to_community
    
    def detect_hierarchical(
        self, 
        graph: nx.Graph, 
        levels: int = 3
    ) -> List[Dict[str, int]]:
        """递归检测分层社区
        
        Args:
            graph: NetworkX 图对象
            levels: 分层级别数
            
        Returns:
            每层的节点到社区ID映射列表
        """
        partitions = []
        current_graph = graph.copy()
        current_partition = self.detect_communities(current_graph)
        partitions.append(current_partition.copy())
        
        for level in range(1, levels):
            communities = {}
            for node, comm_id in current_partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
            
            if len(communities) <= 1:
                for _ in range(levels - level):
                    partitions.append(current_partition.copy())
                break
            
            community_graph = nx.Graph()
            for comm_id in communities:
                community_graph.add_node(comm_id)
            
            for u, v in current_graph.edges():
                comm_u = current_partition[u]
                comm_v = current_partition[v]
                if comm_u != comm_v:
                    if community_graph.has_edge(comm_u, comm_v):
                        community_graph[comm_u][comm_v]["weight"] += 1
                    else:
                        community_graph.add_edge(comm_u, comm_v, weight=1)
            
            if len(community_graph.nodes()) < 2:
                for _ in range(levels - level):
                    partitions.append(current_partition.copy())
                break
            
            higher_partition = self.detect_communities(community_graph)
            
            new_partition = {}
            for node, old_comm in current_partition.items():
                new_comm = higher_partition.get(old_comm, old_comm)
                new_partition[node] = new_comm
            
            current_partition = new_partition
            partitions.append(current_partition.copy())
        
        while len(partitions) < levels:
            partitions.append(partitions[-1].copy())
        
        logger.info(f"Built {len(partitions)} level hierarchical partition")
        return partitions
    
    def compute_modularity(
        self, 
        graph: nx.Graph, 
        partition: Dict[str, int]
    ) -> float:
        """计算分区模块度（质量评估）
        
        Args:
            graph: NetworkX 图对象
            partition: 节点到社区的映射
            
        Returns:
            模块度值（0-1之间，越高越好）
        """
        if len(graph.nodes()) == 0 or len(partition) == 0:
            return 0.0
        
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
        
        m = graph.number_of_edges()
        if m == 0:
            return 0.0
        
        q = 0.0
        for comm_nodes in communities.values():
            comm_subgraph = graph.subgraph(comm_nodes)
            lc = comm_subgraph.number_of_edges()
            
            dc = sum(graph.degree(node) for node in comm_nodes if node in graph)
            
            q += (lc / m) - (dc / (2 * m)) ** 2
        
        return q


_leiden_detector: Optional[LeidenCommunityDetector] = None


def get_leiden_detector() -> LeidenCommunityDetector:
    """获取 Leiden 检测器单例"""
    global _leiden_detector
    if _leiden_detector is None:
        from .config import get_settings
        settings = get_settings()
        _leiden_detector = LeidenCommunityDetector(
            resolution=settings.community_resolution
        )
    return _leiden_detector
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_leiden_detector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/core/leiden_detector.py backend/tests/test_leiden_detector.py
git commit -m "feat: 实现 Leiden 社区检测器"
```

---

## Task 3: 实现分层社区管理器

**Files:**
- Create: `backend/src/core/hierarchical_communities.py`
- Create: `backend/tests/test_hierarchical_communities.py`

- [ ] **Step 1: 编写分层社区管理器测试**

```python
import pytest
from unittest.mock import Mock, patch
from src.core.hierarchical_communities import HierarchicalCommunityManager


class TestHierarchicalCommunityManager:
    
    def test_init(self):
        manager = HierarchicalCommunityManager(levels=3)
        assert manager.levels == 3
        assert manager._partitions == []
    
    def test_build_hierarchy(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        import networkx as nx
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        manager.build_hierarchy(str_graph)
        
        assert len(manager._partitions) == 3
        assert all(len(p) == 34 for p in manager._partitions)
    
    def test_get_community_at_level(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        import networkx as nx
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        comm_id = manager.get_community_at_level("0", level=0)
        assert isinstance(comm_id, int)
        
        comm_id_1 = manager.get_community_at_level("0", level=1)
        assert isinstance(comm_id_1, int)
    
    def test_get_communities_by_level(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        import networkx as nx
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        
        assert isinstance(communities, dict)
        assert len(communities) > 0
        assert all(isinstance(members, list) for members in communities.values())
    
    def test_get_community_members(self):
        manager = HierarchicalCommunityManager(levels=3)
        
        import networkx as nx
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        manager.build_hierarchy(str_graph)
        
        communities = manager.get_communities_by_level(level=0)
        first_comm_id = list(communities.keys())[0]
        
        members = manager.get_community_members(level=0, community_id=first_comm_id)
        
        assert isinstance(members, list)
        assert len(members) > 0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_hierarchical_communities.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: 实现 HierarchicalCommunityManager**

```python
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .leiden_detector import LeidenCommunityDetector, get_leiden_detector


class HierarchicalCommunityManager:
    """分层社区管理器"""
    
    def __init__(
        self, 
        levels: int = 3,
        detector: Optional[LeidenCommunityDetector] = None
    ):
        self.levels = levels
        self._detector = detector or get_leiden_detector()
        self._partitions: List[Dict[str, int]] = []
        self._summaries: Dict[Tuple[int, int], str] = {}
        self._embeddings: Dict[Tuple[int, int], List[float]] = {}
        self._graph: Optional[nx.Graph] = None
    
    def build_hierarchy(self, graph: nx.Graph) -> None:
        """构建分层结构
        
        Args:
            graph: NetworkX 图对象
        """
        self._graph = graph.copy()
        self._partitions = self._detector.detect_hierarchical(graph, self.levels)
        self._summaries.clear()
        self._embeddings.clear()
        
        logger.info(
            f"Built hierarchy with {self.levels} levels, "
            f"communities per level: {[len(set(p.values())) for p in self._partitions]}"
        )
    
    def get_community_at_level(self, entity: str, level: int) -> int:
        """获取实体在指定层级的社区ID
        
        Args:
            entity: 实体名称
            level: 层级（0 到 levels-1）
            
        Returns:
            社区ID
        """
        if not self._partitions:
            raise ValueError("Hierarchy not built. Call build_hierarchy first.")
        
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}")
        
        return self._partitions[level].get(entity, -1)
    
    def get_communities_by_level(self, level: int) -> Dict[int, List[str]]:
        """获取指定层级的所有社区
        
        Args:
            level: 层级
            
        Returns:
            社区ID到成员列表的映射
        """
        if not self._partitions:
            raise ValueError("Hierarchy not built. Call build_hierarchy first.")
        
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}")
        
        communities: Dict[int, List[str]] = {}
        for entity, comm_id in self._partitions[level].items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(entity)
        
        return communities
    
    def get_community_members(self, level: int, community_id: int) -> List[str]:
        """获取指定社区的所有成员
        
        Args:
            level: 层级
            community_id: 社区ID
            
        Returns:
            成员实体列表
        """
        communities = self.get_communities_by_level(level)
        return communities.get(community_id, [])
    
    def get_community_summary(self, level: int, community_id: int) -> str:
        """获取社区摘要（带缓存）
        
        Args:
            level: 层级
            community_id: 社区ID
            
        Returns:
            社区摘要文本
        """
        cache_key = (level, community_id)
        
        if cache_key in self._summaries:
            return self._summaries[cache_key]
        
        members = self.get_community_members(level, community_id)
        if not members:
            return f"社区 {community_id}（层级 {level}）为空"
        
        summary = f"社区 {community_id}（层级 {level}）\n"
        summary += f"成员数量: {len(members)}\n"
        summary += f"成员: {', '.join(members[:10])}"
        if len(members) > 10:
            summary += f"... 等 {len(members)} 个实体"
        
        self._summaries[cache_key] = summary
        return summary
    
    def get_community_embedding(
        self, 
        level: int, 
        community_id: int
    ) -> Optional[List[float]]:
        """获取社区嵌入向量（带缓存）
        
        Args:
            level: 层级
            community_id: 社区ID
            
        Returns:
            嵌入向量
        """
        cache_key = (level, community_id)
        
        if cache_key in self._embeddings:
            return self._embeddings[cache_key]
        
        return None
    
    def set_community_embedding(
        self, 
        level: int, 
        community_id: int, 
        embedding: List[float]
    ) -> None:
        """设置社区嵌入向量
        
        Args:
            level: 层级
            community_id: 社区ID
            embedding: 嵌入向量
        """
        cache_key = (level, community_id)
        self._embeddings[cache_key] = embedding
    
    def find_relevant_communities(
        self, 
        query_embedding: List[float], 
        level: int = 1,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """根据查询嵌入找到最相关的社区
        
        Args:
            query_embedding: 查询嵌入向量
            level: 搜索层级
            top_k: 返回的社区数量
            
        Returns:
            (社区ID, 相似度分数) 列表
        """
        import numpy as np
        
        communities = self.get_communities_by_level(level)
        scores = []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        for comm_id in communities:
            embedding = self.get_community_embedding(level, comm_id)
            if embedding is None:
                continue
            
            comm_vec = np.array(embedding)
            comm_norm = np.linalg.norm(comm_vec)
            
            if query_norm == 0 or comm_norm == 0:
                continue
            
            similarity = np.dot(query_vec, comm_vec) / (query_norm * comm_norm)
            scores.append((comm_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取分层结构统计信息"""
        if not self._partitions:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "levels": self.levels,
            "communities_per_level": [
                len(set(p.values())) for p in self._partitions
            ],
            "total_entities": len(self._partitions[0]) if self._partitions else 0,
            "cached_summaries": len(self._summaries),
            "cached_embeddings": len(self._embeddings),
        }


_hierarchical_manager: Optional[HierarchicalCommunityManager] = None


def get_hierarchical_manager() -> HierarchicalCommunityManager:
    """获取分层社区管理器单例"""
    global _hierarchical_manager
    if _hierarchical_manager is None:
        from .config import get_settings
        settings = get_settings()
        _hierarchical_manager = HierarchicalCommunityManager(
            levels=settings.community_levels
        )
    return _hierarchical_manager
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_hierarchical_communities.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/core/hierarchical_communities.py backend/tests/test_hierarchical_communities.py
git commit -m "feat: 实现分层社区管理器"
```

---

## Task 4: 实现 LLM 缓存层

**Files:**
- Create: `backend/src/core/llm_cache.py`
- Create: `backend/tests/test_llm_cache.py`

- [ ] **Step 1: 编写 LLM 缓存测试**

```python
import pytest
from unittest.mock import Mock
from src.core.llm_cache import LLMCache, llm_cached


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
    
    def test_llm_cached_decorator(self):
        call_count = [0]
        
        @llm_cached(model="test-model")
        def generate_text(prompt: str) -> str:
            call_count[0] += 1
            return f"generated: {prompt}"
        
        result1 = generate_text("hello")
        result2 = generate_text("hello")
        
        assert result1 == result2
        assert call_count[0] == 1
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_llm_cache.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: 实现 LLMCache**

```python
import hashlib
import json
import threading
from typing import Any, Callable, Dict, Optional
from functools import wraps
from loguru import logger

from .cache import LRUCache, CacheBackend


class LLMCache:
    """LLM 调用缓存层"""
    
    def __init__(self, backend: Optional[CacheBackend] = None, max_size: int = 500):
        self.cache = backend or LRUCache(max_size=max_size)
        self._stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()
    
    def _make_key(
        self, 
        prompt: str, 
        model: str, 
        params: Dict[str, Any]
    ) -> str:
        """生成缓存键
        
        Args:
            prompt: 输入提示
            model: 模型标识
            params: 其他参数
            
        Returns:
            缓存键字符串
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            "params": _make_hashable(params)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm:{model}:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get_or_generate(
        self, 
        prompt: str,
        generate_fn: Callable[[], Any],
        model: str = "default",
        **params
    ) -> Any:
        """获取缓存或生成新结果
        
        Args:
            prompt: 输入提示
            generate_fn: 生成函数（缓存未命中时调用）
            model: 模型标识
            **params: 其他参数
            
        Returns:
            缓存或新生成的结果
        """
        cache_key = self._make_key(prompt, model, params)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            with self._lock:
                self._stats["hits"] += 1
            logger.debug(f"LLM cache hit for model {model}")
            return cached_result
        
        result = generate_fn()
        
        self.cache.set(cache_key, result)
        with self._lock:
            self._stats["misses"] += 1
        
        logger.debug(f"LLM cache miss for model {model}, cached result")
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计
        
        Returns:
            包含 hits 和 misses 的统计字典
        """
        with self._lock:
            return self._stats.copy()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        with self._lock:
            self._stats = {"hits": 0, "misses": 0}
        logger.info("LLM cache cleared")
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total == 0:
                return 0.0
            return self._stats["hits"] / total


def _make_hashable(obj: Any) -> Any:
    """将对象转换为可哈希的类型"""
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(item) for item in obj))
    else:
        return obj


_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """获取 LLM 缓存单例"""
    global _llm_cache
    if _llm_cache is None:
        from .config import get_settings
        settings = get_settings()
        _llm_cache = LLMCache(max_size=settings.llm_cache_max_size)
    return _llm_cache


def llm_cached(model: str = "default", **default_params):
    """LLM 调用缓存装饰器
    
    Usage:
        @llm_cached(model="qwen-max")
        def generate_summary(text: str) -> str:
            return llm.generate(f"总结: {text}")
    
    Args:
        model: 模型标识
        **default_params: 默认参数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .config import get_settings
            settings = get_settings()
            
            if not settings.llm_cache_enabled:
                return func(*args, **kwargs)
            
            cache = get_llm_cache()
            
            prompt_parts = [func.__name__, str(args), str(kwargs)]
            prompt = ":".join(prompt_parts)
            
            params = {**default_params, **kwargs}
            
            def generate():
                return func(*args, **kwargs)
            
            return cache.get_or_generate(
                prompt=prompt,
                generate_fn=generate,
                model=model,
                **params
            )
        
        return wrapper
    
    return decorator
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_llm_cache.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/core/llm_cache.py backend/tests/test_llm_cache.py
git commit -m "feat: 实现 LLM 缓存层"
```

---

## Task 5: 集成到现有模块

**Files:**
- Modify: `backend/src/core/community_detector.py`
- Modify: `backend/src/core/summary_generator.py`

- [ ] **Step 1: 修改 community_detector.py 支持 Leiden**

在 `CommunityDetector` 类中添加：

```python
from typing import Optional
from .leiden_detector import LeidenCommunityDetector, get_leiden_detector

class CommunityDetector:
    def __init__(
        self, 
        algorithm: str = "leiden",
        resolution: float = 1.0
    ):
        self.algorithm = algorithm
        self.resolution = resolution
        
        if algorithm == "leiden":
            self._detector = LeidenCommunityDetector(resolution=resolution)
        else:
            import community as community_louvain
            self._louvain = community_louvain
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """检测社区"""
        if self.algorithm == "leiden":
            return self._detector.detect_communities(graph)
        else:
            return self._louvain.best_partition(graph, resolution=self.resolution)
```

- [ ] **Step 2: 修改 summary_generator.py 集成 LLM 缓存**

在 `SummaryGenerator` 类中添加：

```python
from .llm_cache import get_llm_cache, llm_cached

class SummaryGenerator:
    def __init__(self, ...):
        # ... 现有初始化 ...
        self._llm_cache = get_llm_cache()
    
    @llm_cached(model="summary-generator")
    def generate_community_summary_with_cache(
        self, 
        level: int, 
        community_id: int
    ) -> str:
        """生成社区摘要（带 LLM 缓存）"""
        return self.generate_community_summary(community_id, level)
```

- [ ] **Step 3: 运行所有测试验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/src/core/community_detector.py backend/src/core/summary_generator.py
git commit -m "feat: 集成 Leiden 和 LLM 缓存到现有模块"
```

---

## Task 6: 添加 API 端点

**Files:**
- Modify: `backend/src/api/routes.py`

- [ ] **Step 1: 添加社区相关 API 端点**

```python
from ..core.hierarchical_communities import get_hierarchical_manager
from ..core.llm_cache import get_llm_cache

@router.get("/community/stats")
async def get_community_stats():
    """获取分层社区统计信息"""
    manager = get_hierarchical_manager()
    return manager.get_stats()

@router.get("/community/level/{level}")
async def get_communities_at_level(level: int):
    """获取指定层级的所有社区"""
    manager = get_hierarchical_manager()
    communities = manager.get_communities_by_level(level)
    return {
        "level": level,
        "communities": {
            k: {"member_count": len(v), "members": v[:10]}
            for k, v in communities.items()
        }
    }

@router.get("/community/{level}/{community_id}")
async def get_community_detail(level: int, community_id: int):
    """获取社区详情"""
    manager = get_hierarchical_manager()
    members = manager.get_community_members(level, community_id)
    summary = manager.get_community_summary(level, community_id)
    
    return {
        "level": level,
        "community_id": community_id,
        "member_count": len(members),
        "members": members,
        "summary": summary
    }

@router.get("/cache/stats")
async def get_cache_stats():
    """获取 LLM 缓存统计"""
    cache = get_llm_cache()
    stats = cache.get_stats()
    stats["hit_rate"] = cache.get_hit_rate()
    return stats

@router.post("/cache/clear")
async def clear_cache():
    """清空 LLM 缓存"""
    cache = get_llm_cache()
    cache.clear()
    return {"status": "cleared"}
```

- [ ] **Step 2: 测试 API 端点**

Run: `curl http://localhost:8000/api/v1/community/stats`
Expected: JSON response with community statistics

- [ ] **Step 3: Commit**

```bash
git add backend/src/api/routes.py
git commit -m "feat: 添加社区和缓存管理 API 端点"
```

---

## Task 7: 集成测试

**Files:**
- Create: `backend/tests/integration/test_phase1_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
import pytest
import networkx as nx
from src.core.leiden_detector import LeidenCommunityDetector
from src.core.hierarchical_communities import HierarchicalCommunityManager
from src.core.llm_cache import LLMCache


class TestPhase1Integration:
    
    def test_full_pipeline(self):
        detector = LeidenCommunityDetector()
        manager = HierarchicalCommunityManager(levels=3)
        cache = LLMCache()
        
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
        
        partition = detector.detect_communities(str_graph)
        assert len(partition) == 34
        
        modularity = detector.compute_modularity(str_graph, partition)
        assert modularity > 0.3
        
        manager.build_hierarchy(str_graph)
        stats = manager.get_stats()
        
        assert stats["status"] == "built"
        assert stats["levels"] == 3
        
        call_count = [0]
        def generate():
            call_count[0] += 1
            return "test result"
        
        result1 = cache.get_or_generate("prompt", generate, model="test")
        result2 = cache.get_or_generate("prompt", generate, model="test")
        
        assert result1 == result2
        assert call_count[0] == 1
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
```

- [ ] **Step 2: 运行集成测试**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/integration/test_phase1_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/integration/test_phase1_integration.py
git commit -m "test: 添加 Phase 1 集成测试"
```

---

## Task 8: 最终验证和文档更新

- [ ] **Step 1: 运行完整测试套件**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/ -v --cov=src/core`
Expected: All tests PASS, coverage > 80%

- [ ] **Step 2: 启动服务验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m uvicorn src.main:app --reload --port 8000`
Expected: Server starts successfully

- [ ] **Step 3: 验证 API 端点**

Run: `curl http://localhost:8000/api/v1/health`
Expected: `{"status": "healthy"}`

Run: `curl http://localhost:8000/api/v1/cache/stats`
Expected: JSON with cache statistics

- [ ] **Step 4: 更新 README**

在 README.md 中添加 Phase 1 功能说明：

```markdown
## Phase 1 功能

### 分层社区检测
- 使用 Leiden 算法替代 Louvain
- 支持 3 级分层：实体 → 社区 → 全局
- API: `/api/v1/community/level/{level}`

### LLM 缓存层
- 全量 LLM 调用缓存
- 支持内存和 Redis 后端
- API: `/api/v1/cache/stats`
```

- [ ] **Step 5: 最终 Commit**

```bash
git add .
git commit -m "feat: Phase 1 完成 - 分层社区检测 + LLM 缓存层"
```

---

## 成功标准验证

- [ ] Leiden 算法替换完成，模块度 > 0.3
- [ ] 3 级分层结构正确构建
- [ ] LLM 缓存命中率测试 > 50%
- [ ] 所有单元测试和集成测试通过
- [ ] API 端点可访问并返回正确响应
