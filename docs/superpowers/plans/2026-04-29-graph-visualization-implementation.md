# 知识图谱可视化功能实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个功能完整的知识图谱可视化系统，支持图谱浏览、节点交互、搜索筛选和与现有查询功能的集成。

**Architecture:** 采用前后端分离架构，前端使用vis-network进行可视化渲染，后端通过FastAPI提供图谱数据API，Neo4j作为图数据库。新增独立的GraphView页面，并与ChatView和RetrievalView集成。

**Tech Stack:** 
- 前端: React, TypeScript, vis-network, Tailwind CSS
- 后端: FastAPI, Neo4j, Pydantic
- 测试: pytest, Vitest, Playwright

---

## 文件结构

### 后端文件
```
backend/src/
├── api/
│   ├── routes.py              # 修改：添加图谱API路由
│   └── schemas.py             # 修改：添加图谱相关Schema
└── core/
    └── neo4j_client.py        # 修改：添加图谱查询方法
```

### 前端文件
```
frontend/src/
├── components/
│   ├── GraphView.tsx          # 新增：图谱可视化主页面
│   ├── graph/
│   │   ├── GraphCanvas.tsx    # 新增：vis-network封装组件
│   │   ├── GraphControls.tsx  # 新增：控制面板组件
│   │   ├── GraphSearch.tsx    # 新增：搜索组件
│   │   ├── NodeDetails.tsx    # 新增：节点详情面板
│   │   ├── GraphLegend.tsx    # 新增：图例组件
│   │   └── GraphStats.tsx     # 新增：统计信息组件
│   ├── Sidebar.tsx            # 修改：添加图谱导航项
│   ├── ChatView.tsx           # 修改：添加图谱跳转按钮
│   └── RetrievalView.tsx      # 修改：添加图谱跳转按钮
├── lib/
│   ├── api.ts                 # 修改：添加图谱API方法
│   ├── graphUtils.ts          # 新增：图谱数据处理工具
│   └── graphConfig.ts         # 新增：图谱配置
└── types/
    └── graph.ts               # 新增：图谱类型定义
```

---

## Task 1: 后端Schema定义

**Files:**
- Modify: `backend/src/api/schemas.py`

- [ ] **Step 1: 添加图谱相关Schema类**

在 `backend/src/api/schemas.py` 文件末尾添加以下代码：

```python
class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]
    degree: Optional[int] = None


class GraphEdge(BaseModel):
    id: str
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    type: str
    properties: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    node_labels: List[str]
    relationship_types: List[str]


class GraphDataResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats


class GraphSearchResult(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]
    score: Optional[float] = None


class GraphSearchResponse(BaseModel):
    results: List[GraphSearchResult]
    total: int


class NodeRelationships(BaseModel):
    incoming: List[Dict[str, Any]]
    outgoing: List[Dict[str, Any]]


class NodeDetailResponse(BaseModel):
    node: GraphNode
    relationships: NodeRelationships
    neighbors: List[GraphNode]


class QueryResultRequest(BaseModel):
    query: str
    node_ids: List[str]
    max_depth: int = Field(default=2, ge=1, le=3)
```

- [ ] **Step 2: 验证Schema定义**

运行: `cd backend && python -c "from src.api.schemas import GraphNode, GraphEdge, GraphDataResponse; print('Schema imported successfully')"`
预期: 输出 "Schema imported successfully"

- [ ] **Step 3: 提交Schema定义**

```bash
git add backend/src/api/schemas.py
git commit -m "feat: add graph visualization schemas"
```

---

## Task 2: Neo4j图谱查询方法

**Files:**
- Modify: `backend/src/core/neo4j_client.py`

- [ ] **Step 1: 添加get_graph_data方法**

在 `Neo4jClient` 类中添加以下方法：

```python
    def get_graph_data(
        self,
        node_label: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取图谱数据"""
        if node_label:
            node_query = f"""
            MATCH (n:{node_label})
            WITH n, size((n)--()) as degree
            RETURN n, id(n) as node_id, degree, labels(n) as labels
            ORDER BY degree DESC
            SKIP $offset
            LIMIT $limit
            """
        else:
            node_query = """
            MATCH (n)
            WITH n, size((n)--()) as degree
            RETURN n, id(n) as node_id, degree, labels(n) as labels
            ORDER BY degree DESC
            SKIP $offset
            LIMIT $limit
            """
        
        with self.session() as session:
            nodes_result = session.run(node_query, offset=offset, limit=limit)
            nodes = []
            node_ids = []
            
            for record in nodes_result:
                node_data = record.data()
                node_id = str(node_data['node_id'])
                node_ids.append(node_id)
                
                labels = node_data.get('labels', [])
                label = labels[0] if labels else 'Node'
                
                node_props = dict(node_data['n']) if node_data.get('n') else {}
                
                nodes.append({
                    'id': node_id,
                    'label': label,
                    'properties': node_props,
                    'degree': node_data.get('degree', 0)
                })
            
            if not node_ids:
                return {
                    'nodes': [],
                    'edges': [],
                    'stats': {
                        'total_nodes': 0,
                        'total_edges': 0,
                        'node_labels': [],
                        'relationship_types': []
                    }
                }
            
            edge_query = """
            MATCH (n)-[r]->(m)
            WHERE id(n) IN $node_ids AND id(m) IN $node_ids
            RETURN id(r) as edge_id, id(startNode(r)) as from_id, 
                   id(endNode(r)) as to_id, type(r) as rel_type, properties(r) as props
            """
            
            edges_result = session.run(edge_query, node_ids=[int(nid) for nid in node_ids])
            edges = []
            
            for record in edges_result:
                edge_data = record.data()
                edges.append({
                    'id': str(edge_data['edge_id']),
                    'from': str(edge_data['from_id']),
                    'to': str(edge_data['to_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
            
            stats_query = """
            MATCH (n)
            WITH count(n) as total_nodes
            MATCH ()-[r]->()
            WITH total_nodes, count(r) as total_edges
            CALL db.labels() YIELD label
            WITH total_nodes, total_edges, collect(label) as node_labels
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN total_nodes, total_edges, node_labels, collect(relationshipType) as relationship_types
            """
            
            stats_result = session.run(stats_query)
            stats_data = stats_result.single()
            
            stats = {
                'total_nodes': stats_data['total_nodes'] if stats_data else 0,
                'total_edges': stats_data['total_edges'] if stats_data else 0,
                'node_labels': stats_data['node_labels'] if stats_data else [],
                'relationship_types': stats_data['relationship_types'] if stats_data else []
            }
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': stats
            }
```

- [ ] **Step 2: 添加search_nodes方法**

在 `Neo4jClient` 类中添加以下方法：

```python
    def search_nodes(
        self,
        query: str,
        node_label: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """搜索节点"""
        if node_label:
            search_query = f"""
            MATCH (n:{node_label})
            WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $query)
            RETURN n, id(n) as node_id, labels(n) as labels
            LIMIT $limit
            """
        else:
            search_query = """
            MATCH (n)
            WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $query)
            RETURN n, id(n) as node_id, labels(n) as labels
            LIMIT $limit
            """
        
        with self.session() as session:
            result = session.run(search_query, query=query, limit=limit)
            nodes = []
            
            for record in result:
                node_data = record.data()
                labels = node_data.get('labels', [])
                label = labels[0] if labels else 'Node'
                
                node_props = dict(node_data['n']) if node_data.get('n') else {}
                
                nodes.append({
                    'id': str(node_data['node_id']),
                    'label': label,
                    'properties': node_props
                })
            
            return nodes
```

- [ ] **Step 3: 添加get_node_detail方法**

在 `Neo4jClient` 类中添加以下方法：

```python
    def get_node_detail(self, node_id: str) -> Dict[str, Any]:
        """获取节点详情"""
        with self.session() as session:
            node_query = """
            MATCH (n)
            WHERE id(n) = $node_id
            RETURN n, id(n) as node_id, labels(n) as labels
            """
            
            node_result = session.run(node_query, node_id=int(node_id))
            node_data = node_result.single()
            
            if not node_data:
                return None
            
            labels = node_data.get('labels', [])
            label = labels[0] if labels else 'Node'
            node_props = dict(node_data['n']) if node_data.get('n') else {}
            
            node = {
                'id': str(node_data['node_id']),
                'label': label,
                'properties': node_props
            }
            
            incoming_query = """
            MATCH (m)-[r]->(n)
            WHERE id(n) = $node_id
            RETURN id(r) as edge_id, id(m) as from_id, type(r) as rel_type, 
                   properties(r) as props, labels(m) as from_labels
            """
            
            incoming_result = session.run(incoming_query, node_id=int(node_id))
            incoming = []
            neighbors = []
            
            for record in incoming_result:
                edge_data = record.data()
                incoming.append({
                    'from_node': str(edge_data['from_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
                
                from_labels = edge_data.get('from_labels', [])
                neighbor_label = from_labels[0] if from_labels else 'Node'
                
                neighbors.append({
                    'id': str(edge_data['from_id']),
                    'label': neighbor_label,
                    'properties': {}
                })
            
            outgoing_query = """
            MATCH (n)-[r]->(m)
            WHERE id(n) = $node_id
            RETURN id(r) as edge_id, id(m) as to_id, type(r) as rel_type, 
                   properties(r) as props, labels(m) as to_labels
            """
            
            outgoing_result = session.run(outgoing_query, node_id=int(node_id))
            outgoing = []
            
            for record in outgoing_result:
                edge_data = record.data()
                outgoing.append({
                    'to_node': str(edge_data['to_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
                
                to_labels = edge_data.get('to_labels', [])
                neighbor_label = to_labels[0] if to_labels else 'Node'
                
                neighbors.append({
                    'id': str(edge_data['to_id']),
                    'label': neighbor_label,
                    'properties': {}
                })
            
            unique_neighbors = []
            seen_ids = set()
            for neighbor in neighbors:
                if neighbor['id'] not in seen_ids:
                    seen_ids.add(neighbor['id'])
                    unique_neighbors.append(neighbor)
            
            return {
                'node': node,
                'relationships': {
                    'incoming': incoming,
                    'outgoing': outgoing
                },
                'neighbors': unique_neighbors
            }
```

- [ ] **Step 4: 添加get_node_neighbors方法**

在 `Neo4jClient` 类中添加以下方法：

```python
    def get_node_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取节点邻居"""
        if relationship_type:
            neighbor_query = f"""
            MATCH path = (n)-[r:{relationship_type}*1..{depth}]-(m)
            WHERE id(n) = $node_id
            RETURN nodes(path) as path_nodes, relationships(path) as path_rels
            """
        else:
            neighbor_query = f"""
            MATCH path = (n)-[r*1..{depth}]-(m)
            WHERE id(n) = $node_id
            RETURN nodes(path) as path_nodes, relationships(path) as path_rels
            """
        
        with self.session() as session:
            result = session.run(neighbor_query, node_id=int(node_id))
            
            all_nodes = {}
            all_edges = {}
            
            for record in result:
                path_data = record.data()
                
                for node in path_data.get('path_nodes', []):
                    node_id_str = str(node.element_id)
                    if node_id_str not in all_nodes:
                        labels = list(node.labels)
                        all_nodes[node_id_str] = {
                            'id': node_id_str,
                            'label': labels[0] if labels else 'Node',
                            'properties': dict(node)
                        }
                
                for rel in path_data.get('path_rels', []):
                    edge_id = str(rel.element_id)
                    if edge_id not in all_edges:
                        all_edges[edge_id] = {
                            'id': edge_id,
                            'from': str(rel.start_node.element_id),
                            'to': str(rel.end_node.element_id),
                            'type': rel.type,
                            'properties': dict(rel)
                        }
            
            return {
                'nodes': list(all_nodes.values()),
                'edges': list(all_edges.values()),
                'center_node': node_id
            }
```

- [ ] **Step 5: 验证Neo4j方法**

运行: `cd backend && python -c "from src.core.neo4j_client import Neo4jClient; print('Neo4j methods imported successfully')"`
预期: 输出 "Neo4j methods imported successfully"

- [ ] **Step 6: 提交Neo4j方法**

```bash
git add backend/src/core/neo4j_client.py
git commit -m "feat: add graph query methods to Neo4j client"
```

---

## Task 3: 后端API路由

**Files:**
- Modify: `backend/src/api/routes.py`

- [ ] **Step 1: 添加图谱API导入**

在 `backend/src/api/routes.py` 文件顶部的导入部分添加：

```python
from .schemas import (
    QuestionRequest,
    QuestionResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    SchemaResponse,
    MetricsResponse,
    GraphDataResponse,
    GraphSearchResponse,
    NodeDetailResponse,
    QueryResultRequest,
)
```

- [ ] **Step 2: 添加get_graph_data路由**

在 `backend/src/api/routes.py` 文件末尾添加：

```python
@router.get("/graph/data", response_model=GraphDataResponse)
async def get_graph_data(
    node_label: Optional[str] = None,
    limit: int = 500,
    offset: int = 0
):
    """获取图谱数据"""
    try:
        client = get_neo4j_client()
        data = client.get_graph_data(
            node_label=node_label,
            limit=min(limit, 1000),
            offset=offset
        )
        return GraphDataResponse(**data)
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: 添加search_nodes路由**

```python
@router.get("/graph/search", response_model=GraphSearchResponse)
async def search_nodes(
    query: str,
    node_label: Optional[str] = None,
    limit: int = 20
):
    """搜索节点"""
    try:
        client = get_neo4j_client()
        results = client.search_nodes(
            query=query,
            node_label=node_label,
            limit=min(limit, 100)
        )
        return GraphSearchResponse(
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Failed to search nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: 添加get_node_detail路由**

```python
@router.get("/graph/node/{node_id}", response_model=NodeDetailResponse)
async def get_node_detail(node_id: str):
    """获取节点详情"""
    try:
        client = get_neo4j_client()
        data = client.get_node_detail(node_id)
        if not data:
            raise HTTPException(status_code=404, detail="Node not found")
        return NodeDetailResponse(**data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 5: 添加get_node_neighbors路由**

```python
@router.get("/graph/node/{node_id}/neighbors")
async def get_node_neighbors(
    node_id: str,
    depth: int = 1,
    relationship_type: Optional[str] = None
):
    """获取节点邻居"""
    try:
        client = get_neo4j_client()
        data = client.get_node_neighbors(
            node_id=node_id,
            depth=min(depth, 3),
            relationship_type=relationship_type
        )
        return data
    except Exception as e:
        logger.error(f"Failed to get node neighbors: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 6: 添加get_query_result_graph路由**

```python
@router.post("/graph/query-result")
async def get_query_result_graph(request: QueryResultRequest):
    """获取查询结果图谱"""
    try:
        client = get_neo4j_client()
        
        all_nodes = {}
        all_edges = {}
        
        for node_id in request.node_ids:
            neighbor_data = client.get_node_neighbors(
                node_id=node_id,
                depth=request.max_depth
            )
            
            for node in neighbor_data.get('nodes', []):
                all_nodes[node['id']] = node
            
            for edge in neighbor_data.get('edges', []):
                all_edges[edge['id']] = edge
        
        stats_query = """
        MATCH (n)
        WITH count(n) as total_nodes
        MATCH ()-[r]->()
        WITH total_nodes, count(r) as total_edges
        CALL db.labels() YIELD label
        WITH total_nodes, total_edges, collect(label) as node_labels
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN total_nodes, total_edges, node_labels, collect(relationshipType) as relationship_types
        """
        
        stats_result = client.execute_query(stats_query)
        stats_data = stats_result[0] if stats_result else {}
        
        return {
            'nodes': list(all_nodes.values()),
            'edges': list(all_edges.values()),
            'stats': {
                'total_nodes': stats_data.get('total_nodes', 0),
                'total_edges': stats_data.get('total_edges', 0),
                'node_labels': stats_data.get('node_labels', []),
                'relationship_types': stats_data.get('relationship_types', [])
            }
        }
    except Exception as e:
        logger.error(f"Failed to get query result graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 7: 验证API路由**

运行: `cd backend && python -c "from src.api.routes import get_graph_data, search_nodes, get_node_detail; print('API routes imported successfully')"`
预期: 输出 "API routes imported successfully"

- [ ] **Step 8: 提交API路由**

```bash
git add backend/src/api/routes.py
git commit -m "feat: add graph visualization API endpoints"
```

---

## Task 4: 前端类型定义

**Files:**
- Create: `frontend/src/types/graph.ts`

- [ ] **Step 1: 创建图谱类型定义文件**

创建文件 `frontend/src/types/graph.ts` 并添加以下内容：

```typescript
export interface GraphNode {
  id: string;
  label: string;
  properties: Record<string, any>;
  degree?: number;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  properties?: Record<string, any>;
}

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_labels: string[];
  relationship_types: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
}

export interface GraphSearchResult {
  id: string;
  label: string;
  properties: Record<string, any>;
  score?: number;
}

export interface GraphSearchResponse {
  results: GraphSearchResult[];
  total: number;
}

export interface NodeRelationships {
  incoming: Array<{
    from_node: string;
    type: string;
    properties?: Record<string, any>;
  }>;
  outgoing: Array<{
    to_node: string;
    type: string;
    properties?: Record<string, any>;
  }>;
}

export interface NodeDetail {
  node: GraphNode;
  relationships: NodeRelationships;
  neighbors: GraphNode[];
}

export interface GraphFilters {
  nodeLabels: string[];
  relationshipTypes: string[];
  propertyRange?: Record<string, [any, any]>;
}

export interface ViewSettings {
  showLabels: boolean;
  showArrows: boolean;
  layout: LayoutType;
  zoomLevel: number;
}

export type LayoutType = 'force' | 'hierarchical' | 'circular';
```

- [ ] **Step 2: 验证类型定义**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交类型定义**

```bash
git add frontend/src/types/graph.ts
git commit -m "feat: add graph visualization type definitions"
```

---

## Task 5: 图谱配置文件

**Files:**
- Create: `frontend/src/lib/graphConfig.ts`

- [ ] **Step 1: 创建图谱配置文件**

创建文件 `frontend/src/lib/graphConfig.ts` 并添加以下内容：

```typescript
export const nodeColors: Record<string, string> = {
  Person: '#3B82F6',
  Organization: '#10B981',
  Location: '#F59E0B',
  Event: '#8B5CF6',
  Concept: '#EC4899',
  default: '#6B7280'
};

export const edgeColors: Record<string, string> = {
  KNOWS: '#3B82F6',
  WORKS_FOR: '#10B981',
  LOCATED_IN: '#F59E0B',
  PART_OF: '#8B5CF6',
  RELATED_TO: '#EC4899',
  default: '#9CA3AF'
};

export const nodeSizeConfig = {
  min: 20,
  max: 60,
  scale: 'log' as const
};

export const visNetworkOptions = {
  physics: {
    enabled: true,
    barnesHut: {
      gravitationalConstant: -3000,
      centralGravity: 0.3,
      springLength: 100,
      springConstant: 0.05,
      damping: 0.09
    },
    stabilization: {
      enabled: true,
      iterations: 200,
      updateInterval: 25,
      onlyDynamicEdges: false,
      fit: true
    },
    timestep: 0.5,
    adaptiveTimestep: true
  },
  interaction: {
    hover: true,
    tooltipDelay: 200,
    zoomView: true,
    dragView: true
  },
  rendering: {
    hideEdgesOnDrag: true,
    hideNodesOnDrag: false,
    hideEdgesOnZoom: true,
    hideNodesOnZoom: false
  },
  nodes: {
    shape: 'dot',
    scaling: {
      min: nodeSizeConfig.min,
      max: nodeSizeConfig.max,
      label: {
        enabled: true,
        min: 14,
        max: 30,
        maxVisible: 30,
        drawThreshold: 5
      }
    },
    borderWidth: 2,
    borderWidthSelected: 4,
    font: {
      size: 14,
      color: '#1F2937'
    }
  },
  edges: {
    width: 2,
    smooth: {
      enabled: true,
      type: 'continuous',
      roundness: 0.5
    },
    arrows: {
      to: {
        enabled: true,
        scaleFactor: 0.8
      }
    },
    font: {
      size: 12,
      align: 'middle'
    }
  }
};

export function getNodeColor(label: string): string {
  return nodeColors[label] || nodeColors.default;
}

export function getEdgeColor(type: string): string {
  return edgeColors[type] || edgeColors.default;
}

export function calculateNodeSize(degree: number, maxDegree: number): number {
  if (maxDegree === 0) return nodeSizeConfig.min;
  
  const normalizedDegree = degree / maxDegree;
  const logScale = Math.log(1 + normalizedDegree * 9) / Math.log(10);
  
  return nodeSizeConfig.min + logScale * (nodeSizeConfig.max - nodeSizeConfig.min);
}
```

- [ ] **Step 2: 验证配置文件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交配置文件**

```bash
git add frontend/src/lib/graphConfig.ts
git commit -m "feat: add graph visualization configuration"
```

---

## Task 6: 图谱数据处理工具

**Files:**
- Create: `frontend/src/lib/graphUtils.ts`

- [ ] **Step 1: 创建图谱工具文件**

创建文件 `frontend/src/lib/graphUtils.ts` 并添加以下内容：

```typescript
import type { Data, DataSet } from 'vis-data';
import type { Node, Edge, Options } from 'vis-network';
import type { GraphNode, GraphEdge } from '../types/graph';
import { getNodeColor, getEdgeColor, calculateNodeSize } from './graphConfig';

export function convertToVisNodes(
  nodes: GraphNode[]
): DataSet<Node> {
  const vis = require('vis-data');
  const maxDegree = Math.max(...nodes.map(n => n.degree || 0), 1);
  
  const visNodes = nodes.map(node => ({
    id: node.id,
    label: node.properties.name || node.properties.title || node.id,
    color: {
      background: getNodeColor(node.label),
      border: getNodeColor(node.label),
      highlight: {
        background: getNodeColor(node.label),
        border: '#1F2937'
      },
      hover: {
        background: getNodeColor(node.label),
        border: '#374151'
      }
    },
    size: calculateNodeSize(node.degree || 0, maxDegree),
    group: node.label,
    title: `${node.label}: ${node.properties.name || node.id}\n${Object.entries(node.properties)
      .filter(([key]) => key !== 'name' && key !== 'title')
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n')}`,
    font: {
      size: 14,
      color: '#1F2937'
    }
  }));
  
  return new vis.DataSet(visNodes);
}

export function convertToVisEdges(
  edges: GraphEdge[]
): DataSet<Edge> {
  const vis = require('vis-data');
  
  const visEdges = edges.map(edge => ({
    id: edge.id,
    from: edge.from,
    to: edge.to,
    label: edge.type,
    color: {
      color: getEdgeColor(edge.type),
      highlight: '#1F2937',
      hover: '#374151'
    },
    arrows: 'to',
    smooth: {
      enabled: true,
      type: 'curvedCW',
      roundness: 0.2
    },
    title: `${edge.type}${edge.properties && Object.keys(edge.properties).length > 0 
      ? '\n' + Object.entries(edge.properties)
          .map(([key, value]) => `${key}: ${value}`)
          .join('\n')
      : ''}`
  }));
  
  return new vis.DataSet(visEdges);
}

export function filterNodesByLabel(
  nodes: GraphNode[],
  labels: string[]
): GraphNode[] {
  if (labels.length === 0) return nodes;
  return nodes.filter(node => labels.includes(node.label));
}

export function filterEdgesByType(
  edges: GraphEdge[],
  types: string[]
): GraphEdge[] {
  if (types.length === 0) return edges;
  return edges.filter(edge => types.includes(edge.type));
}

export function getConnectedNodes(
  nodeId: string,
  edges: GraphEdge[]
): string[] {
  const connected = new Set<string>();
  
  edges.forEach(edge => {
    if (edge.from === nodeId) {
      connected.add(edge.to);
    } else if (edge.to === nodeId) {
      connected.add(edge.from);
    }
  });
  
  return Array.from(connected);
}

export function highlightNode(
  nodeId: string,
  nodes: DataSet<Node>
): void {
  const node = nodes.get(nodeId);
  if (node) {
    nodes.update({
      id: nodeId,
      borderWidth: 4,
      shadow: true
    });
  }
}

export function unhighlightNode(
  nodeId: string,
  nodes: DataSet<Node>
): void {
  const node = nodes.get(nodeId);
  if (node) {
    nodes.update({
      id: nodeId,
      borderWidth: 2,
      shadow: false
    });
  }
}

export function focusOnNode(
  network: any,
  nodeId: string,
  options?: { scale?: number; animation?: boolean }
): void {
  const scale = options?.scale || 1.0;
  const animation = options?.animation !== false;
  
  network.focus(nodeId, {
    scale,
    animation: animation ? {
      duration: 500,
      easingFunction: 'easeInOutQuad'
    } : false
  });
}

export function fitToView(network: any): void {
  network.fit({
    animation: {
      duration: 500,
      easingFunction: 'easeInOutQuad'
    }
  });
}
```

- [ ] **Step 2: 验证工具文件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交工具文件**

```bash
git add frontend/src/lib/graphUtils.ts
git commit -m "feat: add graph data processing utilities"
```

---

## Task 7: 扩展API服务

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: 添加图谱API方法**

在 `frontend/src/lib/api.ts` 文件中，找到 `api` 对象定义，在末尾添加以下方法：

```typescript
  getGraphData: async (params?: {
    node_label?: string;
    limit?: number;
    offset?: number;
  }): Promise<GraphData> => {
    const queryParams = new URLSearchParams();
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    
    const response = await fetch(`${API_BASE_URL}/graph/data?${queryParams}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch graph data: ${response.statusText}`);
    }
    return response.json();
  },

  searchNodes: async (query: string, params?: {
    node_label?: string;
    limit?: number;
  }): Promise<GraphSearchResponse> => {
    const queryParams = new URLSearchParams();
    queryParams.append('query', query);
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    
    const response = await fetch(`${API_BASE_URL}/graph/search?${queryParams}`);
    if (!response.ok) {
      throw new Error(`Failed to search nodes: ${response.statusText}`);
    }
    return response.json();
  },

  getNodeDetail: async (nodeId: string): Promise<NodeDetail> => {
    const response = await fetch(`${API_BASE_URL}/graph/node/${nodeId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch node detail: ${response.statusText}`);
    }
    return response.json();
  },

  getNodeNeighbors: async (nodeId: string, params?: {
    depth?: number;
    relationship_type?: string;
  }): Promise<GraphData> => {
    const queryParams = new URLSearchParams();
    if (params?.depth) queryParams.append('depth', params.depth.toString());
    if (params?.relationship_type) queryParams.append('relationship_type', params.relationship_type);
    
    const response = await fetch(`${API_BASE_URL}/graph/node/${nodeId}/neighbors?${queryParams}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch node neighbors: ${response.statusText}`);
    }
    return response.json();
  },

  getQueryResultGraph: async (request: {
    query: string;
    node_ids: string[];
    max_depth?: number;
  }): Promise<GraphData> => {
    const response = await fetch(`${API_BASE_URL}/graph/query-result`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch query result graph: ${response.statusText}`);
    }
    return response.json();
  },
```

- [ ] **Step 2: 添加图谱类型导入**

在 `frontend/src/lib/api.ts` 文件顶部添加类型导入：

```typescript
import type { 
  HealthResponse, 
  SchemaResponse, 
  GraphData, 
  GraphSearchResponse, 
  NodeDetail 
} from '../types/graph';
```

- [ ] **Step 3: 验证API扩展**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 4: 提交API扩展**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: add graph visualization API methods"
```

---

## Task 8: GraphCanvas组件

**Files:**
- Create: `frontend/src/components/graph/GraphCanvas.tsx`

- [ ] **Step 1: 创建GraphCanvas组件**

创建文件 `frontend/src/components/graph/GraphCanvas.tsx` 并添加以下内容：

```typescript
import React, { useEffect, useRef, useCallback } from 'react';
import type { Network } from 'vis-network';
import type { GraphNode, GraphEdge } from '../../types/graph';
import { visNetworkOptions } from '../../lib/graphConfig';
import { convertToVisNodes, convertToVisEdges, focusOnNode, fitToView } from '../../lib/graphUtils';

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (nodeId: string) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  highlightNodes?: string[];
  className?: string;
}

export function GraphCanvas({
  nodes,
  edges,
  onNodeClick,
  onNodeDoubleClick,
  highlightNodes = [],
  className = ''
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesRef = useRef<any>(null);
  const edgesRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const initNetwork = async () => {
      const vis = await import('vis-network/standalone');
      
      nodesRef.current = convertToVisNodes(nodes);
      edgesRef.current = convertToVisEdges(edges);
      
      const data = {
        nodes: nodesRef.current,
        edges: edgesRef.current
      };

      networkRef.current = new vis.Network(
        containerRef.current!,
        data,
        visNetworkOptions
      );

      networkRef.current.on('click', (params: any) => {
        if (params.nodes.length > 0 && onNodeClick) {
          onNodeClick(params.nodes[0]);
        }
      });

      networkRef.current.on('doubleClick', (params: any) => {
        if (params.nodes.length > 0 && onNodeDoubleClick) {
          onNodeDoubleClick(params.nodes[0]);
        }
      });

      networkRef.current.once('stabilizationIterationsDone', () => {
        fitToView(networkRef.current!);
      });
    };

    initNetwork();

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!networkRef.current || !nodesRef.current) return;

    nodesRef.current.clear();
    nodesRef.current.add(convertToVisNodes(nodes).get());
    
    edgesRef.current.clear();
    edgesRef.current.add(convertToVisEdges(edges).get());
    
    networkRef.current.redraw();
  }, [nodes, edges]);

  useEffect(() => {
    if (!networkRef.current || !nodesRef.current) return;

    nodesRef.current.forEach((node: any) => {
      const shouldHighlight = highlightNodes.includes(node.id);
      nodesRef.current.update({
        id: node.id,
        borderWidth: shouldHighlight ? 4 : 2,
        shadow: shouldHighlight
      });
    });
  }, [highlightNodes]);

  const handleZoomIn = useCallback(() => {
    if (!networkRef.current) return;
    const scale = networkRef.current.getScale();
    networkRef.current.moveTo({ scale: scale * 1.2 });
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!networkRef.current) return;
    const scale = networkRef.current.getScale();
    networkRef.current.moveTo({ scale: scale / 1.2 });
  }, []);

  const handleResetView = useCallback(() => {
    if (!networkRef.current) return;
    fitToView(networkRef.current);
  }, []);

  const handleFocusNode = useCallback((nodeId: string) => {
    if (!networkRef.current) return;
    focusOnNode(networkRef.current, nodeId, { scale: 1.5 });
  }, []);

  return (
    <div className={`relative w-full h-full ${className}`}>
      <div 
        ref={containerRef} 
        className="w-full h-full"
        style={{ backgroundColor: '#F9FAFB' }}
      />
      
      <div className="absolute bottom-4 right-4 flex gap-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
          title="放大"
        >
          <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
          </svg>
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
          title="缩小"
        >
          <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
          </svg>
        </button>
        <button
          onClick={handleResetView}
          className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
          title="重置视图"
        >
          <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphCanvas组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphCanvas组件**

```bash
git add frontend/src/components/graph/GraphCanvas.tsx
git commit -m "feat: add GraphCanvas component for vis-network"
```

---

## Task 9: GraphSearch组件

**Files:**
- Create: `frontend/src/components/graph/GraphSearch.tsx`

- [ ] **Step 1: 创建GraphSearch组件**

创建文件 `frontend/src/components/graph/GraphSearch.tsx` 并添加以下内容：

```typescript
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import type { GraphSearchResult } from '../../types/graph';
import { api } from '../../lib/api';

interface GraphSearchProps {
  onNodeSelect: (nodeId: string) => void;
  placeholder?: string;
  className?: string;
}

export function GraphSearch({
  onNodeSelect,
  placeholder = '搜索节点...',
  className = ''
}: GraphSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<GraphSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      setShowResults(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await api.searchNodes(searchQuery, { limit: 10 });
      setResults(response.results);
      setShowResults(true);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      handleSearch(value);
    }, 300);
  }, [handleSearch]);

  const handleResultClick = useCallback((result: GraphSearchResult) => {
    onNodeSelect(result.id);
    setQuery('');
    setResults([]);
    setShowResults(false);
  }, [onNodeSelect]);

  const handleClear = useCallback(() => {
    setQuery('');
    setResults([]);
    setShowResults(false);
  }, []);

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        {query && (
          <button
            onClick={handleClear}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {isLoading && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-center text-gray-500">
          搜索中...
        </div>
      )}

      {showResults && !isLoading && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
          {results.map((result) => (
            <button
              key={result.id}
              onClick={() => handleResultClick(result)}
              className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-gray-900">
                    {result.properties.name || result.properties.title || result.id}
                  </div>
                  <div className="text-sm text-gray-500">
                    {result.label}
                  </div>
                </div>
                {result.score !== undefined && (
                  <div className="text-xs text-gray-400">
                    {Math.round(result.score * 100)}%
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
      )}

      {showResults && !isLoading && results.length === 0 && query && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-center text-gray-500">
          未找到匹配的节点
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphSearch组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphSearch组件**

```bash
git add frontend/src/components/graph/GraphSearch.tsx
git commit -m "feat: add GraphSearch component for node search"
```

---

## Task 10: NodeDetails组件

**Files:**
- Create: `frontend/src/components/graph/NodeDetails.tsx`

- [ ] **Step 1: 创建NodeDetails组件**

创建文件 `frontend/src/components/graph/NodeDetails.tsx` 并添加以下内容：

```typescript
import React from 'react';
import { X, ExternalLink } from 'lucide-react';
import type { NodeDetail } from '../../types/graph';

interface NodeDetailsProps {
  nodeDetail: NodeDetail | null;
  onClose: () => void;
  onNavigate: (nodeId: string) => void;
  className?: string;
}

export function NodeDetails({
  nodeDetail,
  onClose,
  onNavigate,
  className = ''
}: NodeDetailsProps) {
  if (!nodeDetail) return null;

  const { node, relationships, neighbors } = nodeDetail;

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">
          节点详情
        </h3>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
        <div>
          <div className="text-sm font-medium text-gray-500 mb-2">基本信息</div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
                {node.label}
              </span>
              <span className="text-sm text-gray-500">ID: {node.id}</span>
            </div>
          </div>
        </div>

        <div>
          <div className="text-sm font-medium text-gray-500 mb-2">属性</div>
          <div className="bg-gray-50 rounded-lg p-3 space-y-2">
            {Object.entries(node.properties).map(([key, value]) => (
              <div key={key} className="flex justify-between text-sm">
                <span className="text-gray-600">{key}:</span>
                <span className="text-gray-900 font-medium">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
            {Object.keys(node.properties).length === 0 && (
              <div className="text-sm text-gray-400 italic">无属性</div>
            )}
          </div>
        </div>

        {relationships.incoming.length > 0 && (
          <div>
            <div className="text-sm font-medium text-gray-500 mb-2">
              入边关系 ({relationships.incoming.length})
            </div>
            <div className="space-y-2">
              {relationships.incoming.slice(0, 5).map((rel, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-2 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded">
                      {rel.type}
                    </span>
                    <span className="text-sm text-gray-600">
                      来自: {rel.from_node}
                    </span>
                  </div>
                </div>
              ))}
              {relationships.incoming.length > 5 && (
                <div className="text-xs text-gray-400 text-center">
                  还有 {relationships.incoming.length - 5} 个关系...
                </div>
              )}
            </div>
          </div>
        )}

        {relationships.outgoing.length > 0 && (
          <div>
            <div className="text-sm font-medium text-gray-500 mb-2">
              出边关系 ({relationships.outgoing.length})
            </div>
            <div className="space-y-2">
              {relationships.outgoing.slice(0, 5).map((rel, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-2 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                      {rel.type}
                    </span>
                    <span className="text-sm text-gray-600">
                      到: {rel.to_node}
                    </span>
                  </div>
                </div>
              ))}
              {relationships.outgoing.length > 5 && (
                <div className="text-xs text-gray-400 text-center">
                  还有 {relationships.outgoing.length - 5} 个关系...
                </div>
              )}
            </div>
          </div>
        )}

        {neighbors.length > 0 && (
          <div>
            <div className="text-sm font-medium text-gray-500 mb-2">
              相关节点 ({neighbors.length})
            </div>
            <div className="space-y-1">
              {neighbors.slice(0, 10).map((neighbor) => (
                <button
                  key={neighbor.id}
                  onClick={() => onNavigate(neighbor.id)}
                  className="w-full flex items-center justify-between p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-left"
                >
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs">
                      {neighbor.label}
                    </span>
                    <span className="text-sm text-gray-700">
                      {neighbor.properties.name || neighbor.properties.title || neighbor.id}
                    </span>
                  </div>
                  <ExternalLink className="w-4 h-4 text-gray-400" />
                </button>
              ))}
              {neighbors.length > 10 && (
                <div className="text-xs text-gray-400 text-center">
                  还有 {neighbors.length - 10} 个节点...
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证NodeDetails组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交NodeDetails组件**

```bash
git add frontend/src/components/graph/NodeDetails.tsx
git commit -m "feat: add NodeDetails component for displaying node information"
```

---

## Task 11: GraphControls组件

**Files:**
- Create: `frontend/src/components/graph/GraphControls.tsx`

- [ ] **Step 1: 创建GraphControls组件**

创建文件 `frontend/src/components/graph/GraphControls.tsx` 并添加以下内容：

```typescript
import React, { useState } from 'react';
import { Filter, ChevronDown } from 'lucide-react';
import type { GraphStats, GraphFilters } from '../../types/graph';

interface GraphControlsProps {
  stats: GraphStats;
  filters: GraphFilters;
  onFilterChange: (filters: GraphFilters) => void;
  className?: string;
}

export function GraphControls({
  stats,
  filters,
  onFilterChange,
  className = ''
}: GraphControlsProps) {
  const [showFilters, setShowFilters] = useState(false);

  const handleNodeLabelToggle = (label: string) => {
    const newLabels = filters.nodeLabels.includes(label)
      ? filters.nodeLabels.filter(l => l !== label)
      : [...filters.nodeLabels, label];
    
    onFilterChange({
      ...filters,
      nodeLabels: newLabels
    });
  };

  const handleRelTypeToggle = (type: string) => {
    const newTypes = filters.relationshipTypes.includes(type)
      ? filters.relationshipTypes.filter(t => t !== type)
      : [...filters.relationshipTypes, type];
    
    onFilterChange({
      ...filters,
      relationshipTypes: newTypes
    });
  };

  const handleClearFilters = () => {
    onFilterChange({
      nodeLabels: [],
      relationshipTypes: []
    });
  };

  const hasActiveFilters = filters.nodeLabels.length > 0 || filters.relationshipTypes.length > 0;

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      <button
        onClick={() => setShowFilters(!showFilters)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">筛选</span>
          {hasActiveFilters && (
            <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full text-xs">
              {filters.nodeLabels.length + filters.relationshipTypes.length}
            </span>
          )}
        </div>
        <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
      </button>

      {showFilters && (
        <div className="border-t border-gray-200 p-4 space-y-4">
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">节点类型</div>
            <div className="flex flex-wrap gap-2">
              {stats.node_labels.map((label) => (
                <button
                  key={label}
                  onClick={() => handleNodeLabelToggle(label)}
                  className={`px-3 py-1 rounded-full text-sm transition-colors ${
                    filters.nodeLabels.includes(label)
                      ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                      : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                  }`}
                >
                  {label}
                </button>
              ))}
              {stats.node_labels.length === 0 && (
                <span className="text-sm text-gray-400 italic">无节点类型</span>
              )}
            </div>
          </div>

          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">关系类型</div>
            <div className="flex flex-wrap gap-2">
              {stats.relationship_types.map((type) => (
                <button
                  key={type}
                  onClick={() => handleRelTypeToggle(type)}
                  className={`px-3 py-1 rounded-full text-sm font-mono transition-colors ${
                    filters.relationshipTypes.includes(type)
                      ? 'bg-emerald-100 text-emerald-700 border border-emerald-300'
                      : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                  }`}
                >
                  {type}
                </button>
              ))}
              {stats.relationship_types.length === 0 && (
                <span className="text-sm text-gray-400 italic">无关系类型</span>
              )}
            </div>
          </div>

          {hasActiveFilters && (
            <button
              onClick={handleClearFilters}
              className="w-full py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
            >
              清除所有筛选
            </button>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphControls组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphControls组件**

```bash
git add frontend/src/components/graph/GraphControls.tsx
git commit -m "feat: add GraphControls component for filtering"
```

---

## Task 12: GraphStats组件

**Files:**
- Create: `frontend/src/components/graph/GraphStats.tsx`

- [ ] **Step 1: 创建GraphStats组件**

创建文件 `frontend/src/components/graph/GraphStats.tsx` 并添加以下内容：

```typescript
import React from 'react';
import { Database, Network } from 'lucide-react';
import type { GraphStats as GraphStatsType } from '../../types/graph';

interface GraphStatsProps {
  stats: GraphStatsType;
  displayedNodes: number;
  displayedEdges: number;
  className?: string;
}

export function GraphStats({
  stats,
  displayedNodes,
  displayedEdges,
  className = ''
}: GraphStatsProps) {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center">
            <Database className="w-5 h-5 text-indigo-600" />
          </div>
          <div>
            <div className="text-sm text-gray-500">节点</div>
            <div className="text-lg font-semibold text-gray-900">
              {displayedNodes.toLocaleString()}
              {stats.total_nodes > displayedNodes && (
                <span className="text-sm text-gray-400 ml-1">
                  / {stats.total_nodes.toLocaleString()}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-emerald-50 flex items-center justify-center">
            <Network className="w-5 h-5 text-emerald-600" />
          </div>
          <div>
            <div className="text-sm text-gray-500">关系</div>
            <div className="text-lg font-semibold text-gray-900">
              {displayedEdges.toLocaleString()}
              {stats.total_edges > displayedEdges && (
                <span className="text-sm text-gray-400 ml-1">
                  / {stats.total_edges.toLocaleString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-100">
        <div className="text-xs text-gray-500">
          节点类型: {stats.node_labels.length} | 关系类型: {stats.relationship_types.length}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphStats组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphStats组件**

```bash
git add frontend/src/components/graph/GraphStats.tsx
git commit -m "feat: add GraphStats component for displaying statistics"
```

---

## Task 13: GraphLegend组件

**Files:**
- Create: `frontend/src/components/graph/GraphLegend.tsx`

- [ ] **Step 1: 创建GraphLegend组件**

创建文件 `frontend/src/components/graph/GraphLegend.tsx` 并添加以下内容：

```typescript
import React from 'react';
import { nodeColors, edgeColors } from '../../lib/graphConfig';

interface GraphLegendProps {
  nodeLabels: string[];
  relationshipTypes: string[];
  className?: string;
}

export function GraphLegend({
  nodeLabels,
  relationshipTypes,
  className = ''
}: GraphLegendProps) {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
      <div className="text-sm font-medium text-gray-700 mb-3">图例</div>
      
      <div className="space-y-3">
        <div>
          <div className="text-xs text-gray-500 mb-2">节点类型</div>
          <div className="flex flex-wrap gap-2">
            {nodeLabels.map((label) => (
              <div key={label} className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: nodeColors[label] || nodeColors.default }}
                />
                <span className="text-xs text-gray-600">{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-gray-500 mb-2">关系类型</div>
          <div className="flex flex-wrap gap-2">
            {relationshipTypes.slice(0, 10).map((type) => (
              <div key={type} className="flex items-center gap-1.5">
                <div
                  className="w-6 h-0.5"
                  style={{ backgroundColor: edgeColors[type] || edgeColors.default }}
                />
                <span className="text-xs text-gray-600 font-mono">{type}</span>
              </div>
            ))}
            {relationshipTypes.length > 10 && (
              <span className="text-xs text-gray-400">
                +{relationshipTypes.length - 10} 更多
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphLegend组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphLegend组件**

```bash
git add frontend/src/components/graph/GraphLegend.tsx
git commit -m "feat: add GraphLegend component for displaying legend"
```

---

## Task 14: GraphView主页面

**Files:**
- Create: `frontend/src/components/GraphView.tsx`

- [ ] **Step 1: 创建GraphView主页面**

创建文件 `frontend/src/components/GraphView.tsx` 并添加以下内容：

```typescript
import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { GraphCanvas } from './graph/GraphCanvas';
import { GraphSearch } from './graph/GraphSearch';
import { NodeDetails } from './graph/NodeDetails';
import { GraphControls } from './graph/GraphControls';
import { GraphStats } from './graph/GraphStats';
import { GraphLegend } from './graph/GraphLegend';
import { api } from '../lib/api';
import { filterNodesByLabel, filterEdgesByType } from '../lib/graphUtils';
import type { GraphData, GraphNode, GraphEdge, GraphFilters, NodeDetail } from '../types/graph';

export function GraphView() {
  const [searchParams] = useSearchParams();
  
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [highlightNodes, setHighlightNodes] = useState<string[]>([]);
  
  const [filters, setFilters] = useState<GraphFilters>({
    nodeLabels: [],
    relationshipTypes: []
  });

  const loadGraphData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getGraphData({ limit: 500 });
      setGraphData(data);
    } catch (err: any) {
      setError(err.message || '加载图谱数据失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraphData();
  }, [loadGraphData]);

  useEffect(() => {
    const highlightParam = searchParams.get('highlight');
    if (highlightParam) {
      const nodeIds = highlightParam.split(',');
      setHighlightNodes(nodeIds);
    }
  }, [searchParams]);

  const handleNodeClick = useCallback(async (nodeId: string) => {
    try {
      const detail = await api.getNodeDetail(nodeId);
      setSelectedNode(detail);
    } catch (error) {
      console.error('Failed to load node detail:', error);
    }
  }, []);

  const handleNodeDoubleClick = useCallback(async (nodeId: string) => {
    try {
      const neighbors = await api.getNodeNeighbors(nodeId, { depth: 1 });
      
      setGraphData(prev => {
        if (!prev) return neighbors;
        
        const existingNodeIds = new Set(prev.nodes.map(n => n.id));
        const existingEdgeIds = new Set(prev.edges.map(e => e.id));
        
        const newNodes = neighbors.nodes.filter(n => !existingNodeIds.has(n.id));
        const newEdges = neighbors.edges.filter(e => !existingEdgeIds.has(e.id));
        
        return {
          nodes: [...prev.nodes, ...newNodes],
          edges: [...prev.edges, ...newEdges],
          stats: prev.stats
        };
      });
    } catch (error) {
      console.error('Failed to load node neighbors:', error);
    }
  }, []);

  const handleNodeSelect = useCallback((nodeId: string) => {
    handleNodeClick(nodeId);
    setHighlightNodes([nodeId]);
  }, [handleNodeClick]);

  const handleNavigate = useCallback((nodeId: string) => {
    handleNodeClick(nodeId);
    setHighlightNodes([nodeId]);
  }, [handleNodeClick]);

  const handleCloseDetails = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const handleFilterChange = useCallback((newFilters: GraphFilters) => {
    setFilters(newFilters);
  }, []);

  const filteredNodes = graphData
    ? filterNodesByLabel(graphData.nodes, filters.nodeLabels)
    : [];
  
  const filteredEdges = graphData
    ? filterEdgesByType(graphData.edges, filters.relationshipTypes)
    : [];

  const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
  const finalEdges = filteredEdges.filter(
    e => visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to)
  );

  return (
    <div className="flex h-full w-full bg-slate-50">
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">知识图谱</h1>
              <p className="text-gray-500 mt-1">可视化展示知识图谱结构</p>
            </div>
            <button
              onClick={loadGraphData}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              刷新
            </button>
          </div>
          
          <div className="mt-4">
            <GraphSearch
              onNodeSelect={handleNodeSelect}
              className="max-w-md"
            />
          </div>
        </div>

        {error && (
          <div className="mx-6 mt-4 bg-rose-50 border border-rose-200 rounded-lg p-4 flex items-start gap-3 text-rose-700">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium">加载图谱数据失败</h4>
              <p className="text-sm mt-1 text-rose-600">{error}</p>
            </div>
          </div>
        )}

        <div className="flex-1 relative">
          {isLoading && !graphData && (
            <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
              <div className="text-center">
                <RefreshCw className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-2" />
                <div className="text-gray-600">加载图谱数据...</div>
              </div>
            </div>
          )}

          {graphData && (
            <GraphCanvas
              nodes={filteredNodes}
              edges={finalEdges}
              onNodeClick={handleNodeClick}
              onNodeDoubleClick={handleNodeDoubleClick}
              highlightNodes={highlightNodes}
            />
          )}
        </div>
      </div>

      <div className="w-80 bg-white border-l border-gray-200 overflow-y-auto">
        <div className="p-4 space-y-4">
          {graphData && (
            <>
              <GraphStats
                stats={graphData.stats}
                displayedNodes={filteredNodes.length}
                displayedEdges={finalEdges.length}
              />

              <GraphControls
                stats={graphData.stats}
                filters={filters}
                onFilterChange={handleFilterChange}
              />

              <GraphLegend
                nodeLabels={graphData.stats.node_labels}
                relationshipTypes={graphData.stats.relationship_types}
              />
            </>
          )}

          {selectedNode && (
            <NodeDetails
              nodeDetail={selectedNode}
              onClose={handleCloseDetails}
              onNavigate={handleNavigate}
            />
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证GraphView组件**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 3: 提交GraphView组件**

```bash
git add frontend/src/components/GraphView.tsx
git commit -m "feat: add GraphView main page component"
```

---

## Task 15: 安装vis-network依赖

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: 安装vis-network依赖**

运行: `cd frontend && npm install vis-network vis-data`

- [ ] **Step 2: 验证依赖安装**

运行: `cd frontend && npm list vis-network vis-data`
预期: 显示已安装的版本

- [ ] **Step 3: 提交package.json更改**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "feat: add vis-network and vis-data dependencies"
```

---

## Task 16: 更新Sidebar导航

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: 添加图谱导航项**

在 `frontend/src/components/Sidebar.tsx` 文件中，找到导航项定义的位置，添加图谱导航项：

```typescript
const navItems = [
  { id: 'chat', label: '对话', icon: MessageSquare },
  { id: 'ingest', label: '导入', icon: Upload },
  { id: 'schema', label: '结构', icon: Database },
  { id: 'graph', label: '图谱', icon: Network },
  { id: 'retrieval', label: '检索', icon: Search },
];
```

- [ ] **Step 2: 添加Network图标导入**

在文件顶部的导入部分添加：

```typescript
import { MessageSquare, Upload, Database, Network, Search, Activity, Settings } from 'lucide-react';
```

- [ ] **Step 3: 验证Sidebar更新**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 4: 提交Sidebar更新**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: add graph navigation item to sidebar"
```

---

## Task 17: 更新App路由

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 添加GraphView路由**

在 `frontend/src/App.tsx` 文件中，导入GraphView组件：

```typescript
import { GraphView } from './components/GraphView';
```

- [ ] **Step 2: 添加GraphView路由渲染**

在主内容区域添加GraphView的路由渲染：

```typescript
{activeTab === 'graph' && <GraphView />}
```

- [ ] **Step 3: 验证App更新**

运行: `cd frontend && npm run lint`
预期: 无类型错误

- [ ] **Step 4: 提交App更新**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add GraphView route to App"
```

---

## Task 18: 测试后端API

**Files:**
- Test: `backend/tests/test_graph_api.py`

- [ ] **Step 1: 创建测试文件**

创建文件 `backend/tests/test_graph_api.py` 并添加以下内容：

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

class TestGraphAPI:
    def test_get_graph_data(self):
        """测试获取图谱数据"""
        response = client.get("/api/graph/data")
        assert response.status_code == 200
        data = response.json()
        assert 'nodes' in data
        assert 'edges' in data
        assert 'stats' in data
    
    def test_get_graph_data_with_limit(self):
        """测试带限制的图谱数据获取"""
        response = client.get("/api/graph/data?limit=100")
        assert response.status_code == 200
        data = response.json()
        assert len(data['nodes']) <= 100
    
    def test_search_nodes(self):
        """测试节点搜索"""
        response = client.get("/api/graph/search?query=test")
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert 'total' in data
    
    def test_get_node_detail_not_found(self):
        """测试获取不存在的节点详情"""
        response = client.get("/api/graph/node/999999")
        assert response.status_code == 404
```

- [ ] **Step 2: 运行测试**

运行: `cd backend && pytest tests/test_graph_api.py -v`
预期: 所有测试通过

- [ ] **Step 3: 提交测试文件**

```bash
git add backend/tests/test_graph_api.py
git commit -m "test: add graph API tests"
```

---

## Task 19: 启动后端服务并验证

**Files:**
- None

- [ ] **Step 1: 启动后端服务**

运行: `cd backend && python -m src.main`

- [ ] **Step 2: 测试图谱API端点**

运行: `curl http://localhost:8000/api/graph/data?limit=10`
预期: 返回JSON格式的图谱数据

- [ ] **Step 3: 测试搜索API**

运行: `curl http://localhost:8000/api/graph/search?query=test`
预期: 返回JSON格式的搜索结果

---

## Task 20: 启动前端服务并验证

**Files:**
- None

- [ ] **Step 1: 启动前端服务**

运行: `cd frontend && npm run dev`

- [ ] **Step 2: 访问图谱页面**

打开浏览器访问: `http://localhost:3000`
点击侧边栏的"图谱"导航项
预期: 显示图谱可视化页面

- [ ] **Step 3: 测试基本功能**

1. 测试图谱加载和显示
2. 测试节点点击和详情显示
3. 测试搜索功能
4. 测试筛选功能
5. 测试缩放和平移

预期: 所有功能正常工作

---

## Task 21: 最终提交和文档更新

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新README文档**

在 `README.md` 文件中添加图谱可视化功能的说明：

```markdown
## 核心功能

- **文档摄取**：支持加载和处理多种格式的文档
- **知识图谱构建**：从文档中提取实体和关系，构建知识图谱
- **图谱可视化**：以图形化方式展示知识图谱，支持交互式探索
- **混合检索**：结合向量检索和图谱检索，提供更准确的结果
- **智能问答**：基于检索到的信息生成准确的回答
- **可视化界面**：提供直观的用户界面，方便用户交互
```

- [ ] **Step 2: 提交最终更改**

```bash
git add README.md
git commit -m "docs: update README with graph visualization feature"
```

- [ ] **Step 3: 创建功能完成标签**

```bash
git tag -a v1.1.0 -m "feat: add graph visualization feature"
```

---

## 自我审查清单

完成所有任务后，请检查以下内容：

- [ ] 所有后端API端点正常工作
- [ ] 所有前端组件正确渲染
- [ ] 图谱数据正确加载和显示
- [ ] 节点交互功能正常
- [ ] 搜索功能正常
- [ ] 筛选功能正常
- [ ] 与ChatView和RetrievalView的集成正常
- [ ] 所有测试通过
- [ ] 代码无lint错误
- [ ] 文档已更新

---

**计划完成！**
