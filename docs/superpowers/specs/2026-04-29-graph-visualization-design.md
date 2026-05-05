# 知识图谱可视化功能设计文档

**文档版本**: 1.0  
**创建日期**: 2026-04-29  
**作者**: AI Assistant  
**状态**: 待审核

---

## 目录

1. [概述](#概述)
2. [需求分析](#需求分析)
3. [技术方案](#技术方案)
4. [系统架构](#系统架构)
5. [前端设计](#前端设计)
6. [后端设计](#后端设计)
7. [数据流和交互](#数据流和交互)
8. [性能优化](#性能优化)
9. [测试策略](#测试策略)
10. [实施计划](#实施计划)
11. [风险评估](#风险评估)

---

## 概述

### 项目背景

GraphRAG是一个基于知识图谱的检索增强生成（RAG）系统，目前支持文档摄取、知识图谱构建、混合检索和智能问答等功能。为了提升用户体验和数据探索能力，需要添加图谱可视化功能，让用户能够直观地查看和交互知识图谱数据。

### 目标

构建一个功能完整、性能优良的知识图谱可视化系统，支持：

- 图谱结构的直观展示
- 节点和关系的交互式探索
- 与现有查询功能的深度集成
- 中型规模数据集（500-5000节点）的流畅渲染

### 成功标准

1. **功能完整性**: 实现所有核心功能（浏览、搜索、筛选、详情查看）
2. **性能指标**: 
   - 初始加载时间 < 3秒（500节点）
   - 交互响应时间 < 100ms
   - 支持2000+节点的流畅渲染
3. **用户体验**: 
   - 直观的交互方式
   - 清晰的视觉反馈
   - 友好的错误处理
4. **集成度**: 与ChatView和RetrievalView无缝集成

---

## 需求分析

### 用户需求

基于与用户的沟通，确定以下需求：

#### 使用场景
- **综合用途**: 浏览图谱结构 + 查询结果展示 + 交互式探索

#### 数据规模
- **中型图谱**: 500-5000个节点

#### 交互功能
- ✅ 基础交互（拖拽、缩放、平移）
- ✅ 节点点击和信息展示
- ✅ 搜索和筛选功能
- ❌ 导出功能（暂不实现）

#### 布局方式
- **力导向布局**: 节点根据相互关系自动分布

#### 集成方式
- **新增独立页面**: 作为新的标签页，与现有页面并列

#### 性能要求
- **平衡方案**: 在功能和性能之间找到平衡

#### 视觉定制
- ✅ 节点按类型着色
- ✅ 节点大小动态调整
- ✅ 关系线样式定制

#### 查询集成
- **全面集成**: 与ChatView和RetrievalView集成

### 功能需求

#### 核心功能

1. **图谱浏览**
   - 显示节点和关系
   - 支持缩放和平移
   - 力导向布局自动排列

2. **节点交互**
   - 单击显示节点详情
   - 双击展开邻居节点
   - 拖拽移动节点

3. **搜索功能**
   - 按节点名称搜索
   - 按属性值搜索
   - 搜索结果定位

4. **筛选功能**
   - 按节点类型筛选
   - 按关系类型筛选
   - 按属性值筛选

5. **详情展示**
   - 节点属性信息
   - 相关关系列表
   - 邻居节点导航

#### 集成功能

1. **ChatView集成**
   - 查询结果高亮显示
   - "在图谱中查看"按钮
   - 自动定位到相关节点

2. **RetrievalView集成**
   - 检索结果可视化
   - 关系路径展示
   - 子图提取显示

### 非功能需求

#### 性能需求
- 初始加载: < 3秒（500节点）
- 搜索响应: < 500ms
- 节点点击响应: < 100ms
- 支持2000+节点流畅渲染

#### 可用性需求
- 直观的用户界面
- 清晰的操作提示
- 友好的错误信息

#### 兼容性需求
- 支持主流浏览器（Chrome、Firefox、Edge、Safari）
- 响应式设计，适配不同屏幕尺寸

---

## 技术方案

### 技术选型

#### 前端技术栈

**核心库**: vis-network

**选择理由**:
1. 专为网络图设计，功能完整
2. 内置力导向布局，性能优秀
3. 支持中型数据集（500-5000节点）
4. 丰富的交互功能
5. 易于实现节点着色和样式定制
6. 社区活跃，文档完善

**替代方案**:
- **Cytoscape.js**: 功能更强大，但学习曲线陡，配置复杂
- **React Flow**: 专为React设计，但主要面向流程图，知识图谱功能需要额外开发

**依赖库**:
```json
{
  "vis-network": "^9.1.6",
  "vis-data": "^7.1.6"
}
```

#### 后端技术栈

**框架**: FastAPI（现有）

**数据库**: Neo4j（现有）

**新增依赖**: 无需新增主要依赖

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  GraphView   │  │  ChatView    │  │ RetrievalView│      │
│  │  (新增)      │  │  (修改)      │  │  (修改)      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  GraphComponent │                        │
│                  │  (vis-network)  │                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │   API Service   │                        │
│                  └────────┬────────┘                        │
└───────────────────────────┼─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────┼─────────────────────────────────┐
│                      后端 (FastAPI)                         │
├───────────────────────────┼─────────────────────────────────┤
│                  ┌────────▼────────┐                        │
│                  │  /graph/data    │  (新增API)             │
│                  │  /graph/search  │                        │
│                  │  /graph/node/:id│                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  Neo4j Client   │                        │
│                  └────────┬────────┘                        │
└───────────────────────────┼─────────────────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │   Neo4j Database  │
                  └───────────────────┘
```

---

## 系统架构

### 整体架构

采用前后端分离的架构，前端负责可视化和交互，后端负责数据查询和业务逻辑。

### 模块划分

#### 前端模块
- **GraphView**: 图谱可视化主页面
- **GraphCanvas**: vis-network封装组件
- **GraphControls**: 控制面板组件
- **GraphSearch**: 搜索组件
- **NodeDetails**: 节点详情组件

#### 后端模块
- **Graph API**: 图谱数据接口
- **Neo4j Client**: 数据库查询封装
- **Cache**: 数据缓存层

### 数据流

```
用户操作 → 前端组件 → API调用 → 后端路由 → Neo4j查询 → 数据返回 → 前端渲染
```

---

## 前端设计

### 组件结构

```
frontend/src/
├── components/
│   ├── GraphView.tsx              # 图谱可视化主页面
│   ├── graph/
│   │   ├── GraphCanvas.tsx        # vis-network封装组件
│   │   ├── GraphControls.tsx      # 控制面板（缩放、布局、筛选）
│   │   ├── GraphSearch.tsx        # 搜索组件
│   │   ├── NodeDetails.tsx        # 节点详情面板
│   │   ├── GraphLegend.tsx        # 图例组件
│   │   └── GraphStats.tsx         # 统计信息组件
│   ├── ChatView.tsx               # 修改：添加图谱跳转按钮
│   └── RetrievalView.tsx          # 修改：添加图谱跳转按钮
├── lib/
│   ├── api.ts                     # 扩展：添加图谱API
│   ├── graphUtils.ts              # 新增：图谱数据处理工具
│   └── graphConfig.ts             # 新增：图谱配置
└── types/
    └── graph.ts                   # 新增：图谱类型定义
```

### 核心组件设计

#### 1. GraphView.tsx

**职责**: 整合所有图谱子组件，管理图谱状态

**接口**:
```typescript
interface GraphViewProps {
  initialNodeLabel?: string;
  initialSearchQuery?: string;
  highlightNodes?: string[];
}
```

**状态管理**:
```typescript
interface GraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
  isLoading: boolean;
  error: string | null;
  selectedNode: GraphNode | null;
  highlightNodes: string[];
  searchQuery: string;
  filters: GraphFilters;
  viewSettings: ViewSettings;
}
```

#### 2. GraphCanvas.tsx

**职责**: 封装vis-network实例，处理可视化渲染

**接口**:
```typescript
interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (nodeId: string) => void;
  onNodeDoubleClick: (nodeId: string) => void;
  highlightNodes?: string[];
}
```

**关键实现**:
- vis-network实例管理
- 力导向布局配置
- 节点和边样式应用
- 事件处理（点击、双击、拖拽）

#### 3. GraphControls.tsx

**职责**: 提供图谱控制功能

**接口**:
```typescript
interface GraphControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  onLayoutChange: (layout: LayoutType) => void;
  onFilterChange: (filter: GraphFilter) => void;
}
```

**功能**:
- 缩放控制按钮
- 布局切换下拉菜单
- 节点类型筛选复选框
- 关系类型筛选复选框

#### 4. NodeDetails.tsx

**职责**: 显示节点详细信息

**接口**:
```typescript
interface NodeDetailsProps {
  node: GraphNode | null;
  relationships: GraphEdge[];
  onClose: () => void;
  onNavigate: (nodeId: string) => void;
}
```

**功能**:
- 节点属性表格
- 相关关系列表
- 邻居节点链接
- 关闭按钮

#### 5. GraphSearch.tsx

**职责**: 提供搜索功能

**接口**:
```typescript
interface GraphSearchProps {
  onSearch: (query: string) => void;
  onClear: () => void;
  placeholder?: string;
}
```

**功能**:
- 搜索输入框
- 防抖处理（300ms）
- 搜索结果下拉列表
- 清除按钮

### 视觉设计

#### 节点样式

```typescript
const nodeStyles = {
  colors: {
    Person: '#3B82F6',      // 蓝色
    Organization: '#10B981', // 绿色
    Location: '#F59E0B',     // 橙色
    Event: '#8B5CF6',        // 紫色
    Concept: '#EC4899',      // 粉色
    default: '#6B7280'       // 灰色
  },
  size: {
    min: 20,
    max: 60,
    scale: 'log'
  },
  shape: 'dot',
  borderWidth: 2,
  borderWidthSelected: 4
};
```

#### 关系样式

```typescript
const edgeStyles = {
  colors: {
    KNOWS: '#3B82F6',
    WORKS_FOR: '#10B981',
    LOCATED_IN: '#F59E0B',
    default: '#9CA3AF'
  },
  width: 2,
  arrows: {
    to: { enabled: true, scaleFactor: 0.8 }
  },
  smooth: {
    type: 'curvedCW',
    roundness: 0.2
  }
};
```

#### 布局配置

```typescript
const layoutOptions = {
  physics: {
    enabled: true,
    barnesHut: {
      gravitationalConstant: -3000,
      centralGravity: 0.3,
      springLength: 100,
      springConstant: 0.05
    },
    stabilization: {
      iterations: 200,
      updateInterval: 25
    }
  }
};
```

### 类型定义

```typescript
// frontend/src/types/graph.ts

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

export interface GraphFilters {
  nodeLabels: string[];
  relationshipTypes: string[];
  propertyRange: Record<string, [any, any]>;
}

export interface ViewSettings {
  showLabels: boolean;
  showArrows: boolean;
  layout: LayoutType;
  zoomLevel: number;
}

export type LayoutType = 'force' | 'hierarchical' | 'circular';
```

---

## 后端设计

### API端点设计

#### 1. 获取图谱数据

```
GET /api/graph/data
```

**查询参数**:
- `node_label` (可选): 节点类型筛选
- `limit` (可选): 返回节点数量限制，默认500
- `offset` (可选): 分页偏移量，默认0

**响应**:
```json
{
  "nodes": [
    {
      "id": "node_1",
      "label": "Person",
      "properties": {
        "name": "张三",
        "age": 30
      },
      "degree": 5
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "from": "node_1",
      "to": "node_2",
      "type": "KNOWS",
      "properties": {}
    }
  ],
  "stats": {
    "total_nodes": 1500,
    "total_edges": 3200,
    "node_labels": ["Person", "Organization"],
    "relationship_types": ["KNOWS", "WORKS_FOR"]
  }
}
```

#### 2. 搜索节点

```
GET /api/graph/search
```

**查询参数**:
- `query`: 搜索关键词
- `node_label` (可选): 节点类型筛选
- `limit` (可选): 返回结果数量，默认20

**响应**:
```json
{
  "results": [
    {
      "id": "node_1",
      "label": "Person",
      "properties": {
        "name": "张三"
      },
      "score": 0.95
    }
  ],
  "total": 5
}
```

#### 3. 获取节点详情

```
GET /api/graph/node/{node_id}
```

**响应**:
```json
{
  "node": {
    "id": "node_1",
    "label": "Person",
    "properties": {
      "name": "张三",
      "age": 30,
      "email": "zhangsan@example.com"
    }
  },
  "relationships": {
    "incoming": [
      {
        "from_node": "node_2",
        "type": "KNOWS",
        "properties": {}
      }
    ],
    "outgoing": [
      {
        "to_node": "node_3",
        "type": "WORKS_FOR",
        "properties": {}
      }
    ]
  },
  "neighbors": [
    {
      "id": "node_2",
      "label": "Person",
      "properties": {"name": "李四"}
    }
  ]
}
```

#### 4. 获取节点邻居

```
GET /api/graph/node/{node_id}/neighbors
```

**查询参数**:
- `depth` (可选): 遍历深度，默认1，最大3
- `relationship_type` (可选): 关系类型筛选

**响应**: 与 `/api/graph/data` 相同格式

#### 5. 获取查询结果图谱

```
POST /api/graph/query-result
```

**请求体**:
```json
{
  "query": "用户的问题",
  "node_ids": ["node_1", "node_2"],
  "max_depth": 2
}
```

**响应**: 与 `/api/graph/data` 相同格式

### Schema定义

```python
# backend/src/api/schemas.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

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

### Neo4j查询方法

```python
# backend/src/core/neo4j_client.py

class Neo4jClient:
    def get_graph_data(
        self,
        node_label: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取图谱数据"""
        
    def search_nodes(
        self,
        query: str,
        node_label: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """搜索节点"""
        
    def get_node_detail(self, node_id: str) -> Dict[str, Any]:
        """获取节点详情"""
        
    def get_node_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取节点邻居"""
        
    def get_query_result_graph(
        self,
        node_ids: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """获取查询结果图谱"""
```

### Cypher查询示例

#### 获取图谱数据

```cypher
// 获取节点
MATCH (n)
WHERE $node_label IS NULL OR $node_label IN labels(n)
WITH n, size((n)--()) as degree
RETURN n, id(n) as node_id, degree
ORDER BY degree DESC
SKIP $offset
LIMIT $limit

// 获取关系
MATCH (n)-[r]->(m)
WHERE id(n) IN $node_ids AND id(m) IN $node_ids
RETURN r, id(r) as edge_id, id(startNode(r)) as from_id, id(endNode(r)) as to_id
```

#### 搜索节点

```cypher
MATCH (n)
WHERE ($node_label IS NULL OR $node_label IN labels(n))
AND ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $query)
RETURN n, id(n) as node_id
LIMIT $limit
```

#### 获取节点详情

```cypher
// 获取节点信息
MATCH (n)
WHERE id(n) = $node_id
RETURN n, labels(n) as labels, id(n) as node_id

// 获取入边关系
MATCH (m)-[r]->(n)
WHERE id(n) = $node_id
RETURN r, id(r) as edge_id, id(m) as from_id, type(r) as rel_type

// 获取出边关系
MATCH (n)-[r]->(m)
WHERE id(n) = $node_id
RETURN r, id(r) as edge_id, id(m) as to_id, type(r) as rel_type
```

---

## 数据流和交互

### 初始加载流程

```
用户打开GraphView页面
         │
         ▼
    检查URL参数
    (node_label, query, highlight)
         │
         ▼
    显示加载状态
         │
         ▼
    调用 GET /api/graph/data
         │
         ▼
    后端查询Neo4j
         │
         ▼
    返回节点和关系数据
         │
         ▼
    前端数据处理
    - 转换为vis-network格式
    - 计算节点颜色和大小
    - 应用筛选条件
         │
         ▼
    渲染图谱
         │
         ▼
    启动力导向布局
         │
         ▼
    布局稳定后隐藏加载状态
```

### 搜索流程

```
用户输入搜索关键词
         │
         ▼
    防抖处理 (300ms)
         │
         ▼
    调用 GET /api/graph/search
         │
         ▼
    返回匹配节点列表
         │
         ▼
    显示搜索结果下拉列表
         │
         ▼
    用户点击结果项
         │
         ▼
    定位到目标节点
    - 缩放到合适级别
    - 高亮节点
    - 显示节点详情
```

### 节点交互流程

```
用户点击节点
         │
         ├─► 单击：显示节点详情面板
         │         │
         │         ▼
         │    调用 GET /api/graph/node/{id}
         │         │
         │         ▼
         │    显示节点属性和关系
         │
         └─► 双击：展开邻居节点
                   │
                   ▼
              检查邻居是否已加载
                   │
                   ├─► 已加载：高亮邻居
                   │
                   └─► 未加载：调用 GET /api/graph/node/{id}/neighbors
                              │
                              ▼
                         添加新节点和关系到图谱
```

### 与其他页面的集成

#### ChatView集成

```typescript
// ChatView.tsx
const handleViewInGraph = () => {
  const nodeIds = result.relevant_nodes.map(n => n.id).join(',');
  navigate(`/graph?highlight=${nodeIds}`);
};

// GraphView.tsx
useEffect(() => {
  const highlightParam = searchParams.get('highlight');
  if (highlightParam) {
    const nodeIds = highlightParam.split(',');
    setHighlightNodes(nodeIds);
    focusOnNodes(nodeIds);
  }
}, [searchParams]);
```

#### RetrievalView集成

```typescript
// RetrievalView.tsx
const handleVisualizeResults = async () => {
  const nodeIds = results.map(r => r.node_id);
  const graphData = await api.getQueryResultGraph(nodeIds);
  navigate('/graph', { state: { graphData, highlightNodes: nodeIds } });
};

// GraphView.tsx
useEffect(() => {
  if (location.state?.graphData) {
    setGraphData(location.state.graphData);
    setHighlightNodes(location.state.highlightNodes);
  }
}, [location.state]);
```

---

## 性能优化

### 前端性能优化

#### vis-network配置优化

```typescript
const optimizedOptions = {
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
      updateInterval: 25
    }
  },
  rendering: {
    hideEdgesOnDrag: true,
    hideEdgesOnZoom: true
  }
};
```

#### 数据分页加载

```typescript
const usePaginatedGraph = () => {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [page, setPage] = useState(0);
  const pageSize = 200;
  
  const loadMore = async () => {
    const newNodes = await api.getGraphData({
      offset: page * pageSize,
      limit: pageSize
    });
    setNodes(prev => [...prev, ...newNodes]);
    setPage(prev => prev + 1);
  };
  
  return { nodes, loadMore };
};
```

### 后端性能优化

#### 查询优化

- 使用Neo4j索引加速搜索
- 限制遍历深度（最大3层）
- 批量查询替代多次单查询

#### 缓存策略

- 后端缓存图谱结构（5分钟TTL）
- 前端缓存已加载的节点和关系
- 使用ETag支持增量更新

#### 数据库索引

```cypher
// 创建索引
CREATE INDEX node_name_index IF NOT EXISTS
FOR (n:Person) ON (n.name);

// 创建全文索引
CALL db.index.fulltext.createNodeIndex(
  'node_fulltext_index',
  ['Person', 'Organization', 'Location'],
  ['name', 'description']
);
```

### 网络传输优化

- 启用Gzip压缩
- 增量更新机制
- 数据预处理减少传输量

---

## 测试策略

### 单元测试

#### 前端组件测试

- GraphCanvas渲染测试
- 节点点击事件测试
- 节点高亮测试
- 筛选功能测试

#### 后端API测试

- 图谱数据获取测试
- 节点搜索测试
- 节点详情获取测试
- 错误处理测试

### 集成测试

- 端到端用户流程测试
- 页面集成测试
- API集成测试

### 性能测试

- 负载测试（使用Locust）
- 性能基准测试
- 渲染性能测试

### 测试覆盖率目标

- 前端单元测试覆盖率: ≥ 80%
- 后端API测试覆盖率: ≥ 85%
- 集成测试覆盖率: ≥ 70%

---

## 实施计划

### 开发阶段

#### 第一阶段：后端API开发（2-3天）

1. 添加图谱相关Schema
2. 实现Neo4j查询方法
3. 创建图谱API端点
4. 编写单元测试

#### 第二阶段：前端基础组件（3-4天）

1. 安装vis-network依赖
2. 创建GraphCanvas组件
3. 实现基础渲染和交互
4. 创建GraphView主页面

#### 第三阶段：前端功能完善（3-4天）

1. 实现搜索功能
2. 实现筛选功能
3. 实现节点详情展示
4. 添加控制面板

#### 第四阶段：集成和优化（2-3天）

1. 与ChatView集成
2. 与RetrievalView集成
3. 性能优化
4. 错误处理完善

#### 第五阶段：测试和文档（2天）

1. 编写测试用例
2. 运行测试并修复问题
3. 编写用户文档
4. 代码审查

### 里程碑

- **M1**: 后端API完成并测试通过
- **M2**: 前端基础可视化功能完成
- **M3**: 所有核心功能完成
- **M4**: 集成测试通过
- **M5**: 性能测试达标，准备发布

---

## 风险评估

### 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| vis-network性能不足 | 高 | 中 | 准备Cytoscape.js备选方案 |
| Neo4j查询性能慢 | 高 | 中 | 添加索引，优化查询 |
| 大数据集渲染卡顿 | 中 | 高 | 实现分页加载和虚拟渲染 |

### 进度风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 需求变更 | 中 | 中 | 采用敏捷开发，快速迭代 |
| 技术难点攻关时间长 | 中 | 低 | 预留缓冲时间 |
| 集成问题 | 中 | 中 | 提前进行集成测试 |

### 质量风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 测试覆盖不足 | 高 | 中 | 制定测试计划，定期检查覆盖率 |
| 性能不达标 | 高 | 中 | 性能测试前置，持续监控 |
| 用户体验不佳 | 中 | 低 | 用户测试，收集反馈 |

---

## 附录

### 参考资料

- [vis-network官方文档](https://visjs.github.io/vis-network/docs/network/)
- [Neo4j Cypher查询语言](https://neo4j.com/docs/cypher-manual/current/)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [React官方文档](https://react.dev/)

### 术语表

- **力导向布局**: 一种图布局算法，节点间模拟物理引力，自动分布位置
- **节点度数**: 节点连接的边数量，反映节点重要性
- **图谱子图**: 从完整图谱中提取的部分节点和关系
- **虚拟渲染**: 只渲染视口内的节点，提升性能

---

**文档结束**
