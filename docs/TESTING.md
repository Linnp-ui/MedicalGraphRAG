# GraphRAG 测试套件

## 测试策略

### 1. 测试范围

#### 后端测试
- **API端点测试**: 所有REST API端点的功能测试
- **数据库测试**: Neo4j客户端方法测试
- **业务逻辑测试**: 文档处理、知识图谱构建、检索功能
- **错误处理测试**: 异常场景和边界条件

#### 前端测试
- **组件测试**: React组件渲染和交互测试
- **API服务测试**: 前端API调用测试
- **用户交互测试**: 用户操作流程测试

#### 端到端测试
- **用户流程测试**: 完整的用户操作流程
- **集成测试**: 前后端集成测试

### 2. 测试框架

- **后端**: pytest + pytest-asyncio + httpx
- **前端**: vitest + @testing-library/react
- **端到端**: Playwright (可选)

### 3. 测试覆盖率目标

- 后端代码覆盖率: ≥ 80%
- 前端代码覆盖率: ≥ 70%
- 关键业务逻辑: 100%

### 4. 测试类型

#### 单元测试
- 测试单个函数或方法
- 隔离依赖，使用mock
- 快速执行

#### 集成测试
- 测试多个组件协作
- 使用测试数据库
- 验证数据流

#### 功能测试
- 测试完整的业务功能
- 端到端的用户场景
- 验证业务规则

#### 边界测试
- 测试边界条件
- 异常输入处理
- 性能边界

### 5. 测试数据

- 使用测试数据库
- 每个测试独立的数据集
- 测试后清理数据

### 6. 测试执行

- 本地开发: 手动执行
- CI/CD: 自动执行
- 定期回归: 每日执行

### 7. 测试报告

- 测试覆盖率报告
- 测试结果报告
- 性能报告
- 问题分析报告

## 测试用例清单

### 后端API测试

#### 健康检查API
- [ ] GET /health - 返回服务状态
- [ ] GET /health - Neo4j连接状态检查

#### 图谱API
- [ ] GET /graph/data - 获取图谱数据
- [ ] GET /graph/data?node_label=Person - 按类型筛选节点
- [ ] GET /graph/data?limit=100 - 限制返回数量
- [ ] GET /graph/search?query=test - 搜索节点
- [ ] GET /graph/node/{id} - 获取节点详情
- [ ] GET /graph/node/{id}/neighbors - 获取节点邻居
- [ ] POST /graph/query-result - 获取查询结果图谱

#### 文档摄取API
- [ ] POST /ingest/file - 摄取单个文件
- [ ] POST /ingest/directory - 摄取目录
- [ ] POST /ingest - 批量摄取

#### 问答API
- [ ] POST /query - 提交问题
- [ ] POST /query - 带历史记录的问答

### 前端组件测试

#### GraphView组件
- [ ] 渲染图谱数据
- [ ] 搜索节点功能
- [ ] 筛选节点类型
- [ ] 错误处理

#### ChatView组件
- [ ] 发送消息
- [ ] 显示回答
- [ ] 显示历史记录

#### API服务
- [ ] getGraphData方法
- [ ] searchNodes方法
- [ ] getNodeDetail方法

### 端到端测试

#### 用户流程
- [ ] 完整的问答流程
- [ ] 图谱可视化流程
- [ ] 文档摄取流程

## 测试执行指南

### 后端测试

```bash
cd backend
pytest tests/ -v --cov=src --cov-report=html
```

### 前端测试

```bash
cd frontend
npm test
```

### 端到端测试

```bash
npm run test:e2e
```

## 测试报告

测试报告将包括：
1. 测试覆盖率统计
2. 通过/失败测试数量
3. 失败测试详情
4. 性能指标
5. 问题分析和建议
