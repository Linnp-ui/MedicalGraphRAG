# GraphRAG

## 项目概述

GraphRAG 是一个基于知识图谱的检索增强生成（RAG）系统，结合了知识图谱和向量检索的优势，提供更准确、更相关的问答能力。

### 技术栈

- **后端**：Python 3.11+, FastAPI, LangChain, LangGraph, Neo4j
- **前端**：React, TypeScript, Vite, Tailwind CSS

## 快速开始

### 后端设置

```bash
# 进入后端目录
cd graphrag

# 安装依赖
pip install -e .

# 运行服务器
python -m src.main
```

### 前端设置

```bash
# 进入前端目录
cd front

# 安装依赖
npm install

# 运行开发服务器
npm run dev

# 构建生产版本
npm run build
```

## 项目结构

```
graphrag/
├── src/
│   ├── api/           # FastAPI 路由和 Pydantic 模式
│   ├── core/          # 配置、Neo4j 客户端、缓存
│   ├── ingestion/     # 文档加载、分块、知识图谱构建、嵌入
│   ├── retrieval/     # 向量、图谱和混合检索
│   ├── chains/        # Cypher 生成、QA 链
│   ├── workflow/      # LangGraph 工作流、路由、状态
│   └── utils/         # 工具（日志记录器）
├── tests/             # 测试文件
├── config/            # YAML 配置文件
└── data/              # 数据目录

front/
├── src/
│   ├── components/    # React 组件
│   ├── pages/         # 页面
│   ├── hooks/         # 自定义钩子
│   └── services/      # API 服务
└── public/            # 静态资产
```

## 核心功能

- **文档摄取**：支持加载和处理多种格式的文档
- **知识图谱构建**：从文档中提取实体和关系，构建知识图谱
- **混合检索**：结合向量检索和图谱检索，提供更准确的结果
- **智能问答**：基于检索到的信息生成准确的回答
- **可视化界面**：提供直观的用户界面，方便用户交互

## 配置

### 后端配置

- **环境变量**：`.env` 文件
- **配置文件**：`config/settings.yaml`
- **LLM 模型**：`OPENAI_MODEL` 用于一般任务，`EXTRACTION_MODEL` 用于实体提取

### 前端配置

- **环境变量**：`.env` 文件，使用 `VITE_` 前缀

## 测试

```bash
# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/test_ingestion.py

# 运行单个测试
pytest tests/test_ingestion.py -k test_load_text_file

# 运行测试并显示详细输出
pytest -v
```

## 代码风格

### Python

- **格式化工具**：Ruff，`line-length = 100`，目标 `py311`
- **导入顺序**：标准库 → 第三方库 → 本地导入（在 `src/` 内使用相对导入）
- **命名约定**：
  - 类：`PascalCase`
  - 函数/方法：`snake_case`
  - 私有方法：前导下划线
  - 常量：`UPPER_SNAKE_CASE`
- **类型提示**：使用 `typing` 模块，Pydantic `BaseModel` 用于 API 模式
- **错误处理**：使用 `loguru` 进行日志记录，在 API 路由中使用 `HTTPException`

## 常见问题

### Neo4j 连接
- 确保 Neo4j 正在运行：`bolt://localhost:7687`
- 检查 `.env` 文件中的凭据
- 验证 `NEO4J_PASSWORD` 是否正确

### LLM API
- 验证 `.env` 中是否设置了 `OPENAI_API_KEY`
- 对于 DashScope：确保 `OPENAI_BASE_URL` 指向兼容的端点
- 模型名称：`qwen-plus`、`qwen-flash`、`gpt-4o-mini`

### 导入错误
- 在 `src/` 内始终使用相对导入（例如：`from ..core.config import ...`）
- 不要使用绝对导入，如 `from src.core.config import ...`

## 开发工作流程

1. **创建分支**用于新功能
2. **编写测试**然后实现（推荐 TDD）
3. **频繁运行测试**：`pytest -v`
4. **提交前格式化代码**：`ruff format .`
5. **提交前检查代码**：`ruff check .`
6. **如果项目结构更改**，更新文档

## 贡献

欢迎贡献！请遵循上述开发工作流程，并确保您的代码符合项目的代码风格指南。

## 许可证

[MIT License](LICENSE)
