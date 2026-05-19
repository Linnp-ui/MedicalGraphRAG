当我需要查看库/应用程序接口文档、进行代码生成、设置或配置步骤时（而无需我主动提出要求），总是使用 Context7 。在进行编码工作时，必须严格遵循karpathy-guidelines这一技能要求。在编写代码前，需仔细研读并充分理解guidelines中的所有规范与标准；编码过程中，要确保每一行代码都符合karpathy-guidelines的具体条款，包括但不限于代码风格、命名规范、架构设计、安全要求等方面；完成编码后，需进行全面检查，验证代码是否完全遵循了karpathy-guidelines的各项规定，确保最终交付的代码符合该技能的所有要求。

## Prometheus 监控

项目已集成 Prometheus 兼容的指标导出功能，所有 HTTP 请求、LLM 调用、缓存访问、Neo4j 查询均自动采集指标。

### 端点

| 端点 | 格式 | 用途 |
|------|------|------|
| `GET /api/v1/metrics` | JSON | 后端内部指标（旧格式） |
| `GET /api/v1/metrics/prometheus` | Prometheus 文本 | 标准 Prometheus 格式，对接 Grafana |

### 指标体系

| 指标名 | 类型 | 标签 | 说明 |
|--------|------|------|------|
| `medicalgraph_http_requests_total` | Counter | `method`, `path`, `status` | HTTP 请求总数 |
| `medicalgraph_http_errors_total` | Counter | `method`, `path`, `status_code` | HTTP 错误数 |
| `medicalgraph_http_request_duration_ms` | Histogram | `method`, `path` | 请求延迟直方图 |
| `medicalgraph_llm_calls_total` | Counter | `status`, `model` | LLM 调用次数 |
| `medicalgraph_llm_call_duration_ms` | Histogram | `status`, `model` | LLM 调用延迟 |
| `medicalgraph_cache_access_total` | Counter | `cache`, `status` | 缓存命中/未命中 |
| `medicalgraph_neo4j_queries_total` | Counter | `query_type` | Neo4j 查询次数 |
| `medicalgraph_neo4j_query_duration_ms` | Histogram | `query_type` | Neo4j 查询延迟 |
| `medicalgraph_circuit_breaker_state` | Gauge | `name` | 熔断器状态 (0=CLOSED, 1=OPEN) |

### 使用方式

**直接查看**：访问 `http://localhost:8000/api/v1/metrics/prometheus`

**Prometheus 配置**：
```yaml
scrape_configs:
  - job_name: medicalgraph
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /api/v1/metrics/prometheus
```

**常用 PromQL**：
```promql
# QPS
rate(medicalgraph_http_requests_total[1m])

# P95 延迟
histogram_quantile(0.95, rate(medicalgraph_http_request_duration_ms_bucket[5m]))

# 错误率
sum(rate(medicalgraph_http_errors_total[5m])) / sum(rate(medicalgraph_http_requests_total[5m]))

# LLM 成功率
rate(medicalgraph_llm_calls_total{status="success"}[5m]) / rate(medicalgraph_llm_calls_total[5m])

# 缓存命中率
rate(medicalgraph_cache_access_total{status="hit"}[5m]) / rate(medicalgraph_cache_access_total[5m])
```

### 代码中使用

```python
from src.core.metrics import get_metrics_middleware

metrics = get_metrics_middleware()

# 记录请求
metrics.record_request(method="GET", path="/api/v1/query", status_code=200, duration_ms=150.5)

# 记录 LLM 调用
metrics.record_llm_call(duration_ms=2300.0, status="success", model="qwen-plus")

# 记录缓存访问
metrics.record_cache_access(hit=True, cache_type="query")

# 记录 Neo4j 查询
metrics.record_neo4j_query(duration_ms=5.2, query_type="entity_lookup")
```
