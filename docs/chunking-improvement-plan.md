# 文本切分策略改进方案

## 一、现状分析

当前 `text_splitter.py` 实现了 7 种策略，整体架构合理但存在若干可优化点：

### 现有架构优势
- 多策略支持（paragraph / sentence / markdown / hybrid / medical / semantic / hierarchical）
- 自动策略选择（`select_chunking_strategy`）
- 流式处理支持

### 现存问题

1. **递归切分缺失** — 没有 LangChain `RecursiveCharacterTextSplitter` 式的按优先级降级切分
2. **语义切分不稳定** — SEMANTIC 策略依赖 `sentence-transformers`，首次加载慢，且阈值固定为 25%，缺乏 buffer 机制
3. **层级切分硬编码** — HIERARCHICAL 的三层大小（1024/512/256）写死，不可配置
4. **无软硬双阈值** — 缺乏 Unstructured 式的 `soft_max` / `hard_max` 控制
5. **重叠策略单一** — 当前只做尾部字符 overlap，没有语义层面的 overlap 控制
6. **无 Late Chunking** — 嵌入阶段未利用全文上下文
7. **中文 CJK 支持不足** — 缺少对中文全角标点的完整分隔符链
8. **流式切分不够智能** — `split_text_streaming` 只做最大 50 字符的回退查找，精细度不足
9. **测试覆盖不足** — 缺少对边界条件（超大段落、空文档、纯代码）的测试
10. **无性能指标** — 未记录各策略的分块耗时、chunk 数量统计

---

## 二、改进目标

| 维度 | 当前 | 目标 |
|------|------|------|
| 召回率 | ~70-80%（估算） | >= 90%（通过 Contextual Retrieval） |
| 语义完整性 | 段落/句子硬切 | 语义边界感知 + 上下文感知嵌入 |
| 配置灵活度 | 固定参数 | 可调软硬阈值、buffer、层级大小 |
| 中文支持 | 基础 | 完整的 CJK 分隔符链 |
| 可观测性 | 日志级别 debug | 统计指标暴露 |
| 微基准 | chunk_size 为字符数 | 支持 token 级计数（tiktoken） |

---

## 三、分阶段改进方案

### 阶段一：架构增强（1-2 天）

#### 1.1 新增 RecursiveSplitStrategy

参考 LangChain 的递归降级策略，替换当前 `_merge_into_chunks` 的线性合并：

**核心思路**：按分隔符优先级链 `["\n\n", "\n", "。", "．", ". ", "，", " ", ""]` 递归尝试直到 chunk 大小达标。

**代码位置**：`TextSplitter._split_recursive()`

```python
def _split_recursive(self, text: str, document_id: str, separators: list[str]) -> list[TextChunk]:
    """按分隔符优先级递归切分"""
```

#### 1.2 引入软硬双阈值

参考 Unstructured 的 `new_after_n_chars`（软阈值）+ `max_characters`（硬阈值）：

```python
chunk_size: int = 512        # 硬阈值（硬上限）
soft_max: int = 384          # 软阈值（到达此值后尝试闭合 chunk）
```

在 `_merge_into_chunks` 中加入判断：达到 `soft_max` 且遇到边界时立即切分，避免硬阈值导致截断语义。

#### 1.3 完善 CJK 分隔符

当前分隔符仅覆盖基础标点，补充全角字符：

```python
CJK_SEPARATORS = [
    "\n\n", "\n",
    "。", "．", "！", "？",
    ". ", "! ", "? ",
    "；", "；\n",
    "，", "、",
    " ", ""
]
```

---

### 阶段二：语义切分升级（2-3 天）

#### 2.1 改进 SemanticSplitter（参考 LlamaIndex）

核心变化：引入 `buffer_size` 和 `breakpoint_percentile_threshold` 替代当前简单的 25% 阈值。

**当前问题**：
- 当前对每对相邻句子计算相似度，阈值为 `np.percentile(similarities, 25)`
- 没有 buffer 窗口，边界检测不够平滑

**改进方案**：

```python
# 新增参数
buffer_size: int = 1          # 比较时两侧各取 N 句
breakpoint_percentile_threshold: int = 95  # 百分比阈值（越大 chunk 越少）
```

算法流程（参考 LlamaIndex `SemanticSplitterNodeParser`）：
1. 按句子切分，合并 `buffer_size` 句为一个窗口
2. 计算相邻窗口的余弦相似度
3. 取相似度分布的 `breakpoint_percentile_threshold` 百分位为阈值
4. 低于阈值处为语义边界

#### 2.2 缓存嵌入模型

当前每次 `_split_semantic` 都重新加载模型，改为类级别的 LRU 缓存：

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str = "shibing624/text2vec-base-chinese"):
    return SentenceTransformer(model_name)
```

---

### 阶段三：上下文感知切分（3-5 天）

#### 3.1 引入 Contextual Retrieval（参考 Anthropic）

对医疗文档场景价值极大：在处理 chunk 之前，用 LLM 生成一段 30-50 字的上下文摘要前置到 chunk 中。

```python
class ContextualChunkEnricher:
    """用 LLM 为每个 chunk 生成上下文"""
    
    async def enrich(self, chunk: TextChunk, document: Document) -> TextChunk:
        prompt = f"""这是文档 <document>{document.content}</document> 中的一个段落。
请用一句话概括这个段落在文档中的上下文位置和主题。
段落：{chunk.content}"""
        context = await self.llm.generate(prompt)
        chunk.content = f"[上下文] {context}\n{chunk.content}"
        return chunk
```

配置开关：
```python
contextual_enrichment: bool = False    # 默认关闭，仅对重要文档启用
contextual_llm_model: str = "qwen-flash"  # 用小模型降低开销
```

#### 3.2 预留 Late Chunking 接口

Late Chunking 的核心思想是：先对整个文档做 transformer 编码，再切分 pool 成 chunk 向量。

```python
class LateChunkingAdapter:
    """延迟分块适配器 - 嵌入时用全文上下文"""
    
    def encode_chunks(self, chunks: list[TextChunk], full_text: str) -> list[list[float]]:
        # 1. 用长上下文嵌入模型编码全文
        # 2. 根据 chunk 边界从 token 序列中提取
        # 3. 对每个 chunk 的 token 做 mean pooling
        pass
```

当前作为预留接口（`late_chunking_enabled: bool = False`），对齐 Jina AI 的 API 风格：`late_chunking=true`。

---

### 阶段四：可观测性与测试（1-2 天）

#### 4.1 添加性能统计

```python
@dataclass
class SplitStats:
    strategy: str
    num_chunks: int
    total_chars: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    split_time_ms: float
    overlap_ratio: float
```

#### 4.2 补充测试用例

| 类型 | 用例 |
|------|------|
| 边界 | 空文档、单字符、单行、超大段落 |
| CJK | 纯中文、中日混排、全角标点 |
| 代码 | Python、JSON、HTML 代码块保持 |
| 医疗 | 多节医疗报告、混合文本 |
| 语义 | 不同主题跳转、指代消解场景 |
| 流式 | 超大文档内存控制 |

---

## 四、配置变更

```python
# config.py 新增字段
chunk_size: int = 512                    # 硬阈值（字符数）
chunk_overlap: int = 75                  # 重叠字符数
soft_max: int = 384                      # 软阈值（新增）
split_strategy: str = "auto"             # 策略
keep_code_blocks: bool = True
keep_headers: bool = True

# 语义切分（阶段二）
semantic_buffer_size: int = 1            # 语义窗口大小
semantic_breakpoint_percentile: int = 95 # 语义断点百分位
semantic_model: str = "shibing624/text2vec-base-chinese"

# 递归切分（阶段一）
recursive_separators: list[str] = [      # 分隔符优先级链
    "\n\n", "\n", "。", "．", ". ", "；", "，", " ", ""
]

# 上下文增强（阶段三）
contextual_enrichment: bool = False
contextual_llm_model: str = "qwen-flash"

# 延迟分块（阶段三，预留）
late_chunking_enabled: bool = False
```

---

## 五、预期效果

| 指标 | 当前 | 阶段一后 | 阶段二后 | 阶段三后 |
|------|------|----------|----------|----------|
| 检索召回率 | ~75% | ~80% | ~85% | ~90%+ |
| 语义完整率 | ~70% | ~80% | ~90% | ~95% |
| 中文兼容性 | 基础 | 良好 | 良好 | 优秀 |
| 配置灵活度 | 固定 | 可调节 | 自适应 | 上下文感知 |

---

## 六、优先级建议

```
P0（本周）: 阶段一 + 阶段二 → 提升基础切分质量
P1（下两周）: 阶段三 → 上下文感知增强（医疗场景重点）
P2（持续）: 阶段四 → 观测 + 测试覆盖
```
