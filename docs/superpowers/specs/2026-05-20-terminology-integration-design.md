# 标准术语库接入设计（ICD-10 + DrugBank）

## 概述

接入 ICD-10 疾病编码库和 DrugBank 药物信息库，为医疗知识图谱提供标准化的术语基础。

## 目标

| 指标 | 当前 | 目标 |
|------|------|------|
| ICD-10 编码映射 | 140 条 | 70,000+ 条 |
| DrugBank 药物映射 | ~100 条 | 15,000+ 条 |
| 术语查询 API | 无 | 支持 |
| 编码转换 API | 无 | 支持 |

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    数据接入架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ ICD-10      │    │ DrugBank    │    │ 数据下载器   │     │
│  │ 编码库      │───▶│ 药物库      │───▶│             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   数据解析器                         │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐       │   │
│  │  │ XML解析   │  │ JSON解析  │  │ CSV解析   │       │   │
│  │  └───────────┘  └───────────┘  └───────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   术语映射器                         │   │
│  │  - 实体对齐                                          │   │
│  │  - 同义词扩展                                        │   │
│  │  - 关系构建                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   存储层                             │   │
│  │  ┌───────────┐  ┌───────────┐                       │   │
│  │  │ Neo4j     │  │ 术语索引  │                       │   │
│  │  │ 知识图谱  │  │ (缓存)    │                       │   │
│  │  └───────────┘  └───────────┘                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   API 层                             │   │
│  │  - 术语查询 API                                      │   │
│  │  - 编码转换 API                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 数据模型

### ICD-10 节点

```cypher
(:ICD10Code {
  code: "I10",              # ICD-10 编码
  name: "高血压",            # 中文名称
  name_en: "Hypertension",  # 英文名称
  category: "循环系统疾病",  # 疾病类别
  chapter: "IX",            # 章节
  block: "I10-I15",         # 编码块
  synonyms: ["原发性高血压", "高血压病"],  # 同义词
  description: "...",       # 详细描述
  related_codes: ["I11", "I12"]  # 相关编码
})
```

### DrugBank 节点

```cypher
(:Drug {
  drugbank_id: "DB00222",       # DrugBank ID
  name: "Glimepiride",          # 通用名
  name_cn: "格列美脲",          # 中文名
  cas_number: "93479-97-1",     # CAS 编号
  atc_code: "A10BB12",          # ATC 编码
  formula: "C24H34N4O5S",       # 分子式
  weight: "490.62 g/mol",       # 分子量
  indications: ["2型糖尿病"],   # 适应症
  contraindications: [...],     # 禁忌症
  side_effects: [...],          # 副作用
  interactions: [...],          # 药物相互作用
  synonyms: ["Amarel", "Amaryl"]  # 同义词
})
```

### 关系类型

```cypher
# ICD-10 相关关系
(:ICD10Code)-[:HAS_PARENT]->(:ICD10Code)      # 层级关系
(:ICD10Code)-[:RELATED_TO]->(:ICD10Code)      # 相关疾病
(:Disease)-[:HAS_ICD10]->(:ICD10Code)         # 疾病-编码

# DrugBank 相关关系
(:Drug)-[:TREATS]->(:Disease)                 # 适应症
(:Drug)-[:HAS_SIDE_EFFECT]->(:Symptom)        # 副作用
(:Drug)-[:INTERACTS_WITH]->(:Drug)            # 药物相互作用
(:Drug)-[:HAS_ATC]->(:ATCCode)                # ATC 编码
```

## 模块设计

### 1. 数据下载器

**文件**: `backend/src/terminology/downloader.py`

```python
class TerminologyDownloader:
    """标准术语库下载器"""
    
    def download_icd10(self, version: str = "2024") -> str:
        """下载 ICD-10 编码库"""
        
    def download_drugbank(self, version: str = "5.1") -> str:
        """下载 DrugBank 药物库"""
```

### 2. 数据解析器

**文件**: `backend/src/terminology/parser.py`

```python
class ICD10Parser:
    """ICD-10 数据解析器"""
    
    def parse(self, file_path: str) -> List[ICD10Code]:
        """解析 ICD-10 数据文件"""

class DrugBankParser:
    """DrugBank 数据解析器"""
    
    def parse(self, file_path: str) -> List[Drug]:
        """解析 DrugBank XML 文件"""
```

### 3. 术语映射器

**文件**: `backend/src/terminology/mapper.py`

```python
class TerminologyMapper:
    """术语映射器 - 将术语映射到现有实体"""
    
    def map_icd10_to_disease(self, icd10_code: ICD10Code) -> Optional[Entity]:
        """将 ICD-10 编码映射到疾病实体"""
        
    def map_drugbank_to_drug(self, drug: Drug) -> Optional[Entity]:
        """将 DrugBank 药物映射到药物实体"""
        
    def expand_synonyms(self, entity: Entity) -> List[str]:
        """扩展实体同义词"""
```

### 4. 术语服务

**文件**: `backend/src/terminology/service.py`

```python
class TerminologyService:
    """术语查询服务"""
    
    def lookup_icd10(self, code: str) -> Optional[ICD10Code]:
        """查询 ICD-10 编码"""
        
    def lookup_drug(self, name: str) -> List[Drug]:
        """查询药物信息"""
        
    def convert_code(self, code: str, from_system: str, to_system: str) -> str:
        """编码转换（如 ICD-10 转 SNOMED-CT）"""
        
    def search_by_name(self, name: str, terminology: str) -> List[Dict]:
        """按名称搜索术语"""
```

## 与现有代码集成

### 扩展 knowledge_fusion.py

```python
# 在 EntityDisambiguator 中扩展映射

class EntityDisambiguator:
    def __init__(self):
        # 现有映射
        self.synonym_rules = self._load_synonym_rules()
        self.icd10_mapping = self._load_icd10_mapping()  # 扩展
        self.drugbank_mapping = self._load_drugbank_mapping()  # 新增
        
    def _load_icd10_mapping(self) -> Dict[str, str]:
        """加载完整 ICD-10 映射（从术语库）"""
        
    def _load_drugbank_mapping(self) -> Dict[str, DrugInfo]:
        """加载 DrugBank 药物映射"""
```

### 新增 API 端点

```python
# backend/src/api/routes.py

@router.get("/terminology/icd10/{code}")
async def lookup_icd10(code: str):
    """查询 ICD-10 编码"""
    
@router.get("/terminology/drug/{name}")
async def lookup_drug(name: str):
    """查询药物信息"""
    
@router.post("/terminology/convert")
async def convert_code(request: CodeConvertRequest):
    """编码转换"""
```

## 数据来源

### ICD-10

| 来源 | 格式 | 说明 |
|------|------|------|
| WHO ICD-10 | XML/CSV | 官方版本 |
| 国家医保局 | Excel | 中国版本（含中文） |
| OpenICD10 | JSON | 开源版本 |

### DrugBank

| 来源 | 格式 | 说明 |
|------|------|------|
| DrugBank 官网 | XML | 需注册账号 |
| DrugBank Open | JSON | 开源子集 |

## 实施步骤

### Task 1: 创建术语模块结构

创建 `backend/src/terminology/` 目录和基础文件

### Task 2: 实现 ICD-10 下载和解析

下载 ICD-10 数据，解析并存储到 Neo4j

### Task 3: 实现 DrugBank 下载和解析

下载 DrugBank 数据，解析并存储到 Neo4j

### Task 4: 实现术语映射器

将术语映射到现有实体，扩展同义词

### Task 5: 实现术语服务 API

提供术语查询和编码转换 API

### Task 6: 集成测试

验证数据完整性和 API 功能

## 成功标准

- [ ] ICD-10 编码映射达到 70,000+ 条
- [ ] DrugBank 药物映射达到 15,000+ 条
- [ ] 术语查询 API 响应时间 < 100ms
- [ ] 与现有实体映射正确率 > 95%
- [ ] 所有测试通过
