# 知识融合对齐与扩展设计

## 概述

修复 `medical_schema.py` 枚举定义与 `knowledge_fusion.py` 实际使用不一致的问题，扩展同义词映射到 600+，完善本体约束和 ICD-10/UMLS 编码映射。

## 目标

| 指标 | 当前 | 目标 |
|------|------|------|
| 实体类型 | 7种 | 11种（+4） |
| 关系类型 | 8种 | 18种（+10） |
| 同义词映射 | ~830条 | 1000+条 |
| ICD-10 编码 | ~140条 | 300+条 |
| UMLS CUI 编码 | ~150条 | 350+条 |
| 关系降级率 | 存在 | 0% |

## 问题分析

### 当前问题

`medical_schema.py` 枚举与 `knowledge_fusion.py` 实际使用不一致：

| 维度 | medical_schema.py | knowledge_fusion.py |
|------|-------------------|---------------------|
| 实体类型 | 7种 | 11种 |
| 关系类型 | 8种 | 18种 |

导致部分关系被降级为 `RELATES_TO`，丢失语义信息。

## 设计方案

### 1. 扩展实体类型枚举

文件：`backend/src/core/medical_schema.py`

```python
class MedicalEntityType(str, Enum):
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    DRUG = "Drug"
    EXAMINATION = "Examination"
    TREATMENT = "Treatment"
    ANATOMY = "Anatomy"
    DEPARTMENT = "Department"
    DIAGNOSTIC_CRITERIA = "DiagnosticCriteria"
    PROGNOSIS = "Prognosis"
    RISK_FACTOR = "RiskFactor"
    PATHOLOGY = "Pathology"
```

### 2. 扩展关系类型枚举

```python
class MedicalRelationshipType(str, Enum):
    HAS_SYMPTOM = "HAS_SYMPTOM"
    CAUSED_BY = "CAUSED_BY"
    TREATED_BY = "TREATED_BY"
    DRUG_FOR = "DRUG_FOR"
    SIDE_EFFECT = "SIDE_EFFECT"
    INDICATES = "INDICATES"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    COMPLICATED_BY = "COMPLICATED_BY"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    PREVENTS = "PREVENTS"
    DIAGNOSED_BY = "DIAGNOSED_BY"
    DIAGNOSTIC_CRITERIA = "DIAGNOSTIC_CRITERIA"
    PROGNOSIS = "PROGNOSIS"
    RISK_FACTOR = "RISK_FACTOR"
    WARNING_SIGN = "WARNING_SIGN"
    DIFFERENTIAL_DIAGNOSIS = "DIFFERENTIAL_DIAGNOSIS"
    SUGGESTS = "SUGGESTS"
```

### 3. 扩展本体约束

```python
"allowed_relations": {
    "Disease": {
        "Symptom": ["HAS_SYMPTOM", "WARNING_SIGN"],
        "Treatment": ["TREATED_BY"],
        "Drug": ["TREATED_BY", "DRUG_FOR"],
        "Department": ["BELONGS_TO"],
        "Anatomy": ["PART_OF"],
        "Disease": ["COMPLICATED_BY", "ASSOCIATED_WITH", "DIFFERENTIAL_DIAGNOSIS"],
        "DiagnosticCriteria": ["DIAGNOSTIC_CRITERIA"],
        "Prognosis": ["PROGNOSIS"],
        "RiskFactor": ["RISK_FACTOR"],
        "Examination": ["DIAGNOSED_BY"],
    },
    "Drug": {
        "Disease": ["DRUG_FOR", "PREVENTS"],
        "Symptom": ["SIDE_EFFECT"],
        "RiskFactor": ["RISK_FACTOR"],
    },
    "Symptom": {
        "Disease": ["INDICATES", "SUGGESTS"],
    },
    "RiskFactor": {
        "Disease": ["RISK_FACTOR"],
    },
    "Examination": {
        "Disease": ["DIAGNOSED_BY"],
    },
    "Prognosis": {
        "Disease": ["PROGNOSIS"],
    },
    "DiagnosticCriteria": {
        "Disease": ["DIAGNOSTIC_CRITERIA"],
    },
}
```

### 4. 扩展同义词映射

目标：1000+ 条同义词映射

| 实体类型 | 当前 | 目标 | 新增 |
|----------|------|------|------|
| Disease | ~150 | 200 | +50 |
| Symptom | ~200 | 250 | +50 |
| Drug | ~100 | 150 | +50 |
| Examination | ~200 | 220 | +20 |
| Anatomy | ~70 | 100 | +30 |
| Treatment | ~40 | 60 | +20 |
| Department | ~45 | 50 | +5 |
| DiagnosticCriteria | 9 | 20 | +11 |
| Prognosis | 7 | 15 | +8 |
| RiskFactor | 16 | 30 | +14 |
| **总计** | **~830** | **~1095** | **~258** |

### 5. 扩展 ICD-10/UMLS 映射

| 编码类型 | 当前 | 目标 |
|----------|------|------|
| ICD-10 | ~140 | 300+ |
| UMLS CUI | ~150 | 350+ |

## 实施步骤

### Task 1: 扩展枚举定义

1. 修改 `medical_schema.py` 添加新枚举值
2. 运行测试验证无破坏性变更

### Task 2: 扩展本体约束

1. 更新 `allowed_relations` 覆盖所有新类型组合
2. 验证约束校验通过

### Task 3: 扩展同义词映射

1. 添加新类型（同义词到现有映射）
2. 扩展各类型同义词数量到目标

### Task 4: 扩展编码映射

1. 扩展 ICD-10 编码映射
2. 扩展 UMLS CUI 编码映射

### Task 5: 集成测试

1. 运行完整测试套件
2. 验证关系降级率降为 0

## 成功标准

- [ ] 所有枚举定义一致，无遗漏
- [ ] 本体约束覆盖所有类型组合
- [ ] 同义词映射达到 1000+ 条
- [ ] ICD-10 编码达到 300+ 条
- [ ] UMLS CUI 编码达到 350+ 条
- [ ] 关系降级率为 0%
- [ ] 所有测试通过
