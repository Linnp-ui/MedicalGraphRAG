# 黄金评估集

## 目录结构

```
golden_set/
├── generated_golden.json   # 从知识库自动生成的黄金评估集 (80条)
├── generate_golden.py      # (可选) 重新生成脚本的引用
└── README.md               # 本文件
```

生成脚本位于: `backend/scripts/generate_golden_set.py`

## 数据统计

| 类别 | 数量 | 说明 |
|------|------|------|
| disease_knowledge | 15 | 疾病定义知识 |
| drug_knowledge | 16 | 药物用途知识 |
| drug_safety | 14 | 药物安全与副作用 |
| diagnosis_assist | 11 | 诊断辅助（症状-疾病）|
| medical_coding | 9 | 医学编码（ICD-10/缩写）|
| treatment_safety | 14 | 治疗安全 |

## 使用方式

```python
from src.evaluation import (
    OfflineEvaluator, 
    load_generated_dataset,
    load_generated_golden_set,
)

# 方式1: 作为 BenchmarkDataset 用于离线评估
evaluator = OfflineEvaluator()
evaluator.load_generated_dataset("golden_set/generated_golden.json")
# 或使用默认路径
# evaluator.load_generated_dataset()
report = evaluator.run_evaluation()

# 方式2: 作为 MedicalGoldenCase 列表用于 RAGAS 安全评估
cases = load_generated_golden_set()
print(f"加载了 {len(cases)} 条黄金用例")
```

## 字段说明

每条用例包含:
- `question`: 问题文本
- `reference_answer`: 参考标准回答
- `expected_intent`: 期望意图分类
- `expected_entities`: 期望识别的实体
- `keywords`: 关键词列表（用于关键词匹配评估）
- `category`: 类别标签
- `difficulty`: 难度等级 (easy/medium/hard)
- `safety_category`: 安全类别
- `forbidden_content`: 回答中不应出现的内容
