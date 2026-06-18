"""黄金评估集加载器 - 加载从知识库生成的评估用例"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .benchmark_dataset import BenchmarkDataset, BenchmarkItem
from .medical_golden_set import MedicalGoldenCase


@dataclass
class GeneratedCase:
    """与 MedicalGoldenCase 兼容的数据结构"""
    question: str
    reference_answer: str
    expected_intent: str
    expected_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    safety_category: str = "general"
    forbidden_content: List[str] = field(default_factory=list)


def to_medical_golden_case(case: GeneratedCase) -> MedicalGoldenCase:
    return MedicalGoldenCase(
        question=case.question,
        reference_answer=case.reference_answer,
        expected_intent=case.expected_intent,
        expected_entities=case.expected_entities,
        keywords=case.keywords,
        category=case.category,
        difficulty=case.difficulty,
        safety_category=case.safety_category,
        forbidden_content=case.forbidden_content,
    )


def to_benchmark_item(case: GeneratedCase) -> BenchmarkItem:
    return BenchmarkItem(
        question=case.question,
        reference_answer=case.reference_answer,
        expected_intent=case.expected_intent,
        expected_entities=case.expected_entities,
        keywords=case.keywords,
        category=case.category,
        difficulty=case.difficulty,
    )


def load_from_json(filepath: str) -> List[GeneratedCase]:
    """从 JSON 文件加载生成的黄金评估集"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"黄金评估集文件不存在: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for item in data.get("items", []):
        cases.append(GeneratedCase(
            question=item.get("question", ""),
            reference_answer=item.get("reference_answer", ""),
            expected_intent=item.get("expected_intent", "general"),
            expected_entities=item.get("expected_entities", []),
            keywords=item.get("keywords", []),
            category=item.get("category", "general"),
            difficulty=item.get("difficulty", "medium"),
            safety_category=item.get("safety_category", "general"),
            forbidden_content=item.get("forbidden_content", []),
        ))
    return cases


def load_generated_dataset(filepath: Optional[str] = None) -> BenchmarkDataset:
    """加载为 BenchmarkDataset，可直接用于评估"""
    if filepath is None:
        filepath = str(Path(__file__).resolve().parent.parent.parent.parent /
                       "golden_set" / "generated_golden.json")

    cases = load_from_json(filepath)
    items = [to_benchmark_item(c) for c in cases]
    dataset = BenchmarkDataset(name="generated_golden_set", items=items)
    
    print(f"[生成集] 加载了 {len(dataset)} 条从知识库生成的评估用例")
    cat_dist = {}
    for item in items:
        cat_dist[item.category] = cat_dist.get(item.category, 0) + 1
    for cat, cnt in sorted(cat_dist.items()):
        print(f"  {cat}: {cnt}")
    
    return dataset


def load_generated_golden_set(filepath: Optional[str] = None) -> List[MedicalGoldenCase]:
    """加载为 MedicalGoldenCase 列表，可用于 RAGAS 安全评估"""
    if filepath is None:
        filepath = str(Path(__file__).resolve().parent.parent.parent.parent /
                       "golden_set" / "generated_golden.json")

    cases = load_from_json(filepath)
    return [to_medical_golden_case(c) for c in cases]
