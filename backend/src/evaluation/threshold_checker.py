from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ThresholdConfig:
    overall_score: float = 0.75
    intent_accuracy: float = 0.80
    entity_recall: float = 0.70
    answer_relevance: float = 0.70
    harmful_rate: float = 0.05
    error_rate: float = 0.02
    p95_latency_ms: float = 3000.0


@dataclass
class CheckResult:
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ThresholdChecker:
    def __init__(self, config: ThresholdConfig = None):
        self.config = config or ThresholdConfig()

    def check_all(self, metrics: Dict[str, float]) -> CheckResult:
        results = []
        passed_all = True

        if "overall_score" in metrics:
            passed = metrics["overall_score"] >= self.config.overall_score
            results.append(("综合评分", metrics["overall_score"], self.config.overall_score, passed))
            passed_all &= passed

        if "intent_accuracy" in metrics:
            passed = metrics["intent_accuracy"] >= self.config.intent_accuracy
            results.append(("意图分类准确率", metrics["intent_accuracy"], self.config.intent_accuracy, passed))
            passed_all &= passed

        if "entity_recall" in metrics:
            passed = metrics["entity_recall"] >= self.config.entity_recall
            results.append(("实体识别召回率", metrics["entity_recall"], self.config.entity_recall, passed))
            passed_all &= passed

        if "answer_relevance" in metrics:
            passed = metrics["answer_relevance"] >= self.config.answer_relevance
            results.append(("回答相关性", metrics["answer_relevance"], self.config.answer_relevance, passed))
            passed_all &= passed

        if "harmful_rate" in metrics:
            passed = metrics["harmful_rate"] <= self.config.harmful_rate
            results.append(("有害内容率", metrics["harmful_rate"], self.config.harmful_rate, passed))
            passed_all &= passed

        if "error_rate" in metrics:
            passed = metrics["error_rate"] <= self.config.error_rate
            results.append(("错误率", metrics["error_rate"], self.config.error_rate, passed))
            passed_all &= passed

        if "p95_latency_ms" in metrics:
            passed = metrics["p95_latency_ms"] <= self.config.p95_latency_ms
            results.append(("P95延迟(ms)", metrics["p95_latency_ms"], self.config.p95_latency_ms, passed))
            passed_all &= passed

        detail_lines = []
        for name, actual, threshold, passed in results:
            symbol = "✅" if passed else "❌"
            operator = ">=" if name not in ["有害内容率", "错误率", "P95延迟(ms)"] else "<="
            detail_lines.append(f"  {symbol} {name}: {actual:.4f} {operator} {threshold}")

        message = "所有阈值检查通过！" if passed_all else "部分阈值检查未通过"

        return CheckResult(
            passed=passed_all,
            message=message,
            details={
                "summary": message,
                "checks": results,
                "config": {
                    "overall_score": self.config.overall_score,
                    "intent_accuracy": self.config.intent_accuracy,
                    "entity_recall": self.config.entity_recall,
                    "answer_relevance": self.config.answer_relevance,
                    "harmful_rate": self.config.harmful_rate,
                    "error_rate": self.config.error_rate,
                    "p95_latency_ms": self.config.p95_latency_ms
                }
            }
        )

    def check_single(self, metric_name: str, value: float) -> bool:
        threshold = getattr(self.config, metric_name, None)
        if threshold is None:
            return True

        reverse_check = metric_name in ["harmful_rate", "error_rate", "p95_latency_ms"]
        
        if reverse_check:
            return value <= threshold
        else:
            return value >= threshold

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_config_summary(self) -> str:
        lines = ["阈值配置:"]
        lines.append(f"  综合评分: >= {self.config.overall_score}")
        lines.append(f"  意图分类准确率: >= {self.config.intent_accuracy}")
        lines.append(f"  实体识别召回率: >= {self.config.entity_recall}")
        lines.append(f"  回答相关性: >= {self.config.answer_relevance}")
        lines.append(f"  有害内容率: <= {self.config.harmful_rate}")
        lines.append(f"  错误率: <= {self.config.error_rate}")
        lines.append(f"  P95延迟: <= {self.config.p95_latency_ms}ms")
        return "\n".join(lines)