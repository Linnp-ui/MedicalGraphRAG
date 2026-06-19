"""Tests for src.evaluation.answer_optimizer"""

import pytest
from unittest.mock import patch, MagicMock

from src.evaluation.answer_optimizer import (
    AnswerStructureOptimizer,
    KeyInformationExtractor,
    LanguageExpressionOptimizer,
    AnswerQualityScorer,
    AnswerOptimizer,
    QualityMetrics,
)


# ---------------------------------------------------------------------------
# Mock jieba for all tests that indirectly call extract_entities
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_jieba():
    """Mock jieba so extract_entities works without the real package."""
    mock_mod = MagicMock()
    mock_mod.cut.return_value = iter(["高血压", "和", "糖尿病"])
    with patch.dict("sys.modules", {"jieba": mock_mod}):
        yield mock_mod


# ---------------------------------------------------------------------------
# TestAnswerStructureOptimizer
# ---------------------------------------------------------------------------
class TestAnswerStructureOptimizer:

    def test_structure_disease_query(self):
        content = [
            "高血压是一种常见的疾病",
            "引起高血压的原因包括遗传和环境因素",
            "症状表现为头痛和头晕",
            "诊断需要检查血压",
            "治疗需要服用药物",
            "预防需要注意饮食",
        ]
        result = AnswerStructureOptimizer.structure_by_intent("disease_query", content)
        assert "定义" in result
        assert "病因" in result
        assert "症状" in result

    def test_structure_drug_query(self):
        content = [
            "阿司匹林用于治疗疼痛",
            "每次服用100mg",
            "禁忌不宜空腹服用",
            "副作用包括胃部不适",
        ]
        result = AnswerStructureOptimizer.structure_by_intent("drug_query", content)
        assert "适应症" in result
        assert "用法用量" in result
        assert "副作用" in result

    def test_structure_symptom_query(self):
        content = [
            "头痛是指头部疼痛的感觉",
            "由于高血压引起",
            "建议及时就医",
        ]
        result = AnswerStructureOptimizer.structure_by_intent("symptom_query", content)
        assert "定义" in result
        assert "可能原因" in result
        assert "建议" in result

    def test_structure_treatment_query(self):
        content = [
            "治疗方法包括药物治疗和手术治疗",
            "首先进行药物治疗然后考虑手术",
            "注意事项是术后护理",
        ]
        result = AnswerStructureOptimizer.structure_by_intent("treatment_query", content)
        assert "治疗方法" in result
        assert "治疗流程" in result
        assert "注意事项" in result

    def test_structure_unknown_intent(self):
        content = ["这是一段内容", "这是另一段内容"]
        result = AnswerStructureOptimizer.structure_by_intent("unknown_intent", content)
        assert "这是一段内容" in result
        assert "这是另一段内容" in result

    def test_structure_empty_content(self):
        result = AnswerStructureOptimizer.structure_by_intent("disease_query", [])
        assert result == ""


# ---------------------------------------------------------------------------
# TestKeyInformationExtractor
# ---------------------------------------------------------------------------
class TestKeyInformationExtractor:

    def test_extract_entities_finds_known_terms(self):
        text = "高血压和糖尿病是常见的慢性疾病"
        entities = KeyInformationExtractor.extract_entities(text)
        assert "高血压" in entities
        assert "糖尿病" in entities

    def test_extract_key_concepts(self):
        text = "高血压是常见的疾病。头痛表现为头部疼痛。"
        concepts = KeyInformationExtractor.extract_key_concepts(text)
        assert len(concepts) > 0

    def test_extract_medical_terms(self):
        text = "患者需要检查血压和血糖，可能需要手术"
        terms = KeyInformationExtractor.extract_medical_terms(text)
        assert "血压" in terms
        assert "血糖" in terms
        assert "手术" in terms

    def test_get_coverage_rate_full(self):
        reference = "高血压和糖尿病"
        prediction = "高血压和糖尿病"
        rate = KeyInformationExtractor.get_coverage_rate(prediction, reference)
        assert rate == 1.0

    def test_get_coverage_rate_partial(self):
        reference = "高血压和糖尿病和心脏病"
        prediction = "高血压"
        rate = KeyInformationExtractor.get_coverage_rate(prediction, reference)
        assert 0 < rate < 1.0

    def test_get_coverage_rate_no_reference_entities(self):
        rate = KeyInformationExtractor.get_coverage_rate("prediction", "普通文本")
        assert rate == 1.0


# ---------------------------------------------------------------------------
# TestLanguageExpressionOptimizer
# ---------------------------------------------------------------------------
class TestLanguageExpressionOptimizer:

    def test_enhance_with_synonyms(self):
        text = "血压高的患者需要注意饮食"
        reference = "高血压是一种常见疾病"
        result = LanguageExpressionOptimizer.enhance_with_synonyms(text, reference)
        assert "高血压" in result

    def test_ensure_term_consistency(self):
        prediction = "糖尿病患者需要控制饮食"
        reference = "血糖是重要的指标"
        result = LanguageExpressionOptimizer.ensure_term_consistency(prediction, reference)
        assert "血糖" in result or "血糖异常" in result


# ---------------------------------------------------------------------------
# TestAnswerQualityScorer
# ---------------------------------------------------------------------------
class TestAnswerQualityScorer:

    @pytest.fixture
    def scorer(self):
        return AnswerQualityScorer()

    def test_score_completeness(self, scorer):
        reference = "高血压和糖尿病是常见疾病"
        prediction = "高血压是常见疾病"
        score = scorer.score_completeness(prediction, reference)
        assert 0 < score <= 1.0

    def test_score_structure_with_markers(self, scorer):
        prediction = "高血压：主要包括以下几点。首先注意饮食；其次注意运动"
        score = scorer.score_structure(prediction, "disease_query")
        assert score == 0.9

    def test_score_structure_without_markers(self, scorer):
        prediction = "高血压是常见疾病"
        score = scorer.score_structure(prediction, "disease_query")
        assert score == 0.5

    def test_score_term_consistency(self, scorer):
        reference = "血压和血糖是重要指标"
        prediction = "血压和血糖是重要指标"
        score = scorer.score_term_consistency(prediction, reference)
        assert score == 1.0

    def test_score_fluency_good_length(self, scorer):
        text = "高血压是一种常见的慢性疾病，需要长期管理和治疗。"
        score = scorer.score_fluency(text)
        assert score == 0.9

    def test_score_fluency_bad_length(self, scorer):
        text = "好"
        score = scorer.score_fluency(text)
        assert score < 0.9

    def test_score_overall(self, scorer):
        prediction = "高血压是一种常见的慢性疾病，需要长期管理"
        reference = "高血压和糖尿病是常见疾病"
        metrics = scorer.score_overall(prediction, reference, "disease_query")
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.completeness <= 1.0
        assert 0 <= metrics.structure_score <= 1.0
        assert 0 <= metrics.term_consistency <= 1.0
        assert 0 <= metrics.fluency <= 1.0
        assert 0 <= metrics.overall_quality <= 1.0


# ---------------------------------------------------------------------------
# TestAnswerOptimizer
# ---------------------------------------------------------------------------
class TestAnswerOptimizer:

    @pytest.fixture
    def optimizer(self):
        return AnswerOptimizer()

    def test_optimize_with_intent(self, optimizer):
        answer = "血压高的患者需要注意饮食和运动"
        reference = "高血压是一种常见疾病，需要注意饮食"
        optimized, quality = optimizer.optimize(answer, reference, intent="disease_query")
        assert isinstance(optimized, str)
        assert isinstance(quality, QualityMetrics)

    def test_optimize_without_intent(self, optimizer):
        answer = "高血压患者需要注意饮食"
        reference = "高血压是一种常见疾病"
        optimized, quality = optimizer.optimize(answer, reference)
        assert isinstance(optimized, str)
        assert isinstance(quality, QualityMetrics)

    def test_batch_optimize(self, optimizer):
        answers = ["血压高的患者", "糖尿病患者"]
        references = ["高血压是常见疾病", "糖尿病是慢性病"]
        results = optimizer.batch_optimize(answers, references)
        assert len(results) == 2
        for opt, quality in results:
            assert isinstance(opt, str)
            assert isinstance(quality, QualityMetrics)

    def test_get_improvement_summary(self, optimizer):
        before = "血压高"
        after = "高血压是一种常见的疾病，需要注意饮食和运动"
        reference = "高血压和糖尿病是常见疾病"
        summary = optimizer.get_improvement_summary(before, after, reference)
        assert "before_quality" in summary
        assert "after_quality" in summary
        assert "improvements" in summary
        assert "overall" in summary["improvements"]
