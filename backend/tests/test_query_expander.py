"""Tests for src.retrieval.query_expander module.

Pure unit tests – MedicalQueryExpander has no external dependencies,
so no mocking is needed. All logic is self-contained.
"""

import pytest

from src.retrieval.query_expander import MedicalQueryExpander


# ===================================================================
# TestMedicalQueryExpander
# ===================================================================

class TestMedicalQueryExpander:

    def test_expand_with_synonyms(self):
        expander = MedicalQueryExpander()
        variants = expander.expand("高血压")
        # Original query should be first
        assert variants[0] == "高血压"
        # Should include synonym variants like "血压升高", "血压高"
        assert any("血压升高" in v for v in variants)
        assert any("血压高" in v for v in variants)

    def test_expand_with_intent_template(self):
        expander = MedicalQueryExpander()
        variants = expander.expand("高血压", intent="disease_query")
        # Should include the disease_query template expansion
        assert any("疾病" in v for v in variants)
        assert any("定义" in v or "病因" in v for v in variants)

    def test_expand_without_intent(self):
        expander = MedicalQueryExpander()
        variants = expander.expand("高血压")
        # Without intent, should still generate synonym variants
        assert len(variants) >= 1
        assert variants[0] == "高血压"

    def test_expand_no_matching_terms(self):
        expander = MedicalQueryExpander()
        variants = expander.expand("天气怎么样")
        # No medical terms match, should just return the original query
        assert variants[0] == "天气怎么样"

    def test_expand_single_returns_string(self):
        expander = MedicalQueryExpander()
        result = expander.expand_single("高血压", intent="disease_query")
        assert isinstance(result, str)
        assert "高血压" in result

    def test_expand_limits_to_6_variants(self):
        expander = MedicalQueryExpander()
        # Use a query that could generate many variants
        variants = expander.expand("高血压糖尿病阿司匹林", intent="disease_query")
        assert len(variants) <= 6


# ===================================================================
# TestFindMatchedTerms
# ===================================================================

class TestFindMatchedTerms:

    def test_find_matching_disease(self):
        expander = MedicalQueryExpander()
        matched = expander._find_matched_terms("高血压的治疗")
        # Should find "高血压" as a matched term
        matched_terms = [term for term, _ in matched]
        assert "高血压" in matched_terms

    def test_find_matching_drug(self):
        expander = MedicalQueryExpander()
        matched = expander._find_matched_terms("阿司匹林的用法")
        matched_terms = [term for term, _ in matched]
        assert "阿司匹林" in matched_terms

    def test_no_match(self):
        expander = MedicalQueryExpander()
        matched = expander._find_matched_terms("今天天气不错")
        assert matched == []


# ===================================================================
# TestGenerateIntentCandidates
# ===================================================================

class TestGenerateIntentCandidates:

    def test_disease_and_symptom_indicators(self):
        expander = MedicalQueryExpander()
        variants = ["心脏病头痛"]
        expander._generate_intent_candidates("心脏病头痛", variants)
        # Should add diagnosis_assist and symptom_query templates
        assert len(variants) > 1
        assert any("诊断" in v for v in variants)
        assert any("症状" in v for v in variants)

    def test_drug_indicators(self):
        expander = MedicalQueryExpander()
        variants = ["阿司匹林"]
        expander._generate_intent_candidates("阿司匹林", variants)
        # Should add drug_query and drug_interaction templates
        assert len(variants) > 1
        assert any("药物" in v for v in variants)
