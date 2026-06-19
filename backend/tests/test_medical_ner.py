"""Tests for src.ingestion.medical_ner"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import fields

from src.ingestion.medical_ner import NEREntity, MedicalNER


# ---------------------------------------------------------------------------
# Controlled synonym / abbreviation fixtures
# ---------------------------------------------------------------------------
SYNONYM_RULES = {
    "Disease": {
        "高血压": ["高血压病", "血压高"],
        "糖尿病": ["DM", "血糖高"],
    },
    "Symptom": {
        "头痛": ["头疼"],
        "头晕": ["头昏"],
    },
}

ABBREVIATION_MAP = {
    "HTN": "高血压",
    "DM": "糖尿病",
}


@pytest.fixture
def mock_disambiguator():
    """Patch EntityDisambiguator to return controlled data."""
    with patch("src.ingestion.medical_ner.EntityDisambiguator") as mock_cls:
        instance = MagicMock()
        instance.synonym_rules = SYNONYM_RULES
        instance.abbreviation_map = ABBREVIATION_MAP
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def ner(mock_disambiguator):
    return MedicalNER()


# ---------------------------------------------------------------------------
# TestNEREntity
# ---------------------------------------------------------------------------
class TestNEREntity:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(NEREntity)}
        assert field_names == {
            "name", "entity_type", "start_pos", "end_pos",
            "confidence", "strategy",
        }

    def test_default_strategy(self):
        e = NEREntity(name="test", entity_type="Disease",
                      start_pos=0, end_pos=2, confidence=0.9)
        assert e.strategy == "unknown"


# ---------------------------------------------------------------------------
# TestMedicalNER
# ---------------------------------------------------------------------------
class TestMedicalNER:

    def test_extract_with_medical_text(self, ner):
        text = "高血压是一种常见的疾病，患者可能出现头痛、头晕等症状"
        entities = ner.extract(text)

        # Should find 高血压, 头痛, 头晕 at minimum via exact match
        names = {e.name for e in entities}
        assert "高血压" in names
        assert "头痛" in names
        assert "头晕" in names

        # All entities should have valid positions
        for e in entities:
            assert e.start_pos >= 0
            assert e.end_pos > e.start_pos
            assert e.confidence > 0

    def test_extract_empty_text(self, ner):
        entities = ner.extract("")
        assert entities == []

    def test_exact_match(self, ner):
        text = "高血压患者需要注意饮食"
        entities = ner._exact_match(text, set())
        names = [e.name for e in entities]
        assert "高血压" in names
        for e in entities:
            assert e.strategy == "exact_dict"
            assert e.confidence == 0.95

    def test_abbreviation_match(self, ner):
        text = "HTN is a common condition"
        entities = ner._abbreviation_match(text, set())
        names = [e.name for e in entities]
        assert "高血压" in names
        for e in entities:
            assert e.strategy == "abbreviation"
            assert e.confidence == 0.90

    def test_suffix_match(self, ner):
        # "心脏病" ends with 病 suffix
        text = "心脏病 是常见的"
        entities = ner._suffix_match(text, set())
        types = [e.entity_type for e in entities]
        assert "Disease" in types
        for e in entities:
            assert e.strategy == "suffix"
            assert e.confidence == 0.70

    def test_cross_validation_boosts_confidence(self, ner):
        e1 = NEREntity(name="高血压", entity_type="Disease",
                       start_pos=0, end_pos=3, confidence=0.95, strategy="exact_dict")
        e2 = NEREntity(name="高血压", entity_type="Disease",
                       start_pos=5, end_pos=8, confidence=0.70, strategy="suffix")

        result = ner._cross_validate([e1, e2])
        # Should keep the best entity with boosted confidence
        assert len(result) == 1
        assert result[0].confidence > 0.95  # boosted

    def test_resolve_overlaps_keeps_highest_confidence(self, ner):
        e1 = NEREntity(name="高血压", entity_type="Disease",
                       start_pos=0, end_pos=3, confidence=0.95, strategy="exact_dict")
        e2 = NEREntity(name="高血", entity_type="Disease",
                       start_pos=0, end_pos=2, confidence=0.70, strategy="suffix")

        result = ner._resolve_overlaps([e1, e2])
        # Should keep the higher confidence entity
        assert len(result) == 1
        assert result[0].name == "高血压"

    def test_extract_as_dict_format(self, ner):
        text = "高血压患者"
        dicts = ner.extract_as_dict(text)
        assert isinstance(dicts, list)
        if dicts:
            assert "name" in dicts[0]
            assert "type" in dicts[0]
            assert "properties" in dicts[0]
            assert "confidence" in dicts[0]["properties"]
            assert "strategy" in dicts[0]["properties"]
