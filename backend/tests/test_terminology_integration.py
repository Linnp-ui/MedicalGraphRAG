import pytest
from fastapi.testclient import TestClient

from src.terminology.service import TerminologyService
from src.ingestion.knowledge_fusion import EntityDisambiguator


class TestEntityDisambiguatorTerminologyIntegration:
    def setup_method(self):
        TerminologyService.reset()

    def teardown_method(self):
        TerminologyService.reset()

    def test_terminology_service_initialization(self):
        disambiguator = EntityDisambiguator()

        assert hasattr(disambiguator, "_terminology_service")

    def test_terminology_service_none_on_error(self, monkeypatch):
        def mock_import(name, *args, **kwargs):
            if "terminology.service" in name:
                raise ImportError("Mocked import error")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        TerminologyService.reset()
        disambiguator = EntityDisambiguator()

        assert disambiguator._terminology_service is None

    def test_disambiguator_has_basic_mappings(self):
        disambiguator = EntityDisambiguator()

        assert "高血压" in disambiguator.icd10_mapping
        assert disambiguator.icd10_mapping["高血压"] == "I10"

    def test_normalize_name_with_synonyms(self):
        disambiguator = EntityDisambiguator()

        result = disambiguator.normalize_name("HTN", "Disease")

        assert result == "高血压"

    def test_normalize_name_with_abbreviation(self):
        disambiguator = EntityDisambiguator()

        result = disambiguator.normalize_name("DM", "Disease")

        assert result == "糖尿病"

    def test_disambiguate_entities(self):
        disambiguator = EntityDisambiguator()

        entities = [
            {"name": "HTN", "type": "Disease", "properties": {}},
            {"name": "高血压", "type": "Disease", "properties": {}},
            {"name": "高血压病", "type": "Disease", "properties": {}},
        ]

        result = disambiguator.disambiguate(entities)

        assert len(result) == 1
        assert result[0]["name"] == "高血压"
        assert "HTN" in result[0]["original_names"]
        assert "高血压" in result[0]["original_names"]


class TestTerminologyAPIIntegration:
    @pytest.fixture
    def client(self):
        from src.main import app
        return TestClient(app)

    def setup_method(self):
        TerminologyService.reset()

    def teardown_method(self):
        TerminologyService.reset()

    def test_terminology_stats_endpoint(self, client):
        response = client.get("/api/v1/terminology/stats")

        assert response.status_code == 200
        data = response.json()
        assert "icd10" in data
        assert "drugbank" in data

    def test_terminology_icd10_not_found(self, client):
        response = client.get("/api/v1/terminology/icd10/NONEXISTENT")

        assert response.status_code == 404

    def test_terminology_drug_not_found(self, client):
        response = client.get("/api/v1/terminology/drug/NONEXISTENT")

        assert response.status_code == 404

    def test_terminology_search_endpoint(self, client):
        response = client.get("/api/v1/terminology/search?query=test")

        assert response.status_code == 200
        data = response.json()
        assert "icd10" in data
        assert "drugs" in data

    def test_terminology_search_with_filter(self, client):
        response = client.get("/api/v1/terminology/search?query=test&terminology=icd10")

        assert response.status_code == 200
        data = response.json()
        assert "icd10" in data
        assert "drugs" in data
        assert data["drugs"] == []

    def test_terminology_search_with_limit(self, client):
        response = client.get("/api/v1/terminology/search?query=test&limit=5")

        assert response.status_code == 200

    def test_terminology_icd10_with_loaded_data(self, client, tmp_path):
        json_file = tmp_path / "icd10.json"
        json_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压", "synonyms": ["原发性高血压"]}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_icd10(str(json_file))

        response = client.get("/api/v1/terminology/icd10/I10")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "I10"
        assert data["name"] == "高血压"

    def test_terminology_drug_with_loaded_data(self, client, tmp_path):
        json_file = tmp_path / "drugs.json"
        json_file.write_text(
            '{"drugs": [{"drugbank_id": "DB00945", "name": "Aspirin", "name_cn": "阿司匹林", "synonyms": ["乙酰水杨酸"]}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_drugbank(str(json_file))

        response = client.get("/api/v1/terminology/drug/Aspirin")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["standard_name"] == "Aspirin"


class TestKnowledgeFusionWithTerminology:
    def setup_method(self):
        TerminologyService.reset()

    def teardown_method(self):
        TerminologyService.reset()

    def test_entity_disambiguator_icd10_mapping(self):
        disambiguator = EntityDisambiguator()

        assert disambiguator.icd10_mapping.get("高血压") == "I10"
        assert disambiguator.icd10_mapping.get("糖尿病") == "E11"
        assert disambiguator.icd10_mapping.get("肺炎") == "J18"

    def test_entity_disambiguator_umls_mapping(self):
        disambiguator = EntityDisambiguator()

        assert disambiguator.umls_mapping.get("高血压") == "C0020481"
        assert disambiguator.umls_mapping.get("糖尿病") == "C0011847"

    def test_link_to_standard_ontology(self):
        from src.ingestion.knowledge_fusion import KnowledgeFusionEngine

        engine = KnowledgeFusionEngine()

        entities = [
            {"name": "高血压", "types": ["Disease"], "properties": {}},
            {"name": "糖尿病", "types": ["Disease"], "properties": {}},
        ]

        result = engine.link_to_standard_ontology(entities)

        assert len(result) == 2
        assert result[0].get("icd10_code") == "I10"
        assert result[0].get("umls_code") == "C0020481"
        assert result[1].get("icd10_code") == "E11"

    def test_fuse_entities_and_relations(self):
        from src.ingestion.knowledge_fusion import KnowledgeFusionEngine

        engine = KnowledgeFusionEngine()

        entities = [
            {"name": "高血压", "type": "Disease", "properties": {}},
            {"name": "头痛", "type": "Symptom", "properties": {}},
        ]

        relationships = [
            {"source": "高血压", "target": "头痛", "type": "HAS_SYMPTOM", "properties": {}},
        ]

        fused_entities, fused_relations = engine.fuse(entities, relationships)

        assert len(fused_entities) == 2
        assert len(fused_relations) == 1
        assert fused_relations[0]["type"] == "HAS_SYMPTOM"
