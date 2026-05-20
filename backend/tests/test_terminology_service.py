import pytest

from src.terminology.service import TerminologyService
from src.terminology.models import ICD10Code, Drug


class TestTerminologyService:
    def setup_method(self):
        TerminologyService.reset()

    def teardown_method(self):
        TerminologyService.reset()

    def test_singleton(self):
        service1 = TerminologyService()
        service2 = TerminologyService()

        assert service1 is service2

    def test_singleton_with_different_params(self):
        service1 = TerminologyService()
        service2 = TerminologyService(icd10_data_path="nonexistent.json")

        assert service1 is service2

    def test_reset(self):
        service1 = TerminologyService()
        TerminologyService.reset()
        service2 = TerminologyService()

        assert service1 is not service2

    def test_lookup_icd10_not_loaded(self):
        service = TerminologyService()

        result = service.lookup_icd10("I10")

        assert result is None

    def test_lookup_icd10_empty_code(self):
        service = TerminologyService()

        result = service.lookup_icd10("")

        assert result is None

        result = service.lookup_icd10(None)

        assert result is None

    def test_lookup_drug_not_loaded(self):
        service = TerminologyService()

        result = service.lookup_drug("Aspirin")

        assert result == []

    def test_lookup_drug_empty_name(self):
        service = TerminologyService()

        result = service.lookup_drug("")

        assert result == []

        result = service.lookup_drug(None)

        assert result == []

    def test_search_empty(self):
        service = TerminologyService()

        result = service.search("")

        assert result == {"icd10": [], "drugs": []}

        result = service.search(None)

        assert result == {"icd10": [], "drugs": []}

    def test_search_not_loaded(self):
        service = TerminologyService()

        result = service.search("高血压")

        assert result == {"icd10": [], "drugs": []}

    def test_search_with_terminology_filter(self):
        service = TerminologyService()

        result = service.search("test", terminology="icd10")

        assert "icd10" in result
        assert "drugs" in result
        assert result["drugs"] == []

        result = service.search("test", terminology="drugbank")

        assert "icd10" in result
        assert "drugs" in result
        assert result["icd10"] == []

    def test_get_stats(self):
        service = TerminologyService()

        stats = service.get_stats()

        assert "icd10" in stats
        assert "drugbank" in stats
        assert stats["icd10"]["loaded"] is False
        assert stats["drugbank"]["loaded"] is False
        assert stats["icd10"]["codes_count"] == 0
        assert stats["drugbank"]["drugs_count"] == 0

    def test_load_icd10_invalid_path(self):
        service = TerminologyService()

        result = service.load_icd10("nonexistent_file.json")

        assert result is False
        assert service._icd10_loaded is False

    def test_load_drugbank_invalid_path(self):
        service = TerminologyService()

        result = service.load_drugbank("nonexistent_file.xml")

        assert result is False
        assert service._drugbank_loaded is False

    def test_lookup_icd10_with_loaded_data(self, tmp_path):
        json_file = tmp_path / "icd10.json"
        json_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压", "synonyms": ["原发性高血压"]}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_icd10(str(json_file))

        result = service.lookup_icd10("I10")

        assert result is not None
        assert result["code"] == "I10"
        assert result["name"] == "高血压"

    def test_lookup_drug_with_loaded_data(self, tmp_path):
        json_file = tmp_path / "drugs.json"
        json_file.write_text(
            '{"drugs": [{"drugbank_id": "DB00945", "name": "Aspirin", "name_cn": "阿司匹林", "synonyms": ["乙酰水杨酸"]}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_drugbank(str(json_file))

        result = service.lookup_drug("Aspirin")

        assert len(result) == 1
        assert result[0]["standard_name"] == "Aspirin"
        assert result[0]["code"] == "DB00945"

    def test_search_with_loaded_data(self, tmp_path):
        json_file = tmp_path / "icd10.json"
        json_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压"}, {"code": "I11", "name": "高血压性心脏病"}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_icd10(str(json_file))

        result = service.search("I1")

        assert len(result["icd10"]) == 2

    def test_search_limit(self, tmp_path):
        json_file = tmp_path / "icd10.json"
        json_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压"}, {"code": "I11", "name": "高血压性心脏病"}, {"code": "I12", "name": "高血压性肾病"}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_icd10(str(json_file))

        result = service.search("I", limit=2)

        assert len(result["icd10"]) == 2

    def test_stats_after_loading(self, tmp_path):
        json_file = tmp_path / "icd10.json"
        json_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压"}]}',
            encoding="utf-8",
        )

        service = TerminologyService()
        service.load_icd10(str(json_file))

        stats = service.get_stats()

        assert stats["icd10"]["loaded"] is True
        assert stats["icd10"]["codes_count"] == 1

    def test_init_with_data_paths(self, tmp_path):
        icd10_file = tmp_path / "icd10.json"
        icd10_file.write_text(
            '{"codes": [{"code": "I10", "name": "高血压"}]}',
            encoding="utf-8",
        )

        TerminologyService.reset()
        service = TerminologyService(icd10_data_path=str(icd10_file))

        stats = service.get_stats()

        assert stats["icd10"]["loaded"] is True
        assert stats["icd10"]["codes_count"] == 1
