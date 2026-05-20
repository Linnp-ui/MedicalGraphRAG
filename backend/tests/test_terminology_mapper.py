import pytest

from src.terminology.mapper import TerminologyMapper, MappingResult
from src.terminology.models import ICD10Code, Drug, ICD10Category


class TestMappingResult:
    def test_to_dict(self):
        result = MappingResult(
            original_name="高血压",
            standard_name="原发性高血压",
            terminology="ICD-10",
            code="I10",
            confidence=1.0,
            synonyms=["高血压病"],
            properties={"chapter": "IX"},
        )
        d = result.to_dict()

        assert d["original_name"] == "高血压"
        assert d["standard_name"] == "原发性高血压"
        assert d["terminology"] == "ICD-10"
        assert d["code"] == "I10"
        assert d["confidence"] == 1.0
        assert d["synonyms"] == ["高血压病"]
        assert d["properties"]["chapter"] == "IX"


class TestTerminologyMapper:
    def test_load_icd10_codes(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压", "高血压病"]),
            ICD10Code(code="E11", name="2型糖尿病", synonyms=["非胰岛素依赖型糖尿病"]),
        ]
        count = mapper.load_icd10_codes(codes)

        assert count == 2
        stats = mapper.get_stats()
        assert stats["icd10_codes"] == 2

    def test_map_to_icd10(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(
                code="I10",
                name="高血压",
                name_en="Essential hypertension",
                category=ICD10Category.CHAPTER_09,
                chapter="IX",
                block="I10-I15",
                synonyms=["原发性高血压", "高血压病"],
                description="特发性高血压",
            ),
        ]
        mapper.load_icd10_codes(codes)

        result = mapper.map_to_icd10("高血压")
        assert result is not None
        assert result.original_name == "高血压"
        assert result.standard_name == "高血压"
        assert result.terminology == "ICD-10"
        assert result.code == "I10"
        assert len(result.synonyms) == 2

    def test_map_to_icd10_by_synonym(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压", "高血压病"]),
        ]
        mapper.load_icd10_codes(codes)

        result = mapper.map_to_icd10("原发性高血压")
        assert result is not None
        assert result.standard_name == "高血压"
        assert result.code == "I10"

    def test_map_to_icd10_not_found(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压"),
        ]
        mapper.load_icd10_codes(codes)

        result = mapper.map_to_icd10("不存在的疾病")
        assert result is None

    def test_map_to_icd10_empty_name(self):
        mapper = TerminologyMapper()
        result = mapper.map_to_icd10("")
        assert result is None

        result = mapper.map_to_icd10(None)
        assert result is None

    def test_load_drugs(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin", name_cn="阿司匹林", synonyms=["乙酰水杨酸"]),
            Drug(drugbank_id="DB00682", name="Warfarin", name_cn="华法林"),
        ]
        count = mapper.load_drugs(drugs)

        assert count == 2
        stats = mapper.get_stats()
        assert stats["drugs"] == 2

    def test_map_to_drug(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(
                drugbank_id="DB00945",
                name="Aspirin",
                name_cn="阿司匹林",
                cas_number="50-78-2",
                atc_code="B01AC06",
                formula="C9H8O4",
                weight=180.16,
                indications=["疼痛", "发热"],
                synonyms=["乙酰水杨酸"],
            ),
        ]
        mapper.load_drugs(drugs)

        result = mapper.map_to_drug("Aspirin")
        assert result is not None
        assert result.original_name == "Aspirin"
        assert result.standard_name == "Aspirin"
        assert result.terminology == "DrugBank"
        assert result.code == "DB00945"
        assert result.properties["cas_number"] == "50-78-2"
        assert result.properties["atc_code"] == "B01AC06"

    def test_map_to_drug_by_chinese_name(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin", name_cn="阿司匹林"),
        ]
        mapper.load_drugs(drugs)

        result = mapper.map_to_drug("阿司匹林")
        assert result is not None
        assert result.standard_name == "Aspirin"
        assert result.code == "DB00945"

    def test_map_to_drug_by_synonym(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin", synonyms=["乙酰水杨酸"]),
        ]
        mapper.load_drugs(drugs)

        result = mapper.map_to_drug("乙酰水杨酸")
        assert result is not None
        assert result.standard_name == "Aspirin"

    def test_map_to_drug_not_found(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin"),
        ]
        mapper.load_drugs(drugs)

        result = mapper.map_to_drug("不存在的药物")
        assert result is None

    def test_map_to_drug_empty_name(self):
        mapper = TerminologyMapper()
        result = mapper.map_to_drug("")
        assert result is None

        result = mapper.map_to_drug(None)
        assert result is None

    def test_expand_synonyms_icd10(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压", "高血压病"]),
        ]
        mapper.load_icd10_codes(codes)

        synonyms = mapper.expand_synonyms("高血压", "icd10")
        assert len(synonyms) == 3
        assert "高血压" in synonyms
        assert "原发性高血压" in synonyms
        assert "高血压病" in synonyms

    def test_expand_synonyms_drug(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin", synonyms=["乙酰水杨酸", "ASA"]),
        ]
        mapper.load_drugs(drugs)

        synonyms = mapper.expand_synonyms("Aspirin", "drug")
        assert len(synonyms) == 3
        assert "Aspirin" in synonyms
        assert "乙酰水杨酸" in synonyms
        assert "ASA" in synonyms

    def test_expand_synonyms_not_found(self):
        mapper = TerminologyMapper()

        synonyms = mapper.expand_synonyms("不存在的名称", "icd10")
        assert synonyms == ["不存在的名称"]

    def test_expand_synonyms_empty_name(self):
        mapper = TerminologyMapper()

        synonyms = mapper.expand_synonyms("", "icd10")
        assert synonyms == []

        synonyms = mapper.expand_synonyms(None, "drug")
        assert synonyms == []

    def test_get_all_icd10_codes(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压"),
            ICD10Code(code="E11", name="2型糖尿病"),
        ]
        mapper.load_icd10_codes(codes)

        all_codes = mapper.get_all_icd10_codes()
        assert len(all_codes) == 2
        codes_str = [c.code for c in all_codes]
        assert "I10" in codes_str
        assert "E11" in codes_str

    def test_get_all_drugs(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin"),
            Drug(drugbank_id="DB00682", name="Warfarin"),
        ]
        mapper.load_drugs(drugs)

        all_drugs = mapper.get_all_drugs()
        assert len(all_drugs) == 2
        drug_ids = [d.drugbank_id for d in all_drugs]
        assert "DB00945" in drug_ids
        assert "DB00682" in drug_ids

    def test_get_stats(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压"]),
        ]
        drugs = [
            Drug(drugbank_id="DB00945", name="Aspirin", name_cn="阿司匹林", synonyms=["乙酰水杨酸"]),
        ]
        mapper.load_icd10_codes(codes)
        mapper.load_drugs(drugs)

        stats = mapper.get_stats()
        assert stats["icd10_codes"] == 1
        assert stats["icd10_name_mappings"] == 2
        assert stats["drugs"] == 1
        assert stats["drug_name_mappings"] == 3

    def test_case_insensitive_mapping(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压"),
        ]
        mapper.load_icd10_codes(codes)

        result = mapper.map_to_icd10("高血压")
        assert result is not None

        result = mapper.map_to_icd10("高血 压 ")
        assert result is not None
        assert result.standard_name == "高血压"

    def test_load_empty_codes(self):
        mapper = TerminologyMapper()
        count = mapper.load_icd10_codes([])
        assert count == 0

    def test_load_empty_drugs(self):
        mapper = TerminologyMapper()
        count = mapper.load_drugs([])
        assert count == 0

    def test_load_codes_without_id(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="", name="无编码"),
        ]
        count = mapper.load_icd10_codes(codes)
        assert count == 0

    def test_load_drugs_without_id(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="", name="无ID药物"),
        ]
        count = mapper.load_drugs(drugs)
        assert count == 0
