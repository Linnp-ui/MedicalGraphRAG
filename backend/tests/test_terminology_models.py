import pytest
from src.terminology.models import ICD10Code, Drug, DrugInteraction, ICD10Category


class TestICD10Code:
    def test_create_icd10_code(self):
        code = ICD10Code(
            code="I10",
            name="高血压",
            name_en="Essential (primary) hypertension",
            category=ICD10Category.CHAPTER_09,
            chapter="循环系统疾病",
            block="高血压",
            synonyms=["原发性高血压", "高血压病"],
            description="无明确原因的高血压",
            related_codes=["I11", "I12"],
            parent_code="I00-I99",
        )

        assert code.code == "I10"
        assert code.name == "高血压"
        assert code.name_en == "Essential (primary) hypertension"
        assert code.category == ICD10Category.CHAPTER_09
        assert code.chapter == "循环系统疾病"
        assert code.block == "高血压"
        assert len(code.synonyms) == 2
        assert "原发性高血压" in code.synonyms
        assert code.description == "无明确原因的高血压"
        assert len(code.related_codes) == 2
        assert code.parent_code == "I00-I99"

    def test_icd10_code_to_dict(self):
        code = ICD10Code(
            code="E11",
            name="2型糖尿病",
            name_en="Type 2 diabetes mellitus",
            category=ICD10Category.CHAPTER_04,
            chapter="内分泌、营养和代谢疾病",
            synonyms=["非胰岛素依赖型糖尿病"],
        )

        result = code.to_dict()

        assert isinstance(result, dict)
        assert result["code"] == "E11"
        assert result["name"] == "2型糖尿病"
        assert result["name_en"] == "Type 2 diabetes mellitus"
        assert result["category"] == "E00-E90"
        assert result["chapter"] == "内分泌、营养和代谢疾病"
        assert result["synonyms"] == ["非胰岛素依赖型糖尿病"]


class TestDrug:
    def test_create_drug(self):
        drug = Drug(
            drugbank_id="DB00945",
            name="Aspirin",
            name_cn="阿司匹林",
            cas_number="50-78-2",
            atc_code="B01AC06",
            formula="C9H8O4",
            weight=180.16,
            indications=["疼痛", "发热", "心血管疾病预防"],
            contraindications=["胃溃疡", "出血性疾病"],
            side_effects=["胃肠道不适", "出血"],
            synonyms=["乙酰水杨酸", "ASA"],
            description="非甾体抗炎药",
        )

        assert drug.drugbank_id == "DB00945"
        assert drug.name == "Aspirin"
        assert drug.name_cn == "阿司匹林"
        assert drug.cas_number == "50-78-2"
        assert drug.atc_code == "B01AC06"
        assert drug.formula == "C9H8O4"
        assert drug.weight == 180.16
        assert len(drug.indications) == 3
        assert "疼痛" in drug.indications
        assert len(drug.contraindications) == 2
        assert len(drug.side_effects) == 2
        assert len(drug.synonyms) == 2
        assert drug.description == "非甾体抗炎药"

    def test_drug_with_interaction(self):
        interaction1 = DrugInteraction(
            drug_name="Warfarin",
            drugbank_id="DB00682",
            description="增加出血风险",
            severity="major",
        )
        interaction2 = DrugInteraction(
            drug_name="Ibuprofen",
            drugbank_id="DB01050",
            description="增加胃肠道不良反应",
            severity="moderate",
        )

        drug = Drug(
            drugbank_id="DB00945",
            name="Aspirin",
            name_cn="阿司匹林",
            interactions=[interaction1, interaction2],
        )

        assert len(drug.interactions) == 2
        assert drug.interactions[0].drug_name == "Warfarin"
        assert drug.interactions[0].severity == "major"
        assert drug.interactions[1].drug_name == "Ibuprofen"
        assert drug.interactions[1].severity == "moderate"

    def test_drug_to_dict(self):
        interaction = DrugInteraction(
            drug_name="Warfarin",
            drugbank_id="DB00682",
            description="增加出血风险",
            severity="major",
        )

        drug = Drug(
            drugbank_id="DB00945",
            name="Aspirin",
            name_cn="阿司匹林",
            cas_number="50-78-2",
            atc_code="B01AC06",
            formula="C9H8O4",
            weight=180.16,
            indications=["疼痛", "发热"],
            contraindications=["胃溃疡"],
            side_effects=["胃肠道不适"],
            interactions=[interaction],
            synonyms=["乙酰水杨酸"],
            description="非甾体抗炎药",
        )

        result = drug.to_dict()

        assert isinstance(result, dict)
        assert result["drugbank_id"] == "DB00945"
        assert result["name"] == "Aspirin"
        assert result["name_cn"] == "阿司匹林"
        assert result["cas_number"] == "50-78-2"
        assert result["atc_code"] == "B01AC06"
        assert result["formula"] == "C9H8O4"
        assert result["weight"] == 180.16
        assert result["indications"] == ["疼痛", "发热"]
        assert result["contraindications"] == ["胃溃疡"]
        assert result["side_effects"] == ["胃肠道不适"]
        assert len(result["interactions"]) == 1
        assert result["interactions"][0]["drug_name"] == "Warfarin"
        assert result["interactions"][0]["severity"] == "major"
        assert result["synonyms"] == ["乙酰水杨酸"]
        assert result["description"] == "非甾体抗炎药"


class TestDrugInteraction:
    def test_create_drug_interaction(self):
        interaction = DrugInteraction(
            drug_name="Warfarin",
            drugbank_id="DB00682",
            description="增加出血风险",
            severity="major",
        )

        assert interaction.drug_name == "Warfarin"
        assert interaction.drugbank_id == "DB00682"
        assert interaction.description == "增加出血风险"
        assert interaction.severity == "major"

    def test_drug_interaction_to_dict(self):
        interaction = DrugInteraction(
            drug_name="Ibuprofen",
            drugbank_id="DB01050",
            description="增加胃肠道不良反应",
            severity="moderate",
        )

        result = interaction.to_dict()

        assert isinstance(result, dict)
        assert result["drug_name"] == "Ibuprofen"
        assert result["drugbank_id"] == "DB01050"
        assert result["description"] == "增加胃肠道不良反应"
        assert result["severity"] == "moderate"
