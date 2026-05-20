import json
import pytest
import tempfile
from pathlib import Path

from src.terminology.parser import (
    ICD10Parser,
    DrugBankParser,
    ParseError,
    UnsupportedFormatError,
)
from src.terminology.models import ICD10Code, Drug


class TestICD10Parser:
    def test_parse_json(self, tmp_path):
        json_file = tmp_path / "test_icd10.json"
        json_data = {
            "codes": [
                {
                    "code": "I10",
                    "name": "高血压",
                    "name_en": "Essential (primary) hypertension",
                    "category": "循环系统疾病",
                    "chapter": "IX",
                    "block": "I10-I15",
                    "synonyms": ["原发性高血压", "高血压病"],
                    "description": "特发性高血压",
                    "related_codes": ["I11", "I12"],
                },
                {
                    "code": "E11",
                    "name": "2型糖尿病",
                    "name_en": "Type 2 diabetes mellitus",
                    "category": "内分泌、营养和代谢疾病",
                    "chapter": "IV",
                    "block": "E10-E14",
                    "synonyms": ["非胰岛素依赖型糖尿病"],
                    "description": "非胰岛素依赖型糖尿病",
                },
            ]
        }
        json_file.write_text(json.dumps(json_data, ensure_ascii=False), encoding="utf-8")

        parser = ICD10Parser(json_file)
        codes = parser.parse()

        assert len(codes) == 2
        assert codes[0].code == "I10"
        assert codes[0].name == "高血压"
        assert codes[0].name_en == "Essential (primary) hypertension"
        assert codes[0].chapter == "IX"
        assert codes[0].block == "I10-I15"
        assert len(codes[0].synonyms) == 2
        assert "原发性高血压" in codes[0].synonyms
        assert codes[0].description == "特发性高血压"
        assert len(codes[0].related_codes) == 2

        assert codes[1].code == "E11"
        assert codes[1].name == "2型糖尿病"

    def test_parse_csv(self, tmp_path):
        csv_file = tmp_path / "test_icd10.csv"
        csv_content = """code,name,name_en,category,chapter,block,synonyms,description,related_codes
I10,高血压,Essential (primary) hypertension,循环系统疾病,IX,I10-I15,原发性高血压|高血压病,特发性高血压,I11|I12
E11,2型糖尿病,Type 2 diabetes mellitus,内分泌、营养和代谢疾病,IV,E10-E14,非胰岛素依赖型糖尿病,非胰岛素依赖型糖尿病,
"""
        csv_file.write_text(csv_content, encoding="utf-8")

        parser = ICD10Parser(csv_file)
        codes = parser.parse()

        assert len(codes) == 2
        assert codes[0].code == "I10"
        assert codes[0].name == "高血压"
        assert len(codes[0].synonyms) == 2
        assert len(codes[0].related_codes) == 2

    def test_parse_xml(self, tmp_path):
        xml_file = tmp_path / "test_icd10.xml"
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<icd10>
    <code>
        <code>I10</code>
        <name>高血压</name>
        <name_en>Essential (primary) hypertension</name_en>
        <chapter>IX</chapter>
        <block>I10-I15</block>
        <synonyms>
            <synonym>原发性高血压</synonym>
            <synonym>高血压病</synonym>
        </synonyms>
        <description>特发性高血压</description>
        <related_codes>
            <code>I11</code>
            <code>I12</code>
        </related_codes>
    </code>
</icd10>
"""
        xml_file.write_text(xml_content, encoding="utf-8")

        parser = ICD10Parser(xml_file)
        codes = parser.parse()

        assert len(codes) == 1
        assert codes[0].code == "I10"
        assert codes[0].name == "高血压"
        assert len(codes[0].synonyms) == 2
        assert len(codes[0].related_codes) == 2

    def test_parse_nonexistent_file(self):
        parser = ICD10Parser("/nonexistent/path/file.json")
        with pytest.raises(ParseError):
            parser.parse()

    def test_parse_unsupported_format(self, tmp_path):
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("some content", encoding="utf-8")

        parser = ICD10Parser(unsupported_file)
        with pytest.raises(UnsupportedFormatError):
            parser.parse()

    def test_get_stats(self, tmp_path):
        json_file = tmp_path / "test_icd10.json"
        json_data = {
            "codes": [
                {"code": "I10", "name": "高血压"},
                {"code": "E11", "name": "2型糖尿病"},
            ]
        }
        json_file.write_text(json.dumps(json_data, ensure_ascii=False), encoding="utf-8")

        parser = ICD10Parser(json_file)
        parser.parse()
        stats = parser.get_stats()

        assert stats.total_records == 2
        assert stats.successful == 2
        assert stats.failed == 0

    def test_get_stats_with_errors(self, tmp_path):
        json_file = tmp_path / "test_icd10.json"
        json_data = {
            "codes": [
                {"code": "I10", "name": "高血压"},
                {"code": ""},
            ]
        }
        json_file.write_text(json.dumps(json_data, ensure_ascii=False), encoding="utf-8")

        parser = ICD10Parser(json_file)
        codes = parser.parse()
        stats = parser.get_stats()

        assert stats.total_records == 2
        assert stats.successful == 2
        assert stats.failed == 0


class TestDrugBankParser:
    def test_parse_json(self, tmp_path):
        json_file = tmp_path / "test_drugbank.json"
        json_data = {
            "drugs": [
                {
                    "drugbank_id": "DB00945",
                    "name": "Aspirin",
                    "name_cn": "阿司匹林",
                    "cas_number": "50-78-2",
                    "atc_code": "B01AC06",
                    "formula": "C9H8O4",
                    "weight": 180.16,
                    "indications": ["疼痛", "发热"],
                    "contraindications": ["胃溃疡"],
                    "side_effects": ["胃肠道不适"],
                    "interactions": [
                        {
                            "drug_name": "Warfarin",
                            "drugbank_id": "DB00682",
                            "description": "增加出血风险",
                            "severity": "major",
                        }
                    ],
                    "synonyms": ["乙酰水杨酸"],
                    "description": "非甾体抗炎药",
                }
            ]
        }
        json_file.write_text(json.dumps(json_data, ensure_ascii=False), encoding="utf-8")

        parser = DrugBankParser(json_file)
        drugs = parser.parse()

        assert len(drugs) == 1
        assert drugs[0].drugbank_id == "DB00945"
        assert drugs[0].name == "Aspirin"
        assert drugs[0].name_cn == "阿司匹林"
        assert drugs[0].cas_number == "50-78-2"
        assert drugs[0].atc_code == "B01AC06"
        assert drugs[0].formula == "C9H8O4"
        assert drugs[0].weight == 180.16
        assert len(drugs[0].indications) == 2
        assert len(drugs[0].interactions) == 1
        assert drugs[0].interactions[0].drug_name == "Warfarin"
        assert drugs[0].interactions[0].severity == "major"

    def test_parse_xml(self, tmp_path):
        xml_file = tmp_path / "test_drugbank.xml"
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank>
    <drug>
        <drugbank-id>DB00945</drugbank-id>
        <name>Aspirin</name>
        <name_cn>阿司匹林</name_cn>
        <cas-number>50-78-2</cas-number>
        <atc-code>B01AC06</atc-code>
        <formula>C9H8O4</formula>
        <weight>180.16</weight>
        <indications>
            <item>疼痛</item>
            <item>发热</item>
        </indications>
        <synonyms>
            <item>乙酰水杨酸</item>
        </synonyms>
        <description>非甾体抗炎药</description>
    </drug>
</drugbank>
"""
        xml_file.write_text(xml_content, encoding="utf-8")

        parser = DrugBankParser(xml_file)
        drugs = parser.parse()

        assert len(drugs) == 1
        assert drugs[0].drugbank_id == "DB00945"
        assert drugs[0].name == "Aspirin"
        assert len(drugs[0].indications) == 2

    def test_parse_nonexistent_file(self):
        parser = DrugBankParser("/nonexistent/path/file.json")
        with pytest.raises(ParseError):
            parser.parse()

    def test_parse_unsupported_format(self, tmp_path):
        unsupported_file = tmp_path / "test.csv"
        unsupported_file.write_text("some content", encoding="utf-8")

        parser = DrugBankParser(unsupported_file)
        with pytest.raises(UnsupportedFormatError):
            parser.parse()

    def test_get_stats(self, tmp_path):
        json_file = tmp_path / "test_drugbank.json"
        json_data = {
            "drugs": [
                {"drugbank_id": "DB00945", "name": "Aspirin"},
                {"drugbank_id": "DB00682", "name": "Warfarin"},
            ]
        }
        json_file.write_text(json.dumps(json_data, ensure_ascii=False), encoding="utf-8")

        parser = DrugBankParser(json_file)
        parser.parse()
        stats = parser.get_stats()

        assert stats.total_records == 2
        assert stats.successful == 2
        assert stats.failed == 0

    def test_parse_empty_file(self, tmp_path):
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}", encoding="utf-8")

        parser = DrugBankParser(json_file)
        drugs = parser.parse()

        assert len(drugs) == 0
        stats = parser.get_stats()
        assert stats.total_records == 0
        assert stats.successful == 0


class TestParseStats:
    def test_to_dict(self, tmp_path):
        json_file = tmp_path / "test.json"
        json_file.write_text('{"codes": [{"code": "I10", "name": "高血压"}]}', encoding="utf-8")

        parser = ICD10Parser(json_file)
        parser.parse()
        stats = parser.get_stats()
        result = stats.to_dict()

        assert "total_records" in result
        assert "successful" in result
        assert "failed" in result
        assert "errors" in result
