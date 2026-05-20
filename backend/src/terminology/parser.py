import csv
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .models import Drug, DrugInteraction, ICD10Code, ICD10Category


@dataclass
class ParseStats:
    total_records: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "successful": self.successful,
            "failed": self.failed,
            "errors": self.errors,
        }


class ParseError(Exception):
    pass


class UnsupportedFormatError(ParseError):
    pass


class ParserFileNotFoundError(ParseError):
    pass


class BaseParser(ABC):
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._stats = ParseStats()

    @abstractmethod
    def parse(self) -> List[Any]:
        pass

    def get_stats(self) -> ParseStats:
        return self._stats

    def _validate_file_exists(self) -> None:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise ParserFileNotFoundError(f"File not found: {self.file_path}")

    def _get_format(self) -> str:
        return self.file_path.suffix.lower().lstrip(".")


class ICD10Parser(BaseParser):
    SUPPORTED_FORMATS = {"json", "xml", "csv"}

    def parse(self) -> List[ICD10Code]:
        self._validate_file_exists()
        file_format = self._get_format()

        if file_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {file_format}")
            raise UnsupportedFormatError(
                f"Unsupported format: {file_format}. Supported formats: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Parsing ICD-10 file: {self.file_path}")
        
        if file_format == "json":
            result = self._parse_json()
        elif file_format == "xml":
            result = self._parse_xml()
        else:
            result = self._parse_csv()

        logger.info(
            f"Parsed {self._stats.successful} ICD-10 codes, "
            f"failed: {self._stats.failed}"
        )
        return result

    def _parse_json(self) -> List[ICD10Code]:
        codes = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_list = data.get("codes", [])
            self._stats.total_records = len(code_list)

            for item in code_list:
                try:
                    code = self._create_icd10_code(item)
                    codes.append(code)
                    self._stats.successful += 1
                except Exception as e:
                    self._stats.failed += 1
                    error_msg = f"Failed to parse code {item.get('code', 'unknown')}: {str(e)}"
                    self._stats.errors.append(error_msg)
                    logger.warning(error_msg)

        except json.JSONDecodeError as e:
            self._stats.errors.append(f"JSON decode error: {str(e)}")
            logger.error(f"JSON decode error: {str(e)}")
            raise ParseError(f"Invalid JSON format: {str(e)}")

        return codes

    def _parse_xml(self) -> List[ICD10Code]:
        codes = []
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            code_elements = root.findall("code")
            self._stats.total_records = len(code_elements)

            for elem in code_elements:
                try:
                    code_value = elem.findtext("code") or elem.get("code", "")
                    if not code_value:
                        continue
                    item = {
                        "code": code_value,
                        "name": elem.findtext("name") or "",
                        "name_en": elem.findtext("name_en"),
                        "category": elem.findtext("category"),
                        "chapter": elem.findtext("chapter"),
                        "block": elem.findtext("block"),
                        "synonyms": self._parse_synonyms(elem.find("synonyms")),
                        "description": elem.findtext("description"),
                        "related_codes": self._parse_related_codes(elem.find("related_codes")),
                    }
                    code = self._create_icd10_code(item)
                    codes.append(code)
                    self._stats.successful += 1
                except Exception as e:
                    self._stats.failed += 1
                    error_msg = f"Failed to parse XML element: {str(e)}"
                    self._stats.errors.append(error_msg)
                    logger.warning(error_msg)

        except ET.ParseError as e:
            self._stats.errors.append(f"XML parse error: {str(e)}")
            logger.error(f"XML parse error: {str(e)}")
            raise ParseError(f"Invalid XML format: {str(e)}")

        return codes

    def _parse_csv(self) -> List[ICD10Code]:
        codes = []
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self._stats.total_records = len(rows)

                for row in rows:
                    try:
                        item = {
                            "code": row.get("code", ""),
                            "name": row.get("name", ""),
                            "name_en": row.get("name_en") or None,
                            "category": row.get("category") or None,
                            "chapter": row.get("chapter") or None,
                            "block": row.get("block") or None,
                            "synonyms": self._parse_list_field(row.get("synonyms", "")),
                            "description": row.get("description") or None,
                            "related_codes": self._parse_list_field(row.get("related_codes", "")),
                        }
                        code = self._create_icd10_code(item)
                        codes.append(code)
                        self._stats.successful += 1
                    except Exception as e:
                        self._stats.failed += 1
                        error_msg = f"Failed to parse CSV row: {str(e)}"
                        self._stats.errors.append(error_msg)
                        logger.warning(error_msg)

        except Exception as e:
            self._stats.errors.append(f"CSV read error: {str(e)}")
            logger.error(f"CSV read error: {str(e)}")
            raise ParseError(f"Failed to read CSV: {str(e)}")

        return codes

    def _create_icd10_code(self, item: Dict[str, Any]) -> ICD10Code:
        category = None
        if item.get("category"):
            category = self._get_category_enum(item["category"])

        return ICD10Code(
            code=item.get("code", ""),
            name=item.get("name", ""),
            name_en=item.get("name_en"),
            category=category,
            chapter=item.get("chapter"),
            block=item.get("block"),
            synonyms=item.get("synonyms", []),
            description=item.get("description"),
            related_codes=item.get("related_codes", []),
            parent_code=item.get("parent_code"),
        )

    def _get_category_enum(self, category_value: str) -> Optional[ICD10Category]:
        for cat in ICD10Category:
            if cat.value == category_value:
                return cat
        return None

    def _parse_synonyms(self, elem) -> List[str]:
        if elem is None:
            return []
        return [s.text for s in elem.findall("synonym") if s.text]

    def _parse_related_codes(self, elem) -> List[str]:
        if elem is None:
            return []
        return [c.text for c in elem.findall("code") if c.text]

    def _parse_list_field(self, value: str) -> List[str]:
        if not value:
            return []
        return [v.strip() for v in value.split("|") if v.strip()]


class DrugBankParser(BaseParser):
    SUPPORTED_FORMATS = {"json", "xml"}

    def parse(self) -> List[Drug]:
        self._validate_file_exists()
        file_format = self._get_format()

        if file_format not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {file_format}")
            raise UnsupportedFormatError(
                f"Unsupported format: {file_format}. Supported formats: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Parsing DrugBank file: {self.file_path}")

        if file_format == "json":
            result = self._parse_json()
        else:
            result = self._parse_xml()

        logger.info(
            f"Parsed {self._stats.successful} drugs, "
            f"failed: {self._stats.failed}"
        )
        return result

    def _parse_json(self) -> List[Drug]:
        drugs = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            drug_list = data.get("drugs", [])
            self._stats.total_records = len(drug_list)

            for item in drug_list:
                try:
                    drug = self._create_drug(item)
                    drugs.append(drug)
                    self._stats.successful += 1
                except Exception as e:
                    self._stats.failed += 1
                    error_msg = f"Failed to parse drug {item.get('drugbank_id', 'unknown')}: {str(e)}"
                    self._stats.errors.append(error_msg)
                    logger.warning(error_msg)

        except json.JSONDecodeError as e:
            self._stats.errors.append(f"JSON decode error: {str(e)}")
            logger.error(f"JSON decode error: {str(e)}")
            raise ParseError(f"Invalid JSON format: {str(e)}")

        return drugs

    def _parse_xml(self) -> List[Drug]:
        drugs = []
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            drug_elements = root.findall(".//drug") or root.findall("drug")
            self._stats.total_records = len(drug_elements)

            for elem in drug_elements:
                try:
                    item = {
                        "drugbank_id": elem.findtext("drugbank-id") or elem.findtext("drugbank_id") or "",
                        "name": elem.findtext("name") or "",
                        "name_cn": elem.findtext("name_cn"),
                        "cas_number": elem.findtext("cas-number") or elem.findtext("cas_number"),
                        "atc_code": elem.findtext("atc-code") or elem.findtext("atc_code"),
                        "formula": elem.findtext("formula"),
                        "weight": self._parse_float(elem.findtext("weight")),
                        "indications": self._parse_list_element(elem.find("indications")),
                        "contraindications": self._parse_list_element(elem.find("contraindications")),
                        "side_effects": self._parse_list_element(elem.find("side-effects") or elem.find("side_effects")),
                        "interactions": self._parse_interactions(elem.find("interactions")),
                        "synonyms": self._parse_list_element(elem.find("synonyms")),
                        "description": elem.findtext("description"),
                    }
                    drug = self._create_drug(item)
                    drugs.append(drug)
                    self._stats.successful += 1
                except Exception as e:
                    self._stats.failed += 1
                    error_msg = f"Failed to parse XML element: {str(e)}"
                    self._stats.errors.append(error_msg)
                    logger.warning(error_msg)

        except ET.ParseError as e:
            self._stats.errors.append(f"XML parse error: {str(e)}")
            logger.error(f"XML parse error: {str(e)}")
            raise ParseError(f"Invalid XML format: {str(e)}")

        return drugs

    def _create_drug(self, item: Dict[str, Any]) -> Drug:
        interactions = []
        for inter in item.get("interactions", []):
            interactions.append(
                DrugInteraction(
                    drug_name=inter.get("drug_name", ""),
                    drugbank_id=inter.get("drugbank_id"),
                    description=inter.get("description"),
                    severity=inter.get("severity"),
                )
            )

        return Drug(
            drugbank_id=item.get("drugbank_id", ""),
            name=item.get("name", ""),
            name_cn=item.get("name_cn"),
            cas_number=item.get("cas_number"),
            atc_code=item.get("atc_code"),
            formula=item.get("formula"),
            weight=item.get("weight"),
            indications=item.get("indications", []),
            contraindications=item.get("contraindications", []),
            side_effects=item.get("side_effects", []),
            interactions=interactions,
            synonyms=item.get("synonyms", []),
            description=item.get("description"),
        )

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_list_element(self, elem) -> List[str]:
        if elem is None:
            return []
        return [e.text for e in elem.findall("item") if e.text]

    def _parse_interactions(self, elem) -> List[Dict[str, Any]]:
        if elem is None:
            return []
        interactions = []
        for inter in elem.findall("interaction"):
            interactions.append({
                "drug_name": inter.findtext("drug-name") or inter.findtext("drug_name") or "",
                "drugbank_id": inter.findtext("drugbank-id") or inter.findtext("drugbank_id"),
                "description": inter.findtext("description"),
                "severity": inter.findtext("severity"),
            })
        return interactions
