from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .mapper import TerminologyMapper
from .models import Drug, ICD10Code
from .parser import DrugBankParser, ICD10Parser


class TerminologyService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        icd10_data_path: Optional[str] = None,
        drugbank_data_path: Optional[str] = None,
    ):
        if self._initialized:
            return

        self._mapper = TerminologyMapper()
        self._icd10_data_path = icd10_data_path
        self._drugbank_data_path = drugbank_data_path
        self._icd10_loaded = False
        self._drugbank_loaded = False

        if icd10_data_path:
            self._load_icd10_data(icd10_data_path)
        if drugbank_data_path:
            self._load_drugbank_data(drugbank_data_path)

        self._initialized = True
        logger.info("TerminologyService initialized")

    def _load_icd10_data(self, path: str) -> None:
        try:
            parser = ICD10Parser(path)
            codes = parser.parse()
            count = self._mapper.load_icd10_codes(codes)
            self._icd10_loaded = True
            logger.info(f"Loaded {count} ICD-10 codes from {path}")
        except Exception as e:
            logger.error(f"Failed to load ICD-10 data: {e}")
            self._icd10_loaded = False

    def _load_drugbank_data(self, path: str) -> None:
        try:
            parser = DrugBankParser(path)
            drugs = parser.parse()
            count = self._mapper.load_drugs(drugs)
            self._drugbank_loaded = True
            logger.info(f"Loaded {count} drugs from {path}")
        except Exception as e:
            logger.error(f"Failed to load DrugBank data: {e}")
            self._drugbank_loaded = False

    def load_icd10(self, path: str) -> bool:
        self._icd10_data_path = path
        self._load_icd10_data(path)
        return self._icd10_loaded

    def load_drugbank(self, path: str) -> bool:
        self._drugbank_data_path = path
        self._load_drugbank_data(path)
        return self._drugbank_loaded

    def lookup_icd10(self, code: str) -> Optional[Dict[str, Any]]:
        if not code:
            return None

        mapping = self._mapper.map_to_icd10(code)
        if mapping:
            return mapping.to_dict()

        all_codes = self._mapper.get_all_icd10_codes()
        for icd10_code in all_codes:
            if icd10_code.code == code:
                return icd10_code.to_dict()

        return None

    def lookup_drug(self, name: str) -> List[Dict[str, Any]]:
        if not name:
            return []

        results = []

        mapping = self._mapper.map_to_drug(name)
        if mapping:
            results.append(mapping.to_dict())

        all_drugs = self._mapper.get_all_drugs()
        name_lower = name.lower()
        for drug in all_drugs:
            if drug.drugbank_id and drug.drugbank_id.lower() == name_lower:
                if not any(r.get("code") == drug.drugbank_id for r in results):
                    results.append(
                        {
                            "original_name": name,
                            "standard_name": drug.name,
                            "terminology": "DrugBank",
                            "code": drug.drugbank_id,
                            "confidence": 1.0,
                            "synonyms": drug.synonyms,
                            "properties": {
                                "name_cn": drug.name_cn,
                                "cas_number": drug.cas_number,
                                "atc_code": drug.atc_code,
                                "formula": drug.formula,
                                "weight": drug.weight,
                                "indications": drug.indications,
                                "contraindications": drug.contraindications,
                                "side_effects": drug.side_effects,
                            },
                        }
                    )

        return results

    def search(
        self, query: str, terminology: str = "all", limit: int = 10
    ) -> Dict[str, List[Dict]]:
        if not query:
            return {"icd10": [], "drugs": []}

        result = {"icd10": [], "drugs": []}
        query_lower = query.lower()

        if terminology in ("all", "icd10"):
            icd10_results = self._search_icd10(query_lower, limit)
            result["icd10"] = icd10_results

        if terminology in ("all", "drugbank", "drugs"):
            drug_results = self._search_drugs(query_lower, limit)
            result["drugs"] = drug_results

        return result

    def _search_icd10(self, query: str, limit: int) -> List[Dict[str, Any]]:
        results = []
        all_codes = self._mapper.get_all_icd10_codes()

        for code in all_codes:
            score = self._calculate_match_score_icd10(code, query)
            if score > 0:
                results.append(
                    {
                        "code": code.code,
                        "name": code.name,
                        "name_en": code.name_en,
                        "category": code.category.value if code.category else None,
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _calculate_match_score_icd10(self, code: ICD10Code, query: str) -> float:
        if code.code.lower() == query:
            return 1.0
        if code.code.lower().startswith(query):
            return 0.9
        if code.name and query in code.name.lower():
            return 0.8
        if code.name_en and query in code.name_en.lower():
            return 0.7
        for synonym in code.synonyms:
            if query in synonym.lower():
                return 0.6
        if code.description and query in code.description.lower():
            return 0.5
        return 0.0

    def _search_drugs(self, query: str, limit: int) -> List[Dict[str, Any]]:
        results = []
        all_drugs = self._mapper.get_all_drugs()

        for drug in all_drugs:
            score = self._calculate_match_score_drug(drug, query)
            if score > 0:
                results.append(
                    {
                        "drugbank_id": drug.drugbank_id,
                        "name": drug.name,
                        "name_cn": drug.name_cn,
                        "atc_code": drug.atc_code,
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _calculate_match_score_drug(self, drug: Drug, query: str) -> float:
        if drug.drugbank_id and drug.drugbank_id.lower() == query:
            return 1.0
        if drug.name and drug.name.lower() == query:
            return 0.95
        if drug.name_cn and drug.name_cn.lower() == query:
            return 0.95
        if drug.drugbank_id and drug.drugbank_id.lower().startswith(query):
            return 0.9
        if drug.name and drug.name.lower().startswith(query):
            return 0.85
        if drug.name_cn and drug.name_cn.lower().startswith(query):
            return 0.85
        if drug.name and query in drug.name.lower():
            return 0.8
        if drug.name_cn and query in drug.name_cn.lower():
            return 0.8
        for synonym in drug.synonyms:
            if query in synonym.lower():
                return 0.7
        for indication in drug.indications:
            if query in indication.lower():
                return 0.5
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        mapper_stats = self._mapper.get_stats()
        return {
            "icd10": {
                "loaded": self._icd10_loaded,
                "codes_count": mapper_stats["icd10_codes"],
                "name_mappings_count": mapper_stats["icd10_name_mappings"],
                "data_path": self._icd10_data_path,
            },
            "drugbank": {
                "loaded": self._drugbank_loaded,
                "drugs_count": mapper_stats["drugs"],
                "name_mappings_count": mapper_stats["drug_name_mappings"],
                "data_path": self._drugbank_data_path,
            },
        }

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            logger.info("Resetting TerminologyService instance")
            cls._instance = None
