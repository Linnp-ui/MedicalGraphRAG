from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from loguru import logger

from .models import Drug, ICD10Code


@dataclass
class MappingResult:
    original_name: str
    standard_name: str
    terminology: str
    code: Optional[str] = None
    confidence: float = 1.0
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TerminologyMapper:
    def __init__(self):
        self._icd10_index: Dict[str, ICD10Code] = {}
        self._icd10_name_index: Dict[str, str] = {}
        self._drug_index: Dict[str, Drug] = {}
        self._drug_name_index: Dict[str, str] = {}

    def load_icd10_codes(self, codes: List[ICD10Code]) -> int:
        count = 0
        for code in codes:
            if not code.code:
                continue
            self._icd10_index[code.code] = code
            self._icd10_name_index[self._normalize_name(code.name)] = code.code
            for synonym in code.synonyms:
                normalized = self._normalize_name(synonym)
                if normalized not in self._icd10_name_index:
                    self._icd10_name_index[normalized] = code.code
            count += 1
        logger.info(f"Loaded {count} ICD-10 codes into mapper")
        return count

    def load_drugs(self, drugs: List[Drug]) -> int:
        count = 0
        for drug in drugs:
            if not drug.drugbank_id:
                continue
            self._drug_index[drug.drugbank_id] = drug
            self._drug_name_index[self._normalize_name(drug.name)] = drug.drugbank_id
            if drug.name_cn:
                self._drug_name_index[self._normalize_name(drug.name_cn)] = drug.drugbank_id
            for synonym in drug.synonyms:
                normalized = self._normalize_name(synonym)
                if normalized not in self._drug_name_index:
                    self._drug_name_index[normalized] = drug.drugbank_id
            count += 1
        logger.info(f"Loaded {count} drugs into mapper")
        return count

    def map_to_icd10(self, name: str) -> Optional[MappingResult]:
        if not name:
            return None
        normalized = self._normalize_name(name)
        code_id = self._icd10_name_index.get(normalized)
        if not code_id:
            logger.debug(f"No ICD-10 mapping found for: {name}")
            return None
        code = self._icd10_index.get(code_id)
        if not code:
            return None
        return MappingResult(
            original_name=name,
            standard_name=code.name,
            terminology="ICD-10",
            code=code.code,
            confidence=1.0,
            synonyms=code.synonyms,
            properties={
                "name_en": code.name_en,
                "category": code.category.value if code.category else None,
                "chapter": code.chapter,
                "block": code.block,
                "description": code.description,
            },
        )

    def map_to_drug(self, name: str) -> Optional[MappingResult]:
        if not name:
            return None
        normalized = self._normalize_name(name)
        drug_id = self._drug_name_index.get(normalized)
        if not drug_id:
            logger.debug(f"No drug mapping found for: {name}")
            return None
        drug = self._drug_index.get(drug_id)
        if not drug:
            return None
        return MappingResult(
            original_name=name,
            standard_name=drug.name,
            terminology="DrugBank",
            code=drug.drugbank_id,
            confidence=1.0,
            synonyms=drug.synonyms,
            properties={
                "name_cn": drug.name_cn,
                "cas_number": drug.cas_number,
                "atc_code": drug.atc_code,
                "formula": drug.formula,
                "weight": drug.weight,
                "indications": drug.indications,
                "contraindications": drug.contraindications,
                "side_effects": drug.side_effects,
            },
        )

    def expand_synonyms(self, name: str, entity_type: str) -> List[str]:
        if not name:
            return []
        result = [name]
        if entity_type.lower() in ("icd10", "disease", "diagnosis"):
            mapping = self.map_to_icd10(name)
            if mapping:
                result.extend(mapping.synonyms)
        elif entity_type.lower() in ("drug", "medication", "drugbank"):
            mapping = self.map_to_drug(name)
            if mapping:
                result.extend(mapping.synonyms)
        return list(dict.fromkeys(result))

    def get_all_icd10_codes(self) -> List[ICD10Code]:
        return list(self._icd10_index.values())

    def get_all_drugs(self) -> List[Drug]:
        return list(self._drug_index.values())

    def get_stats(self) -> Dict[str, int]:
        return {
            "icd10_codes": len(self._icd10_index),
            "icd10_name_mappings": len(self._icd10_name_index),
            "drugs": len(self._drug_index),
            "drug_name_mappings": len(self._drug_name_index),
        }

    def _normalize_name(self, name: str) -> str:
        if not name:
            return ""
        return name.lower().strip().replace(" ", "")
