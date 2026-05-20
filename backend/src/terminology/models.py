from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class ICD10Category(str, Enum):
    CHAPTER_01 = "A00-B99"
    CHAPTER_02 = "C00-D48"
    CHAPTER_03 = "D50-D89"
    CHAPTER_04 = "E00-E90"
    CHAPTER_05 = "F00-F99"
    CHAPTER_06 = "G00-G99"
    CHAPTER_07 = "H00-H59"
    CHAPTER_08 = "H60-H95"
    CHAPTER_09 = "I00-I99"
    CHAPTER_10 = "J00-J99"
    CHAPTER_11 = "K00-K93"
    CHAPTER_12 = "L00-L99"
    CHAPTER_13 = "M00-M99"
    CHAPTER_14 = "N00-N99"
    CHAPTER_15 = "O00-O99"
    CHAPTER_16 = "P00-P96"
    CHAPTER_17 = "Q00-Q99"
    CHAPTER_18 = "R00-R99"
    CHAPTER_19 = "S00-T98"
    CHAPTER_20 = "V00-Y98"
    CHAPTER_21 = "Z00-Z99"
    CHAPTER_22 = "U00-U99"


@dataclass
class ICD10Code:
    code: str
    name: str
    name_en: Optional[str] = None
    category: Optional[ICD10Category] = None
    chapter: Optional[str] = None
    block: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None
    related_codes: List[str] = field(default_factory=list)
    parent_code: Optional[str] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.category is not None:
            result["category"] = self.category.value
        return result


@dataclass
class DrugInteraction:
    drug_name: str
    drugbank_id: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Drug:
    drugbank_id: str
    name: str
    name_cn: Optional[str] = None
    cas_number: Optional[str] = None
    atc_code: Optional[str] = None
    formula: Optional[str] = None
    weight: Optional[float] = None
    indications: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    interactions: List[DrugInteraction] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "drugbank_id": self.drugbank_id,
            "name": self.name,
            "name_cn": self.name_cn,
            "cas_number": self.cas_number,
            "atc_code": self.atc_code,
            "formula": self.formula,
            "weight": self.weight,
            "indications": self.indications,
            "contraindications": self.contraindications,
            "side_effects": self.side_effects,
            "interactions": [i.to_dict() for i in self.interactions],
            "synonyms": self.synonyms,
            "description": self.description,
        }
        return result
