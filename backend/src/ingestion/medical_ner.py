"""Medical NER (Named Entity Recognition) module.

Lightweight rule + dictionary based medical entity extraction as an alternative
to pure LLM-based extraction. Uses the synonym dictionary from knowledge_fusion
for high-recall entity matching.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from ..core.medical_schema import MedicalEntityType
from ..ingestion.knowledge_fusion import EntityDisambiguator


@dataclass
class NEREntity:
    name: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float


class MedicalNER:
    """Rule + dictionary based medical NER engine.

    Uses a multi-strategy approach:
    1. Exact dictionary match (highest priority, confidence 0.95)
    2. Abbreviation match (confidence 0.90)
    3. Regex pattern match for specific entity types (confidence 0.75)
    4. Suffix/prefix pattern match (confidence 0.70)
    """

    def __init__(self):
        self.disambiguator = EntityDisambiguator()
        self._entity_dict = self._build_entity_dict()
        self._abbreviation_dict = self._build_abbreviation_dict()
        self._regex_patterns = self._build_regex_patterns()
        self._suffix_patterns = self._build_suffix_patterns()

    def _build_entity_dict(self) -> Dict[str, str]:
        """Build reverse lookup: entity_name -> entity_type"""
        entity_dict = {}
        synonym_rules = self.disambiguator.synonym_rules

        for entity_type, entities in synonym_rules.items():
            for canonical_name, synonyms in entities.items():
                entity_dict[canonical_name] = entity_type
                for synonym in synonyms:
                    if len(synonym) >= 2:
                        entity_dict[synonym] = entity_type

        return entity_dict

    def _build_abbreviation_dict(self) -> Dict[str, Tuple[str, str]]:
        """Build abbreviation lookup: abbr -> (canonical_name, entity_type)"""
        abbr_dict = {}
        abbr_map = self.disambiguator.abbreviation_map

        for abbr, canonical in abbr_map.items():
            entity_type = self._entity_dict.get(canonical, "Disease")
            abbr_dict[abbr.upper()] = (canonical, entity_type)

        return abbr_dict

    def _build_regex_patterns(self) -> List[Tuple[str, str, re.Pattern]]:
        """Build regex patterns for specific medical entity types"""
        patterns = [
            (
                "disease_code_icd10",
                "Disease",
                re.compile(r'\b[A-Z][0-9]{2}(?:\.[0-9]{1,2})?\b'),
            ),
            (
                "drug_dosage",
                "Drug",
                re.compile(r'[\u4e00-\u9fa5]+(?:片|胶囊|注射液|口服液|颗粒|软膏|栓剂|贴剂|喷雾)[\u4e00-\u9fa5]*\d+(?:mg|g|ml|μg|IU)'),
            ),
            (
                "measurement",
                "Examination",
                re.compile(r'(?:血压|血糖|体温|心率|白细胞)\s*(?:为|是|达|高达)?\s*\d+(?:\.?\d*)?\s*(?:mmHg|mg/dL|℃|次/分|×10\^9/L)?'),
            ),
        ]
        return patterns

    def _build_suffix_patterns(self) -> List[Tuple[str, str, re.Pattern]]:
        """Build suffix/prefix patterns for entity type hints"""
        patterns = [
            ("disease_suffix", "Disease", re.compile(r'[\u4e00-\u9fa5]{2,10}(?:病|症|炎|癌|瘤|疾)(?:\s|$)')),
            ("symptom_suffix", "Symptom", re.compile(r'[\u4e00-\u9fa5]{1,8}(?:痛|痒|晕|胀|麻|咳|喘|烧)(?:\s|$)')),
            ("treatment_suffix", "Treatment", re.compile(r'[\u4e00-\u9fa5]{2,10}(?:手术|治疗|疗法|化疗|放疗|移植)(?:\s|$)')),
            ("department_suffix", "Department", re.compile(r'[\u4e00-\u9fa5]{2,8}(?:科|科室|门诊)(?:\s|$)')),
            ("anatomy_suffix", "Anatomy", re.compile(r'[\u4e00-\u9fa5]{1,6}(?:脏|肌|骨|神经|血管|组织)(?:\s|$)')),
        ]
        return patterns

    def extract(self, text: str) -> List[NEREntity]:
        """Extract medical entities from text using multi-strategy matching.

        Args:
            text: Input text to extract entities from

        Returns:
            List of NEREntity objects sorted by start position
        """
        entities = []
        seen_spans = set()

        entities.extend(self._exact_match(text, seen_spans))
        entities.extend(self._abbreviation_match(text, seen_spans))
        entities.extend(self._regex_match(text, seen_spans))
        entities.extend(self._suffix_match(text, seen_spans))

        entities.sort(key=lambda e: e.start_pos)
        entities = self._resolve_overlaps(entities)

        logger.info(f"NER extracted {len(entities)} entities from {len(text)} chars")
        return entities

    def _exact_match(self, text: str, seen_spans: set) -> List[NEREntity]:
        """Exact dictionary match - highest priority"""
        entities = []
        text_lower = text.lower()

        sorted_entities = sorted(self._entity_dict.items(), key=lambda x: -len(x[0]))

        for entity_name, entity_type in sorted_entities:
            name_lower = entity_name.lower()
            if len(name_lower) < 2:
                continue

            start = 0
            while True:
                pos = text_lower.find(name_lower, start)
                if pos == -1:
                    break

                span_key = (pos, pos + len(entity_name))
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(NEREntity(
                        name=entity_name,
                        entity_type=entity_type,
                        start_pos=pos,
                        end_pos=pos + len(entity_name),
                        confidence=0.95,
                    ))
                start = pos + 1

        return entities

    def _abbreviation_match(self, text: str, seen_spans: set) -> List[NEREntity]:
        """Abbreviation match"""
        entities = []
        words = re.findall(r'\b[A-Z]{2,5}\b', text)

        for word in words:
            if word in self._abbreviation_dict:
                pos = text.find(word)
                if pos != -1:
                    span_key = (pos, pos + len(word))
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        canonical, entity_type = self._abbreviation_dict[word]
                        entities.append(NEREntity(
                            name=canonical,
                            entity_type=entity_type,
                            start_pos=pos,
                            end_pos=pos + len(word),
                            confidence=0.90,
                        ))

        return entities

    def _regex_match(self, text: str, seen_spans: set) -> List[NEREntity]:
        """Regex pattern match"""
        entities = []

        for name, entity_type, pattern in self._regex_patterns:
            for match in pattern.finditer(text):
                span_key = (match.start(), match.end())
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(NEREntity(
                        name=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.75,
                    ))

        return entities

    def _suffix_match(self, text: str, seen_spans: set) -> List[NEREntity]:
        """Suffix/prefix pattern match"""
        entities = []

        for name, entity_type, pattern in self._suffix_patterns:
            for match in pattern.finditer(text):
                span_key = (match.start(), match.end())
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(NEREntity(
                        name=match.group().strip(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.70,
                    ))

        return entities

    def _resolve_overlaps(self, entities: List[NEREntity]) -> List[NEREntity]:
        """Resolve overlapping entities by keeping highest confidence"""
        if not entities:
            return entities

        entities.sort(key=lambda e: (-e.confidence, e.start_pos))
        result = []
        occupied = set()

        for entity in entities:
            span = set(range(entity.start_pos, entity.end_pos))
            if not span & occupied:
                result.append(entity)
                occupied.update(span)

        result.sort(key=lambda e: e.start_pos)
        return result

    def extract_as_dict(self, text: str) -> List[Dict[str, str]]:
        """Extract entities and return as list of dicts (compatible with KG builder)"""
        entities = self.extract(text)
        return [
            {"name": e.name, "type": e.entity_type, "properties": {"confidence": e.confidence}}
            for e in entities
        ]


def extract_medical_entities(text: str) -> List[NEREntity]:
    """Convenience function for medical NER extraction"""
    ner = MedicalNER()
    return ner.extract(text)
