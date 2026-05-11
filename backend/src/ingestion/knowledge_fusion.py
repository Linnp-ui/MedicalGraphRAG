import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from loguru import logger

from ..core.medical_schema import MedicalEntityType


class EntityDisambiguator:
    """实体消歧模块 - 识别并合并指称同一实体的不同表述"""

    def __init__(self):
        self.synonym_rules = self._load_synonym_rules()
        self.abbreviation_map = self._load_abbreviation_map()

    def _load_synonym_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """加载医疗实体同义词规则"""
        return {
            "Disease": {
                "高血压": ["高血压病", "原发性高血压", "HTN", "high blood pressure"],
                "糖尿病": ["糖尿病 mellitus", "DM", "高血糖"],
                "肺炎": ["肺部感染", "pneumonia"],
                "冠心病": ["冠状动脉粥样硬化性心脏病", "coronary heart disease"],
                "哮喘": ["支气管哮喘", "asthma"],
                "脑梗死": ["脑卒中", "中风", "cerebral infarction"],
                "心肌梗死": ["心梗", "myocardial infarction", "MI"],
                "胃炎": ["慢性胃炎", "胃黏膜炎症"],
                "关节炎": ["关节炎症"],
                "肝炎": ["肝脏炎症"],
            },
            "Symptom": {
                "咳嗽": ["干咳", "湿咳", "cough"],
                "发热": ["发烧", "体温升高", "fever"],
                "头痛": ["头疼", "headache"],
                "胸痛": ["胸闷", "chest pain"],
                "乏力": ["疲倦", "疲劳", "fatigue"],
                "恶心": ["想吐", "nausea"],
                "呕吐": ["呕", "vomiting"],
                "腹泻": ["拉肚子", "diarrhea"],
                "呼吸困难": ["气短", "喘不上气", "dyspnea"],
                "头晕": ["眩晕", "dizziness"],
            },
            "Drug": {
                "阿司匹林": ["乙酰水杨酸", "aspirin"],
                "布洛芬": ["ibuprofen"],
                "青霉素": ["盘尼西林", "penicillin"],
                "胰岛素": ["insulin"],
                "肝素": ["heparin"],
                "硝苯地平": ["心痛定", "nifedipine"],
                "硝酸甘油": ["nitroglycerin"],
                "阿莫西林": ["amoxicillin"],
            },
            "Anatomy": {
                "心脏": ["心", "heart"],
                "肺": ["肺部", "lungs"],
                "肝脏": ["肝", "liver"],
                "肾脏": ["肾", "kidney"],
                "大脑": ["脑", "brain"],
                "胃": ["胃部", "stomach"],
                "肠道": ["肠", "intestine"],
            },
            "Department": {
                "内科": ["internal medicine"],
                "外科": ["surgery"],
                "急诊科": ["急诊", "emergency"],
                "妇产科": ["妇科", "obstetrics and gynecology"],
                "儿科": ["pediatrics"],
                "神经科": ["神经内科", "neurology"],
                "心内科": ["心血管内科", "cardiology"],
            },
        }

    def _load_abbreviation_map(self) -> Dict[str, str]:
        """加载医疗缩写映射"""
        return {
            "HTN": "高血压",
            "DM": "糖尿病",
            "MI": "心肌梗死",
            "CHD": "冠心病",
            "CVA": "脑梗死",
            "COPD": "慢性阻塞性肺疾病",
            "TB": "肺结核",
            "HIV": "艾滋病",
            "ICU": "重症监护室",
            "MRI": "磁共振成像",
            "CT": "计算机断层扫描",
            "ECG": "心电图",
            "BP": "血压",
            "HR": "心率",
            "RR": "呼吸频率",
            "T": "体温",
        }

    def normalize_name(self, entity_name: str, entity_type: str) -> str:
        """标准化实体名称"""
        name = entity_name.strip()
        name_lower = name.lower()

        if name_lower in self.abbreviation_map:
            return self.abbreviation_map[name_lower]

        entity_synonyms = self.synonym_rules.get(entity_type, {})
        for canonical, synonyms in entity_synonyms.items():
            if name_lower == canonical.lower():
                return canonical
            for synonym in synonyms:
                if name_lower == synonym.lower():
                    return canonical

        return entity_name

    def compute_similarity(self, name1: str, name2: str) -> float:
        """计算两个实体名称的相似度"""
        name1 = name1.lower()
        name2 = name2.lower()

        if name1 == name2:
            return 1.0

        if name1 in name2 or name2 in name1:
            return 0.7 + min(len(name1), len(name2)) / max(len(name1), len(name2)) * 0.3

        common_chars = set(name1) & set(name2)
        if not common_chars:
            return 0.0

        jaccard = len(common_chars) / len(set(name1) | set(name2))
        edit_dist = self._levenshtein_distance(name1, name2)
        edit_sim = 1 - edit_dist / max(len(name1), len(name2))

        return (jaccard + edit_sim) / 2

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def disambiguate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对实体列表进行消歧处理"""
        normalized_entities = []
        entity_groups = defaultdict(list)

        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "")

            normalized_name = self.normalize_name(name, entity_type)
            normalized_entity = {
                **entity,
                "normalized_name": normalized_name,
                "original_names": [name],
            }

            group_key = (entity_type, normalized_name)
            entity_groups[group_key].append(normalized_entity)

        for group_key, group in entity_groups.items():
            if len(group) == 1:
                normalized_entities.append(group[0])
            else:
                merged = self._merge_entities(group)
                normalized_entities.append(merged)

        logger.info(f"Disambiguated {len(entities)} entities into {len(normalized_entities)} unique entities")
        return normalized_entities

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个指称同一实体的记录"""
        merged = {
            "name": entities[0]["normalized_name"],
            "type": entities[0]["type"],
            "properties": {},
            "original_names": [],
        }

        for entity in entities:
            merged["original_names"].extend(entity.get("original_names", []))
            merged["properties"].update(entity.get("properties", {}))

        merged["original_names"] = list(set(merged["original_names"]))

        return merged


class RelationAligner:
    """关系对齐模块 - 对齐和规范化关系类型"""

    def __init__(self):
        self.relation_mapping = self._load_relation_mapping()

    def _load_relation_mapping(self) -> Dict[str, Dict[str, str]]:
        """加载关系映射规则"""
        return {
            "HAS_SYMPTOM": {
                "表现为": "HAS_SYMPTOM",
                "有症状": "HAS_SYMPTOM",
                "症状包括": "HAS_SYMPTOM",
                "症状是": "HAS_SYMPTOM",
                "主要症状": "HAS_SYMPTOM",
                "常见症状": "HAS_SYMPTOM",
            },
            "CAUSED_BY": {
                "由...引起": "CAUSED_BY",
                "病因是": "CAUSED_BY",
                "原因是": "CAUSED_BY",
                "由于": "CAUSED_BY",
                "因": "CAUSED_BY",
                "诱发": "CAUSED_BY",
            },
            "TREATED_BY": {
                "治疗方案": "TREATED_BY",
                "治疗方法": "TREATED_BY",
                "用...治疗": "TREATED_BY",
                "治疗": "TREATED_BY",
                "疗法": "TREATED_BY",
                "手术治疗": "TREATED_BY",
            },
            "DRUG_FOR": {
                "用于治疗": "DRUG_FOR",
                "适应症": "DRUG_FOR",
                "适用": "DRUG_FOR",
                "治疗": "DRUG_FOR",
                "针对": "DRUG_FOR",
            },
            "SIDE_EFFECT": {
                "副作用": "SIDE_EFFECT",
                "不良反应": "SIDE_EFFECT",
                "不良反应包括": "SIDE_EFFECT",
                "副作用有": "SIDE_EFFECT",
                "可能引起": "SIDE_EFFECT",
            },
            "INDICATES": {
                "提示": "INDICATES",
                "表明": "INDICATES",
                "可能是": "INDICATES",
                "可能患有": "INDICATES",
                "提示可能": "INDICATES",
            },
            "PART_OF": {
                "属于": "PART_OF",
                "位于": "PART_OF",
                "构成": "PART_OF",
                "组成": "PART_OF",
            },
            "BELONGS_TO": {
                "属于科室": "BELONGS_TO",
                "由...治疗": "BELONGS_TO",
                "就诊科室": "BELONGS_TO",
                "对应科室": "BELONGS_TO",
            },
        }

    def align_relation(self, relation_type: str) -> str:
        """对齐关系类型"""
        relation_type = relation_type.strip().upper()

        for canonical, mappings in self.relation_mapping.items():
            if relation_type == canonical:
                return canonical
            for alias, target in mappings.items():
                if relation_type == alias.upper():
                    return target

        return relation_type

    def validate_relation(self, source_type: str, target_type: str, relation_type: str) -> bool:
        """验证关系类型是否符合医疗本体约束"""
        valid_relations = {
            ("Disease", "Symptom"): ["HAS_SYMPTOM"],
            ("Disease", "Disease"): ["CAUSED_BY"],
            ("Disease", "Treatment"): ["TREATED_BY"],
            ("Disease", "Department"): ["BELONGS_TO"],
            ("Drug", "Disease"): ["DRUG_FOR"],
            ("Drug", "Symptom"): ["SIDE_EFFECT"],
            ("Symptom", "Disease"): ["INDICATES"],
            ("Anatomy", "Anatomy"): ["PART_OF"],
        }

        key = (source_type, target_type)
        if key in valid_relations:
            return relation_type in valid_relations[key]

        return True

    def align_relations(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对关系列表进行对齐处理"""
        aligned = []
        for rel in relationships:
            aligned_type = self.align_relation(rel.get("type", ""))
            source_type = rel.get("source_type", "")
            target_type = rel.get("target_type", "")

            if source_type and target_type:
                if not self.validate_relation(source_type, target_type, aligned_type):
                    logger.warning(f"Invalid relation: {source_type} -[{aligned_type}]-> {target_type}")
                    continue

            aligned.append({
                **rel,
                "aligned_type": aligned_type,
            })

        logger.info(f"Aligned {len(relationships)} relationships")
        return aligned


class KnowledgeFusionEngine:
    """知识融合引擎 - 整合实体消歧和关系对齐"""

    def __init__(self):
        self.disambiguator = EntityDisambiguator()
        self.relation_aligner = RelationAligner()

    def fuse(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """执行完整的知识融合流程"""
        logger.info("Starting knowledge fusion...")

        disambiguated_entities = self.disambiguator.disambiguate(entities)

        entity_name_map = {
            ent.get("name", ""): ent.get("normalized_name", ent.get("name", ""))
            for ent in disambiguated_entities
        }

        normalized_relationships = []
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")

            normalized_source = entity_name_map.get(source, source)
            normalized_target = entity_name_map.get(target, target)

            normalized_relationships.append({
                **rel,
                "source": normalized_source,
                "target": normalized_target,
            })

        aligned_relationships = self.relation_aligner.align_relations(normalized_relationships)

        logger.info(f"Knowledge fusion completed: {len(disambiguated_entities)} entities, {len(aligned_relationships)} relationships")
        return disambiguated_entities, aligned_relationships

    def link_to_standard_ontology(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """链接实体到标准医学本体（如ICD-10、UMLS）"""
        linked_entities = []
        for entity in entities:
            linked = self._link_entity(entity)
            linked_entities.append(linked)
        return linked_entities

    def _link_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """链接单个实体到标准本体"""
        name = entity.get("name", "")
        entity_type = entity.get("type", "")

        icd10_mapping = {
            "高血压": "I10",
            "糖尿病": "E11",
            "肺炎": "J18",
            "冠心病": "I25",
            "脑梗死": "I63",
            "心肌梗死": "I21",
            "哮喘": "J45",
        }

        umls_mapping = {
            "阿司匹林": "C0005890",
            "布洛芬": "C0013227",
            "青霉素": "C0031236",
            "胰岛素": "C0022847",
        }

        linked = {**entity}

        if entity_type == "Disease" and name in icd10_mapping:
            linked["icd10_code"] = icd10_mapping[name]
        elif entity_type == "Drug" and name in umls_mapping:
            linked["umls_code"] = umls_mapping[name]

        return linked