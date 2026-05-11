import pytest
from src.ingestion.knowledge_fusion import EntityDisambiguator, RelationAligner, KnowledgeFusionEngine
from src.chains.medical_intent import MedicalIntentClassifier, MedicalIntent


class TestEntityDisambiguator:
    """测试实体消歧模块"""

    def test_normalize_disease_name(self):
        disambiguator = EntityDisambiguator()
        
        assert disambiguator.normalize_name("高血压病", "Disease") == "高血压"
        assert disambiguator.normalize_name("原发性高血压", "Disease") == "高血压"
        assert disambiguator.normalize_name("HTN", "Disease") == "高血压"
        assert disambiguator.normalize_name("糖尿病", "Disease") == "糖尿病"
        assert disambiguator.normalize_name("DM", "Disease") == "糖尿病"

    def test_normalize_symptom_name(self):
        disambiguator = EntityDisambiguator()
        
        assert disambiguator.normalize_name("发烧", "Symptom") == "发热"
        assert disambiguator.normalize_name("头疼", "Symptom") == "头痛"
        assert disambiguator.normalize_name("干咳", "Symptom") == "咳嗽"

    def test_normalize_drug_name(self):
        disambiguator = EntityDisambiguator()
        
        assert disambiguator.normalize_name("乙酰水杨酸", "Drug") == "阿司匹林"
        assert disambiguator.normalize_name("aspirin", "Drug") == "阿司匹林"

    def test_compute_similarity(self):
        disambiguator = EntityDisambiguator()
        
        assert disambiguator.compute_similarity("高血压", "高血压病") > 0.8
        assert disambiguator.compute_similarity("糖尿病", "糖尿病 mellitus") > 0.5
        assert disambiguator.compute_similarity("阿司匹林", "青霉素") < 0.3

    def test_disambiguate_entities(self):
        disambiguator = EntityDisambiguator()
        
        entities = [
            {"name": "高血压病", "type": "Disease", "properties": {}},
            {"name": "HTN", "type": "Disease", "properties": {"source": "doc1"}},
            {"name": "原发性高血压", "type": "Disease", "properties": {"source": "doc2"}},
        ]
        
        result = disambiguator.disambiguate(entities)
        
        assert len(result) == 1
        assert result[0]["name"] == "高血压"
        assert len(result[0]["original_names"]) == 3


class TestRelationAligner:
    """测试关系对齐模块"""

    def test_align_relation(self):
        aligner = RelationAligner()
        
        assert aligner.align_relation("表现为") == "HAS_SYMPTOM"
        assert aligner.align_relation("症状包括") == "HAS_SYMPTOM"
        assert aligner.align_relation("由...引起") == "CAUSED_BY"
        assert aligner.align_relation("治疗方案") == "TREATED_BY"
        assert aligner.align_relation("用于治疗") == "DRUG_FOR"
        assert aligner.align_relation("副作用") == "SIDE_EFFECT"

    def test_validate_relation(self):
        aligner = RelationAligner()
        
        assert aligner.validate_relation("Disease", "Symptom", "HAS_SYMPTOM") == True
        assert aligner.validate_relation("Disease", "Disease", "CAUSED_BY") == True
        assert aligner.validate_relation("Drug", "Disease", "DRUG_FOR") == True
        assert aligner.validate_relation("Disease", "Symptom", "DRUG_FOR") == False


class TestKnowledgeFusionEngine:
    """测试知识融合引擎"""

    def test_fuse_entities_and_relations(self):
        engine = KnowledgeFusionEngine()
        
        entities = [
            {"name": "高血压病", "type": "Disease", "properties": {}},
            {"name": "头痛", "type": "Symptom", "properties": {}},
            {"name": "HTN", "type": "Disease", "properties": {"source": "doc1"}},
        ]
        
        relationships = [
            {"source": "高血压病", "target": "头痛", "type": "表现为", "properties": {}},
            {"source": "HTN", "target": "头痛", "type": "HAS_SYMPTOM", "properties": {}},
        ]
        
        fused_entities, fused_rels = engine.fuse(entities, relationships)
        
        assert len(fused_entities) == 2
        assert "高血压" in [e["name"] for e in fused_entities]
        assert "头痛" in [rel["target"] for rel in fused_rels]

    def test_link_to_standard_ontology(self):
        engine = KnowledgeFusionEngine()
        
        entities = [
            {"name": "高血压", "type": "Disease", "properties": {}},
            {"name": "阿司匹林", "type": "Drug", "properties": {}},
        ]
        
        linked = engine.link_to_standard_ontology(entities)
        
        assert linked[0].get("icd10_code") == "I10"
        assert linked[1].get("umls_code") == "C0005890"


class TestMedicalIntentClassifier:
    """测试医疗意图分类器"""

    @pytest.mark.skip(reason="Requires LLM API access")
    def test_classify_disease_query(self):
        classifier = MedicalIntentClassifier()
        
        result = classifier.classify("高血压是什么疾病？")
        
        assert result.intent == MedicalIntent.DISEASE_QUERY
        assert result.confidence > 0.7
        assert "高血压" in result.entities

    @pytest.mark.skip(reason="Requires LLM API access")
    def test_classify_symptom_query(self):
        classifier = MedicalIntentClassifier()
        
        result = classifier.classify("头痛是什么原因引起的？")
        
        assert result.intent == MedicalIntent.SYMPTOM_QUERY
        assert "头痛" in result.entities

    @pytest.mark.skip(reason="Requires LLM API access")
    def test_classify_drug_query(self):
        classifier = MedicalIntentClassifier()
        
        result = classifier.classify("阿司匹林有什么副作用？")
        
        assert result.intent == MedicalIntent.DRUG_QUERY
        assert "阿司匹林" in result.entities

    @pytest.mark.skip(reason="Requires LLM API access")
    def test_classify_diagnosis_assist(self):
        classifier = MedicalIntentClassifier()
        
        result = classifier.classify("我最近经常头痛、头晕，可能是什么病？")
        
        assert result.intent == MedicalIntent.DIAGNOSIS_ASSIST
        assert "头痛" in result.entities
        assert "头晕" in result.entities

    def test_get_intent_prompt(self):
        classifier = MedicalIntentClassifier()
        
        prompt = classifier.get_intent_prompt(MedicalIntent.DISEASE_QUERY)
        
        assert "疾病的定义和概述" in prompt
        assert "主要病因和危险因素" in prompt

    def test_route_to_agent(self):
        classifier = MedicalIntentClassifier()
        
        assert classifier.route_to_agent(MedicalIntent.DISEASE_QUERY) == "disease_agent"
        assert classifier.route_to_agent(MedicalIntent.DRUG_QUERY) == "drug_agent"
        assert classifier.route_to_agent(MedicalIntent.SYMPTOM_QUERY) == "symptom_agent"
        assert classifier.route_to_agent(MedicalIntent.UNKNOWN) == "general_agent"