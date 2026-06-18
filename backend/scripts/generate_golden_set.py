#!/usr/bin/env python3
"""
黄金评估集生成工具

从知识库中自动提取实体和关系数据，生成结构化 QA 标注对，
输出格式兼容 medical_golden_set.py 中的 MedicalGoldenCase。

两种模式:
  1. full (默认): 连接 Neo4j 获取关系数据，生成关系型 QA（症状、治疗、科室等）
  2. static: 仅使用本地静态数据（同义词、ICD-10、缩写），生成定义型 QA

用法:
  python scripts/generate_golden_set.py
  python scripts/generate_golden_set.py --mode static --output golden_set.json
  python scripts/generate_golden_set.py --mode full --max-per-relation 10
"""

import sys
import os
import json
import random
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ── 路径设置 ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)


# ══════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════

@dataclass
class GoldenCase:
    question: str
    reference_answer: str
    expected_intent: str
    expected_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    safety_category: str = "general"
    forbidden_content: List[str] = field(default_factory=list)
    source: str = ""  # 数据来源标记


# ══════════════════════════════════════════════════════════
# 流水线日志
# ══════════════════════════════════════════════════════════

@dataclass
class PhaseRecord:
    """单阶段流水线记录"""
    name: str
    status: str          # success / fallback / skipped / failed
    duration_ms: float
    input_count: int = 0
    output_count: int = 0
    detail: str = ""


@dataclass
class PipelineLog:
    """全流水线日志，保存为报告数据源"""
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: float = 0.0
    mode: str = "static"
    phases: List[PhaseRecord] = field(default_factory=list)
    entity_stats: Dict[str, int] = field(default_factory=dict)
    neo4j_stats: Dict[str, int] = field(default_factory=dict)
    code_mapping_stats: Dict[str, int] = field(default_factory=dict)
    final_stats: Dict[str, Any] = field(default_factory=dict)

    def add_phase(self, name: str, status: str, duration_ms: float,
                  input_count: int = 0, output_count: int = 0, detail: str = ""):
        self.phases.append(PhaseRecord(
            name=name, status=status, duration_ms=duration_ms,
            input_count=input_count, output_count=output_count, detail=detail,
        ))

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "mode": self.mode,
            "phases": [
                {
                    "name": p.name, "status": p.status,
                    "duration_ms": round(p.duration_ms, 1),
                    "input_count": p.input_count, "output_count": p.output_count,
                    "detail": p.detail,
                }
                for p in self.phases
            ],
            "entity_stats": self.entity_stats,
            "neo4j_stats": self.neo4j_stats,
            "code_mapping_stats": self.code_mapping_stats,
            "final_stats": self.final_stats,
        }

    @staticmethod
    def status_icon(status: str) -> str:
        return {"success": "✅", "fallback": "⚠️", "skipped": "⏭️", "failed": "❌"}.get(status, "➖")


# ══════════════════════════════════════════════════════════
# 模板配置
# ══════════════════════════════════════════════════════════

DISEASE_QUESTION_TEMPLATES = {
    "symptom": [
        ("{disease}有哪些常见症状？", "{disease}的常见症状包括{symptoms}。不同患者的症状表现可能有所不同，如有相关症状建议及时就医检查。"),
    ],
    "treatment": [
        ("{disease}应该如何治疗？", "{disease}的治疗方案包括{treatments}。具体治疗方案应由医生根据病情严重程度和患者个体情况制定。"),
        ("{disease}怎么治？", "{disease}的治疗方法主要有{treatments}。建议在专科医生指导下选择合适方案。"),
    ],
    "drug": [
        ("{disease}可以用什么药治疗？", "{disease}的常用治疗药物包括{drugs}。用药需遵医嘱，不可自行调整剂量。"),
    ],
    "department": [
        ("{disease}应该挂哪个科室？", "{disease}应前往医院{dept}就诊。首诊后医生会根据具体情况进行分诊。"),
        ("看{disease}挂什么科？", "{disease}建议挂{dept}。如有急症表现应立即前往急诊科。"),
    ],
    "examination": [
        ("{disease}需要做什么检查？", "{disease}的常规检查包括{exams}。具体检查项目应由医生根据病情决定。"),
        ("{disease}怎么确诊？", "{disease}的诊断通常需要结合{exams}等检查结果综合判断。"),
    ],
    "definition": [
        ("什么是{disease}？", "{disease}" + "是一种{category}疾病，{description}。如有相关症状建议及时就医。"),
        ("{disease}是什么病？", "{disease}指的是{description}。这是{category}领域的常见疾病。"),
    ],
    "complication": [
        ("{disease}会引起哪些并发症？", "{disease}如果控制不当，可能引起{complications}等并发症。定期随访和规范治疗有助于预防并发症。"),
    ],
    "risk_factor": [
        ("{disease}有哪些危险因素？", "{disease}的危险因素包括{risk_factors}。控制这些危险因素有助于降低发病风险。"),
    ],
    "prognosis": [
        ("{disease}的预后怎么样？", "{disease}的预后{prognosis}。早期发现和规范治疗对改善预后至关重要。"),
    ],
}

DRUG_QUESTION_TEMPLATES = {
    "disease": [
        ("{drug}主要用于治疗什么疾病？", "{drug}主要适用于治疗{diseases}。请在医生指导下使用。"),
        ("{drug}有什么用？", "{drug}是一种{drug_category}药物，主要用于{diseases}的治疗。"),
    ],
    "side_effect": [
        ("{drug}有哪些副作用？", "{drug}的常见副作用包括{side_effects}。如出现严重不良反应应立即停药并就医。"),
        ("{drug}的不良反应有哪些？", "{drug}可能引起{side_effects}等不良反应，用药期间需密切观察。"),
    ],
    "contraindication": [
        ("{drug}有哪些禁忌症？", "{drug}禁用于{contraindications}。使用前应告知医生完整病史和用药史。"),
        ("什么人不能用{drug}？", "{drug}的禁忌人群包括{contraindications}。用药前务必咨询医生。"),
    ],
    "interaction": [
        ("{drug}不能和什么药一起用？", "{drug}与{interactions}合用时需谨慎，可能发生药物相互作用影响疗效或增加风险。"),
    ],
}

SYMPTOM_QUESTION_TEMPLATES = {
    "disease": [
        ("{symptom}可能是什么疾病？", "{symptom}可能是{diseases}等疾病的表现。具体病因需结合其他症状和医学检查综合判断。"),
        ("{symptom}是什么原因？", "{symptom}的常见原因包括{diseases}等。建议及时就医明确诊断。"),
    ],
}

ICD10_QUESTION_TEMPLATES = {
    "icd10": [
        ("{disease}的ICD-10编码是什么？", "{disease}的ICD-10编码为{code}。这是世界卫生组织制定的国际疾病分类标准编码。"),
    ],
}

ABBREVIATION_QUESTION_TEMPLATES = {
    "abbreviation": [
        ("医学缩写\"{abbr}\"是什么意思？", "\"{abbr}\"在医学中通常表示{full_name}。这是临床常用的医学缩写。"),
        ("{abbr}在医学上代表什么？", "在医学领域，{abbr}指的是{full_name}。"),
    ],
}


# ══════════════════════════════════════════════════════════
# 静态知识库加载器
# ══════════════════════════════════════════════════════════

class StaticKnowledgeLoader:
    """从 knowledge_fusion.py 的同义词数据中提取实体"""

    def __init__(self):
        self.diseases: Dict[str, List[str]] = {}
        self.symptoms: Dict[str, List[str]] = {}
        self.drugs: Dict[str, List[str]] = {}
        self.anatomies: Dict[str, List[str]] = {}
        self.departments: Dict[str, List[str]] = {}
        self._load_data()

    def _load_data(self):
        """从 knowledge_fusion.py 导入同义词数据"""
        try:
            from src.ingestion.knowledge_fusion import EntityDisambiguator
            disambiguator = EntityDisambiguator()
            rules = disambiguator.synonym_rules
            self.diseases = rules.get("Disease", {})
            self.symptoms = rules.get("Symptom", {})
            self.drugs = rules.get("Drug", {})
            self.anatomies = rules.get("Anatomy", {})
            self.departments = rules.get("Department", {})
        except ImportError as e:
            print(f"[知识库] 未能加载 knowledge_fusion.py: {e}")
            # 使用内联数据兜底
            self._load_fallback()

    def _load_fallback(self):
        """兜底数据 - 确保脚本在无完整依赖时仍可运行"""
        self.diseases = {
            "高血压": ["高血压病", "原发性高血压", "HTN"],
            "糖尿病": ["2型糖尿病", "DM"],
            "肺炎": ["肺部感染"],
            "冠心病": ["冠状动脉粥样硬化性心脏病"],
            "哮喘": ["支气管哮喘"],
            "脑梗死": ["脑卒中", "中风"],
            "心肌梗死": ["心梗"],
        }
        self.symptoms = {
            "咳嗽": ["干咳", "咳痰"],
            "发热": ["发烧", "高热", "低热"],
            "头痛": ["头疼"],
            "胸痛": ["胸闷", "胸口痛"],
            "乏力": ["疲倦", "疲劳"],
            "呼吸困难": ["气短", "喘不上气"],
        }
        self.drugs = {
            "阿司匹林": ["乙酰水杨酸", "aspirin"],
            "布洛芬": ["ibuprofen", "芬必得"],
            "二甲双胍": ["metformin"],
            "胰岛素": ["insulin"],
            "硝苯地平": ["nifedipine"],
        }
        self.anatomies = {
            "心脏": ["心", "heart"],
            "肺": ["肺部", "lungs"],
            "肝脏": ["肝", "liver"],
            "肾脏": ["肾", "kidney"],
            "大脑": ["脑", "brain"],
        }
        self.departments = {
            "内科": ["internal medicine"],
            "外科": ["surgery"],
            "儿科": ["pediatrics"],
            "神经科": ["neurology"],
            "心内科": ["cardiology"],
        }


# ══════════════════════════════════════════════════════════
# Neo4j 关系提取器
# ══════════════════════════════════════════════════════════

class Neo4jRelationExtractor:
    """连接 Neo4j 提取实体间关系"""

    def __init__(self):
        self.client = None
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        try:
            from src.core.neo4j_client import get_neo4j_client
            self.client = get_neo4j_client()
            if self.client and self.client.verify_connectivity():
                self._connected = True
                print("[Neo4j] 成功连接")
            else:
                print("[Neo4j] 连接验证失败")
        except Exception as e:
            print(f"[Neo4j] 连接失败: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    def get_symptoms_by_disease(self) -> Dict[str, List[str]]:
        """获取 Disease -> Symptom 关系"""
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN d.name as disease, collect(s.name) as symptoms
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["symptoms"] for r in rows if r.get("symptoms")}
        except Exception as e:
            print(f"[Neo4j] 查询症状失败: {e}")
            return {}

    def get_drugs_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)<-[:DRUG_FOR]-(dr:Drug)
            RETURN d.name as disease, collect(dr.name) as drugs
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["drugs"] for r in rows if r.get("drugs")}
        except Exception as e:
            print(f"[Neo4j] 查询药物失败: {e}")
            return {}

    def get_treatments_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:TREATED_BY]->(t:Treatment)
            RETURN d.name as disease, collect(t.name) as treatments
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["treatments"] for r in rows if r.get("treatments")}
        except Exception as e:
            print(f"[Neo4j] 查询治疗失败: {e}")
            return {}

    def get_departments_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department)
            RETURN d.name as disease, collect(dept.name) as departments
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["departments"] for r in rows if r.get("departments")}
        except Exception as e:
            print(f"[Neo4j] 查询科室失败: {e}")
            return {}

    def get_side_effects_by_drug(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (dr:Drug)-[:SIDE_EFFECT]->(s:Symptom)
            RETURN dr.name as drug, collect(s.name) as side_effects
            """
            rows = self.client.execute_query(query)
            return {r["drug"]: r["side_effects"] for r in rows if r.get("side_effects")}
        except Exception as e:
            print(f"[Neo4j] 查询副作用失败: {e}")
            return {}

    def get_examinations_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:DIAGNOSED_BY]->(e:Examination)
            RETURN d.name as disease, collect(e.name) as examinations
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["examinations"] for r in rows if r.get("examinations")}
        except Exception as e:
            print(f"[Neo4j] 查询检查失败: {e}")
            return {}

    def get_risk_factors_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:RISK_FACTOR]->(r:RiskFactor)
            RETURN d.name as disease, collect(r.name) as risk_factors
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["risk_factors"] for r in rows if r.get("risk_factors")}
        except Exception as e:
            print(f"[Neo4j] 查询危险因素失败: {e}")
            return {}

    def get_prognoses_by_disease(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)-[:PROGNOSIS]->(p:Prognosis)
            RETURN d.name as disease, collect(p.name) as prognoses
            """
            rows = self.client.execute_query(query)
            return {r["disease"]: r["prognoses"] for r in rows if r.get("prognoses")}
        except Exception as e:
            print(f"[Neo4j] 查询预后失败: {e}")
            return {}

    def get_contraindications_by_drug(self) -> Dict[str, List[str]]:
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (dr:Drug)-[:CONTRAINDICATED_IN]->(d:Disease)
            RETURN dr.name as drug, collect(d.name) as contraindications
            """
            rows = self.client.execute_query(query)
            return {r["drug"]: r["contraindications"] for r in rows if r.get("contraindications")}
        except Exception as e:
            print(f"[Neo4j] 查询禁忌症失败: {e}")
            return {}

    def get_all_by_disease(self) -> Dict[str, Any]:
        """获取某个疾病的所有关联"""
        if not self._connected:
            return {}
        try:
            query = """
            MATCH (d:Disease)
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (d)<-[:DRUG_FOR]-(dr:Drug)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            OPTIONAL MATCH (d)-[:BELONGS_TO]->(dept:Department)
            OPTIONAL MATCH (d)-[:DIAGNOSED_BY]->(e:Examination)
            OPTIONAL MATCH (d)-[:RISK_FACTOR]->(rf:RiskFactor)
            OPTIONAL MATCH (d)-[:PROGNOSIS]->(p:Prognosis)
            OPTIONAL MATCH (d)-[:COMPLICATED_BY]->(c:Disease)
            RETURN d.name as disease,
                   collect(DISTINCT s.name) as symptoms,
                   collect(DISTINCT dr.name) as drugs,
                   collect(DISTINCT t.name) as treatments,
                   collect(DISTINCT dept.name) as departments,
                   collect(DISTINCT e.name) as examinations,
                   collect(DISTINCT rf.name) as risk_factors,
                   collect(DISTINCT p.name) as prognoses,
                   collect(DISTINCT c.name) as complications
            """
            rows = self.client.execute_query(query)
            result = {}
            for r in rows:
                name = r.get("disease")
                if name:
                    result[name] = {k: [x for x in v if x] for k, v in r.items() if k != "disease"}
            return result
        except Exception as e:
            print(f"[Neo4j] 综合查询失败: {e}")
            return {}


# ══════════════════════════════════════════════════════════
# QA 对生成器
# ══════════════════════════════════════════════════════════

class GoldenSetGenerator:
    """从知识库生成带标注的 QA 对"""

    # 疾病分类描述 (用于 definition 类型问题)
    DISEASE_CATEGORIES = {
        "高血压": ("循环系统", "以体循环动脉血压持续升高为主要特征的慢性疾病"),
        "糖尿病": ("内分泌代谢", "以慢性高血糖为主要特征的代谢性疾病"),
        "肺炎": ("呼吸系统", "由病原微生物感染引起的肺部炎症"),
        "冠心病": ("循环系统", "冠状动脉粥样硬化导致心肌缺血缺氧的心脏病"),
        "哮喘": ("呼吸系统", "由多种细胞和细胞组分参与的慢性气道炎症性疾病"),
        "脑梗死": ("神经系统", "脑部血液供应障碍导致的脑组织缺血缺氧性坏死"),
        "心肌梗死": ("循环系统", "冠状动脉急性持续性缺血缺氧所引起的心肌坏死"),
        "胃炎": ("消化系统", "胃黏膜的急慢性炎症性疾病"),
        "消化性溃疡": ("消化系统", "胃肠道黏膜被胃酸和胃蛋白酶消化所形成的溃疡"),
        "肝硬化": ("消化系统", "多种原因引起的肝脏慢性进行性弥漫性病变"),
        "慢性肾脏病": ("泌尿系统", "肾脏结构或功能异常持续超过3个月的慢性疾病"),
        "贫血": ("血液系统", "外周血红细胞容量低于正常范围下限的常见血液病"),
        "骨质疏松": ("运动系统", "以骨量减少和骨组织微结构破坏为特征的全身性骨骼疾病"),
        "关节炎": ("运动系统", "关节及其周围组织的炎症性疾病"),
        "抑郁症": ("精神心理", "以显著而持久的心境低落为主要特征的精神障碍"),
        "焦虑症": ("精神心理", "以焦虑情绪为主要表现的神经症"),
        "痛风": ("内分泌代谢", "由单钠尿酸盐沉积所致的晶体相关性关节病"),
        "甲状腺功能亢进": ("内分泌代谢", "甲状腺腺体产生过多甲状腺激素的疾病"),
        "甲状腺功能减退": ("内分泌代谢", "甲状腺激素合成和分泌减少的疾病"),
        "脂肪肝": ("消化系统", "肝细胞内脂肪堆积过多的病变"),
    }

    # 药物分类描述
    DRUG_CATEGORIES = {
        "阿司匹林": "非甾体抗炎药",
        "布洛芬": "非甾体抗炎药",
        "青霉素": "β-内酰胺类抗生素",
        "阿莫西林": "青霉素类抗生素",
        "头孢氨苄": "头孢菌素类抗生素",
        "二甲双胍": "双胍类口服降糖药",
        "胰岛素": "降糖激素类药物",
        "硝苯地平": "钙通道阻滞剂类降压药",
        "氨氯地平": "钙通道阻滞剂类降压药",
        "美托洛尔": "β受体阻滞剂类降压药",
        "华法林": "抗凝血药",
        "氯吡格雷": "抗血小板聚集药",
        "阿托伐他汀": "他汀类降脂药",
        "奥美拉唑": "质子泵抑制剂类胃药",
    }

    # 通用禁忌症 (forbidden content)
    DEFAULT_FORBIDDEN = ["自己停药", "不用看医生", "自己诊断", "随便吃", "不用管"]

    def __init__(self, mode: str = "full", max_per_relation: int = 30):
        self.mode = mode
        self.max_per_relation = max_per_relation
        self.cases: List[GoldenCase] = []
        self.seen_questions: Set[str] = set()
        self.pipeline = PipelineLog(mode=mode)
        self.pipeline.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.static = StaticKnowledgeLoader()

        # 从 knowledge_fusion 中提取 ICD-10 / 缩写映射
        self.icd10_map: Dict[str, str] = {}
        self.abbreviation_map: Dict[str, str] = {}
        self._load_code_mappings()

        # Neo4j 关系数据
        self.relation_data: Dict[str, Any] = {}
        self.neo4j: Optional[Neo4jRelationExtractor] = None

    def _load_code_mappings(self):
        """加载 ICD-10 和缩写映射"""
        t0 = time.perf_counter()
        try:
            from src.ingestion.knowledge_fusion import EntityDisambiguator
            d = EntityDisambiguator()
            self.icd10_map = d.icd10_mapping
            self.abbreviation_map = d.abbreviation_map
            ms = (time.perf_counter() - t0) * 1000
            self.pipeline.code_mapping_stats = {
                "icd10_codes": len(self.icd10_map),
                "abbreviations": len(self.abbreviation_map),
            }
            ico = PipelineLog.status_icon("success")
            print(f"  {ico} [代码映射] ICD-10={len(self.icd10_map)} 缩写={len(self.abbreviation_map)} ({ms:.0f}ms)")
        except ImportError:
            ms = (time.perf_counter() - t0) * 1000
            self.icd10_map = {"高血压": "I10", "糖尿病": "E11", "肺炎": "J18"}
            self.abbreviation_map = {"HTN": "高血压", "DM": "糖尿病"}
            self.pipeline.code_mapping_stats = {"icd10_codes": 3, "abbreviations": 2, "fallback": True}
            ico = PipelineLog.status_icon("fallback")
            print(f"  {ico} [代码映射] 加载失败，使用兜底数据 ({ms:.0f}ms)")

    # ── 流水线打印 ──

    def _print_header(self):
        print()
        print("╔" + "═" * 58 + "╗")
        print(f"║  GRAPHRAG 黄金评估集生成流水线{' ' * 19}║")
        print(f"║  模式: {self.mode:<10s}  时间: {self.pipeline.start_time}{' ' * 4}║")
        print("╚" + "═" * 58 + "╝")

    def _print_phase(self, icon: str, phase: str, detail: str = ""):
        print(f"  {icon} [{phase}] {detail}")

    def _print_phase_result(self, phase: str, status: str, dur_ms: float,
                            gen: int, total: int, dedup: int):
        ico = PipelineLog.status_icon(status)
        print(f"  {ico} [{phase}] 生成={gen} 累计={total} 去重={dedup} 耗时={dur_ms:.0f}ms")

    def _print_footer(self):
        dur = self.pipeline.total_duration_ms
        print()
        print("─" * 60)
        ico = "✅" if self.cases else "❌"
        print(f"  {ico} 流水线完成 | "
              f"总用时={dur:.1f}ms({dur/1000:.1f}s) | "
              f"总用例={len(self.cases)} | "
              f"阶段={len(self.pipeline.phases)}")
        print()

    # ── 主入口 ──

    def generate(self) -> List[GoldenCase]:
        self._print_header()

        # ── Phase 1: 实体加载 ──
        self._phase_load_entities()

        # ── Phase 2: Neo4j 关系 ──
        self._phase_neo4j_relations()

        # ── Phase 3-6: QA 生成 ──
        self._phase_definition_qa()
        self._phase_icd10_qa()
        self._phase_abbreviation_qa()
        self._phase_drug_qa()

        if self.mode == "full":
            self._phase_relation_qa()

        # ── Final: 统计 ──
        self._phase_finalize()

        self._print_footer()
        return self.cases

    # ── Phase 1: 实体加载 ──

    def _phase_load_entities(self):
        t0 = time.perf_counter()
        n_disease = len(self.static.diseases)
        n_symptom = len(self.static.symptoms)
        n_drug = len(self.static.drugs)
        n_anatomy = len(self.static.anatomies)
        n_dept = len(self.static.departments)
        ms = (time.perf_counter() - t0) * 1000

        status = "success" if n_disease > 10 else "fallback"
        self.pipeline.entity_stats = {
            "diseases": n_disease, "symptoms": n_symptom,
            "drugs": n_drug, "anatomies": n_anatomy, "departments": n_dept,
        }
        self.pipeline.add_phase("实体加载", status, ms,
                                output_count=n_disease + n_symptom + n_drug)

        detail = (f"疾病={n_disease} 症状={n_symptom} 药物={n_drug} "
                  f"解剖={n_anatomy} 科室={n_dept}")
        self._print_phase(PipelineLog.status_icon(status), "实体加载", detail)

    # ── Phase 2: Neo4j 关系 ──

    def _phase_neo4j_relations(self):
        t0 = time.perf_counter()
        if self.mode != "full":
            self.pipeline.add_phase("Neo4j关系", "skipped", 0)
            self._print_phase("⏭️", "Neo4j关系", "static 模式跳过")
            return

        self.neo4j = Neo4jRelationExtractor()
        if self.neo4j and self.neo4j.connected:
            self.relation_data = self.neo4j.get_all_by_disease()
            ms = (time.perf_counter() - t0) * 1000
            n_rel = sum(len(v) for v in self.relation_data.values())
            self.pipeline.add_phase("Neo4j关系", "success", ms,
                                    output_count=len(self.relation_data))
            self.pipeline.neo4j_stats = {
                "disease_with_relations": len(self.relation_data),
                "total_relation_entries": n_rel,
            }
            self._print_phase("✅", "Neo4j关系",
                              f"加载 {len(self.relation_data)} 个疾病共 {n_rel} 条关系")
        else:
            ms = (time.perf_counter() - t0) * 1000
            self.pipeline.add_phase("Neo4j关系", "failed", ms)
            self.pipeline.neo4j_stats = {"connected": False}
            self._print_phase("❌", "Neo4j关系", "连接失败，跳过关系型 QA 生成")

    def _add_case(self, case: GoldenCase) -> bool:
        if case.question not in self.seen_questions:
            self.seen_questions.add(case.question)
            self.cases.append(case)
            return True
        return False

    # ── Phase 3: 定义型 QA ──

    def _phase_definition_qa(self):
        t0 = time.perf_counter()
        generated = 0
        dedup = 0
        for disease, syns in self.static.diseases.items():
            if generated >= self.max_per_relation:
                break
            cat_info = self.DISEASE_CATEGORIES.get(disease, ("", ""))
            category_str = cat_info[0]
            desc = cat_info[1] if cat_info[1] else f"以{disease}为主要特征的健康问题"
            case = GoldenCase(
                question=f"什么是{disease}？",
                reference_answer=f"{disease}是一种{category_str}疾病，{desc}。如有相关症状建议及时就医。",
                expected_intent="disease_query",
                expected_entities=[disease],
                keywords=[disease] + (list(syns)[:2] if syns else []),
                category="disease_knowledge",
                difficulty="easy",
                source="static_definition"
            )
            if self._add_case(case):
                generated += 1
            else:
                dedup += 1
        ms = (time.perf_counter() - t0) * 1000
        status = "success" if generated > 0 else "failed"
        self.pipeline.add_phase("定义型QA", status, ms,
                                input_count=len(self.static.diseases),
                                output_count=generated)
        self._print_phase_result("定义型QA", status, ms, generated, len(self.cases), dedup)

    # ── Phase 4: ICD-10 编码 QA ──

    def _phase_icd10_qa(self):
        t0 = time.perf_counter()
        generated = 0
        dedup = 0
        for disease, code in self.icd10_map.items():
            if generated >= self.max_per_relation:
                break
            case = GoldenCase(
                question=f"{disease}的ICD-10编码是什么？",
                reference_answer=f"{disease}的ICD-10编码为{code}。这是世界卫生组织制定的国际疾病分类标准编码，用于医疗统计和医保报销。",
                expected_intent="examination_query",
                expected_entities=[disease],
                keywords=[disease, code, "ICD-10"],
                category="medical_coding",
                difficulty="medium",
                source="icd10"
            )
            if self._add_case(case):
                generated += 1
            else:
                dedup += 1
        ms = (time.perf_counter() - t0) * 1000
        status = "success" if generated > 0 else "failed"
        self.pipeline.add_phase("ICD-10编码QA", status, ms,
                                input_count=len(self.icd10_map),
                                output_count=generated)
        self._print_phase_result("ICD-10编码QA", status, ms, generated, len(self.cases), dedup)

    # ── Phase 5: 缩写 QA ──

    def _phase_abbreviation_qa(self):
        t0 = time.perf_counter()
        generated = 0
        dedup = 0
        skipped = 0
        for abbr, full in self.abbreviation_map.items():
            if generated >= self.max_per_relation:
                break
            if len(abbr) <= 1 or len(abbr) > 6:
                skipped += 1
                continue
            forbidden = ["没听过", "不重要"]
            case = GoldenCase(
                question=f"医学缩写\"{abbr}\"是什么意思？",
                reference_answer=f"在医学领域，\"{abbr}\"通常表示{full}。这是临床常用的医学缩写。",
                expected_intent="examination_query",
                expected_entities=[],
                keywords=[abbr, full],
                category="medical_coding",
                difficulty="easy",
                forbidden_content=forbidden,
                source="abbreviation"
            )
            if self._add_case(case):
                generated += 1
            else:
                dedup += 1
        ms = (time.perf_counter() - t0) * 1000
        status = "success" if generated > 0 else "failed"
        self.pipeline.add_phase("缩写QA", status, ms,
                                input_count=len(self.abbreviation_map) - skipped,
                                output_count=generated,
                                detail=f"跳过{skipped}条(长度过滤)")
        self._print_phase_result("缩写QA", status, ms, generated, len(self.cases), dedup)

    # ── Phase 6: 药物 QA ──

    def _phase_drug_qa(self):
        t0 = time.perf_counter()
        known_drug_disease = {
            "阿司匹林": ["感冒", "发热", "头痛", "心脑血管疾病"],
            "布洛芬": ["发热", "头痛", "关节痛", "肌肉痛"],
            "二甲双胍": ["2型糖尿病"],
            "胰岛素": ["1型糖尿病", "2型糖尿病"],
            "硝苯地平": ["高血压", "心绞痛"],
            "氨氯地平": ["高血压"],
            "美托洛尔": ["高血压", "心绞痛", "心律失常"],
            "华法林": ["房颤", "深静脉血栓", "肺栓塞"],
            "氯吡格雷": ["冠心病", "心肌梗死", "脑梗死"],
            "阿托伐他汀": ["高脂血症", "冠心病"],
            "奥美拉唑": ["胃溃疡", "胃食管反流"],
            "沙丁胺醇": ["哮喘", "支气管痉挛"],
            "阿莫西林": ["细菌感染", "肺炎", "扁桃体炎"],
            "左氧氟沙星": ["细菌感染", "肺炎", "尿路感染"],
            "甲硝唑": ["厌氧菌感染", "牙周炎"],
            "青霉素": ["细菌感染", "肺炎", "咽炎"],
            "头孢曲松": ["细菌感染", "肺炎"],
            "地塞米松": ["炎症", "过敏"],
            "环孢素": ["移植排斥"],
            "肝素": ["血栓", "抗凝"],
            "硝酸甘油": ["心绞痛"],
            "呋塞米": ["水肿", "心力衰竭"],
            "氢氯噻嗪": ["高血压", "水肿"],
            "卡托普利": ["高血压", "心力衰竭"],
        }
        known_side_effects = {
            "阿司匹林": ["胃肠道出血", "恶心", "胃溃疡"],
            "布洛芬": ["胃肠道不适", "恶心", "头晕"],
            "二甲双胍": ["恶心", "腹泻", "乳酸酸中毒"],
            "硝苯地平": ["头痛", "面部潮红", "踝部水肿"],
            "美托洛尔": ["疲劳", "头晕", "心动过缓"],
            "华法林": ["出血", "瘀斑"],
            "阿托伐他汀": ["肌肉酸痛", "肝酶升高"],
            "奥美拉唑": ["头痛", "腹泻", "恶心"],
            "左氧氟沙星": ["恶心", "腹泻", "头痛"],
            "地塞米松": ["血糖升高", "骨质疏松", "免疫抑制"],
        }

        # Sub-phase: 药物用途
        t1 = time.perf_counter()
        gen1, dedup1 = 0, 0
        for drug, diseases in known_drug_disease.items():
            if gen1 >= self.max_per_relation:
                break
            diseases_str = "、".join(diseases)
            drug_cat = self.DRUG_CATEGORIES.get(drug, "药物")
            case = GoldenCase(
                question=f"{drug}主要用于治疗什么疾病？",
                reference_answer=f"{drug}是一种{drug_cat}，主要适用于治疗{diseases_str}。请在医生指导下使用，不可自行调整剂量。",
                expected_intent="drug_query",
                expected_entities=[drug] + diseases[:2],
                keywords=[drug] + diseases[:3],
                category="drug_knowledge",
                difficulty="easy",
                forbidden_content=["自己用药", "不用看医生", "随便吃"],
                source="static_drug_disease"
            )
            if self._add_case(case):
                gen1 += 1
            else:
                dedup1 += 1
        ms1 = (time.perf_counter() - t1) * 1000

        # Sub-phase: 副作用
        t2 = time.perf_counter()
        gen2, dedup2 = 0, 0
        for drug, effects in known_side_effects.items():
            if gen2 >= self.max_per_relation:
                break
            effects_str = "、".join(effects)
            case = GoldenCase(
                question=f"{drug}有哪些副作用？",
                reference_answer=f"{drug}的常见副作用包括{effects_str}。如出现严重不良反应应立即停药并就医。长期用药需定期监测相关指标。",
                expected_intent="drug_query",
                expected_entities=[drug],
                keywords=[drug] + effects[:3],
                category="drug_safety",
                difficulty="medium",
                safety_category="adverse_reaction",
                forbidden_content=["没有副作用", "很安全", "不用担心"],
                source="static_side_effects"
            )
            if self._add_case(case):
                gen2 += 1
            else:
                dedup2 += 1
        ms2 = (time.perf_counter() - t2) * 1000

        ms = ms1 + ms2
        total_gen = gen1 + gen2
        total_dedup = dedup1 + dedup2
        status = "success" if total_gen > 0 else "failed"
        detail = f"用途={gen1}条 副作用={gen2}条"
        self.pipeline.add_phase("药物QA", status, ms,
                                input_count=len(known_drug_disease) + len(known_side_effects),
                                output_count=total_gen, detail=detail)
        self._print_phase_result("药物QA", status, ms, total_gen, len(self.cases), total_dedup)

    # ── Phase 7: 关系型 QA (Neo4j) ──

    def _phase_relation_qa(self):
        t0 = time.perf_counter()
        if not self.relation_data:
            self.pipeline.add_phase("关系型QA", "skipped", 0)
            self._print_phase("⏭️", "关系型QA", "无 Neo4j 关系数据")
            return

        disease_names = list(self.relation_data.keys())
        random.shuffle(disease_names)
        generated = 0
        dedup = 0
        type_counts = {"symptom": 0, "drug": 0, "department": 0, "examination": 0, "complication": 0}

        for disease in disease_names:
            if generated >= self.max_per_relation:
                break
            rel = self.relation_data[disease]

            if rel.get("symptoms") and type_counts["symptom"] < self.max_per_relation // 5:
                s = rel["symptoms"]
                case = GoldenCase(
                    question=f"{disease}有哪些常见症状？",
                    reference_answer=f"{disease}的常见症状包括{'、'.join(s)}。不同患者的症状表现可能有所不同，如有相关症状建议及时就医检查。",
                    expected_intent="diagnosis_assist",
                    expected_entities=[disease],
                    keywords=[disease] + s[:3],
                    category="diagnosis_assist",
                    difficulty="medium",
                    safety_category="early_warning",
                    source="neo4j_symptom"
                )
                if self._add_case(case):
                    generated += 1; type_counts["symptom"] += 1
                else:
                    dedup += 1

            if rel.get("drugs") and type_counts["drug"] < self.max_per_relation // 5:
                dr = rel["drugs"]
                case = GoldenCase(
                    question=f"{disease}可以用什么药治疗？",
                    reference_answer=f"{disease}的常用治疗药物包括{'、'.join(dr)}。具体用药方案应由医生根据病情制定，不可自行用药。",
                    expected_intent="drug_query", expected_entities=[disease] + dr[:1],
                    keywords=[disease] + dr[:3], category="treatment_safety",
                    difficulty="medium", safety_category="drug_safety",
                    forbidden_content=["自己买药", "随便吃"], source="neo4j_drug"
                )
                if self._add_case(case):
                    generated += 1; type_counts["drug"] += 1
                else:
                    dedup += 1

            if rel.get("departments") and type_counts["department"] < self.max_per_relation // 5:
                dept = rel["departments"]
                case = GoldenCase(
                    question=f"{disease}应该挂哪个科室？",
                    reference_answer=f"{disease}应前往医院{'、'.join(dept)}就诊。首诊后医生会根据具体情况进行分诊。如有急症表现应立即前往急诊科。",
                    expected_intent="health_advice", expected_entities=[disease],
                    keywords=[disease] + dept[:2], category="treatment_safety",
                    difficulty="easy", source="neo4j_department"
                )
                if self._add_case(case):
                    generated += 1; type_counts["department"] += 1
                else:
                    dedup += 1

            if rel.get("examinations") and type_counts["examination"] < self.max_per_relation // 5:
                exams = rel["examinations"]
                case = GoldenCase(
                    question=f"{disease}需要做什么检查？",
                    reference_answer=f"{disease}的诊断通常需要结合{'、'.join(exams)}等检查进行综合判断。具体检查项目应由医生根据病情决定。",
                    expected_intent="examination_query", expected_entities=[disease] + exams[:1],
                    keywords=[disease] + exams[:3], category="diagnosis_assist",
                    difficulty="medium", source="neo4j_examination"
                )
                if self._add_case(case):
                    generated += 1; type_counts["examination"] += 1
                else:
                    dedup += 1

            if rel.get("complications") and type_counts["complication"] < self.max_per_relation // 5:
                comps = rel["complications"]
                case = GoldenCase(
                    question=f"{disease}会引起哪些并发症？",
                    reference_answer=f"{disease}如果控制不当，可能引起{'、'.join(comps)}等并发症。定期随访和规范治疗有助于预防并发症的发生。",
                    expected_intent="disease_query", expected_entities=[disease] + comps[:1],
                    keywords=[disease] + comps[:3], category="diagnosis_assist",
                    difficulty="hard", source="neo4j_complication"
                )
                if self._add_case(case):
                    generated += 1; type_counts["complication"] += 1
                else:
                    dedup += 1

        ms = (time.perf_counter() - t0) * 1000
        status = "success" if generated > 0 else "skipped"
        detail = " | ".join(f"{k}={v}" for k, v in type_counts.items() if v > 0)
        self.pipeline.add_phase("关系型QA", status, ms,
                                input_count=len(self.relation_data),
                                output_count=generated, detail=detail)
        self._print_phase_result("关系型QA", status, ms, generated, len(self.cases), dedup)

    # ── Phase Final: 统计 ──

    def _phase_finalize(self):
        t0 = time.perf_counter()
        before = len(self.cases)
        random.shuffle(self.cases)
        after = len(self.cases)

        cat_dist: Dict[str, int] = {}
        intent_dist: Dict[str, int] = {}
        diff_dist: Dict[str, int] = {}
        source_dist: Dict[str, int] = {}
        for c in self.cases:
            cat_dist[c.category] = cat_dist.get(c.category, 0) + 1
            intent_dist[c.expected_intent] = intent_dist.get(c.expected_intent, 0) + 1
            diff_dist[c.difficulty] = diff_dist.get(c.difficulty, 0) + 1
            source_dist[c.source] = source_dist.get(c.source, 0) + 1

        ms = (time.perf_counter() - t0) * 1000

        # 更新 pipeline
        self.pipeline.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pipeline.total_duration_ms = (datetime.strptime(self.pipeline.end_time, "%Y-%m-%d %H:%M:%S") -
                                           datetime.strptime(self.pipeline.start_time, "%Y-%m-%d %H:%M:%S")).total_seconds() * 1000
        self.pipeline.final_stats = {
            "total_cases": after,
            "categories": cat_dist,
            "intents": intent_dist,
            "difficulties": diff_dist,
            "sources": source_dist,
            "shuffled": True,
        }
        self.pipeline.add_phase("统计汇总", "success", ms,
                                input_count=before, output_count=after)

        # 打印汇总
        print()
        print("─" * 60)
        print("  📊 [统计汇总]")
        print(f"    总用例: {after}")
        print(f"    类别分布: {dict(sorted(cat_dist.items()))}")
        print(f"    难度分布: {dict(sorted(diff_dist.items()))}")
        print(f"    来源分布: {dict(sorted(source_dist.items()))}")
        self._print_phase_result("统计汇总", "success", ms, after, after, 0)


# ══════════════════════════════════════════════════════════
# 导出工具
# ══════════════════════════════════════════════════════════

def to_medical_golden_case(case: GoldenCase) -> dict:
    """转换为 MedicalGoldenCase 兼容的字典格式"""
    return {
        "question": case.question,
        "reference_answer": case.reference_answer,
        "expected_intent": case.expected_intent,
        "expected_entities": case.expected_entities,
        "keywords": case.keywords,
        "category": case.category,
        "difficulty": case.difficulty,
        "safety_category": case.safety_category,
        "forbidden_content": case.forbidden_content,
    }


def to_benchmark_item(case: GoldenCase) -> dict:
    """转换为 BenchmarkItem 兼容的字典格式"""
    return {
        "question": case.question,
        "reference_answer": case.reference_answer,
        "expected_intent": case.expected_intent,
        "expected_entities": case.expected_entities,
        "keywords": case.keywords,
        "category": case.category,
        "difficulty": case.difficulty,
    }


def save_as_python(cases: List[GoldenCase], filepath: str):
    """保存为 Python 模块格式（兼容 medical_golden_set.py）"""
    lines = [
        "from dataclasses import dataclass, field",
        "from typing import List",
        "",
        "",
        "@dataclass",
        "class GeneratedGoldenCase:",
        "    question: str",
        "    reference_answer: str",
        "    expected_intent: str",
        "    expected_entities: List[str]",
        "    keywords: List[str] = field(default_factory=list)",
        "    category: str = \"general\"",
        "    difficulty: str = \"medium\"",
        "    safety_category: str = \"general\"",
        "    forbidden_content: List[str] = field(default_factory=list)",
        "",
        "",
        f"GENERATED_GOLDEN_CASES: List[GeneratedGoldenCase] = [",
    ]

    for case in cases:
        lines.append("    GeneratedGoldenCase(")
        lines.append(f'        question="{case.question}",')
        lines.append(f'        reference_answer="{case.reference_answer}",')
        lines.append(f'        expected_intent="{case.expected_intent}",')
        lines.append(f'        expected_entities={case.expected_entities},')
        lines.append(f'        keywords={case.keywords},')
        lines.append(f'        category="{case.category}",')
        lines.append(f'        difficulty="{case.difficulty}",')
        lines.append(f'        safety_category="{case.safety_category}",')
        lines.append(f'        forbidden_content={case.forbidden_content},')
        lines.append("    ),")

    lines.append("]")
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Python 模块已保存: {filepath}")


def save_as_json(cases: List[GoldenCase], filepath: str):
    """保存为 JSON 格式"""
    data = {
        "name": "generated_golden_set",
        "description": "从知识库自动生成的黄金评估集",
        "version": "1.0",
        "total": len(cases),
        "items": [to_medical_golden_case(c) for c in cases]
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存: {filepath}")


def save_markdown_report(cases: List[GoldenCase], filepath: str, pipeline: Optional[PipelineLog] = None):
    """保存 Markdown 格式的报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统计数据
    categories: Dict[str, int] = {}
    intents: Dict[str, int] = {}
    difficulties: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for c in cases:
        categories[c.category] = categories.get(c.category, 0) + 1
        intents[c.expected_intent] = intents.get(c.expected_intent, 0) + 1
        difficulties[c.difficulty] = difficulties.get(c.difficulty, 0) + 1
        sources[c.source] = sources.get(c.source, 0) + 1

    bar = "─" * 60

    with open(filepath, "w", encoding="utf-8") as f:
        # ── Header ──
        f.write("# GRAPHRAG 黄金评估集生成报告\n\n")
        f.write(f"**生成时间**: {now}  \n")
        if pipeline:
            f.write(f"**开始时间**: {pipeline.start_time}  \n")
            f.write(f"**运行模式**: {pipeline.mode}  \n")
        f.write(f"**总用例数**: {len(cases)}  \n")
        f.write(f"{bar}\n\n")

        # ── 流水线日志 ──
        if pipeline and pipeline.phases:
            f.write("## 一、生成流水线\n\n")
            f.write("| 阶段 | 状态 | 耗时(ms) | 输入 | 输出 | 详情 |\n")
            f.write("|------|------|----------|------|------|------|\n")
            for p in pipeline.phases:
                icon = PipelineLog.status_icon(p.status)
                detail = p.detail.replace("|", "/") if p.detail else ""
                f.write(f"| {icon} {p.name} | {p.status} | {p.duration_ms:.1f} | "
                        f"{p.input_count} | {p.output_count} | {detail} |\n")
            f.write(f"| **合计** | **{pipeline.phases[-1].status}** | "
                    f"**{pipeline.total_duration_ms:.1f}** | | **{len(cases)}** | |\n")
            f.write(f"\n{bar}\n\n")

        # ── 知识库统计 ──
        if pipeline:
            f.write("## 二、知识库统计\n\n")
            es = pipeline.entity_stats
            if es:
                f.write("### 实体加载\n\n")
                f.write("| 类型 | 数量 |\n|------|------|\n")
                for k, v in sorted(es.items()):
                    f.write(f"| {k} | {v} |\n")
            cs = pipeline.code_mapping_stats
            if cs:
                f.write("\n### 代码映射\n\n")
                f.write("| 类型 | 数量 |\n|------|------|\n")
                for k, v in sorted(cs.items()):
                    f.write(f"| {k} | {v} |\n")
            ns = pipeline.neo4j_stats
            if ns:
                f.write("\n### Neo4j 关系\n\n")
                f.write("| 指标 | 值 |\n|------|------|\n")
                for k, v in sorted(ns.items()):
                    f.write(f"| {k} | {v} |\n")
            f.write(f"\n{bar}\n\n")

        # ── 数据分布 ──
        f.write("## 三、数据分布\n\n")
        f.write("### 3.1 类别分布\n\n")
        f.write("| 类别 | 数量 | 占比 |\n|------|------|------|\n")
        for cat, n in sorted(categories.items()):
            pct = n / len(cases) * 100
            f.write(f"| {cat} | {n} | {pct:.1f}% |\n")

        f.write("\n### 3.2 意图分布\n\n")
        f.write("| 意图 | 数量 | 占比 |\n|------|------|------|\n")
        for intent, n in sorted(intents.items()):
            pct = n / len(cases) * 100
            f.write(f"| {intent} | {n} | {pct:.1f}% |\n")

        f.write("\n### 3.3 难度分布\n\n")
        f.write("| 难度 | 数量 | 占比 |\n|------|------|------|\n")
        for diff, n in sorted(difficulties.items()):
            pct = n / len(cases) * 100
            f.write(f"| {diff} | {n} | {pct:.1f}% |\n")

        f.write("\n### 3.4 来源分布\n\n")
        f.write("| 来源 | 数量 | 占比 |\n|------|------|------|\n")
        for src, n in sorted(sources.items()):
            pct = n / len(cases) * 100
            f.write(f"| {src} | {n} | {pct:.1f}% |\n")

        f.write(f"\n{bar}\n\n")

        # ── 用例列表 ──
        f.write("## 四、用例列表\n\n")
        for i, c in enumerate(cases, 1):
            f.write(f"### {i}. {c.question}\n\n")
            f.write(f"- **意图**: {c.expected_intent} | **类别**: {c.category} | "
                    f"**难度**: {c.difficulty}  \n")
            f.write(f"- **实体**: {', '.join(c.expected_entities) if c.expected_entities else '无'}  \n")
            f.write(f"- **来源**: {c.source}  \n")
            f.write(f"- **参考回答**: {c.reference_answer}  \n\n")

    print(f"Markdown 报告已保存: {filepath}")


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="黄金评估集生成工具")
    parser.add_argument(
        "--mode", choices=["full", "static"], default="full",
        help="运行模式: full(默认,含Neo4j关系), static(仅静态数据)"
    )
    parser.add_argument(
        "--max-per-relation", type=int, default=30,
        help="每类关系最多生成的 QA 对数"
    )
    parser.add_argument(
        "--output-dir", type=str, default="golden_set",
        help="输出目录 (默认: golden_set/)"
    )
    parser.add_argument(
        "--formats", nargs="+", default=["json", "py", "md"],
        choices=["json", "py", "md"],
        help="输出格式"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成
    generator = GoldenSetGenerator(mode=args.mode, max_per_relation=args.max_per_relation)
    cases = generator.generate()
    pipeline = generator.pipeline

    # 输出 JSON 黄金集
    if "json" in args.formats:
        save_as_json(cases, str(output_dir / "generated_golden.json"))

    # 输出 Python 模块
    if "py" in args.formats:
        save_as_python(cases, str(output_dir / "generated_golden.py"))

    # 输出 Markdown 报告 (含流水线日志)
    if "md" in args.formats:
        save_markdown_report(cases, str(output_dir / "generated_golden_report.md"), pipeline)

    # 输出流水线 JSON 日志
    pipeline_log_path = output_dir / "pipeline_log.json"
    with open(pipeline_log_path, "w", encoding="utf-8") as f:
        json.dump(pipeline.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"流水线日志已保存: {pipeline_log_path}")

    # 汇总
    print("\n" + "=" * 60)
    print(f"生成完成! 共 {len(cases)} 条黄金评估用例")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    print("\n提示: 可使用以下方式集成到评估系统:")
    print("  from evaluation.generated_golden import GENERATED_GOLDEN_CASES")
    print("  evaluator.load_dataset(MyDataset(GENERATED_GOLDEN_CASES))")


if __name__ == "__main__":
    main()
