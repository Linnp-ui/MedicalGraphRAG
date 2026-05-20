from enum import Enum
from typing import List, Dict

class MedicalEntityType(str, Enum):
    DISEASE = "Disease"            # 疾病
    SYMPTOM = "Symptom"            # 症状
    DRUG = "Drug"                  # 药物
    EXAMINATION = "Examination"    # 检查/检验
    TREATMENT = "Treatment"        # 治疗
    ANATOMY = "Anatomy"            # 解剖部位
    DEPARTMENT = "Department"        # 科室
    DIAGNOSTIC_CRITERIA = "DiagnosticCriteria"  # 诊断标准
    PROGNOSIS = "Prognosis"        # 预后
    RISK_FACTOR = "RiskFactor"     # 危险因素
    PATHOLOGY = "Pathology"         # 病理

class MedicalRelationshipType(str, Enum):
    HAS_SYMPTOM = "HAS_SYMPTOM"    # 疾病-症状: 表现为
    CAUSED_BY = "CAUSED_BY"        # 疾病-病因: 由...引起
    TREATED_BY = "TREATED_BY"      # 疾病-治疗: 治疗方案为
    DRUG_FOR = "DRUG_FOR"          # 药物-适应症: 用于治疗
    SIDE_EFFECT = "SIDE_EFFECT"    # 药物-副作用: 副作用包括
    INDICATES = "INDICATES"        # 症状-提示: 提示可能患有
    PART_OF = "PART_OF"            # 解剖-整体: 属于...部位
    BELONGS_TO = "BELONGS_TO"      # 疾病-科室: 属于...科室
    COMPLICATED_BY = "COMPLICATED_BY"    # 并发: 由...引起并发症
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # 关联: 与...相关
    PREVENTS = "PREVENTS"          # 预防: 可预防
    DIAGNOSED_BY = "DIAGNOSED_BY"  # 诊断: 通过...诊断
    DIAGNOSTIC_CRITERIA = "DIAGNOSTIC_CRITERIA"  # 诊断标准
    PROGNOSIS = "PROGNOSIS"        # 预后: 预后为
    RISK_FACTOR = "RISK_FACTOR"    # 危险因素: 是...的危险因素
    WARNING_SIGN = "WARNING_SIGN"   # 预警信号: 需要警惕
    DIFFERENTIAL_DIAGNOSIS = "DIFFERENTIAL_DIAGNOSIS"  # 鉴别诊断
    SUGGESTS = "SUGGESTS"          # 提示: 提示可能

MEDICAL_SCHEMA = {
    "entities": [e.value for e in MedicalEntityType],
    "relationships": [r.value for r in MedicalRelationshipType],
    "descriptions": {
        MedicalEntityType.DISEASE: "各种人类疾病、病症，如：高血压、糖尿病、新冠肺炎等",
        MedicalEntityType.SYMPTOM: "患者的主观感受或客观表现，如：咳嗽、发热、头痛等",
        MedicalEntityType.DRUG: "用于预防、治疗、诊断疾病的物质，如：阿司匹林、胰岛素等",
        MedicalEntityType.EXAMINATION: "临床检查、化验项目，如：血常规、CT、MRI等",
        MedicalEntityType.TREATMENT: "非药物类的治疗方法，如：手术、放疗、理疗等",
        MedicalEntityType.ANATOMY: "人体部位或器官，如：肺部、肝脏、神经系统等",
        MedicalEntityType.DEPARTMENT: "医院的科室，如：内科、外科、眼科等",
        MedicalEntityType.DIAGNOSTIC_CRITERIA: "诊断标准或检查指标，如：空腹血糖>7.0mmol/L等",
        MedicalEntityType.PROGNOSIS: "疾病预后信息，如：5年生存率、复发风险等",
        MedicalEntityType.RISK_FACTOR: "疾病危险因素，如：吸烟、肥胖、高血压等",
        MedicalEntityType.PATHOLOGY: "病理特征，如：细胞变性、坏死、增生等",
    },
    "allowed_relations": {
        "Disease": {
            "Symptom": ["HAS_SYMPTOM", "WARNING_SIGN"],
            "Treatment": ["TREATED_BY"],
            "Drug": ["TREATED_BY", "DRUG_FOR"],
            "Department": ["BELONGS_TO"],
            "Anatomy": ["PART_OF"],
            "Disease": ["COMPLICATED_BY", "ASSOCIATED_WITH", "DIFFERENTIAL_DIAGNOSIS"],
            "DiagnosticCriteria": ["DIAGNOSTIC_CRITERIA"],
            "Prognosis": ["PROGNOSIS"],
            "RiskFactor": ["RISK_FACTOR"],
            "Examination": ["DIAGNOSED_BY"],
            "Pathology": ["ASSOCIATED_WITH"],
        },
        "Drug": {
            "Disease": ["DRUG_FOR", "PREVENTS"],
            "Symptom": ["SIDE_EFFECT"],
            "RiskFactor": ["RISK_FACTOR"],
        },
        "Symptom": {
            "Disease": ["INDICATES", "SUGGESTS"],
        },
        "Anatomy": {
            "Disease": ["PART_OF"],
            "Anatomy": ["PART_OF"],
        },
        "RiskFactor": {
            "Disease": ["RISK_FACTOR"],
        },
        "Examination": {
            "Disease": ["DIAGNOSED_BY"],
            "DiagnosticCriteria": ["DIAGNOSTIC_CRITERIA"],
        },
        "Prognosis": {
            "Disease": ["PROGNOSIS"],
        },
        "DiagnosticCriteria": {
            "Disease": ["DIAGNOSTIC_CRITERIA"],
        },
    }
}
