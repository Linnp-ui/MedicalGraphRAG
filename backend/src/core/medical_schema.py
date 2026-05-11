from enum import Enum
from typing import List, Dict

class MedicalEntityType(str, Enum):
    DISEASE = "Disease"            # 疾病
    SYMPTOM = "Symptom"            # 症状
    DRUG = "Drug"                  # 药物
    EXAMINATION = "Examination"    # 检查/检验
    TREATMENT = "Treatment"        # 治疗
    ANATOMY = "Anatomy"            # 解剖部位
    DEPARTMENT = "Department"      # 科室

class MedicalRelationshipType(str, Enum):
    HAS_SYMPTOM = "HAS_SYMPTOM"    # 疾病-症状: 表现为
    CAUSED_BY = "CAUSED_BY"        # 疾病-病因: 由...引起
    TREATED_BY = "TREATED_BY"      # 疾病-治疗: 治疗方案为
    DRUG_FOR = "DRUG_FOR"          # 药物-适应症: 用于治疗
    SIDE_EFFECT = "SIDE_EFFECT"    # 药物-副作用: 副作用包括
    INDICATES = "INDICATES"        # 症状-提示: 提示可能患有
    PART_OF = "PART_OF"            # 解剖-整体: 属于...部位
    BELONGS_TO = "BELONGS_TO"      # 疾病-科室: 属于...科室

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
        MedicalEntityType.DEPARTMENT: "医院的科室，如：内科、外科、眼科等"
    }
}
