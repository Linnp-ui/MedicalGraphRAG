# 标准术语库接入实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 ICD-10 疾病编码库和 DrugBank 药物信息库，为医疗知识图谱提供标准化术语基础。

**Architecture:** 创建独立的 terminology 模块，包含数据下载器、解析器、映射器和服务层。通过批量导入方式将术语数据存入 Neo4j，并扩展现有 knowledge_fusion.py 的映射能力。

**Tech Stack:** Python, Neo4j, requests, xml.etree.ElementTree, pytest

---

## File Structure

```
backend/src/terminology/
├── __init__.py           # 模块初始化
├── models.py             # 数据模型定义
├── downloader.py         # 数据下载器
├── parser.py             # 数据解析器
├── mapper.py             # 术语映射器
└── service.py            # 术语服务 API

backend/tests/
├── test_terminology_models.py
├── test_terminology_parser.py
├── test_terminology_mapper.py
└── test_terminology_service.py

data/terminology/
├── icd10/                # ICD-10 数据文件
└── drugbank/             # DrugBank 数据文件
```

---

### Task 1: 创建术语模块结构和数据模型

**Files:**
- Create: `backend/src/terminology/__init__.py`
- Create: `backend/src/terminology/models.py`
- Create: `backend/tests/test_terminology_models.py`

- [ ] **Step 1: 创建模块目录和初始化文件**

```python
# backend/src/terminology/__init__.py
from .models import ICD10Code, Drug, DrugInteraction
from .downloader import TerminologyDownloader
from .parser import ICD10Parser, DrugBankParser
from .mapper import TerminologyMapper
from .service import TerminologyService

__all__ = [
    "ICD10Code",
    "Drug",
    "DrugInteraction",
    "TerminologyDownloader",
    "ICD10Parser",
    "DrugBankParser",
    "TerminologyMapper",
    "TerminologyService",
]
```

- [ ] **Step 2: 创建数据模型**

```python
# backend/src/terminology/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ICD10Category(str, Enum):
    """ICD-10 章节分类"""
    INFECTIOUS = "I"          # 某些传染病和寄生虫病
    NEOPLASMS = "II"          # 肿瘤
    BLOOD = "III"             # 血液及造血器官疾病
    ENDOCRINE = "IV"          # 内分泌、营养和代谢疾病
    MENTAL = "V"              # 精神和行为障碍
    NERVOUS = "VI"            # 神经系统疾病
    EYE = "VII"               # 眼和附器疾病
    EAR = "VIII"              # 耳和乳突疾病
    CIRCULATORY = "IX"        # 循环系统疾病
    RESPIRATORY = "X"         # 呼吸系统疾病
    DIGESTIVE = "XI"          # 消化系统疾病
    SKIN = "XII"              # 皮肤和皮下组织疾病
    MUSCULOSKELETAL = "XIII"  # 肌肉骨骼系统和结缔组织疾病
    GENITOURINARY = "XIV"     # 泌尿生殖系统疾病
    PREGNANCY = "XV"          # 妊娠、分娩和产褥期
    PERINATAL = "XVI"         # 起源于围生期的某些情况
    CONGENITAL = "XVII"       # 先天性畸形、变形和染色体异常
    SYMPTOMS = "XVIII"        # 症状、体征和临床与实验室异常所见
    INJURY = "XIX"            # 损伤、中毒和外因的某些其他后果
    EXTERNAL = "XX"           # 疾病和死亡的外因
    HEALTH_FACTORS = "XXI"    # 影响健康状态和与保健机构接触的因素


@dataclass
class ICD10Code:
    """ICD-10 编码数据模型"""
    code: str                              # ICD-10 编码 (如 "I10")
    name: str                              # 中文名称
    name_en: Optional[str] = None          # 英文名称
    category: Optional[str] = None         # 疾病类别
    chapter: Optional[str] = None          # 章节
    block: Optional[str] = None            # 编码块
    synonyms: List[str] = field(default_factory=list)  # 同义词
    description: Optional[str] = None      # 详细描述
    related_codes: List[str] = field(default_factory=list)  # 相关编码
    parent_code: Optional[str] = None      # 父编码
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category,
            "chapter": self.chapter,
            "block": self.block,
            "synonyms": self.synonyms,
            "description": self.description,
            "related_codes": self.related_codes,
            "parent_code": self.parent_code,
        }


@dataclass
class DrugInteraction:
    """药物相互作用"""
    drug_name: str                         # 相互作用药物名称
    drugbank_id: Optional[str] = None      # 相互作用药物 DrugBank ID
    description: str = ""                  # 相互作用描述
    severity: Optional[str] = None         # 严重程度 (mild/moderate/severe)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_name": self.drug_name,
            "drugbank_id": self.drugbank_id,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class Drug:
    """DrugBank 药物数据模型"""
    drugbank_id: str                       # DrugBank ID
    name: str                              # 通用名
    name_cn: Optional[str] = None          # 中文名
    cas_number: Optional[str] = None       # CAS 编号
    atc_code: Optional[str] = None         # ATC 编码
    formula: Optional[str] = None          # 分子式
    weight: Optional[str] = None           # 分子量
    indications: List[str] = field(default_factory=list)      # 适应症
    contraindications: List[str] = field(default_factory=list) # 禁忌症
    side_effects: List[str] = field(default_factory=list)     # 副作用
    interactions: List[DrugInteraction] = field(default_factory=list)  # 药物相互作用
    synonyms: List[str] = field(default_factory=list)          # 同义词
    description: Optional[str] = None      # 描述
    
    def to_dict(self) -> Dict[str, Any]:
        return {
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
```

- [ ] **Step 3: 创建测试文件**

```python
# backend/tests/test_terminology_models.py
import pytest
from src.terminology.models import ICD10Code, Drug, DrugInteraction, ICD10Category


class TestICD10Code:
    def test_create_icd10_code(self):
        code = ICD10Code(
            code="I10",
            name="高血压",
            name_en="Hypertension",
            category="循环系统疾病",
            chapter="IX",
        )
        assert code.code == "I10"
        assert code.name == "高血压"
        assert code.name_en == "Hypertension"
        
    def test_icd10_code_to_dict(self):
        code = ICD10Code(
            code="I10",
            name="高血压",
            synonyms=["原发性高血压", "高血压病"],
        )
        result = code.to_dict()
        assert result["code"] == "I10"
        assert result["name"] == "高血压"
        assert "原发性高血压" in result["synonyms"]


class TestDrug:
    def test_create_drug(self):
        drug = Drug(
            drugbank_id="DB00222",
            name="Glimepiride",
            name_cn="格列美脲",
            atc_code="A10BB12",
        )
        assert drug.drugbank_id == "DB00222"
        assert drug.name == "Glimepiride"
        
    def test_drug_with_interaction(self):
        interaction = DrugInteraction(
            drug_name="Aspirin",
            description="可能增加低血糖风险",
            severity="moderate",
        )
        drug = Drug(
            drugbank_id="DB00222",
            name="Glimepiride",
            interactions=[interaction],
        )
        assert len(drug.interactions) == 1
        assert drug.interactions[0].drug_name == "Aspirin"
        
    def test_drug_to_dict(self):
        drug = Drug(
            drugbank_id="DB00222",
            name="Glimepiride",
            indications=["2型糖尿病"],
        )
        result = drug.to_dict()
        assert result["drugbank_id"] == "DB00222"
        assert "2型糖尿病" in result["indications"]
```

- [ ] **Step 4: 运行测试验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_terminology_models.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add backend/src/terminology/ backend/tests/test_terminology_models.py
git commit -m "feat: 创建术语模块结构和数据模型"
```

---

### Task 2: 实现 ICD-10 数据解析器

**Files:**
- Create: `backend/src/terminology/parser.py`
- Create: `backend/tests/test_terminology_parser.py`
- Create: `data/terminology/icd10/sample_icd10.json`

- [ ] **Step 1: 创建 ICD-10 示例数据**

```json
// data/terminology/icd10/sample_icd10.json
{
  "codes": [
    {
      "code": "I10",
      "name": "高血压",
      "name_en": "Essential (primary) hypertension",
      "category": "循环系统疾病",
      "chapter": "IX",
      "block": "I10-I15",
      "synonyms": ["原发性高血压", "高血压病"],
      "description": "特发性高血压",
      "related_codes": ["I11", "I12", "I13", "I15"]
    },
    {
      "code": "E11",
      "name": "2型糖尿病",
      "name_en": "Type 2 diabetes mellitus",
      "category": "内分泌、营养和代谢疾病",
      "chapter": "IV",
      "block": "E10-E14",
      "synonyms": ["非胰岛素依赖型糖尿病", "成人发病型糖尿病"],
      "description": "非胰岛素依赖型糖尿病",
      "related_codes": ["E10", "E12", "E13", "E14"]
    },
    {
      "code": "J18",
      "name": "肺炎",
      "name_en": "Pneumonia",
      "category": "呼吸系统疾病",
      "chapter": "X",
      "block": "J12-J18",
      "synonyms": ["肺部感染", "肺部炎症"],
      "description": "肺炎，病原体未特指"
    }
  ]
}
```

- [ ] **Step 2: 实现 ICD-10 解析器**

```python
# backend/src/terminology/parser.py
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from .models import ICD10Code, Drug, DrugInteraction


class ICD10Parser:
    """ICD-10 数据解析器
    
    支持多种数据格式：
    - JSON: 官方或自定义格式
    - XML: WHO 官方格式
    - CSV: 简化格式
    """
    
    def __init__(self):
        self._parsed_count = 0
        self._error_count = 0
    
    def parse(self, file_path: str) -> List[ICD10Code]:
        """解析 ICD-10 数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            ICD10Code 列表
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"ICD-10 data file not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return self._parse_json(file_path)
        elif suffix == ".xml":
            return self._parse_xml(file_path)
        elif suffix == ".csv":
            return self._parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _parse_json(self, file_path: str) -> List[ICD10Code]:
        """解析 JSON 格式"""
        codes = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        code_list = data.get("codes", data) if isinstance(data, dict) else data
        
        for item in code_list:
            try:
                code = ICD10Code(
                    code=item.get("code", ""),
                    name=item.get("name", ""),
                    name_en=item.get("name_en"),
                    category=item.get("category"),
                    chapter=item.get("chapter"),
                    block=item.get("block"),
                    synonyms=item.get("synonyms", []),
                    description=item.get("description"),
                    related_codes=item.get("related_codes", []),
                    parent_code=item.get("parent_code"),
                )
                codes.append(code)
                self._parsed_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse ICD-10 code: {item}, error: {e}")
                self._error_count += 1
        
        logger.info(f"Parsed {self._parsed_count} ICD-10 codes, {self._error_count} errors")
        return codes
    
    def _parse_xml(self, file_path: str) -> List[ICD10Code]:
        """解析 XML 格式（WHO 官方格式）"""
        codes = []
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for code_elem in root.findall(".//Class"):
            try:
                code = ICD10Code(
                    code=code_elem.get("code", ""),
                    name=code_elem.findtext("Rubric/Label", ""),
                    name_en=code_elem.findtext("Rubric/Label[@lang='en']", ""),
                )
                codes.append(code)
                self._parsed_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse XML element: {e}")
                self._error_count += 1
        
        return codes
    
    def _parse_csv(self, file_path: str) -> List[ICD10Code]:
        """解析 CSV 格式"""
        import csv
        codes = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    synonyms = row.get("synonyms", "").split("|") if row.get("synonyms") else []
                    code = ICD10Code(
                        code=row.get("code", ""),
                        name=row.get("name", ""),
                        name_en=row.get("name_en"),
                        category=row.get("category"),
                        chapter=row.get("chapter"),
                        synonyms=synonyms,
                    )
                    codes.append(code)
                    self._parsed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse CSV row: {row}, error: {e}")
                    self._error_count += 1
        
        return codes
    
    def get_stats(self) -> Dict[str, int]:
        """获取解析统计"""
        return {
            "parsed": self._parsed_count,
            "errors": self._error_count,
        }


class DrugBankParser:
    """DrugBank 数据解析器
    
    支持格式：
    - XML: DrugBank 官方格式
    - JSON: 转换后的格式
    """
    
    def __init__(self):
        self._parsed_count = 0
        self._error_count = 0
    
    def parse(self, file_path: str) -> List[Drug]:
        """解析 DrugBank 数据文件"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"DrugBank data file not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return self._parse_json(file_path)
        elif suffix == ".xml":
            return self._parse_xml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _parse_json(self, file_path: str) -> List[Drug]:
        """解析 JSON 格式"""
        drugs = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        drug_list = data.get("drugs", data) if isinstance(data, dict) else data
        
        for item in drug_list:
            try:
                interactions = []
                for inter in item.get("interactions", []):
                    interactions.append(DrugInteraction(
                        drug_name=inter.get("drug_name", ""),
                        drugbank_id=inter.get("drugbank_id"),
                        description=inter.get("description", ""),
                        severity=inter.get("severity"),
                    ))
                
                drug = Drug(
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
                drugs.append(drug)
                self._parsed_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse drug: {item}, error: {e}")
                self._error_count += 1
        
        logger.info(f"Parsed {self._parsed_count} drugs, {self._error_count} errors")
        return drugs
    
    def _parse_xml(self, file_path: str) -> List[Drug]:
        """解析 XML 格式（DrugBank 官方格式）"""
        drugs = []
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        ns = {"db": "http://drugbank.ca"}
        
        for drug_elem in root.findall(".//db:drug", ns):
            try:
                drugbank_id = drug_elem.findtext("db:drugbank-id", "", ns)
                name = drug_elem.findtext("db:name", "", ns)
                
                indications = []
                for ind in drug_elem.findall(".//db:indication", ns):
                    if ind.text:
                        indications.append(ind.text)
                
                interactions = []
                for inter in drug_elem.findall(".//db:drug-interaction", ns):
                    interactions.append(DrugInteraction(
                        drug_name=inter.findtext("db:name", "", ns),
                        description=inter.findtext("db:description", "", ns),
                    ))
                
                drug = Drug(
                    drugbank_id=drugbank_id,
                    name=name,
                    indications=indications,
                    interactions=interactions,
                )
                drugs.append(drug)
                self._parsed_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse drug XML: {e}")
                self._error_count += 1
        
        return drugs
    
    def get_stats(self) -> Dict[str, int]:
        """获取解析统计"""
        return {
            "parsed": self._parsed_count,
            "errors": self._error_count,
        }
```

- [ ] **Step 3: 创建解析器测试**

```python
# backend/tests/test_terminology_parser.py
import pytest
from pathlib import Path
from src.terminology.parser import ICD10Parser, DrugBankParser
from src.terminology.models import ICD10Code, Drug


class TestICD10Parser:
    def test_parse_json(self):
        parser = ICD10Parser()
        file_path = Path(__file__).parent.parent.parent / "data" / "terminology" / "icd10" / "sample_icd10.json"
        
        if file_path.exists():
            codes = parser.parse(str(file_path))
            assert len(codes) > 0
            assert codes[0].code == "I10"
            assert codes[0].name == "高血压"
    
    def test_parse_nonexistent_file(self):
        parser = ICD10Parser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.json")
    
    def test_parse_unsupported_format(self):
        parser = ICD10Parser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse("test.xyz")
    
    def test_get_stats(self):
        parser = ICD10Parser()
        parser._parsed_count = 10
        parser._error_count = 2
        stats = parser.get_stats()
        assert stats["parsed"] == 10
        assert stats["errors"] == 2


class TestDrugBankParser:
    def test_parse_json(self):
        parser = DrugBankParser()
        # 测试空列表
        drugs = parser._parse_json_from_data({"drugs": []})
        assert len(drugs) == 0
    
    def test_parse_nonexistent_file(self):
        parser = DrugBankParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.xml")
```

- [ ] **Step 4: 运行测试验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_terminology_parser.py -v`
Expected: Tests pass

- [ ] **Step 5: 提交**

```bash
git add backend/src/terminology/parser.py backend/tests/test_terminology_parser.py data/terminology/
git commit -m "feat: 实现 ICD-10 和 DrugBank 数据解析器"
```

---

### Task 3: 实现术语映射器

**Files:**
- Create: `backend/src/terminology/mapper.py`
- Create: `backend/tests/test_terminology_mapper.py`

- [ ] **Step 1: 实现术语映射器**

```python
# backend/src/terminology/mapper.py
from typing import List, Dict, Optional, Any, Set
from loguru import logger
from dataclasses import dataclass, field

from .models import ICD10Code, Drug


@dataclass
class MappingResult:
    """映射结果"""
    original_name: str
    standard_name: str
    terminology: str
    code: Optional[str] = None
    confidence: float = 1.0
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


class TerminologyMapper:
    """术语映射器
    
    将标准术语库映射到现有实体系统：
    - ICD-10 编码 → 疾病实体
    - DrugBank 药物 → 药物实体
    """
    
    def __init__(self):
        self._icd10_index: Dict[str, ICD10Code] = {}
        self._icd10_name_index: Dict[str, str] = {}
        self._drug_index: Dict[str, Drug] = {}
        self._drug_name_index: Dict[str, str] = {}
    
    def load_icd10_codes(self, codes: List[ICD10Code]) -> int:
        """加载 ICD-10 编码到索引
        
        Args:
            codes: ICD-10 编码列表
            
        Returns:
            加载的编码数量
        """
        count = 0
        for code in codes:
            self._icd10_index[code.code] = code
            self._icd10_name_index[code.name.lower()] = code.code
            
            for synonym in code.synonyms:
                self._icd10_name_index[synonym.lower()] = code.code
            
            count += 1
        
        logger.info(f"Loaded {count} ICD-10 codes into mapper")
        return count
    
    def load_drugs(self, drugs: List[Drug]) -> int:
        """加载药物到索引
        
        Args:
            drugs: 药物列表
            
        Returns:
            加载的药物数量
        """
        count = 0
        for drug in drugs:
            self._drug_index[drug.drugbank_id] = drug
            self._drug_name_index[drug.name.lower()] = drug.drugbank_id
            
            if drug.name_cn:
                self._drug_name_index[drug.name_cn.lower()] = drug.drugbank_id
            
            for synonym in drug.synonyms:
                self._drug_name_index[synonym.lower()] = drug.drugbank_id
            
            count += 1
        
        logger.info(f"Loaded {count} drugs into mapper")
        return count
    
    def map_to_icd10(self, name: str) -> Optional[MappingResult]:
        """将名称映射到 ICD-10 编码
        
        Args:
            name: 疾病名称或同义词
            
        Returns:
            映射结果，未找到返回 None
        """
        name_lower = name.lower()
        
        if name_lower not in self._icd10_name_index:
            return None
        
        code_value = self._icd10_name_index[name_lower]
        icd10_code = self._icd10_index.get(code_value)
        
        if not icd10_code:
            return None
        
        return MappingResult(
            original_name=name,
            standard_name=icd10_code.name,
            terminology="ICD-10",
            code=icd10_code.code,
            confidence=1.0,
            synonyms=icd10_code.synonyms,
            properties={
                "name_en": icd10_code.name_en,
                "category": icd10_code.category,
                "chapter": icd10_code.chapter,
            }
        )
    
    def map_to_drug(self, name: str) -> Optional[MappingResult]:
        """将名称映射到 DrugBank 药物
        
        Args:
            name: 药物名称或同义词
            
        Returns:
            映射结果，未找到返回 None
        """
        name_lower = name.lower()
        
        if name_lower not in self._drug_name_index:
            return None
        
        drugbank_id = self._drug_name_index[name_lower]
        drug = self._drug_index.get(drugbank_id)
        
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
                "atc_code": drug.atc_code,
                "cas_number": drug.cas_number,
                "indications": drug.indications,
            }
        )
    
    def expand_synonyms(self, name: str, entity_type: str) -> List[str]:
        """扩展实体同义词
        
        Args:
            name: 实体名称
            entity_type: 实体类型
            
        Returns:
            同义词列表
        """
        synonyms = [name]
        
        if entity_type == "Disease":
            result = self.map_to_icd10(name)
            if result:
                synonyms.extend(result.synonyms)
        elif entity_type == "Drug":
            result = self.map_to_drug(name)
            if result:
                synonyms.extend(result.synonyms)
        
        return list(set(synonyms))
    
    def get_all_icd10_codes(self) -> List[ICD10Code]:
        """获取所有 ICD-10 编码"""
        return list(self._icd10_index.values())
    
    def get_all_drugs(self) -> List[Drug]:
        """获取所有药物"""
        return list(self._drug_index.values())
    
    def get_stats(self) -> Dict[str, int]:
        """获取映射器统计"""
        return {
            "icd10_codes": len(self._icd10_index),
            "icd10_names": len(self._icd10_name_index),
            "drugs": len(self._drug_index),
            "drug_names": len(self._drug_name_index),
        }
```

- [ ] **Step 2: 创建映射器测试**

```python
# backend/tests/test_terminology_mapper.py
import pytest
from src.terminology.mapper import TerminologyMapper, MappingResult
from src.terminology.models import ICD10Code, Drug


class TestTerminologyMapper:
    def test_load_icd10_codes(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压"]),
            ICD10Code(code="E11", name="2型糖尿病", synonyms=["非胰岛素依赖型糖尿病"]),
        ]
        
        count = mapper.load_icd10_codes(codes)
        assert count == 2
        assert len(mapper._icd10_index) == 2
    
    def test_map_to_icd10(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压", "高血压病"]),
        ]
        mapper.load_icd10_codes(codes)
        
        result = mapper.map_to_icd10("高血压")
        assert result is not None
        assert result.standard_name == "高血压"
        assert result.code == "I10"
        
        result2 = mapper.map_to_icd10("原发性高血压")
        assert result2 is not None
        assert result2.standard_name == "高血压"
    
    def test_map_to_icd10_not_found(self):
        mapper = TerminologyMapper()
        result = mapper.map_to_icd10("不存在的疾病")
        assert result is None
    
    def test_load_drugs(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(drugbank_id="DB00222", name="Glimepiride", name_cn="格列美脲"),
        ]
        
        count = mapper.load_drugs(drugs)
        assert count == 1
    
    def test_map_to_drug(self):
        mapper = TerminologyMapper()
        drugs = [
            Drug(
                drugbank_id="DB00222",
                name="Glimepiride",
                name_cn="格列美脲",
                synonyms=["Amaryl"],
            ),
        ]
        mapper.load_drugs(drugs)
        
        result = mapper.map_to_drug("格列美脲")
        assert result is not None
        assert result.standard_name == "Glimepiride"
        assert result.code == "DB00222"
        
        result2 = mapper.map_to_drug("Amaryl")
        assert result2 is not None
    
    def test_expand_synonyms(self):
        mapper = TerminologyMapper()
        codes = [
            ICD10Code(code="I10", name="高血压", synonyms=["原发性高血压", "高血压病"]),
        ]
        mapper.load_icd10_codes(codes)
        
        synonyms = mapper.expand_synonyms("高血压", "Disease")
        assert "高血压" in synonyms
        assert "原发性高血压" in synonyms
    
    def test_get_stats(self):
        mapper = TerminologyMapper()
        codes = [ICD10Code(code="I10", name="高血压")]
        drugs = [Drug(drugbank_id="DB001", name="Aspirin")]
        
        mapper.load_icd10_codes(codes)
        mapper.load_drugs(drugs)
        
        stats = mapper.get_stats()
        assert stats["icd10_codes"] == 1
        assert stats["drugs"] == 1
```

- [ ] **Step 3: 运行测试验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_terminology_mapper.py -v`
Expected: 8 passed

- [ ] **Step 4: 提交**

```bash
git add backend/src/terminology/mapper.py backend/tests/test_terminology_mapper.py
git commit -m "feat: 实现术语映射器"
```

---

### Task 4: 实现术语服务 API

**Files:**
- Create: `backend/src/terminology/service.py`
- Create: `backend/tests/test_terminology_service.py`

- [ ] **Step 1: 实现术语服务**

```python
# backend/src/terminology/service.py
from typing import List, Dict, Optional, Any
from loguru import logger
from functools import lru_cache

from .models import ICD10Code, Drug
from .parser import ICD10Parser, DrugBankParser
from .mapper import TerminologyMapper, MappingResult


class TerminologyService:
    """术语查询服务
    
    提供统一的术语查询接口：
    - ICD-10 编码查询
    - DrugBank 药物查询
    - 编码转换
    - 名称搜索
    """
    
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
        self._icd10_codes: List[ICD10Code] = []
        self._drugs: List[Drug] = []
        
        if icd10_data_path:
            self._load_icd10(icd10_data_path)
        
        if drugbank_data_path:
            self._load_drugbank(drugbank_data_path)
        
        self._initialized = True
        logger.info(f"TerminologyService initialized with {len(self._icd10_codes)} ICD-10 codes, {len(self._drugs)} drugs")
    
    def _load_icd10(self, file_path: str) -> int:
        """加载 ICD-10 数据"""
        parser = ICD10Parser()
        self._icd10_codes = parser.parse(file_path)
        return self._mapper.load_icd10_codes(self._icd10_codes)
    
    def _load_drugbank(self, file_path: str) -> int:
        """加载 DrugBank 数据"""
        parser = DrugBankParser()
        self._drugs = parser.parse(file_path)
        return self._mapper.load_drugs(self._drugs)
    
    def lookup_icd10(self, code: str) -> Optional[Dict[str, Any]]:
        """查询 ICD-10 编码
        
        Args:
            code: ICD-10 编码或名称
            
        Returns:
            ICD-10 编码信息
        """
        result = self._mapper.map_to_icd10(code)
        if result:
            return {
                "code": result.code,
                "name": result.standard_name,
                "synonyms": result.synonyms,
                "properties": result.properties,
            }
        
        icd10_code = self._mapper._icd10_index.get(code)
        if icd10_code:
            return icd10_code.to_dict()
        
        return None
    
    def lookup_drug(self, name: str) -> List[Dict[str, Any]]:
        """查询药物信息
        
        Args:
            name: 药物名称或 DrugBank ID
            
        Returns:
            匹配的药物列表
        """
        results = []
        
        result = self._mapper.map_to_drug(name)
        if result:
            drug = self._mapper._drug_index.get(result.code)
            if drug:
                results.append(drug.to_dict())
        
        drug = self._mapper._drug_index.get(name)
        if drug:
            results.append(drug.to_dict())
        
        return results
    
    def search(self, query: str, terminology: str = "all", limit: int = 10) -> Dict[str, List[Dict]]:
        """搜索术语
        
        Args:
            query: 搜索关键词
            terminology: 术语库类型 (icd10/drugbank/all)
            limit: 返回数量限制
            
        Returns:
            搜索结果
        """
        results = {
            "icd10": [],
            "drugbank": [],
        }
        
        query_lower = query.lower()
        
        if terminology in ["icd10", "all"]:
            for code in self._icd10_codes:
                if query_lower in code.name.lower() or query_lower in code.code.lower():
                    results["icd10"].append(code.to_dict())
                    if len(results["icd10"]) >= limit:
                        break
        
        if terminology in ["drugbank", "all"]:
            for drug in self._drugs:
                if query_lower in drug.name.lower() or (drug.name_cn and query_lower in drug.name_cn.lower()):
                    results["drugbank"].append(drug.to_dict())
                    if len(results["drugbank"]) >= limit:
                        break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "icd10_codes": len(self._icd10_codes),
            "drugs": len(self._drugs),
            "mapper_stats": self._mapper.get_stats(),
        }
    
    @classmethod
    def reset(cls):
        """重置单例实例"""
        cls._instance = None
```

- [ ] **Step 2: 创建服务测试**

```python
# backend/tests/test_terminology_service.py
import pytest
from src.terminology.service import TerminologyService
from src.terminology.models import ICD10Code, Drug


class TestTerminologyService:
    def test_singleton(self):
        TerminologyService.reset()
        service1 = TerminologyService()
        service2 = TerminologyService()
        assert service1 is service2
        TerminologyService.reset()
    
    def test_lookup_icd10_not_loaded(self):
        TerminologyService.reset()
        service = TerminologyService()
        result = service.lookup_icd10("I10")
        assert result is None
        TerminologyService.reset()
    
    def test_search_empty(self):
        TerminologyService.reset()
        service = TerminologyService()
        results = service.search("高血压")
        assert results["icd10"] == []
        assert results["drugbank"] == []
        TerminologyService.reset()
    
    def test_get_stats(self):
        TerminologyService.reset()
        service = TerminologyService()
        stats = service.get_stats()
        assert "icd10_codes" in stats
        assert "drugs" in stats
        TerminologyService.reset()
```

- [ ] **Step 3: 运行测试验证**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/test_terminology_service.py -v`
Expected: 4 passed

- [ ] **Step 4: 提交**

```bash
git add backend/src/terminology/service.py backend/tests/test_terminology_service.py
git commit -m "feat: 实现术语服务 API"
```

---

### Task 5: 集成到现有系统

**Files:**
- Modify: `backend/src/ingestion/knowledge_fusion.py`
- Modify: `backend/src/api/routes.py`
- Create: `backend/tests/test_terminology_integration.py`

- [ ] **Step 1: 扩展 knowledge_fusion.py**

在 `EntityDisambiguator` 类中添加术语库支持：

```python
# 在 EntityDisambiguator.__init__ 中添加
from src.terminology.service import TerminologyService

class EntityDisambiguator:
    def __init__(self):
        # ... 现有代码 ...
        
        # 加载术语服务
        try:
            self._terminology_service = TerminologyService()
        except Exception as e:
            logger.warning(f"Failed to load terminology service: {e}")
            self._terminology_service = None
    
    def normalize_name(self, name: str, entity_type: str = "") -> str:
        """规范化名称 - 扩展支持术语库"""
        # 先查术语库
        if self._terminology_service:
            if entity_type == "Disease":
                result = self._terminology_service.lookup_icd10(name)
                if result:
                    return result.get("name", name)
            elif entity_type == "Drug":
                results = self._terminology_service.lookup_drug(name)
                if results:
                    return results[0].get("name", name)
        
        # 现有逻辑
        # ... 保持原有代码 ...
```

- [ ] **Step 2: 添加 API 端点**

在 `routes.py` 中添加术语查询 API：

```python
# backend/src/api/routes.py

from src.terminology.service import TerminologyService

@router.get("/terminology/icd10/{code}")
async def lookup_icd10(code: str):
    """查询 ICD-10 编码"""
    service = TerminologyService()
    result = service.lookup_icd10(code)
    if not result:
        raise HTTPException(status_code=404, detail=f"ICD-10 code not found: {code}")
    return result


@router.get("/terminology/drug/{name}")
async def lookup_drug(name: str):
    """查询药物信息"""
    service = TerminologyService()
    results = service.lookup_drug(name)
    if not results:
        raise HTTPException(status_code=404, detail=f"Drug not found: {name}")
    return results


@router.get("/terminology/search")
async def search_terminology(
    query: str,
    terminology: str = "all",
    limit: int = 10,
):
    """搜索术语"""
    service = TerminologyService()
    return service.search(query, terminology, limit)


@router.get("/terminology/stats")
async def get_terminology_stats():
    """获取术语库统计"""
    service = TerminologyService()
    return service.get_stats()
```

- [ ] **Step 3: 创建集成测试**

```python
# backend/tests/test_terminology_integration.py
import pytest
from src.terminology.service import TerminologyService


class TestTerminologyIntegration:
    def test_service_initialization(self):
        TerminologyService.reset()
        service = TerminologyService()
        stats = service.get_stats()
        assert isinstance(stats, dict)
        TerminologyService.reset()
```

- [ ] **Step 4: 运行完整测试**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 集成术语库到现有系统

- 扩展 knowledge_fusion.py 支持术语库查询
- 添加术语查询 API 端点
- 添加集成测试"
```

---

### Task 6: 最终验证和文档更新

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-20-terminology-integration-design.md`

- [ ] **Step 1: 更新 README.md**

在 README.md 中添加术语库使用说明。

- [ ] **Step 2: 更新设计文档**

标记所有成功标准为已完成。

- [ ] **Step 3: 运行完整测试**

Run: `cd d:\code\project\GRAPHRAG\backend && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "docs: 更新术语库文档和验证"
```

---

## 成功标准

- [ ] ICD-10 解析器支持 JSON/XML/CSV 格式
- [ ] DrugBank 解析器支持 JSON/XML 格式
- [ ] 术语映射器支持名称到编码的双向映射
- [ ] 术语服务提供查询/搜索 API
- [ ] 与现有 knowledge_fusion.py 集成
- [ ] API 端点正常工作
- [ ] 所有测试通过
