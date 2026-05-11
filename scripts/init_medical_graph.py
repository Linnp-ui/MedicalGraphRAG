import sys
import os

# Add backend/src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.core.neo4j_client import get_neo4j_client
from src.core.medical_schema import MedicalEntityType
from loguru import logger

def init_medical_constraints():
    """初始化医疗知识图谱的约束和索引"""
    client = get_neo4j_client()
    
    logger.info("Starting medical graph initialization...")
    
    # 1. 为所有医疗实体创建唯一性约束（基于名称）
    # 在 Neo4j 中，通常 MERGE 实体时使用名称作为主键
    for entity_type in MedicalEntityType:
        label = entity_type.value
        query = f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
        try:
            client.execute_query(query)
            logger.info(f"Created uniqueness constraint for label: {label}")
        except Exception as e:
            logger.error(f"Failed to create constraint for {label}: {e}")
            
    # 2. 为通用 Entity 标签创建索引（如果 kg_builder 还在用它的话）
    try:
        client.execute_query("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")
        logger.info("Created index for general Entity label")
    except Exception as e:
        logger.error(f"Failed to create index for Entity: {e}")

    # 3. 为 Document 和 Chunk 创建约束
    try:
        client.execute_query("CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        client.execute_query("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        logger.info("Created constraints for Document and Chunk")
    except Exception as e:
        logger.error(f"Failed to create constraints for Document/Chunk: {e}")

    logger.info("Medical graph initialization completed.")

if __name__ == "__main__":
    init_medical_constraints()
