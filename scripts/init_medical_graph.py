import sys
import os

# Add backend/src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.core.neo4j_client import get_neo4j_client
from src.core.medical_schema import MedicalEntityType, MedicalRelationshipType
from src.ingestion.knowledge_fusion import KnowledgeFusionEngine
from src.ingestion.kg_builder import KnowledgeGraphBuilder
from src.ingestion.document_loader import load_document, load_documents_from_directory
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


def load_sample_medical_data():
    """加载示例医疗数据到知识图谱"""
    logger.info("Loading sample medical data...")
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "input", "medical_sample.txt")
    
    if not os.path.exists(data_path):
        logger.warning(f"Sample data file not found at {data_path}")
        return False
    
    try:
        document = load_document(data_path)
        builder = KnowledgeGraphBuilder()
        result = builder.ingest_document(document, extract_entities=True, create_embeddings=True)
        
        logger.info(f"Successfully ingested sample data: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to load sample medical data: {e}")
        return False


def query_medical_graph():
    """查询医疗知识图谱示例"""
    client = get_neo4j_client()
    
    logger.info("Querying medical graph...")
    
    # 1. 查询所有疾病
    try:
        diseases = client.execute_query("MATCH (d:Disease) RETURN d.name AS name")
        logger.info(f"Found {len(diseases)} diseases: {[d['name'] for d in diseases]}")
        
        symptoms = client.execute_query("MATCH (s:Symptom) RETURN s.name AS name")
        logger.info(f"Found {len(symptoms)} symptoms: {[s['name'] for s in symptoms]}")
        
        drugs = client.execute_query("MATCH (dr:Drug) RETURN dr.name AS name")
        logger.info(f"Found {len(drugs)} drugs: {[d['name'] for d in drugs]}")
        
        relations = client.execute_query("MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) RETURN d.name, r.type, s.name")
        logger.info(f"Found {len(relations)} disease-symptom relations")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")


def init_sample_medical_graph():
    """初始化示例医疗图谱初始化"""
    init_medical_constraints()
    success = load_sample_medical_data()
    if success:
        query_medical_graph()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical knowledge graph management")
    parser.add_argument("--init", action="store_true", help="Initialize constraints only")
    parser.add_argument("--load", action="store_true", help="Load sample data")
    parser.add_argument("--query", action="store_true", help="Query the graph")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    
    args = parser.parse_args()
    
    if args.all:
        init_sample_medical_graph()
    elif args.init:
        init_medical_constraints()
    elif args.load:
        load_sample_medical_data()
    elif args.query:
        query_medical_graph()
    else:
        print("Usage: python init_medical_graph.py --all | --init | --load | --query")
