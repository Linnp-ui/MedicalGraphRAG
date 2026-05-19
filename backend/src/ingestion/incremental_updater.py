from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from loguru import logger
from dataclasses import dataclass

from ..core.neo4j_client import get_neo4j_client
from ..ingestion.text_splitter import TextSplitter, SplitStrategy
from ..ingestion.knowledge_fusion import KnowledgeFusionEngine
from .embedding import EmbeddingClient


class UpdateStrategy(str, Enum):
    FULL_REBUILD = "full_rebuild"
    INCREMENTAL = "incremental"
    LAZY_UPDATE = "lazy_update"


@dataclass
class UpdateResult:
    document_id: str
    success: bool
    message: str = ""
    updated_entities: int = 0
    updated_relations: int = 0
    updated_vectors: int = 0


@dataclass
class TextChunk:
    id: str
    document_id: str
    content: str
    index: int = 0


class IncrementalUpdater:
    """增量更新管理器 - 支持文档的增量更新和版本管理"""

    def __init__(self, strategy: UpdateStrategy = UpdateStrategy.INCREMENTAL):
        self.strategy = strategy
        self.neo4j_client = get_neo4j_client()
        self.fusion_engine = KnowledgeFusionEngine()
        self.text_splitter = TextSplitter(strategy=SplitStrategy.SEMANTIC)

    def update_document(self, document_id: str, content: str) -> UpdateResult:
        """增量更新单篇文档"""
        logger.info(f"开始增量更新文档: {document_id}")
        
        try:
            if self.strategy == UpdateStrategy.FULL_REBUILD:
                return self._full_rebuild(document_id, content)

            # 1. 删除旧数据
            self._cleanup_document(document_id)

            # 2. 处理新内容
            chunks = self._process_content(content, document_id)

            # 3. 更新图谱
            entity_count, relation_count = self._update_graph(chunks, document_id)

            # 4. 更新向量
            vector_count = self._update_vectors(chunks)

            # 5. 更新版本记录
            self._update_version_record(document_id)

            logger.info(f"文档 {document_id} 增量更新成功")
            return UpdateResult(
                document_id=document_id,
                success=True,
                message="增量更新成功",
                updated_entities=entity_count,
                updated_relations=relation_count,
                updated_vectors=vector_count
            )

        except Exception as e:
            logger.error(f"文档 {document_id} 更新失败: {e}")
            return UpdateResult(
                document_id=document_id,
                success=False,
                message=str(e)
            )

    def batch_update(self, documents: List[Dict[str, str]]) -> List[UpdateResult]:
        """批量增量更新文档"""
        results = []
        for doc in documents:
            result = self.update_document(doc["document_id"], doc["content"])
            results.append(result)
        return results

    def delete_document(self, document_id: str) -> UpdateResult:
        """删除文档及其相关数据"""
        logger.info(f"删除文档: {document_id}")
        
        try:
            self._cleanup_document(document_id)
            self._delete_version_record(document_id)
            
            return UpdateResult(
                document_id=document_id,
                success=True,
                message="文档删除成功"
            )
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return UpdateResult(
                document_id=document_id,
                success=False,
                message=str(e)
            )

    def _cleanup_document(self, document_id: str):
        """清理文档相关的旧数据"""
        # 删除图谱中该文档创建的节点和关系
        cypher = """
        MATCH (n) WHERE n.source_document = $doc_id
        DETACH DELETE n
        """
        self.neo4j_client.execute_query(cypher, {"doc_id": document_id})

        # 删除向量库中的相关向量（通过Cypher查询）
        cypher = """
        MATCH (c:Chunk) WHERE c.document_id = $doc_id
        DETACH DELETE c
        """
        self.neo4j_client.execute_query(cypher, {"doc_id": document_id})

    def _process_content(self, content: str, document_id: str) -> List[TextChunk]:
        """处理文档内容，生成文本块"""
        chunks = self.text_splitter.split(content)
        
        return [
            TextChunk(
                id=f"{document_id}_{i}",
                document_id=document_id,
                content=chunk,
                index=i
            )
            for i, chunk in enumerate(chunks)
        ]

    def _update_graph(self, chunks: List[TextChunk], document_id: str) -> Tuple[int, int]:
        """更新知识图谱"""
        entity_count = 0
        relation_count = 0

        for chunk in chunks:
            entities = self.fusion_engine.extract_entities(chunk.content)
            relations = self.fusion_engine.extract_relations(chunk.content)

            # 更新实体
            for entity in entities:
                self._upsert_entity(entity, document_id)
                entity_count += 1

            # 更新关系
            for relation in relations:
                if self._upsert_relation(relation):
                    relation_count += 1

        return entity_count, relation_count

    def _upsert_entity(self, entity: Dict[str, Any], document_id: str):
        """插入或更新实体"""
        label = entity.get("label", "Entity")
        name = entity.get("name", "")
        
        cypher = f"""
        MERGE (e:{label} {{name: $name}})
        ON CREATE SET 
            e.created_at = $timestamp,
            e.source_document = $doc_id,
            e.version = 1,
            e.confidence = $confidence
        ON MATCH SET 
            e.updated_at = $timestamp,
            e.version = e.version + 1,
            e.source_document = $doc_id,
            e.confidence = $confidence
        """

        self.neo4j_client.execute_query(cypher, {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "doc_id": document_id,
            "confidence": entity.get("confidence", 1.0)
        })

    def _upsert_relation(self, relation: Dict[str, Any]) -> bool:
        """插入或更新关系"""
        try:
            source_label = relation.get("source", {}).get("label", "Entity")
            source_name = relation.get("source", {}).get("name", "")
            target_label = relation.get("target", {}).get("label", "Entity")
            target_name = relation.get("target", {}).get("name", "")
            relation_type = relation.get("type", "RELATED_TO")

            cypher = f"""
            MATCH (source:{source_label} {{name: $source_name}})
            MATCH (target:{target_label} {{name: $target_name}})
            MERGE (source)-[r:{relation_type}]->(target)
            ON CREATE SET r.created_at = $timestamp, r.confidence = $confidence
            ON MATCH SET r.updated_at = $timestamp, r.confidence = $confidence
            """

            self.neo4j_client.execute_query(cypher, {
                "source_name": source_name,
                "target_name": target_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": relation.get("confidence", 1.0)
            })
            return True
        except Exception as e:
            logger.warning(f"创建关系失败: {e}")
            return False

    def _update_vectors(self, chunks: List[TextChunk]) -> int:
        """更新向量库"""
        embedding_client = EmbeddingClient()
        
        for chunk in chunks:
            embedding = embedding_client.embed_text(chunk.content)
            
            # 同时创建Chunk节点存储向量和元数据
            cypher = """
            CREATE (c:Chunk {
                id: $chunk_id,
                document_id: $doc_id,
                content: $content,
                index: $index,
                embedding: $embedding,
                created_at: $timestamp
            })
            """
            self.neo4j_client.execute_query(cypher, {
                "chunk_id": chunk.id,
                "doc_id": chunk.document_id,
                "content": chunk.content,
                "index": chunk.index,
                "embedding": embedding,
                "timestamp": datetime.now().isoformat()
            })
        
        return len(chunks)

    def _update_version_record(self, document_id: str):
        """更新版本记录"""
        cypher = """
        MERGE (d:DocumentVersion {document_id: $doc_id})
        ON CREATE SET 
            d.version = 1,
            d.created_at = $timestamp,
            d.updated_at = $timestamp,
            d.status = 'active'
        ON MATCH SET 
            d.version = d.version + 1,
            d.updated_at = $timestamp
        """

        self.neo4j_client.execute_query(cypher, {
            "doc_id": document_id,
            "timestamp": datetime.now().isoformat()
        })

    def _delete_version_record(self, document_id: str):
        """删除版本记录"""
        cypher = """
        MATCH (d:DocumentVersion {document_id: $doc_id})
        DELETE d
        """
        self.neo4j_client.execute_query(cypher, {"doc_id": document_id})

    def _full_rebuild(self, document_id: str, content: str) -> UpdateResult:
        """全量重建模式"""
        logger.info(f"使用全量重建模式更新文档: {document_id}")
        self._cleanup_document(document_id)
        return self.update_document(document_id, content)

    def get_document_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档版本信息"""
        cypher = """
        MATCH (d:DocumentVersion {document_id: $doc_id})
        RETURN d.document_id as document_id, d.version as version, 
               d.created_at as created_at, d.updated_at as updated_at, d.status as status
        """
        
        results = self.neo4j_client.execute_query(cypher, {"doc_id": document_id})
        
        if results:
            return {
                "document_id": results[0].get("document_id"),
                "version": results[0].get("version"),
                "created_at": results[0].get("created_at"),
                "updated_at": results[0].get("updated_at"),
                "status": results[0].get("status")
            }
        return None


def create_incremental_updater(strategy: str = "incremental") -> IncrementalUpdater:
    """创建增量更新器的便捷函数"""
    strategy_enum = UpdateStrategy(strategy.lower())
    return IncrementalUpdater(strategy=strategy_enum)