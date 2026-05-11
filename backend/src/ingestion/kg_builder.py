import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from ..core.config import get_settings
from ..core.neo4j_client import Neo4jClient
from .document_loader import Document, DocumentLoader
from .text_splitter import TextChunk
from .embedding import EmbeddingClient, get_embedding_client
from ..core.medical_schema import MEDICAL_SCHEMA
from .medical_processor import MedicalTextProcessor


@dataclass
class Entity:
    """Entity data class"""

    name: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Relationship:
    """Relationship data class"""

    source: str
    target: str
    type: str
    properties: Dict[str, Any]


class KnowledgeGraphBuilder:
    """Build knowledge graph from documents"""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.neo4j_client = neo4j_client
        self.embedding_client = embedding_client
        self.medical_processor = MedicalTextProcessor()

    def _get_neo4j_client(self) -> Neo4jClient:
        if self.neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client

            self.neo4j_client = get_neo4j_client()
        return self.neo4j_client

    def _get_embedding_client(self) -> EmbeddingClient:
        if self.embedding_client is None:
            self.embedding_client = get_embedding_client()
        return self.embedding_client

    def ingest_document(
        self,
        document: Document,
        extract_entities: bool = True,
        create_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """将文档摄入到知识图谱

        Args:
            document: 要摄入的文档
            extract_entities: 是否提取实体和关系
            create_embeddings: 是否创建文本块嵌入

        Returns:
            摄入结果，包含文档ID、创建的文本块数、提取的实体数和创建的关系数
        """
        results = {
            "document_id": document.id,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
        }

        # 如果是医疗领域，进行预处理
        settings = get_settings()
        if settings.domain == "medical":
            document = self.medical_processor.process_document(document)

        # 创建文档节点
        self._create_document_node(document)

        # 分割文档为文本块
        from .text_splitter import TextSplitter

        splitter = TextSplitter()
        chunks = splitter.split_text(document.content, document.id)
        results["chunks_created"] = len(chunks)

        # 收集所有实体和关系用于去重
        entity_map: Dict[str, Entity] = {}
        rel_set: set = set()
        chunk_entity_links: List[tuple] = []
        relationships: List[Relationship] = []

        # 先创建文本块节点
        for chunk in chunks:
            self._create_chunk_node(chunk, document.id)
            self._link_document_to_chunk(document.id, chunk.id)

        # 批量创建嵌入（一次API调用处理所有文本块）
        if create_embeddings and chunks:
            self._create_chunk_embeddings_batch(chunks)

        # 按文本块提取实体（必须按文本块提取以保证上下文质量）
        if extract_entities:
            for chunk in chunks:
                chunk_entities, chunk_rels = self._extract_entities_from_chunk(chunk)

                # 按名称（不区分大小写）去重实体
                for entity in chunk_entities:
                    key = entity.name.lower()
                    if key not in entity_map:
                        entity_map[key] = entity
                    else:
                        existing = entity_map[key]
                        existing.properties.update(entity.properties)
                    chunk_entity_links.append((key, chunk.id))

                # 按源实体+目标实体+关系类型去重关系
                for rel in chunk_rels:
                    rel_key = (rel.source.lower(), rel.target.lower(), rel.type)
                    if rel_key not in rel_set:
                        rel_set.add(rel_key)
                        relationships.append(rel)

        # 将去重后的实体写入Neo4j
        for entity in entity_map.values():
            self._create_entity_node(entity)

        # 将去重后的关系写入Neo4j
        for rel in relationships:
            self._create_relationship(rel)

        # 将文本块与实体关联
        for entity_key, chunk_id in chunk_entity_links:
            if entity_key in entity_map:
                self._link_entity_to_chunk(entity_map[entity_key].name, chunk_id)

        results["entities_extracted"] = len(entity_map)
        results["relationships_created"] = len(relationships)

        logger.info(f"Document {document.id} ingested: {results}")
        return results

    def _create_document_node(self, document: Document):
        """Create document node in Neo4j"""
        client = self._get_neo4j_client()
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.content = $content,
            d.metadata = $metadata
        """
        client.execute_query(
            query,
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "metadata": json.dumps(document.metadata),
            },
        )

    def _create_chunk_node(self, chunk: TextChunk, document_id: str):
        """Create chunk node in Neo4j"""
        client = self._get_neo4j_client()
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.content = $content,
            c.document_id = $document_id,
            c.index = $index,
            c.metadata = $metadata
        """
        client.execute_query(
            query,
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "index": chunk.index,
                "metadata": json.dumps(chunk.metadata),
            },
        )

    def _link_document_to_chunk(self, document_id: str, chunk_id: str):
        """Link document to chunk"""
        client = self._get_neo4j_client()
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        client.execute_query(query, {"document_id": document_id, "chunk_id": chunk_id})

    def _extract_entities_from_chunk(
        self, chunk: TextChunk
    ) -> tuple[List[Entity], List[Relationship]]:
        """使用LLM从文本块中提取实体和关系

        Args:
            chunk: 要提取实体和关系的文本块

        Returns:
            提取的实体列表和关系列表
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        import json as _json

        settings = get_settings()

        llm = ChatOpenAI(
            model=settings.extraction_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "https://api.openai.com/v1",
        )

        if settings.domain == "medical":
            system_prompt = f"""你是一个医疗实体关系抽取专家。从给定的文本中提取医疗相关的实体和关系。

请以JSON格式返回，结构如下：
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "源实体", "target": "目标实体", "type": "关系类型", "properties": {{}}}}
  ]
}}

可用的实体类型包括：{", ".join(MEDICAL_SCHEMA['entities'])}
可用的关系类型包括：{", ".join(MEDICAL_SCHEMA['relationships'])}

实体类型说明：
{chr(10).join([f"- {k}: {v}" for k, v in MEDICAL_SCHEMA['descriptions'].items()])}

只提取文本中明确提到的医疗实体和关系。只返回JSON，不要有其他解释文字。"""
        else:
            system_prompt = """你是一个实体关系抽取专家。从给定的文本中提取实体和关系。

请以JSON格式返回，结构如下：
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "源实体", "target": "目标实体", "type": "关系类型", "properties": {{}}}}
  ]
}}

实体类型可以是：Person, Organization, Location, Product, Concept, System, Database, Framework 等
关系类型可以是：USES, HAS, WORKS_AT, LOCATED_IN, RELATED_TO 等
只提取文本中明确提到的实体和关系。只返回JSON，不要有其他解释文字。"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "从以下文本中提取实体和关系：\n\n{text}"),
            ]
        )

        chain = prompt | llm

        try:
            result = chain.invoke({"text": chunk.content})
            content = result.content

            # 解析响应中的JSON
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = _json.loads(json_str)
            else:
                data = {"entities": [], "relationships": []}

            entities = [
                Entity(name=e["name"], type=e["type"], properties=e.get("properties", {}))
                for e in data.get("entities", [])
            ]

            relationships = [
                Relationship(
                    source=r["source"],
                    target=r["target"],
                    type=r["type"],
                    properties=r.get("properties", {}),
                )
                for r in data.get("relationships", [])
            ]

            logger.info(
                f"LLM extracted {len(entities)} entities and {len(relationships)} relationships"
            )
            return entities, relationships

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return [], []

    def _create_entity_node(self, entity: Entity):
        """在Neo4j中创建实体节点（基于名称合并，无需APOC）

        Args:
            entity: 要创建的实体
        """
        client = self._get_neo4j_client()
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.properties = $properties
        ON MATCH SET
            e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END,
            e.properties = CASE
                WHEN e.properties IS NULL THEN $properties
                ELSE e.properties + $merge_props
            END
        """
        client.execute_query(
            query,
            {
                "name": entity.name,
                "type": entity.type,
                "properties": json.dumps(entity.properties),
                "merge_props": json.dumps(entity.properties),
            },
        )

    def _link_entity_to_chunk(self, entity_name: str, chunk_id: str):
        """Link entity to chunk in Neo4j"""
        client = self._get_neo4j_client()
        query = """
        MATCH (e:Entity {name: $entity_name})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (c)-[:CONTAINS_ENTITY]->(e)
        """
        client.execute_query(
            query,
            {"entity_name": entity_name, "chunk_id": chunk_id},
        )

    def _create_relationship(self, relationship: Relationship):
        """在Neo4j中创建关系

        Args:
            relationship: 要创建的关系
        """
        client = self._get_neo4j_client()
        query = """
        MATCH (e1:Entity {name: $source})
        MATCH (e2:Entity {name: $target})
        MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
        ON CREATE SET r.properties = $properties
        """
        client.execute_query(
            query,
            {
                "source": relationship.source,
                "target": relationship.target,
                "rel_type": relationship.type,
                "properties": json.dumps(relationship.properties),
            },
        )

    def _create_chunk_embeddings_batch(self, chunks: List[TextChunk]):
        """批量创建所有文本块的嵌入（单次API调用）

        Args:
            chunks: 要创建嵌入的文本块列表
        """
        client = self._get_neo4j_client()
        embedding_client = self._get_embedding_client()

        texts = [chunk.content for chunk in chunks]
        try:
            embeddings = embedding_client.embed_texts(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed, falling back to serial: {e}")
            # 回退：逐个嵌入
            embeddings = []
            for text in texts:
                try:
                    embeddings.append(embedding_client.embed_text(text))
                except Exception as inner_e:
                    logger.error(f"Single embedding failed: {inner_e}")
                    embeddings.append([])

        # 将嵌入写入Neo4j
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        SET c.embedding = $embedding
        """
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                try:
                    client.execute_query(query, {"chunk_id": chunk.id, "embedding": embedding})
                except Exception as e:
                    logger.error(f"Failed to store embedding for chunk {chunk.id}: {e}")

        logger.info(f"Stored {len([e for e in embeddings if e])} embeddings in batch")

    def create_vector_index(self, index_name: str = "chunk_vector_index"):
        """为文本块创建向量索引（Neo4j 5.x / 2026.x 语法）

        Args:
            index_name: 向量索引名称
        """
        client = self._get_neo4j_client()
        settings = get_settings()
        dim = settings.embedding_dimensions

        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
        """
        try:
            client.execute_query(query)
            logger.info(f"Vector index '{index_name}' created or already exists")
        except Exception as e:
            logger.warning(f"Vector index creation failed: {e}")


def ingest_document(
    document: Document,
    extract_entities: bool = True,
    create_embeddings: bool = True,
) -> Dict[str, Any]:
    """摄入单个文档的便捷函数

    Args:
        document: 要摄入的文档
        extract_entities: 是否提取实体和关系
        create_embeddings: 是否创建文本块嵌入

    Returns:
        摄入结果
    """
    builder = KnowledgeGraphBuilder()
    return builder.ingest_document(document, extract_entities, create_embeddings)


def ingest_documents_from_directory(
    directory: str,
    extract_entities: bool = True,
    create_embeddings: bool = True,
) -> List[Dict[str, Any]]:
    """从目录摄入所有文档

    Args:
        directory: 文档目录路径
        extract_entities: 是否提取实体和关系
        create_embeddings: 是否创建文本块嵌入

    Returns:
        所有文档的摄入结果列表
    """
    loader = DocumentLoader()
    builder = KnowledgeGraphBuilder()

    documents = loader.load_batch(directory)
    results = []

    for doc in documents:
        result = builder.ingest_document(doc, extract_entities, create_embeddings)
        results.append(result)

    # 摄入完成后创建向量索引
    builder.create_vector_index()

    return results
