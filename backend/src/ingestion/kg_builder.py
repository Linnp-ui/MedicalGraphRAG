import hashlib
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dataclasses import field

from loguru import logger

from ..core.config import get_settings
from ..core.neo4j_client import Neo4jClient
from ..core.medical_schema import MEDICAL_SCHEMA, MedicalEntityType, MedicalRelationshipType
from .document_loader import Document, DocumentLoader
from .text_splitter import TextChunk
from .embedding import EmbeddingClient, get_embedding_client
from .medical_processor import MedicalTextProcessor
from .knowledge_fusion import KnowledgeFusionEngine
from .medical_ner import MedicalNER
from ..utils.process_monitor import track_process, get_structured_logger

_structured_logger = get_structured_logger("ingestion")


@dataclass
class Entity:
    """Entity data class with multi-type support"""

    name: str
    type: str
    properties: Dict[str, Any]
    types: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.types:
            self.types = [self.type] if self.type else []
        elif self.type not in self.types:
            self.types.insert(0, self.type)


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
        use_ner: bool = True,
    ):
        self.neo4j_client = neo4j_client
        self.embedding_client = embedding_client
        self.medical_processor = MedicalTextProcessor()
        self.fusion_engine = KnowledgeFusionEngine()
        self.ner = MedicalNER() if use_ner else None
        self.use_ner = use_ner
        self._medical_entity_types = {e.value for e in MedicalEntityType}
        self._medical_relation_types = {r.value for r in MedicalRelationshipType}
        self._stats = {
            "documents_skipped": 0,
            "chunks_skipped": 0,
            "entities_skipped": 0,
            "embeddings_cached_hits": 0,
        }

    def _get_neo4j_client(self) -> Neo4jClient:
        if self.neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client

            self.neo4j_client = get_neo4j_client()
        return self.neo4j_client

    def _get_embedding_client(self) -> EmbeddingClient:
        if self.embedding_client is None:
            self.embedding_client = get_embedding_client()
        return self.embedding_client

    def get_stats(self) -> Dict[str, int]:
        """获取缓存命中统计"""
        return self._stats.copy()

    def reset_stats(self):
        """重置统计计数器"""
        self._stats = {
            "documents_skipped": 0,
            "chunks_skipped": 0,
            "entities_skipped": 0,
            "embeddings_cached_hits": 0,
        }

    def _check_document_duplicate(self, content_hash: str) -> Optional[str]:
        """检查是否存在相同内容的文档

        Args:
            content_hash: 文档内容的SHA256哈希

        Returns:
            已存在文档的ID，如果不存在则返回None
        """
        client = self._get_neo4j_client()
        query = """
        MATCH (d:Document)
        WHERE d.content_hash = $content_hash
        RETURN d.id as existing_id
        LIMIT 1
        """
        try:
            results = client.execute_query(query, {"content_hash": content_hash})
            if results:
                return results[0]["existing_id"]
        except Exception as e:
            logger.warning(f"Failed to check document duplicate by content_hash: {e}")
        
        return None

    def _check_document_exists_by_id(self, document_id: str) -> bool:
        """检查文档ID是否已存在

        Args:
            document_id: 文档ID

        Returns:
            如果文档存在则返回True
        """
        client = self._get_neo4j_client()
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.id as existing_id
        LIMIT 1
        """
        try:
            results = client.execute_query(query, {"document_id": document_id})
            return bool(results)
        except Exception as e:
            logger.warning(f"Failed to check document by ID: {e}")
        return False

    @track_process("ingestion.ingest_document")
    def ingest_document(
        self,
        document: Document,
        extract_entities: bool = True,
        create_embeddings: bool = True,
        skip_duplicate: bool = True,
    ) -> Dict[str, Any]:
        """将文档摄入到知识图谱

        Args:
            document: 要摄入的文档
            extract_entities: 是否提取实体和关系
            create_embeddings: 是否创建文本块嵌入
            skip_duplicate: 是否跳过重复文档（基于内容hash）

        Returns:
            摄入结果，包含文档ID、创建的文本块数、提取的实体数和创建的关系数
        """
        _structured_logger.info(
            "ingest_document_started",
            document_id=document.id,
            content_length=len(document.content),
            extract_entities=extract_entities,
            create_embeddings=create_embeddings,
        )
        
        content_hash = hashlib.sha256(document.content.encode()).hexdigest()

        if skip_duplicate:
            existing_by_id = self._check_document_exists_by_id(document.id)
            if existing_by_id:
                logger.info(f"Document with ID '{document.id}' already exists, skipping")
                self._stats["documents_skipped"] += 1
                return {
                    "document_id": document.id,
                    "status": "duplicate_skipped",
                    "existing_document_id": document.id,
                    "chunks_created": 0,
                    "entities_extracted": 0,
                    "relationships_created": 0,
                    "skipped": True,
                }
            
            existing_by_hash = self._check_document_duplicate(content_hash)
            if existing_by_hash and existing_by_hash != document.id:
                logger.info(f"Document with same content already exists: {existing_by_hash}, skipping")
                self._stats["documents_skipped"] += 1
                return {
                    "document_id": document.id,
                    "status": "duplicate_skipped",
                    "existing_document_id": existing_by_hash,
                    "chunks_created": 0,
                    "entities_extracted": 0,
                    "relationships_created": 0,
                    "skipped": True,
                }

        results = {
            "document_id": document.id,
            "status": "success",
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
        from .text_splitter import TextSplitter, SplitStrategy, select_chunking_strategy

        strategy_name = settings.split_strategy
        if strategy_name == "auto":
            strategy = select_chunking_strategy(document.content, settings.domain)
        else:
            strategy = SplitStrategy(strategy_name)

        logger.info(f"Using chunking strategy: {strategy.value} for document {document.id}")

        splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            soft_max=settings.soft_max,
            strategy=strategy,
        )
        chunks = splitter.split_text(document.content, document.id)
        results["chunks_created"] = len(chunks)

        # 上下文增强（可选，默认关闭；需传入 llm_call 函数）
        if settings.contextual_enrichment:
            logger.info("Contextual enrichment is enabled but requires llm_call integration")

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

        # 知识融合：实体消歧和关系对齐（仅医疗领域）
        if settings.domain == "medical" and (entity_map or relationships):
            entities_for_fusion = [
                {"name": e.name, "type": e.type, "properties": e.properties}
                for e in entity_map.values()
            ]
            rels_for_fusion = [
                {"source": r.source, "target": r.target, "type": r.type, "properties": r.properties}
                for r in relationships
            ]

            fused_entities, fused_rels = self.fusion_engine.fuse(entities_for_fusion, rels_for_fusion)

            fused_entities = self.fusion_engine.link_to_standard_ontology(fused_entities)

            fused_entity_map = {e["name"]: e for e in fused_entities}
            fused_relationships = fused_rels

            logger.info(f"Knowledge fusion completed: {len(fused_entity_map)} entities, {len(fused_relationships)} relationships")
        else:
            fused_entity_map = {e.name: {"name": e.name, "type": e.type, "properties": e.properties} for e in entity_map.values()}
            fused_relationships = [{"source": r.source, "target": r.target, "type": r.type, "properties": r.properties} for r in relationships]

        # 将融合后的实体写入Neo4j
        for entity in fused_entity_map.values():
            entity_types = entity.get("types", [entity.get("type", "Entity")])
            primary_type = entity_types[0] if entity_types else entity.get("type", "Entity")
            entity_obj = Entity(
                name=entity["name"], 
                type=primary_type, 
                properties=entity.get("properties", {}),
                types=entity_types
            )
            self._create_entity_node(entity_obj)

        # 将融合后的关系写入Neo4j
        for rel in fused_relationships:
            rel_obj = Relationship(source=rel["source"], target=rel["target"], type=rel.get("aligned_type", rel["type"]), properties=rel["properties"])
            self._create_relationship(rel_obj)

        # 将文本块与实体关联（使用融合后的实体名称）
        for entity_key, chunk_id in chunk_entity_links:
            if entity_key in entity_map:
                original_entity = entity_map[entity_key]
                if settings.domain == "medical" and original_entity.name in fused_entity_map:
                    fused_name = fused_entity_map[original_entity.name]["name"]
                    self._link_entity_to_chunk(fused_name, chunk_id)
                else:
                    self._link_entity_to_chunk(original_entity.name, chunk_id)

        results["entities_extracted"] = len(fused_entity_map)
        results["relationships_created"] = len(fused_relationships)

        _structured_logger.info(
            "ingest_document_completed",
            document_id=document.id,
            chunks_created=results["chunks_created"],
            entities_extracted=results["entities_extracted"],
            relationships_created=results["relationships_created"],
        )

        logger.info(f"Document {document.id} ingested: {results}")
        return results

    def _create_document_node(self, document: Document):
        """Create document node in Neo4j"""
        client = self._get_neo4j_client()
        content_hash = hashlib.sha256(document.content.encode()).hexdigest()
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.content = $content,
            d.metadata = $metadata,
            d.content_hash = $content_hash,
            d.created_at = CASE WHEN d.created_at IS NULL THEN timestamp() ELSE d.created_at END,
            d.updated_at = timestamp()
        """
        client.execute_query(
            query,
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "metadata": json.dumps(document.metadata),
                "content_hash": content_hash,
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
        """使用 NER + LLM 从文本块中提取实体和关系

        医疗领域优先使用 NER 模型（快速、低成本），LLM 作为补充。
        通用领域使用 LLM 提取。

        Args:
            chunk: 要提取实体和关系的文本块

        Returns:
            提取的实体列表和关系列表
        """
        settings = get_settings()

        if settings.domain == "medical" and self.use_ner and self.ner:
            return self._extract_with_ner(chunk)

        return self._extract_with_llm(chunk)

    def _extract_with_ner(
        self, chunk: TextChunk
    ) -> tuple[List[Entity], List[Relationship]]:
        """使用 NER 模型提取实体，基于共现关系推断简单关系"""
        ner_entities = self.ner.extract(chunk.content)

        entities = [
            Entity(name=e.name, type=e.entity_type, properties={"confidence": e.confidence})
            for e in ner_entities
        ]

        relationships = self._infer_relationships_from_cooccurrence(ner_entities, chunk.content)

        logger.info(
            f"NER extracted {len(entities)} entities and {len(relationships)} relationships"
        )
        return entities, relationships

    def _extract_with_llm(
        self, chunk: TextChunk
    ) -> tuple[List[Entity], List[Relationship]]:
        """使用 LLM 从文本块中提取实体和关系"""
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        import json as _json

        settings = get_settings()

        llm = ChatOpenAI(
            model=settings.extraction_model,
            temperature=0,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        json_example = (
            '{{'
            '  "entities": ['
            '    {{"name": "实体名", "type": "实体类型", "properties": {{}}}}'
            '  ],'
            '  "relationships": ['
            '    {{"source": "源实体", "target": "目标实体", "type": "关系类型", "properties": {{}}}}'
            '  ]'
            '}}'
        )

        if settings.domain == "medical":
            entity_types = ", ".join(MEDICAL_SCHEMA['entities'])
            rel_types = ", ".join(MEDICAL_SCHEMA['relationships'])
            entity_descriptions = "\n".join([f"- {k}: {v}" for k, v in MEDICAL_SCHEMA['descriptions'].items()])

            system_message = (
                "你是一个医疗实体关系抽取专家。从给定的文本中提取医疗相关的实体和关系。\n\n"
                "请以JSON格式返回，结构如下：\n"
                + json_example +
                "\n\n可用的实体类型包括：" + entity_types +
                "\n可用的关系类型包括：" + rel_types +
                "\n\n实体类型说明：\n" + entity_descriptions +
                "\n\n只提取文本中明确提到的医疗实体和关系。只返回JSON，不要有其他解释文字。"
            )
        else:
            system_message = (
                "你是一个实体关系抽取专家。从给定的文本中提取实体和关系。\n\n"
                "请以JSON格式返回，结构如下：\n"
                + json_example +
                "\n\n实体类型可以是：Person, Organization, Location, Product, Concept, System, Database, Framework 等"
                "\n关系类型可以是：USES, HAS, WORKS_AT, LOCATED_IN, RELATED_TO 等"
                "\n只提取文本中明确提到的实体和关系。只返回JSON，不要有其他解释文字。"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "从以下文本中提取实体和关系：\n\n{text}"),
            ]
        )

        chain = prompt | llm

        try:
            result = chain.invoke({"text": chunk.content})
            content = result.content

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

    def _infer_relationships_from_cooccurrence(
        self, entities: list, text: str
    ) -> List[Relationship]:
        """基于共现和医疗本体约束推断简单关系"""
        from ..core.medical_schema import MEDICAL_SCHEMA

        relationships = []
        seen_rels = set()

        allowed_relations = MEDICAL_SCHEMA.get('allowed_relations', {})

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                src_type = e1.entity_type
                tgt_type = e2.entity_type

                if src_type in allowed_relations and tgt_type in allowed_relations.get(src_type, {}):
                    rel_types = allowed_relations[src_type][tgt_type]
                    if rel_types:
                        rel_key = (e1.name.lower(), e2.name.lower(), rel_types[0])
                        if rel_key not in seen_rels:
                            seen_rels.add(rel_key)
                            relationships.append(Relationship(
                                source=e1.name,
                                target=e2.name,
                                type=rel_types[0],
                                properties={"inferred": True, "confidence": 0.6},
                            ))

                        rel_key_rev = (e2.name.lower(), e1.name.lower(), rel_types[0])
                        if rel_key_rev not in seen_rels:
                            seen_rels.add(rel_key_rev)
                            relationships.append(Relationship(
                                source=e2.name,
                                target=e1.name,
                                type=rel_types[0],
                                properties={"inferred": True, "confidence": 0.6},
                            ))

        return relationships

    def _create_entity_node(self, entity: Entity):
        """在Neo4j中创建实体节点（支持多标签）

        如果实体已存在且包含嵌入，则跳过嵌入生成以节省API调用。

        Args:
            entity: 要创建的实体（支持 types 多标签）
        """
        client = self._get_neo4j_client()
        settings = get_settings()

        embedding = None
        entity_existed = False

        check_query = """
        MATCH (e:Entity {name: $name})
        RETURN e.embedding as embedding
        """

        try:
            existing_results = client.execute_query(check_query, {"name": entity.name})
            if existing_results and existing_results[0].get("embedding"):
                entity_existed = True
                logger.debug(f"Entity '{entity.name}' already exists with embedding, skipping")
                self._stats["entities_skipped"] += 1
                self._stats["embeddings_cached_hits"] += 1
        except Exception as e:
            logger.warning(f"Failed to check existing entity: {e}")

        if not entity_existed:
            try:
                embedding_client = self.embedding_client or get_embedding_client()
                embedding = embedding_client.embed_text(entity.name)
            except Exception as e:
                logger.warning(f"Failed to create embedding for entity '{entity.name}': {e}")

        valid_types = [t for t in entity.types if t in self._medical_entity_types] if settings.domain == "medical" else entity.types
        if not valid_types:
            valid_types = [entity.type] if entity.type else ["Entity"]

        primary_type = valid_types[0] if valid_types else "Entity"
        labels_str = ":".join(["Entity"] + valid_types)

        if embedding is not None:
            query = f"""
            MERGE (e:Entity {{name: $name}})
            ON CREATE SET 
                e.properties = $properties, 
                e.type = $type, 
                e.types = $types,
                e.embedding = $embedding
            ON MATCH SET 
                e.properties = CASE
                    WHEN e.properties IS NULL THEN $properties
                    ELSE e.properties + $merge_props
                END,
                e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END,
                e.types = CASE 
                    WHEN e.types IS NULL THEN $types 
                    ELSE apoc.coll.toSet(e.types + $types) 
                END,
                e.embedding = CASE WHEN e.embedding IS NULL THEN $embedding ELSE e.embedding END
            WITH e
            CALL apoc.create.addLabels(e, $additional_labels) YIELD node
            RETURN node
            """
        else:
            query = f"""
            MERGE (e:Entity {{name: $name}})
            ON CREATE SET 
                e.properties = $properties, 
                e.type = $type,
                e.types = $types
            ON MATCH SET 
                e.properties = CASE
                    WHEN e.properties IS NULL THEN $properties
                    ELSE e.properties + $merge_props
                END,
                e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END,
                e.types = CASE 
                    WHEN e.types IS NULL THEN $types 
                    ELSE apoc.coll.toSet(e.types + $types) 
                END
            WITH e
            CALL apoc.create.addLabels(e, $additional_labels) YIELD node
            RETURN node
            """

        params = {
            "name": entity.name,
            "type": primary_type,
            "types": valid_types,
            "properties": json.dumps(entity.properties),
            "merge_props": json.dumps(entity.properties),
            "additional_labels": valid_types,
        }
        if embedding is not None:
            params["embedding"] = embedding

        try:
            client.execute_query(query, params)
        except Exception as e:
            logger.warning(f"Failed to add labels with APOC, falling back to basic query: {e}")
            fallback_query = f"""
            MERGE (e:{labels_str} {{name: $name}})
            ON CREATE SET 
                e.properties = $properties, 
                e.type = $type,
                e.types = $types
            ON MATCH SET 
                e.properties = CASE
                    WHEN e.properties IS NULL THEN $properties
                    ELSE e.properties + $merge_props
                END,
                e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END,
                e.types = CASE 
                    WHEN e.types IS NULL THEN $types 
                    ELSE $types 
                END
            """
            if embedding is not None:
                fallback_query = fallback_query.replace(
                    "e.types = $types",
                    "e.types = $types, e.embedding = CASE WHEN e.embedding IS NULL THEN $embedding ELSE e.embedding END"
                )
                params["embedding"] = embedding
            client.execute_query(fallback_query, params)

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
        """在Neo4j中创建关系（支持医疗领域的特定关系类型）

        Args:
            relationship: 要创建的关系
        """
        client = self._get_neo4j_client()
        settings = get_settings()

        if settings.domain == "medical" and relationship.type in self._medical_relation_types:
            rel_label = relationship.type
            query = f"""
            MATCH (e1:Entity {{name: $source}})
            MATCH (e2:Entity {{name: $target}})
            MERGE (e1)-[r:{rel_label}]->(e2)
            ON CREATE SET r.properties = $properties
            ON MATCH SET r.properties = CASE
                WHEN r.properties IS NULL THEN $properties
                ELSE r.properties + $properties
            END
            """
        else:
            query = """
            MATCH (e1:Entity {name: $source})
            MATCH (e2:Entity {name: $target})
            MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
            ON CREATE SET r.properties = $properties
            ON MATCH SET r.properties = CASE
                WHEN r.properties IS NULL THEN $properties
                ELSE r.properties + $properties
            END
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
        """批量创建文本块的嵌入（带缓存检查）

        对于已有嵌入的文本块，跳过API调用以节省资源。

        Args:
            chunks: 要创建嵌入的文本块列表
        """
        client = self._get_neo4j_client()
        embedding_client = self._get_embedding_client()

        chunks_to_embed = []
        chunks_to_embed_indices = []
        results = []

        for i, chunk in enumerate(chunks):
            check_query = """
            MATCH (c:Chunk {id: $chunk_id})
            RETURN c.embedding as embedding
            """
            try:
                existing = client.execute_query(check_query, {"chunk_id": chunk.id})
                if existing and existing[0].get("embedding"):
                    results.append((i, existing[0]["embedding"]))
                    self._stats["embeddings_cached_hits"] += 1
                    logger.debug(f"Chunk '{chunk.id}' already has embedding, using cached")
                    continue
            except Exception as e:
                logger.warning(f"Failed to check chunk embedding: {e}")

            chunks_to_embed.append(chunk)
            chunks_to_embed_indices.append(i)

        self._stats["chunks_skipped"] += len(results)

        if not chunks_to_embed:
            logger.info(f"All {len(chunks)} chunks already have embeddings, skipping API calls")
            return results

        logger.info(f"Creating embeddings for {len(chunks_to_embed)}/{len(chunks)} chunks (cached: {len(results)})")

        texts = [chunk.content for chunk in chunks_to_embed]
        try:
            new_embeddings = embedding_client.embed_texts(texts)
            for idx, embedding in zip(chunks_to_embed_indices, new_embeddings):
                if embedding:
                    results.append((idx, embedding))
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            new_embeddings = []
            for text in texts:
                try:
                    new_embeddings.append(embedding_client.embed_text(text))
                except Exception as inner_e:
                    logger.error(f"Single embedding failed: {inner_e}")
                    new_embeddings.append([])

            for idx, embedding in zip(chunks_to_embed_indices, new_embeddings):
                if embedding:
                    results.append((idx, embedding))

        embed_query = """
        MATCH (c:Chunk {id: $chunk_id})
        SET c.embedding = $embedding
        """
        new_results = results[len(chunks_to_embed_indices):]
        for chunk, (idx, embedding) in zip(chunks_to_embed, new_results):
            if embedding:
                try:
                    client.execute_query(embed_query, {"chunk_id": chunk.id, "embedding": embedding})
                except Exception as e:
                    logger.error(f"Failed to store embedding for chunk {chunk.id}: {e}")

        logger.info(f"Stored {len([e for _, e in results if e])} embeddings in batch")

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
