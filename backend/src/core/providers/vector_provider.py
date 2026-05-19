from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from loguru import logger


@runtime_checkable
class VectorProvider(Protocol):
    """向量存储提供者协议"""
    
    def create_index(self, index_name: str, dimension: int):
        """创建向量索引"""
        ...
    
    def add_vectors(self, index_name: str, vectors: List[List[float]], ids: List[str]):
        """添加向量"""
        ...
    
    def search(self, index_name: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """向量搜索"""
        ...
    
    def delete_vectors(self, index_name: str, ids: List[str]):
        """删除向量"""
        ...
    
    def get_vector(self, index_name: str, id: str) -> Optional[List[float]]:
        """获取单个向量"""
        ...


class Neo4jVectorProvider:
    """Neo4j向量存储提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.neo4j_config = config
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from ..neo4j_client import Neo4jClient
            self._client = Neo4jClient(
                uri=self.neo4j_config.get("uri"),
                username=self.neo4j_config.get("username"),
                password=self.neo4j_config.get("password"),
            )
        return self._client
    
    def create_index(self, index_name: str, dimension: int):
        client = self._get_client()
        query = f"""
        CREATE VECTOR INDEX {index_name}
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                vector: {{
                    dimension: {dimension},
                    similarityFunction: 'cosine'
                }}
            }}
        }}
        """
        try:
            client.execute_query(query)
            logger.info(f"Created vector index: {index_name}")
        except Exception as e:
            logger.warning(f"Index creation failed (may already exist): {e}")
    
    def add_vectors(self, index_name: str, vectors: List[List[float]], ids: List[str]):
        client = self._get_client()
        for i, (vector, chunk_id) in enumerate(zip(vectors, ids)):
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            SET c.embedding = $embedding
            """
            client.execute_query(query, {"chunk_id": chunk_id, "embedding": vector})
    
    def search(self, index_name: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        client = self._get_client()
        query = f"""
        CALL db.index.vector.queryNodes('{index_name}', {top_k}, $embedding)
        YIELD node AS c, score AS similarity
        RETURN c.id AS chunk_id,
               c.content AS content,
               c.document_id AS document_id,
               c.index AS index,
               similarity
        """
        return client.execute_query(query, {"embedding": query_vector})
    
    def delete_vectors(self, index_name: str, ids: List[str]):
        client = self._get_client()
        for chunk_id in ids:
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            SET c.embedding = NULL
            """
            client.execute_query(query, {"chunk_id": chunk_id})
    
    def get_vector(self, index_name: str, id: str) -> Optional[List[float]]:
        client = self._get_client()
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        RETURN c.embedding AS embedding
        """
        results = client.execute_query(query, {"chunk_id": id})
        if results:
            return results[0].get("embedding")
        return None


class LanceDBProvider:
    """LanceDB向量存储提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.db_path = config.get("path", "./lancedb")
        self._db = None
    
    def _get_db(self):
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    def create_index(self, index_name: str, dimension: int):
        db = self._get_db()
        if index_name not in db.table_names():
            import pyarrow as pa
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), dimension)),
                ("content", pa.string()),
                ("document_id", pa.string()),
                ("index", pa.int32()),
            ])
            db.create_table(index_name, schema=schema)
            logger.info(f"Created LanceDB table: {index_name}")
    
    def add_vectors(self, index_name: str, vectors: List[List[float]], ids: List[str]):
        db = self._get_db()
        table = db.open_table(index_name)
        data = [
            {"id": ids[i], "vector": vectors[i]}
            for i in range(len(vectors))
        ]
        table.add(data)
    
    def search(self, index_name: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        db = self._get_db()
        table = db.open_table(index_name)
        results = table.search(query_vector).limit(top_k).to_list()
        return [
            {
                "chunk_id": r["id"],
                "content": r.get("content", ""),
                "document_id": r.get("document_id", ""),
                "index": r.get("index", 0),
                "similarity": r["_distance"],
            }
            for r in results
        ]
    
    def delete_vectors(self, index_name: str, ids: List[str]):
        db = self._get_db()
        table = db.open_table(index_name)
        table.delete(f"id IN {ids}")
    
    def get_vector(self, index_name: str, id: str) -> Optional[List[float]]:
        db = self._get_db()
        table = db.open_table(index_name)
        results = table.search().where(f"id = '{id}'").limit(1).to_list()
        if results:
            return results[0].get("vector")
        return None


class MockVectorProvider:
    """Mock向量存储提供者（用于测试）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self._vectors: Dict[str, Dict[str, Any]] = {}
    
    def create_index(self, index_name: str, dimension: int):
        if index_name not in self._vectors:
            self._vectors[index_name] = {}
        logger.info(f"Created mock index: {index_name}")
    
    def add_vectors(self, index_name: str, vectors: List[List[float]], ids: List[str]):
        if index_name not in self._vectors:
            self._vectors[index_name] = {}
        for i, (vector, id) in enumerate(zip(vectors, ids)):
            self._vectors[index_name][id] = {
                "vector": vector,
                "content": f"Content for {id}",
                "document_id": f"doc_{i}",
                "index": i,
            }
    
    def search(self, index_name: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        if index_name not in self._vectors:
            return []
        
        results = []
        for id, data in self._vectors[index_name].items():
            results.append({
                "chunk_id": id,
                "content": data["content"],
                "document_id": data["document_id"],
                "index": data["index"],
                "similarity": 0.8,
            })
        return results[:top_k]
    
    def delete_vectors(self, index_name: str, ids: List[str]):
        if index_name in self._vectors:
            for id in ids:
                self._vectors[index_name].pop(id, None)
    
    def get_vector(self, index_name: str, id: str) -> Optional[List[float]]:
        if index_name in self._vectors and id in self._vectors[index_name]:
            return self._vectors[index_name][id].get("vector")
        return None


class VectorFactory:
    """向量存储提供者工厂"""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册向量存储提供者"""
        cls._providers[name] = provider_class
        logger.info(f"Registered vector provider: {name}")
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> VectorProvider:
        """创建向量存储提供者实例"""
        provider_type = config.get("type", "neo4j")
        
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown vector provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        logger.info(f"Creating vector provider: {provider_type}")
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取可用的提供者列表"""
        return list(cls._providers.keys())


VectorFactory.register("neo4j", Neo4jVectorProvider)
VectorFactory.register("lancedb", LanceDBProvider)
VectorFactory.register("mock", MockVectorProvider)


def get_vector_provider(config: Optional[Dict[str, Any]] = None) -> VectorProvider:
    """获取向量存储提供者单例"""
    global _vector_provider
    if _vector_provider is None:
        if config is None:
            from ..config import get_settings
            settings = get_settings()
            config = {
                "type": settings.vector_provider,
                "uri": settings.neo4j_uri,
                "username": settings.neo4j_username,
                "password": settings.neo4j_password,
                "path": settings.lancedb_path,
            }
        _vector_provider = VectorFactory.create(config)
    return _vector_provider


_vector_provider: Optional[VectorProvider] = None