import time
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Generator, Optional
import re

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    Driver,
    GraphDatabase,
    Result,
)
from loguru import logger

from .config import get_settings

# Schema cache: (schema_string, expires_at_timestamp)
_schema_cache: tuple[str, float] | None = None
_SCHEMA_CACHE_TTL = 300  # 5 minutes
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class Neo4jClient:
    """Neo4j client with connection pooling"""

    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self._driver: Optional[Driver] = None
        self._async_driver: Optional[AsyncDriver] = None

    def _normalize_identifier(self, identifier: str, field_name: str) -> str:
        if not _IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(f"Invalid {field_name}")
        return identifier

    def _get_driver(self) -> Driver:
        """Get or create Neo4j driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(
                    self.settings.neo4j_username,
                    self.settings.neo4j_password,
                ),
                max_connection_lifetime=1800,
                max_connection_pool_size=100,
                connection_acquisition_timeout=30,
                connection_timeout=30,
                max_transaction_retry_time=30,
            )
            logger.info(f"Neo4j driver created: {self.settings.neo4j_uri}")
        return self._driver

    def _get_async_driver(self) -> AsyncDriver:
        """Get or create async Neo4j driver"""
        if self._async_driver is None:
            self._async_driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(
                    self.settings.neo4j_username,
                    self.settings.neo4j_password,
                ),
                max_connection_lifetime=1800,
                max_connection_pool_size=100,
                connection_acquisition_timeout=30,
                connection_timeout=30,
                max_transaction_retry_time=30,
            )
            logger.info(f"Async Neo4j driver created: {self.settings.neo4j_uri}")
        return self._async_driver

    @contextmanager
    def session(self, **kwargs) -> Generator:
        """Get a Neo4j session"""
        driver = self._get_driver()
        session = driver.session(**kwargs)
        try:
            yield session
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self, **kwargs):
        """Get an async Neo4j session"""
        driver = self._get_async_driver()
        session = driver.session(**kwargs)
        try:
            yield session
        finally:
            await session.close()

    def execute_query(self, query: str, parameters: Optional[dict] = None, **kwargs) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts"""
        import time
        from .metrics import get_metrics_middleware
        start = time.perf_counter()
        with self.session() as session:
            result: Result = session.run(query, parameters or {}, **kwargs)
            results = [record.data() for record in result]
        duration_ms = (time.perf_counter() - start) * 1000
        try:
            get_metrics_middleware().record_neo4j_query(duration_ms, query_type="execute")
        except Exception as e:
            logger.warning(f"Failed to record Neo4j metrics: {e}")
        return results

    async def execute_query_async(
        self, query: str, parameters: Optional[dict] = None, **kwargs
    ) -> list[dict]:
        """Execute a Cypher query asynchronously"""
        async with self.async_session() as session:
            result = await session.run(query, parameters or {}, **kwargs)
            return [record.data() async for record in result]

    def verify_connectivity(self) -> bool:
        """Verify Neo4j connection"""
        try:
            with self.session() as session:
                result = session.run("RETURN 1")
                result.consume()
            logger.info("Neo4j connection verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def get_schema(self, use_cache: bool = True) -> str:
        """Get Neo4j database schema, cached for 5 minutes to reduce DB load"""
        global _schema_cache

        if use_cache and _schema_cache is not None:
            schema_str, expires_at = _schema_cache
            if time.time() < expires_at:
                logger.debug("Returning cached schema")
                return schema_str

        schema_parts = []

        # Get node labels
        node_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        result = self.execute_query(node_query)
        labels = result[0].get("labels", []) if result else []

        # Get relationship types
        rel_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        result = self.execute_query(rel_query)
        rel_types = result[0].get("types", []) if result else []

        # Get sample properties for each label
        properties_query = """
        MATCH (n)
        WITH labels(n)[0] as label, keys(n) as keys
        RETURN label, collect(DISTINCT keys)[0..3] as sample_properties
        LIMIT 10
        """
        result = self.execute_query(properties_query)

        schema_parts.append("Node labels:")
        for label in labels:
            props = next(
                (r["sample_properties"] for r in result if r["label"] == label),
                [],
            )
            schema_parts.append(f"  - {label}: {props}")

        schema_parts.append("\nRelationship types:")
        for rel_type in rel_types:
            schema_parts.append(f"  - {rel_type}")

        schema_str = "\n".join(schema_parts)

        # Store in cache
        _schema_cache = (schema_str, time.time() + _SCHEMA_CACHE_TTL)
        logger.debug("Schema cached")

        return schema_str

    def invalidate_schema_cache(self):
        """Force-invalidate the schema cache (call after ingestion)"""
        global _schema_cache
        _schema_cache = None
        logger.info("Schema cache invalidated")

    def get_graph_data(
        self,
        node_label: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> dict[str, Any]:
        """获取图谱数据"""
        if node_label:
            safe_node_label = self._normalize_identifier(node_label, "node_label")
            node_query = f"""
            MATCH (n:`{safe_node_label}`)
            WITH n, COUNT {{ (n)--() }} as degree
            RETURN n, id(n) as node_id, degree, labels(n) as labels
            ORDER BY degree DESC
            SKIP $offset
            LIMIT $limit
            """
        else:
            node_query = """
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            RETURN n, id(n) as node_id, degree, labels(n) as labels
            ORDER BY degree DESC
            SKIP $offset
            LIMIT $limit
            """
        
        with self.session() as session:
            nodes_result = session.run(node_query, offset=offset, limit=limit)
            nodes = []
            node_ids = []
            
            for record in nodes_result:
                node_data = record.data()
                node_id = str(node_data['node_id'])
                node_ids.append(node_id)
                
                labels = node_data.get('labels', [])
                label = labels[0] if labels else 'Node'
                
                node_props = dict(node_data['n']) if node_data.get('n') else {}
                
                nodes.append({
                    'id': node_id,
                    'label': label,
                    'properties': node_props,
                    'degree': node_data.get('degree', 0)
                })
            
            if not node_ids:
                return {
                    'nodes': [],
                    'edges': [],
                    'stats': {
                        'total_nodes': 0,
                        'total_edges': 0,
                        'node_labels': [],
                        'relationship_types': []
                    }
                }
            
            edge_query = """
            MATCH (n)-[r]->(m)
            WHERE id(n) IN $node_ids AND id(m) IN $node_ids
            RETURN id(r) as edge_id, id(startNode(r)) as from_id, 
                   id(endNode(r)) as to_id, type(r) as rel_type, properties(r) as props
            """
            
            edges_result = session.run(edge_query, node_ids=[int(nid) for nid in node_ids])
            edges = []
            
            for record in edges_result:
                edge_data = record.data()
                edges.append({
                    'id': str(edge_data['edge_id']),
                    'from': str(edge_data['from_id']),
                    'to': str(edge_data['to_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
            
            stats_query = """
            MATCH (n)
            WITH count(n) as total_nodes
            MATCH ()-[r]->()
            WITH total_nodes, count(r) as total_edges
            CALL db.labels() YIELD label
            WITH total_nodes, total_edges, collect(label) as node_labels
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN total_nodes, total_edges, node_labels, collect(relationshipType) as relationship_types
            """
            
            stats_result = session.run(stats_query)
            stats_data = stats_result.single()
            
            stats = {
                'total_nodes': stats_data['total_nodes'] if stats_data else 0,
                'total_edges': stats_data['total_edges'] if stats_data else 0,
                'node_labels': stats_data['node_labels'] if stats_data else [],
                'relationship_types': stats_data['relationship_types'] if stats_data else []
            }
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': stats
            }

    def search_nodes(
        self,
        search_text: str,
        node_label: Optional[str] = None,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """搜索节点 - 优先使用全全文索引，回退到属性匹配"""
        searchable_props = ['name', 'title', 'description', 'text', 'content', 'label', 'type']

        fulltext_query = """
        CALL db.index.fulltext.queryNodes("entity_fulltext_idx", $search_text)
        YIELD node, score
        RETURN node, id(node) as node_id, labels(node) as labels, score
        LIMIT $limit
        """

        if node_label:
            safe_node_label = self._normalize_identifier(node_label, "node_label")
            search_query = f"""
            MATCH (n:`{safe_node_label}`)
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) CONTAINS $search_text)
            RETURN n, id(n) as node_id, labels(n) as labels
            LIMIT $limit
            """
        else:
            search_query = """
            MATCH (n)
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) CONTAINS $search_text)
            RETURN n, id(n) as node_id, labels(n) as labels
            LIMIT $limit
            """

        with self.session() as session:
            try:
                result = session.run(fulltext_query, search_text=search_text, limit=limit)
                nodes = []
                found = False

                for record in result:
                    found = True
                    node_data = record.data()
                    labels = node_data.get('labels', [])
                    label = labels[0] if labels else 'Node'
                    node_props = dict(node_data['node']) if node_data.get('node') else {}

                    nodes.append({
                        'id': str(node_data['node_id']),
                        'label': label,
                        'properties': node_props,
                        'score': node_data.get('score', 0),
                    })

                if found:
                    return nodes

                result = session.run(
                    search_query,
                    search_text=search_text,
                    searchable_props=searchable_props,
                    limit=limit
                )
                nodes = []

                for record in result:
                    node_data = record.data()
                    labels = node_data.get('labels', [])
                    label = labels[0] if labels else 'Node'

                    node_props = dict(node_data['n']) if node_data.get('n') else {}

                    nodes.append({
                        'id': str(node_data['node_id']),
                        'label': label,
                        'properties': node_props
                    })

                return nodes
            except Exception as e:
                logger.warning(f"Fulltext search failed, falling back: {e}")
                try:
                    result = session.run(
                        search_query,
                        search_text=search_text,
                        searchable_props=searchable_props,
                        limit=limit
                    )
                    nodes = []

                    for record in result:
                        node_data = record.data()
                        labels = node_data.get('labels', [])
                        label = labels[0] if labels else 'Node'

                        node_props = dict(node_data['n']) if node_data.get('n') else {}

                        nodes.append({
                            'id': str(node_data['node_id']),
                            'label': label,
                            'properties': node_props
                        })

                    return nodes
                except Exception as e2:
                    logger.warning(f"Search failed with error: {e2}")
                    return []

    def fuzzy_search_nodes(
        self,
        search_text: str,
        node_label: Optional[str] = None,
        limit: int = 20,
        fuzzy_mode: str = "contains",
        min_similarity: float = 0.6,
    ) -> list[dict[str, Any]]:
        """模糊搜索节点

        Args:
            search_text: 搜索文本
            node_label: 节点标签过滤
            limit: 返回数量限制
            fuzzy_mode: 模糊匹配模式
                - "contains": 包含匹配（默认）
                - "prefix": 前缀匹配
                - "suffix": 后缀匹配
                - "regex": 正则表达式匹配
                - "fuzzy": 编辑距离模糊匹配
            min_similarity: 最小相似度（仅用于 fuzzy 模式，0-1之间）

        Returns:
            匹配的节点列表
        """
        searchable_props = ['name', 'title', 'description', 'text', 'content', 'label', 'type']

        label_filter = ""
        if node_label:
            safe_node_label = self._normalize_identifier(node_label, "node_label")
            label_filter = f":`{safe_node_label}`"

        if fuzzy_mode == "prefix":
            where_clause = f"""
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) STARTS WITH $search_text)
            """
        elif fuzzy_mode == "suffix":
            where_clause = f"""
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) ENDS WITH $search_text)
            """
        elif fuzzy_mode == "regex":
            where_clause = f"""
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) =~ $search_text)
            """
        elif fuzzy_mode == "fuzzy":
            where_clause = f"""
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toLower(toString(n[prop])) CONTAINS toLower($search_text))
            """
        else:
            where_clause = f"""
            WHERE ANY(prop IN $searchable_props 
                      WHERE prop IN keys(n) 
                      AND n[prop] IS NOT NULL 
                      AND toString(n[prop]) CONTAINS $search_text)
            """

        search_query = f"""
        MATCH (n{label_filter})
        {where_clause}
        RETURN n, id(n) as node_id, labels(n) as labels
        LIMIT $limit
        """

        with self.session() as session:
            try:
                result = session.run(
                    search_query,
                    search_text=search_text,
                    searchable_props=searchable_props,
                    limit=limit
                )
                nodes = []

                for record in result:
                    node_data = record.data()
                    labels = node_data.get('labels', [])
                    label = labels[0] if labels else 'Node'
                    node_props = dict(node_data['n']) if node_data.get('n') else {}

                    prop_values = [str(v) for v in node_props.values() if v is not None]
                    matched_text = ""
                    for prop in searchable_props:
                        if prop in node_props and node_props[prop]:
                            val = str(node_props[prop])
                            if search_text.lower() in val.lower():
                                matched_text = val
                                break

                    nodes.append({
                        'id': str(node_data['node_id']),
                        'label': label,
                        'properties': node_props,
                        'matched_property': matched_text[:200] if matched_text else None,
                    })

                return nodes
            except Exception as e:
                logger.warning(f"Fuzzy search failed: {e}")
                return []

    def search_nodes_with_score(
        self,
        search_text: str,
        node_label: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """带相似度评分的搜索节点

        使用多种策略进行搜索并计算相似度分数：
        1. 精确匹配（分数 1.0）
        2. 前缀匹配（分数 0.9）
        3. 包含匹配（分数 0.7）
        4. 模糊匹配（分数 0.5-0.7）
        """
        all_results = {}
        search_text_lower = search_text.lower()

        exact_results = self.fuzzy_search_nodes(search_text, node_label, limit, "contains")
        for node in exact_results:
            node_id = node['id']
            props = node.get('properties', {})
            name = str(props.get('name', props.get('title', ''))).lower()

            if name == search_text_lower:
                score = 1.0
            elif name.startswith(search_text_lower):
                score = 0.9
            elif search_text_lower in name:
                score = 0.7 + (len(search_text_lower) / max(len(name), 1)) * 0.2
            else:
                score = 0.5

            if node_id not in all_results or score > all_results[node_id].get('score', 0):
                node['score'] = round(score, 3)
                all_results[node_id] = node

        sorted_results = sorted(all_results.values(), key=lambda x: x.get('score', 0), reverse=True)
        return sorted_results[:limit]

    def get_node_detail(self, node_id: str) -> Optional[dict[str, Any]]:
        """获取节点详情"""
        with self.session() as session:
            node_query = """
            MATCH (n)
            WHERE id(n) = $node_id
            RETURN n, id(n) as node_id, labels(n) as labels
            """
            
            node_result = session.run(node_query, node_id=int(node_id))
            node_data = node_result.single()
            
            if not node_data:
                return None
            
            labels = node_data.get('labels', [])
            label = labels[0] if labels else 'Node'
            node_props = dict(node_data['n']) if node_data.get('n') else {}
            
            node = {
                'id': str(node_data['node_id']),
                'label': label,
                'properties': node_props
            }
            
            incoming_query = """
            MATCH (m)-[r]->(n)
            WHERE id(n) = $node_id
            RETURN id(r) as edge_id, id(m) as from_id, type(r) as rel_type, 
                   properties(r) as props, labels(m) as from_labels
            """
            
            incoming_result = session.run(incoming_query, node_id=int(node_id))
            incoming = []
            neighbors = []
            
            for record in incoming_result:
                edge_data = record.data()
                incoming.append({
                    'from_node': str(edge_data['from_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
                
                from_labels = edge_data.get('from_labels', [])
                neighbor_label = from_labels[0] if from_labels else 'Node'
                
                neighbors.append({
                    'id': str(edge_data['from_id']),
                    'label': neighbor_label,
                    'properties': {}
                })
            
            outgoing_query = """
            MATCH (n)-[r]->(m)
            WHERE id(n) = $node_id
            RETURN id(r) as edge_id, id(m) as to_id, type(r) as rel_type, 
                   properties(r) as props, labels(m) as to_labels
            """
            
            outgoing_result = session.run(outgoing_query, node_id=int(node_id))
            outgoing = []
            
            for record in outgoing_result:
                edge_data = record.data()
                outgoing.append({
                    'to_node': str(edge_data['to_id']),
                    'type': edge_data['rel_type'],
                    'properties': edge_data.get('props') or {}
                })
                
                to_labels = edge_data.get('to_labels', [])
                neighbor_label = to_labels[0] if to_labels else 'Node'
                
                neighbors.append({
                    'id': str(edge_data['to_id']),
                    'label': neighbor_label,
                    'properties': {}
                })
            
            unique_neighbors = []
            seen_ids = set()
            for neighbor in neighbors:
                if neighbor['id'] not in seen_ids:
                    seen_ids.add(neighbor['id'])
                    unique_neighbors.append(neighbor)
            
            return {
                'node': node,
                'relationships': {
                    'incoming': incoming,
                    'outgoing': outgoing
                },
                'neighbors': unique_neighbors
            }

    def get_node_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relationship_type: Optional[str] = None
    ) -> dict[str, Any]:
        """获取节点邻居"""
        if relationship_type:
            safe_relationship_type = self._normalize_identifier(
                relationship_type, "relationship_type"
            )
            neighbor_query = f"""
            MATCH path = (n)-[r:`{safe_relationship_type}`*1..{depth}]-(m)
            WHERE id(n) = $node_id
            UNWIND nodes(path) as node
            WITH DISTINCT node, id(node) as node_id, labels(node) as node_labels
            RETURN node, node_id, node_labels
            """
        else:
            neighbor_query = f"""
            MATCH path = (n)-[r*1..{depth}]-(m)
            WHERE id(n) = $node_id
            UNWIND nodes(path) as node
            WITH DISTINCT node, id(node) as node_id, labels(node) as node_labels
            RETURN node, node_id, node_labels
            """
        
        with self.session() as session:
            nodes_result = session.run(neighbor_query, node_id=int(node_id))
            
            all_nodes = {}
            for record in nodes_result:
                node_data = record.data()
                node_id_str = str(node_data['node_id'])
                
                if node_id_str not in all_nodes:
                    labels = node_data.get('node_labels', [])
                    node_props = dict(node_data['node']) if node_data.get('node') else {}
                    
                    all_nodes[node_id_str] = {
                        'id': node_id_str,
                        'label': labels[0] if labels else 'Node',
                        'properties': node_props
                    }
            
            if relationship_type:
                edge_query = f"""
                MATCH path = (n)-[r:`{safe_relationship_type}`*1..{depth}]-(m)
                WHERE id(n) = $node_id
                UNWIND relationships(path) as rel
                WITH DISTINCT rel, id(rel) as edge_id, id(startNode(rel)) as from_id, 
                       id(endNode(rel)) as to_id, type(rel) as rel_type
                RETURN edge_id, from_id, to_id, rel_type, properties(rel) as props
                """
            else:
                edge_query = f"""
                MATCH path = (n)-[r*1..{depth}]-(m)
                WHERE id(n) = $node_id
                UNWIND relationships(path) as rel
                WITH DISTINCT rel, id(rel) as edge_id, id(startNode(rel)) as from_id, 
                       id(endNode(rel)) as to_id, type(rel) as rel_type
                RETURN edge_id, from_id, to_id, rel_type, properties(rel) as props
                """
            
            edges_result = session.run(edge_query, node_id=int(node_id))
            
            all_edges = {}
            for record in edges_result:
                edge_data = record.data()
                edge_id = str(edge_data['edge_id'])
                
                if edge_id not in all_edges:
                    all_edges[edge_id] = {
                        'id': edge_id,
                        'from': str(edge_data['from_id']),
                        'to': str(edge_data['to_id']),
                        'type': edge_data['rel_type'],
                        'properties': edge_data.get('props') or {}
                    }
            
            return {
                'nodes': list(all_nodes.values()),
                'edges': list(all_edges.values()),
                'center_node': node_id
            }

    def close(self):
        """Close Neo4j driver"""
        if self._driver:
            self._driver.close()
            self._driver = None
        if self._async_driver:
            self._async_driver.close()
            self._async_driver = None
        logger.info("Neo4j drivers closed")


# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get singleton Neo4j client"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
