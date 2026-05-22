from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .community_detector import CommunityDetector, get_community_detector
from .neo4j_client import Neo4jClient
from .llm_cache import get_llm_cache, llm_cached, LLMCache


class SummaryGenerator:
    """分层摘要生成模块 - 为社区和实体生成摘要"""

    def __init__(
        self,
        community_detector: Optional[CommunityDetector] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        llm_cache: Optional[LLMCache] = None
    ):
        self._community_detector = community_detector or get_community_detector()
        self._neo4j_client = neo4j_client
        self._llm_cache = llm_cache or get_llm_cache()
        self._summaries = {}

    def _get_neo4j_client(self) -> Neo4jClient:
        if self._neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client
            self._neo4j_client = get_neo4j_client()
        return self._neo4j_client

    def _get_llm_client(self):
        from ..ingestion.embedding import get_embedding_client
        return get_embedding_client()

    def generate_entity_summary(self, entity_name: str) -> str:
        """为单个实体生成摘要"""
        client = self._get_neo4j_client()

        query = """
        MATCH (e:Entity {name: $entity_name})
        OPTIONAL MATCH (e)-[r]->(related)
        RETURN e.name as name,
               e.type as type,
               e.properties as properties,
               collect({target: related.name, relation: type(r)}) as relationships
        LIMIT 1
        """

        results = client.execute_query(query, {"entity_name": entity_name})
        
        if not results:
            return f"未找到实体: {entity_name}"

        entity = results[0]
        relationships = entity.get("relationships", [])
        
        summary_parts = [f"实体: {entity_name}"]
        entity_type = entity.get("type")
        if entity_type:
            summary_parts.append(f"类型: {entity_type}")
        
        # 格式化属性，过滤空属性
        props = entity.get("properties")
        if props:
            if isinstance(props, dict):
                prop_strs = []
                for k, v in props.items():
                    if v is not None and v != "" and v != {} and v != []:
                        # 确保值是字符串
                        prop_strs.append(f"{k}: {v}")
                if prop_strs:
                    summary_parts.append(f"属性: {', '.join(prop_strs)}")
        
        # 格式化关系，按关系类型分组 - 确保None安全
        if relationships and isinstance(relationships, list):
            # 按关系类型分组
            rel_groups = {}
            for r in relationships:
                if not r or not isinstance(r, dict):
                    continue
                rel_type = r.get("relation", "RELATED") or "RELATED"
                target = r.get("target", "") or ""
                # 确保是字符串
                rel_type = str(rel_type)
                target = str(target)
                if target:
                    if rel_type not in rel_groups:
                        rel_groups[rel_type] = []
                    rel_groups[rel_type].append(target)
            
            if rel_groups:
                rel_strs = []
                for rel_type, targets in rel_groups.items():
                    # 过滤掉空的目标
                    valid_targets = [t for t in targets if t]
                    if not valid_targets:
                        continue
                    targets_str = "、".join(valid_targets[:3])
                    if len(valid_targets) > 3:
                        targets_str += f"等{len(valid_targets)}个"
                    rel_strs.append(f"{rel_type}: {targets_str}")
                
                if rel_strs:
                    summary_parts.append(f"关系: {'; '.join(rel_strs)}")
        
        return " | ".join(summary_parts)

    def generate_community_summary(self, community_id: int, level: int = 0) -> str:
        """为社区生成摘要"""
        members = self._community_detector.get_community_members(community_id)
        
        if not members:
            return f"社区 {community_id} 没有成员"

        entity_summaries = []
        for member in members[:10]:
            entity_summaries.append(self.generate_entity_summary(member))

        centrality = self._community_detector.compute_community_centrality(community_id)
        key_entities = list(centrality.keys())[:3]
        
        summary = f"""社区 {community_id} (层级 {level})
成员数量: {len(members)}
核心实体: {', '.join(key_entities)}
实体列表: {', '.join(members[:5])}{'...' if len(members) > 5 else ''}

实体详情:
{chr(10).join(entity_summaries)}"""

        return summary

    def generate_community_summary_with_cache(self, community_id: int, level: int = 0) -> str:
        """为社区生成摘要（带缓存）"""
        cache_key = f"community_summary:{community_id}:{level}"
        
        cached = self._llm_cache.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for community {community_id} summary")
            return cached
        
        summary = self.generate_community_summary(community_id, level)
        self._llm_cache.cache.set(cache_key, summary)
        
        return summary

    def generate_hierarchical_summaries(self, levels: int = 3) -> Dict[int, str]:
        """生成分层摘要"""
        summaries = {}
        hierarchical_communities = self._community_detector.detect_hierarchical_communities(levels)

        for level, communities in enumerate(hierarchical_communities):
            for community_id in set(communities.values()):
                summaries[(level, community_id)] = self.generate_community_summary(community_id, level)

        return summaries

    def generate_global_summary(self) -> str:
        """生成全局摘要"""
        top_communities = self._community_detector.get_top_communities(top_n=3)
        
        summary_parts = ["知识图谱全局摘要"]
        
        for comm_id, count in top_communities:
            comm_summary = self.generate_community_summary(comm_id, 0)
            summary_parts.append(f"\n--- 社区 {comm_id} ({count}个实体) ---")
            summary_parts.append(comm_summary)

        return "\n".join(summary_parts)

    def get_summary(self, entity_name: Optional[str] = None, community_id: Optional[int] = None) -> str:
        """获取摘要（实体或社区）"""
        if entity_name:
            return self.generate_entity_summary(entity_name)
        elif community_id is not None:
            return self.generate_community_summary(community_id)
        else:
            return self.generate_global_summary()

    def summarize_query_context(self, query: str) -> Dict[str, Any]:
        """根据查询生成相关摘要上下文"""
        client = self._get_neo4j_client()

        keywords = [w for w in query.split() if len(w) > 2]
        
        related_entities = []
        for keyword in keywords[:5]:
            query_cypher = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $keyword OR e.name =~ $regex
            RETURN e.name as name, e.type as type
            LIMIT 3
            """
            results = client.execute_query(
                query_cypher,
                {"keyword": keyword, "regex": f".*{keyword}.*"}
            )
            related_entities.extend(results)

        entity_summaries = {}
        for entity in related_entities[:5]:
            entity_summaries[entity["name"]] = self.generate_entity_summary(entity["name"])

        communities = set()
        for entity in related_entities:
            comm_id = self._community_detector.get_entity_community(entity["name"])
            if comm_id is not None:
                communities.add(comm_id)

        community_summaries = {}
        for comm_id in communities:
            community_summaries[comm_id] = self.generate_community_summary(comm_id)

        return {
            "query": query,
            "related_entities": entity_summaries,
            "related_communities": community_summaries,
            "global_summary": self.generate_global_summary() if len(communities) > 0 else None
        }


def get_summary_generator() -> SummaryGenerator:
    """获取摘要生成器单例"""
    global _summary_generator
    if _summary_generator is None:
        _summary_generator = SummaryGenerator()
    return _summary_generator


_summary_generator: Optional[SummaryGenerator] = None