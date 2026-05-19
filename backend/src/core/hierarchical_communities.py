import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .leiden_detector import LeidenCommunityDetector, get_leiden_detector


class HierarchicalCommunityManager:
    def __init__(
        self, 
        levels: int = 3,
        detector: Optional[LeidenCommunityDetector] = None
    ):
        self.levels = levels
        self._detector = detector or get_leiden_detector()
        self._partitions: List[Dict[str, int]] = []
        self._summaries: Dict[Tuple[int, int], str] = {}
        self._embeddings: Dict[Tuple[int, int], List[float]] = {}
        self._graph: Optional[nx.Graph] = None
    
    def build_hierarchy(self, graph: nx.Graph) -> None:
        self._graph = graph.copy()
        self._partitions = self._detector.detect_hierarchical(graph, self.levels)
        self._summaries.clear()
        self._embeddings.clear()
        
        logger.info(
            f"Built hierarchy with {self.levels} levels, "
            f"communities per level: {[len(set(p.values())) for p in self._partitions]}"
        )
    
    def get_community_at_level(self, entity: str, level: int) -> int:
        if not self._partitions:
            raise ValueError("Hierarchy not built. Call build_hierarchy first.")
        
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}")
        
        return self._partitions[level].get(entity, -1)
    
    def get_communities_by_level(self, level: int) -> Dict[int, List[str]]:
        if not self._partitions:
            raise ValueError("Hierarchy not built. Call build_hierarchy first.")
        
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}")
        
        communities: Dict[int, List[str]] = {}
        for entity, comm_id in self._partitions[level].items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(entity)
        
        return communities
    
    def get_community_members(self, level: int, community_id: int) -> List[str]:
        communities = self.get_communities_by_level(level)
        return communities.get(community_id, [])
    
    def get_community_summary(self, level: int, community_id: int) -> str:
        cache_key = (level, community_id)
        
        if cache_key in self._summaries:
            return self._summaries[cache_key]
        
        members = self.get_community_members(level, community_id)
        if not members:
            return f"社区 {community_id}（层级 {level}）为空"
        
        summary = f"社区 {community_id}（层级 {level}）\n"
        summary += f"成员数量: {len(members)}\n"
        summary += f"成员: {', '.join(members[:10])}"
        if len(members) > 10:
            summary += f"... 等 {len(members)} 个实体"
        
        self._summaries[cache_key] = summary
        return summary
    
    def get_community_embedding(
        self, 
        level: int, 
        community_id: int
    ) -> Optional[List[float]]:
        cache_key = (level, community_id)
        
        if cache_key in self._embeddings:
            return self._embeddings[cache_key]
        
        return None
    
    def set_community_embedding(
        self, 
        level: int, 
        community_id: int, 
        embedding: List[float]
    ) -> None:
        cache_key = (level, community_id)
        self._embeddings[cache_key] = embedding
    
    def find_relevant_communities(
        self, 
        query_embedding: List[float], 
        level: int = 1,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        import numpy as np
        
        communities = self.get_communities_by_level(level)
        scores = []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        for comm_id in communities:
            embedding = self.get_community_embedding(level, comm_id)
            if embedding is None:
                continue
            
            comm_vec = np.array(embedding)
            comm_norm = np.linalg.norm(comm_vec)
            
            if query_norm == 0 or comm_norm == 0:
                continue
            
            similarity = np.dot(query_vec, comm_vec) / (query_norm * comm_norm)
            scores.append((comm_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._partitions:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "levels": self.levels,
            "communities_per_level": [
                len(set(p.values())) for p in self._partitions
            ],
            "total_entities": len(self._partitions[0]) if self._partitions else 0,
            "cached_summaries": len(self._summaries),
            "cached_embeddings": len(self._embeddings),
        }


_hierarchical_manager: Optional[HierarchicalCommunityManager] = None


def get_hierarchical_manager() -> HierarchicalCommunityManager:
    global _hierarchical_manager
    if _hierarchical_manager is None:
        from .config import get_settings
        settings = get_settings()
        _hierarchical_manager = HierarchicalCommunityManager(
            levels=settings.community_levels
        )
    return _hierarchical_manager
