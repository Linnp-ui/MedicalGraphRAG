from typing import Any, Dict, List, Literal, Optional, Tuple
import networkx as nx
from community import community_louvain
from loguru import logger

from ..core.neo4j_client import Neo4jClient
from .leiden_detector import LeidenCommunityDetector


class CommunityDetector:
    """社区检测模块 - 使用Leiden/Louvain算法对知识图谱进行社区聚类"""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        algorithm: Literal["leiden", "louvain"] = "louvain",
        resolution: float = 1.0
    ):
        self._neo4j_client = neo4j_client
        self._algorithm = algorithm
        self._resolution = resolution
        self._leiden_detector = LeidenCommunityDetector(resolution=resolution) if algorithm == "leiden" else None
        self._communities = {}
        self._community_summaries = {}

    def _get_neo4j_client(self) -> Neo4jClient:
        if self._neo4j_client is None:
            from ..core.neo4j_client import get_neo4j_client
            self._neo4j_client = get_neo4j_client()
        return self._neo4j_client

    def build_graph_from_neo4j(self) -> nx.Graph:
        """从Neo4j构建NetworkX图"""
        client = self._get_neo4j_client()

        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN e1.name as source, e2.name as target, type(r) as relation_type
        """

        results = client.execute_query(query)
        G = nx.Graph()

        for result in results:
            source = result["source"]
            target = result["target"]
            G.add_edge(source, target, relation_type=result["relation_type"])

        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def detect_communities(self, graph: Optional[nx.Graph] = None) -> Dict[str, int]:
        """使用Leiden或Louvain算法检测社区"""
        if graph is None:
            graph = self.build_graph_from_neo4j()

        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping community detection")
            return {}

        try:
            if self._algorithm == "leiden" and self._leiden_detector is not None:
                partition = self._leiden_detector.detect_communities(graph)
            else:
                partition = community_louvain.best_partition(graph, resolution=self._resolution)
            self._communities = partition

            community_counts = {}
            for node, community_id in partition.items():
                community_counts[community_id] = community_counts.get(community_id, 0) + 1

            logger.info(f"Detected {len(community_counts)} communities using {self._algorithm}")
            for comm_id, count in community_counts.items():
                logger.debug(f"Community {comm_id}: {count} nodes")

            return partition
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}

    def get_communities(self) -> Dict[int, List[str]]:
        """获取按社区分组的实体列表"""
        if not self._communities:
            self.detect_communities()

        result = {}
        for node, community_id in self._communities.items():
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node)

        return result

    def get_community_members(self, community_id: int) -> List[str]:
        """获取指定社区的成员"""
        communities = self.get_communities()
        return communities.get(community_id, [])

    def get_entity_community(self, entity_name: str) -> Optional[int]:
        """获取实体所属的社区"""
        return self._communities.get(entity_name)

    def get_top_communities(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """获取最大的N个社区"""
        communities = self.get_communities()
        sorted_communities = sorted(
            communities.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        return [(comm_id, len(members)) for comm_id, members in sorted_communities[:top_n]]

    def compute_community_centrality(self, community_id: int) -> Dict[str, float]:
        """计算社区内节点的中心性"""
        graph = self.build_graph_from_neo4j()
        members = self.get_community_members(community_id)
        
        if not members:
            return {}

        subgraph = graph.subgraph(members)
        betweenness = nx.betweenness_centrality(subgraph)
        degree = nx.degree_centrality(subgraph)

        result = {}
        for member in members:
            result[member] = {
                "betweenness": betweenness.get(member, 0),
                "degree": degree.get(member, 0),
                "combined": (betweenness.get(member, 0) + degree.get(member, 0)) / 2
            }

        return dict(sorted(result.items(), key=lambda x: x[1]["combined"], reverse=True))

    def detect_hierarchical_communities(self, levels: int = 3) -> List[Dict[int, List[str]]]:
        """检测分层社区结构"""
        communities = []
        current_partition = self.detect_communities()
        
        communities.append(current_partition)
        
        for level in range(1, levels):
            community_groups = {}
            for node, comm_id in current_partition.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(node)
            
            meta_graph = nx.Graph()
            for comm_id, members in community_groups.items():
                meta_graph.add_node(comm_id)
            
            for comm_id1, members1 in community_groups.items():
                for comm_id2, members2 in community_groups.items():
                    if comm_id1 < comm_id2:
                        has_connection = False
                        for m1 in members1:
                            for m2 in members2:
                                if m1 in self._communities and m2 in self._communities:
                                    has_connection = True
                                    break
                            if has_connection:
                                break
                        if has_connection:
                            meta_graph.add_edge(comm_id1, comm_id2)
            
            if meta_graph.number_of_nodes() > 1:
                try:
                    current_partition = community_louvain.best_partition(meta_graph)
                    communities.append(current_partition)
                except Exception as e:
                    logger.warning(f"Hierarchical detection level {level} failed: {e}")
                    break
            else:
                break
        
        return communities


def get_community_detector(
    algorithm: str = "louvain",
    resolution: float = 1.0
) -> CommunityDetector:
    """获取社区检测单例"""
    global _community_detector
    if _community_detector is None:
        _community_detector = CommunityDetector(algorithm=algorithm, resolution=resolution)
    return _community_detector


_community_detector: Optional[CommunityDetector] = None