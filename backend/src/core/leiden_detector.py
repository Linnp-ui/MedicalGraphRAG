import igraph as ig
import networkx as nx
from typing import Dict, List, Optional
from loguru import logger


class LeidenCommunityDetector:
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution

    def _nx_to_igraph(self, graph: nx.Graph) -> ig.Graph:
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
        g = ig.Graph(n=len(graph.nodes()), edges=edges)
        g.vs["name"] = list(graph.nodes())
        return g

    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        if len(graph.nodes()) == 0:
            return {}

        ig_graph = self._nx_to_igraph(graph)

        import leidenalg
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.CPMVertexPartition,
            resolution_parameter=self.resolution
        )

        node_to_community = {}
        for community_id, community in enumerate(partition):
            for node_idx in community:
                node_name = ig_graph.vs[node_idx]["name"]
                node_to_community[node_name] = community_id

        logger.info(f"Detected {len(set(node_to_community.values()))} communities")
        return node_to_community

    def detect_hierarchical(
        self,
        graph: nx.Graph,
        levels: int = 3
    ) -> List[Dict[str, int]]:
        partitions = []
        current_graph = graph.copy()
        current_partition = self.detect_communities(current_graph)
        partitions.append(current_partition.copy())

        for level in range(1, levels):
            communities = {}
            for node, comm_id in current_partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)

            if len(communities) <= 1:
                for _ in range(levels - level):
                    partitions.append(current_partition.copy())
                break

            community_graph = nx.Graph()
            for comm_id in communities:
                community_graph.add_node(comm_id)

            for u, v in current_graph.edges():
                comm_u = current_partition[u]
                comm_v = current_partition[v]
                if comm_u != comm_v:
                    if community_graph.has_edge(comm_u, comm_v):
                        community_graph[comm_u][comm_v]["weight"] += 1
                    else:
                        community_graph.add_edge(comm_u, comm_v, weight=1)

            if len(community_graph.nodes()) < 2:
                for _ in range(levels - level):
                    partitions.append(current_partition.copy())
                break

            higher_partition = self.detect_communities(community_graph)

            new_partition = {}
            for node, old_comm in current_partition.items():
                new_comm = higher_partition.get(old_comm, old_comm)
                new_partition[node] = new_comm

            current_partition = new_partition
            partitions.append(current_partition.copy())

        while len(partitions) < levels:
            partitions.append(partitions[-1].copy())

        logger.info(f"Built {len(partitions)} level hierarchical partition")
        return partitions

    def compute_modularity(
        self,
        graph: nx.Graph,
        partition: Dict[str, int]
    ) -> float:
        if len(graph.nodes()) == 0 or len(partition) == 0:
            return 0.0

        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)

        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        q = 0.0
        for comm_nodes in communities.values():
            comm_subgraph = graph.subgraph(comm_nodes)
            lc = comm_subgraph.number_of_edges()

            dc = sum(graph.degree(node) for node in comm_nodes if node in graph)

            q += (lc / m) - (dc / (2 * m)) ** 2

        return q


_leiden_detector: Optional[LeidenCommunityDetector] = None


def get_leiden_detector() -> LeidenCommunityDetector:
    global _leiden_detector
    if _leiden_detector is None:
        from .config import get_settings
        settings = get_settings()
        _leiden_detector = LeidenCommunityDetector(
            resolution=settings.community_resolution
        )
    return _leiden_detector
