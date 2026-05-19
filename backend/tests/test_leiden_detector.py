import pytest
import networkx as nx
from src.core.leiden_detector import LeidenCommunityDetector


class TestLeidenCommunityDetector:

    def test_detect_communities_basic(self):
        detector = LeidenCommunityDetector()
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})

        partition = detector.detect_communities(str_graph)

        assert isinstance(partition, dict)
        assert len(partition) == 34
        assert all(isinstance(v, int) for v in partition.values())

    def test_detect_hierarchical(self):
        detector = LeidenCommunityDetector()
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})

        partitions = detector.detect_hierarchical(str_graph, levels=3)

        assert isinstance(partitions, list)
        assert len(partitions) == 3
        assert all(isinstance(p, dict) for p in partitions)

    def test_compute_modularity(self):
        detector = LeidenCommunityDetector(resolution=0.1)
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})

        partition = detector.detect_communities(str_graph)
        modularity = detector.compute_modularity(str_graph, partition)

        assert isinstance(modularity, float)
        assert -0.5 <= modularity <= 1

    def test_resolution_parameter(self):
        detector_low = LeidenCommunityDetector(resolution=0.5)
        detector_high = LeidenCommunityDetector(resolution=2.0)
        graph = nx.karate_club_graph()
        str_graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})

        partition_low = detector_low.detect_communities(str_graph)
        partition_high = detector_high.detect_communities(str_graph)

        num_communities_low = len(set(partition_low.values()))
        num_communities_high = len(set(partition_high.values()))

        assert num_communities_low <= num_communities_high
