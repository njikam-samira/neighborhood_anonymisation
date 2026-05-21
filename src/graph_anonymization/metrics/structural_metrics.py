from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np


def calculate_apl(graph: nx.Graph) -> float:
    """Compute average path length over reachable node pairs."""
    total_distance = 0
    num_pairs = 0
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                distance = nx.shortest_path_length(graph, source=nodes[i], target=nodes[j])
                total_distance += distance
                num_pairs += 1
            except nx.NetworkXNoPath:
                continue
    if num_pairs == 0:
        return float("inf")
    n = graph.number_of_nodes()
    return (2 * total_distance) / (n * (n - 1))


def calculate_il(profile_original: List[Dict[int, int]], profile_new: List[Dict[int, int]]) -> float:
    """RMSE information loss on node degrees."""
    return float(np.sqrt(np.mean([(orig["degree"] - new["degree"]) ** 2 for orig, new in zip(profile_original, profile_new)])))


def calculate_clustering_coefficient(graph: nx.Graph) -> float:
    """Average clustering coefficient."""
    return float(nx.average_clustering(graph))


def calculate_edge_intersection(graph_original: nx.Graph, graph_anonymized: nx.Graph) -> float:
    """Edge-intersection ratio between original and anonymized graphs."""
    original_edges = set(graph_original.edges())
    anonymized_edges = set(graph_anonymized.edges())
    if not original_edges:
        return 0.0
    intersection = len(original_edges.intersection(anonymized_edges))
    return float(intersection / len(original_edges))


def verify_k_degree_anonymity(profile_new: Sequence, k: int) -> bool:
    """
    Verify k-degree anonymity for either:
    - list of tuples: [(node_id, degree), ...]
    - list of dicts: [{'id': ..., 'degree': ...}, ...]
    """
    degree_counts: Dict[int, int] = {}
    for item in profile_new:
        if isinstance(item, dict):
            degree = int(item["degree"])
        else:
            degree = int(item[1])
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    return all(count >= k for count in degree_counts.values())

