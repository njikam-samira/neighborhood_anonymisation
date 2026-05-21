from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np


def average_path_length_reachable(graph: nx.Graph) -> float:
    """Average shortest-path length computed on reachable node pairs."""
    if graph.number_of_nodes() < 2:
        return 0.0
    ordered_nodes = sorted(int(node) for node in graph.nodes())
    index = {node: idx for idx, node in enumerate(ordered_nodes)}
    total_distance = 0.0
    pair_count = 0
    for source, distances in nx.all_pairs_shortest_path_length(graph):
        src_idx = index[int(source)]
        for target, distance in distances.items():
            if index[int(target)] > src_idx:
                total_distance += float(distance)
                pair_count += 1
    if pair_count == 0:
        return float("inf")
    return total_distance / float(pair_count)


def degree_mae(original_graph: nx.Graph, anonymized_graph: nx.Graph) -> float:
    """Mean absolute degree error between original and anonymized graphs."""
    if original_graph.number_of_nodes() == 0:
        return 0.0
    errors = []
    for node in original_graph.nodes():
        original_degree = float(original_graph.degree[node])
        anonymized_degree = float(anonymized_graph.degree[node]) if anonymized_graph.has_node(node) else 0.0
        errors.append(abs(original_degree - anonymized_degree))
    return float(np.mean(errors))


def compute_graph_stats(graph: nx.Graph) -> Dict[str, float]:
    """Compute generic structural statistics for a graph."""
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "clustering": float(nx.average_clustering(graph)),
        "apl": float(average_path_length_reachable(graph)),
    }


def compute_metrics(
    original_graph: nx.Graph,
    anonymized_graph: nx.Graph,
    original_stats: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Compute structural deltas and utility metrics against the original graph."""
    if original_stats is None:
        original_stats = compute_graph_stats(original_graph)
    anonymized_stats = compute_graph_stats(anonymized_graph)

    return {
        "original_nodes": original_stats["nodes"],
        "anonymized_nodes": anonymized_stats["nodes"],
        "node_variation": anonymized_stats["nodes"] - original_stats["nodes"],
        "original_edges": original_stats["edges"],
        "anonymized_edges": anonymized_stats["edges"],
        "edge_variation": anonymized_stats["edges"] - original_stats["edges"],
        "original_density": original_stats["density"],
        "anonymized_density": anonymized_stats["density"],
        "density_variation": anonymized_stats["density"] - original_stats["density"],
        "mae": degree_mae(original_graph, anonymized_graph),
        "original_clustering": original_stats["clustering"],
        "anonymized_clustering": anonymized_stats["clustering"],
        "clustering_variation": anonymized_stats["clustering"] - original_stats["clustering"],
        "original_apl": original_stats["apl"],
        "anonymized_apl": anonymized_stats["apl"],
        "apl_variation": anonymized_stats["apl"] - original_stats["apl"],
    }

