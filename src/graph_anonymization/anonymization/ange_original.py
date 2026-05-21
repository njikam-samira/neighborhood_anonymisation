from __future__ import annotations

import random
from typing import Dict, Sequence

import networkx as nx
import numpy as np

from .cluster_formation_mae import cluster_formation_MAE
from .reconstruction_optimize_mean import graph_reconstruction_optimize_mean


def create_weight_matrix(graph: nx.Graph, seed: int) -> np.ndarray:
    """Create a symmetric random weight matrix aligned with node ids."""
    if graph.number_of_nodes() == 0:
        return np.zeros((0, 0), dtype=np.int16)
    max_id = max(int(node) for node in graph.nodes())
    matrix = np.zeros((max_id + 1, max_id + 1), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for u, v in graph.edges():
        weight = int(rng.integers(1, 100))
        matrix[int(u), int(v)] = weight
        matrix[int(v), int(u)] = weight
    return matrix


def anonymize_ange_original(graph: nx.Graph, k: int, seed: int) -> nx.Graph:
    """Run Ange original anonymization with reconstruction stage."""
    if graph.number_of_nodes() > 1000:
        return anonymize_ange_original_scalable(graph, k=k, seed=seed)

    profile = [{"id": int(node), "degree": int(graph.degree[node])} for node in graph.nodes()]
    profile.sort(key=lambda item: item["degree"], reverse=True)
    clusters, cluster_degrees, _ = cluster_formation_MAE(profile, k)
    weights = create_weight_matrix(graph, seed=seed)
    _, _, _, _, _, updated_graph = graph_reconstruction_optimize_mean(
        clusters=clusters,
        cluster_degrees=cluster_degrees,
        CW=weights,
        G=graph.copy(),
    )
    return nx.Graph(updated_graph)


def anonymize_ange_original_scalable(graph: nx.Graph, k: int, seed: int, max_rounds: int = 3) -> nx.Graph:
    """Approximate scalable fallback for larger graphs."""
    if graph.number_of_nodes() == 0:
        return graph.copy()

    profile = [{"id": int(node), "degree": int(graph.degree[node])} for node in graph.nodes()]
    profile.sort(key=lambda item: item["degree"], reverse=True)
    clusters, cluster_degrees, _ = cluster_formation_MAE(profile, k)

    target_degree: Dict[int, int] = {}
    for index, cluster in enumerate(clusters):
        target = int(cluster_degrees[index])
        for node in cluster:
            target_degree[int(node["id"])] = target

    working = graph.copy()
    rng = random.Random(seed)
    nodes = [int(node) for node in working.nodes()]
    degree = {node: int(working.degree[node]) for node in nodes}
    for node in nodes:
        if node not in target_degree:
            target_degree[node] = degree[node]

    for _ in range(max_rounds):
        over_nodes = [node for node in nodes if degree[node] > target_degree[node]]
        rng.shuffle(over_nodes)
        for u in over_nodes:
            while degree[u] > target_degree[u]:
                candidates = [v for v in working.neighbors(u) if degree[int(v)] > target_degree[int(v)]]
                if not candidates:
                    break
                v = max(candidates, key=lambda x: (degree[int(x)] - target_degree[int(x)], degree[int(x)]))
                v = int(v)
                if working.has_edge(u, v):
                    working.remove_edge(u, v)
                    degree[u] -= 1
                    degree[v] -= 1

        under_nodes = [node for node in nodes if degree[node] < target_degree[node]]
        under_nodes.sort(key=lambda node: (target_degree[node] - degree[node], degree[node]), reverse=True)
        for u in under_nodes:
            while degree[u] < target_degree[u]:
                partner = None
                for v in under_nodes:
                    if v == u:
                        continue
                    if degree[v] < target_degree[v] and not working.has_edge(u, v):
                        partner = v
                        break
                if partner is None:
                    candidates = [v for v in nodes if v != u and not working.has_edge(u, v)]
                    if not candidates:
                        break
                    partner = min(
                        candidates,
                        key=lambda v: (max(0, degree[v] - target_degree[v]), degree[v], v),
                    )
                working.add_edge(u, partner)
                degree[u] += 1
                degree[partner] += 1

    return nx.Graph(working)

