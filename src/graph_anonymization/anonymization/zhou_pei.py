from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np


def compute_structural_signatures(graph: nx.Graph) -> Dict[int, np.ndarray]:
    """Compute local structural signatures used by Zhou-Pei style grouping."""
    features: Dict[int, np.ndarray] = {}
    triangles = nx.triangles(graph)
    clustering = nx.clustering(graph)
    for node in graph.nodes():
        degree = int(graph.degree[node])
        neighbors = list(graph.neighbors(node))
        avg_neighbor_degree = float(np.mean([graph.degree[n] for n in neighbors])) if neighbors else 0.0
        neighborhood = graph.subgraph(neighbors).copy()
        component_sizes = (
            sorted((len(component) for component in nx.connected_components(neighborhood)), reverse=True)
            if neighborhood.number_of_nodes()
            else []
        )
        features[int(node)] = np.array(
            [
                float(degree),
                float(triangles[node]),
                float(clustering[node]),
                float(avg_neighbor_degree),
                float(neighborhood.number_of_edges()),
                float((component_sizes + [0, 0, 0])[:3][0]),
                float((component_sizes + [0, 0, 0])[:3][1]),
            ],
            dtype=float,
        )
    return features


def harmonize_group_neighborhoods_add_only(graph: nx.Graph, group: Sequence[int]) -> None:
    """Harmonize neighborhoods by adding edges only."""
    if len(group) < 2:
        return
    group_nodes = sorted(int(node) for node in group)
    group_set = set(group_nodes)

    ordered_by_degree = sorted(group_nodes, key=lambda node: graph.degree[node], reverse=True)
    anchor_count = min(3, max(1, len(group_nodes) - 1))
    anchors = ordered_by_degree[:anchor_count]
    for node in group_nodes:
        for anchor in anchors:
            if node != anchor and not graph.has_edge(node, anchor):
                graph.add_edge(node, anchor)

    external_counter: Counter[int] = Counter()
    external_sizes: List[int] = []
    for node in group_nodes:
        ext_neighbors = [nbr for nbr in graph.neighbors(node) if nbr not in group_set]
        external_sizes.append(len(ext_neighbors))
        external_counter.update(ext_neighbors)
    target_external = int(round(float(np.median(external_sizes)))) if external_sizes else 0
    target_external = max(target_external, 0)
    ranked_candidates = sorted(
        external_counter.items(),
        key=lambda item: (-item[1], int(graph.degree[item[0]]), int(item[0])),
    )
    template_external = [node for node, _ in ranked_candidates[:target_external]]
    for node in group_nodes:
        current_external = {nbr for nbr in graph.neighbors(node) if nbr not in group_set}
        for nbr in template_external:
            if nbr != node and nbr not in current_external:
                graph.add_edge(node, nbr)


def anonymize_zhou_pei(graph: nx.Graph, k: int, seed: int) -> nx.Graph:
    """Simplified Zhou-Pei inspired anonymization."""
    rng = random.Random(seed)
    working = graph.copy()
    nodes = sorted(int(node) for node in working.nodes())
    features = compute_structural_signatures(working)
    unprocessed = set(nodes)

    while unprocessed:
        if len(unprocessed) <= k:
            group = sorted(unprocessed)
            harmonize_group_neighborhoods_add_only(working, group)
            break
        seed_node = max(unprocessed, key=lambda node: (working.degree[node], -node))
        seed_vec = features[seed_node]
        candidates = [node for node in unprocessed if node != seed_node]
        candidates.sort(
            key=lambda node: (
                float(np.linalg.norm(features[node] - seed_vec)),
                -working.degree[node],
                node,
            )
        )
        group = [seed_node] + candidates[: max(0, k - 1)]
        harmonize_group_neighborhoods_add_only(working, group)
        unprocessed -= set(group)

    _ = rng.random()
    return nx.Graph(working)

