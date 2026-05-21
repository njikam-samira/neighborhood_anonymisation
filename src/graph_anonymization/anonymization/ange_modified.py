from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random

import networkx as nx
import numpy as np


Signature = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class AngeModifieConfig:
    k: int
    alpha: float = 1.0
    beta: float = 1.0
    passes: int = 2


def _ncc_signature(graph: nx.Graph, node: int, max_components: int = 4) -> Signature:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return (tuple([0] * max_components), tuple([0] * max_components))

    neighborhood = graph.subgraph(neighbors)
    component_nodes = [set(component) for component in nx.connected_components(neighborhood)]
    component_sizes = sorted((len(component) for component in component_nodes), reverse=True)
    component_edges = sorted(
        int(neighborhood.subgraph(component).number_of_edges()) for component in component_nodes
    )

    padded_sizes = tuple((component_sizes + [0] * max_components)[:max_components])
    padded_edges = tuple((component_edges + [0] * max_components)[:max_components])
    return padded_sizes, padded_edges


def _ncc_distance(left: Signature, right: Signature) -> float:
    left_sizes, left_edges = left
    right_sizes, right_edges = right
    size_diff = sum(abs(float(a) - float(b)) for a, b in zip(left_sizes, right_sizes))
    edge_diff = sum(abs(float(a) - float(b)) for a, b in zip(left_edges, right_edges))
    size_norm = 1.0 + float(max(sum(left_sizes), sum(right_sizes), 1))
    edge_norm = 1.0 + float(max(sum(left_edges), sum(right_edges), 1))
    return (size_diff / size_norm) + (edge_diff / edge_norm)


def _clusterisation_ncc_fast(graph: nx.Graph, k: int) -> List[List[int]]:
    nodes = sorted(int(node) for node in graph.nodes())
    if not nodes:
        return []
    if k <= 1:
        return [[node] for node in nodes]

    degrees = {node: int(graph.degree[node]) for node in nodes}
    signatures = {node: _ncc_signature(graph, node) for node in nodes}
    ordered = sorted(nodes, key=lambda node: (-degrees[node], signatures[node], node))
    groups = [ordered[i : i + k] for i in range(0, len(ordered), k)]
    if len(groups) > 1 and len(groups[-1]) < k:
        groups[-2].extend(groups[-1])
        groups.pop()
    return [sorted(group) for group in groups if group]


def _node_distance(
    node_a: int,
    node_b: int,
    degrees: Dict[int, int],
    signatures: Dict[int, Signature],
    alpha: float,
    beta: float,
    max_degree: float,
) -> float:
    degree_gap = abs(float(degrees[node_a]) - float(degrees[node_b])) / max_degree
    ncc_gap = _ncc_distance(signatures[node_a], signatures[node_b])
    return (alpha * degree_gap) + (beta * ncc_gap)


def clusterisation_ncc(
    graph: nx.Graph,
    k: int,
    alpha: float,
    beta: float,
    fast_mode: bool = False,
) -> List[List[int]]:
    if fast_mode:
        return _clusterisation_ncc_fast(graph, k)

    nodes = sorted(int(node) for node in graph.nodes())
    if not nodes:
        return []
    if k <= 1:
        return [[node] for node in nodes]

    degrees = {node: int(graph.degree[node]) for node in nodes}
    signatures = {node: _ncc_signature(graph, node) for node in nodes}
    max_degree = max(float(max(degrees.values()) if degrees else 1), 1.0)

    remaining = set(nodes)
    clusters: List[List[int]] = []

    while len(remaining) >= k:
        seed = max(remaining, key=lambda node: (degrees[node], -node))
        remaining.remove(seed)
        candidates = sorted(
            remaining,
            key=lambda node: (_node_distance(seed, node, degrees, signatures, alpha, beta, max_degree), -degrees[node], node),
        )
        selected = candidates[: max(0, k - 1)]
        for node in selected:
            remaining.remove(node)
        clusters.append(sorted([seed] + selected))

    leftovers = sorted(remaining)
    if leftovers:
        if not clusters:
            clusters = [leftovers]
        else:
            for idx, node in enumerate(leftovers):
                clusters[idx % len(clusters)].append(node)

    return [sorted(cluster) for cluster in clusters if cluster]


def _internal_template_edges(group: Sequence[int]) -> set[Tuple[int, int]]:
    ordered = sorted(int(node) for node in group)
    size = len(ordered)
    if size < 2:
        return set()
    ring_radius = 1 if size < 6 else 2
    expected: set[Tuple[int, int]] = set()
    for i, src in enumerate(ordered):
        for step in range(1, ring_radius + 1):
            dst = ordered[(i + step) % size]
            if src != dst:
                expected.add((min(src, dst), max(src, dst)))
    return expected


def _harmonize_cluster_internal_template(graph: nx.Graph, cluster_nodes: Sequence[int]) -> None:
    if len(cluster_nodes) < 2:
        return
    desired_internal = _internal_template_edges(cluster_nodes)
    current_internal = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in graph.subgraph(cluster_nodes).edges()
    }
    for u, v in current_internal - desired_internal:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
    for u, v in desired_internal - current_internal:
        graph.add_edge(u, v)


def _harmonize_cluster_external_template(graph: nx.Graph, cluster_nodes: Sequence[int]) -> None:
    if len(cluster_nodes) < 2:
        return

    group_nodes = sorted(int(node) for node in cluster_nodes)
    group_set = set(group_nodes)
    external_counter: Counter[int] = Counter()
    external_sizes: List[int] = []
    for node in group_nodes:
        ext_neighbors = [int(nbr) for nbr in graph.neighbors(node) if int(nbr) not in group_set]
        external_sizes.append(len(ext_neighbors))
        external_counter.update(ext_neighbors)

    target_external = int(round(float(np.median(external_sizes)))) if external_sizes else 0
    target_external = max(target_external, 0)
    ranked_candidates = sorted(
        external_counter.items(),
        key=lambda item: (-item[1], int(graph.degree[item[0]]), int(item[0])),
    )
    template_external = [node for node, _ in ranked_candidates[:target_external]]
    template_set = set(template_external)

    for node in group_nodes:
        current_external = {int(nbr) for nbr in graph.neighbors(node) if int(nbr) not in group_set}
        for nbr in list(current_external - template_set):
            if graph.has_edge(node, nbr):
                graph.remove_edge(node, nbr)
        for nbr in template_set - current_external:
            if nbr != node:
                graph.add_edge(node, nbr)


def _build_cluster_profile(graph: nx.Graph, cluster: Sequence[int]) -> Tuple[int, Signature, List[int]]:
    ordered = sorted(int(node) for node in cluster)
    if not ordered:
        return 0, ((0, 0, 0, 0), (0, 0, 0, 0)), []
    degrees = [int(graph.degree[node]) for node in ordered]
    target_degree = int(round(float(np.median(degrees))))
    signatures = [_ncc_signature(graph, node) for node in ordered]
    target_ncc = Counter(signatures).most_common(1)[0][0]
    return target_degree, target_ncc, ordered


def _score_node_to_profile(
    graph: nx.Graph,
    node: int,
    target_degree: int,
    target_ncc: Signature,
    alpha: float,
    beta: float,
    max_degree: float,
) -> float:
    node_degree = int(graph.degree[node])
    node_ncc = _ncc_signature(graph, node)
    degree_gap = abs(float(node_degree) - float(target_degree)) / max_degree
    ncc_gap = _ncc_distance(node_ncc, target_ncc)
    return (alpha * degree_gap) + (beta * ncc_gap)


def _candidate_add_neighbors(graph: nx.Graph, node: int, group_set: set[int], rng: random.Random) -> List[int]:
    neighbors = set(graph.neighbors(node))
    cluster_candidates = [v for v in group_set if v != node and v not in neighbors]
    two_hop: set[int] = set()
    for nbr in neighbors:
        two_hop.update(int(x) for x in graph.neighbors(nbr))
    two_hop = {v for v in two_hop if v != node and v not in neighbors}
    all_nodes = list(graph.nodes())
    rng.shuffle(all_nodes)
    random_pool = [int(v) for v in all_nodes[:64] if int(v) != node and int(v) not in neighbors]
    merged = list(dict.fromkeys(cluster_candidates + sorted(two_hop)[:64] + random_pool))
    return merged[:96]


def _candidate_remove_neighbors(graph: nx.Graph, node: int, group_set: set[int]) -> List[int]:
    neighbors = [int(nbr) for nbr in graph.neighbors(node)]
    outside = [nbr for nbr in neighbors if nbr not in group_set]
    inside = [nbr for nbr in neighbors if nbr in group_set]
    outside.sort(key=lambda nbr: (graph.degree[nbr], nbr))
    inside.sort(key=lambda nbr: (graph.degree[nbr], nbr))
    return outside + inside


def _apply_best_ange_operation(
    graph: nx.Graph,
    node: int,
    cluster_nodes: Sequence[int],
    target_degree: int,
    target_ncc: Signature,
    alpha: float,
    beta: float,
    max_degree: float,
    rng: random.Random,
) -> bool:
    group_set = set(int(v) for v in cluster_nodes)
    current_score = _score_node_to_profile(graph, node, target_degree, target_ncc, alpha, beta, max_degree)
    best_delta = 0.0
    best_op: Tuple[str, int] | None = None

    for candidate in _candidate_add_neighbors(graph, node, group_set, rng):
        if graph.has_edge(node, candidate):
            continue
        graph.add_edge(node, candidate)
        new_score = _score_node_to_profile(graph, node, target_degree, target_ncc, alpha, beta, max_degree)
        graph.remove_edge(node, candidate)
        delta = current_score - new_score
        if delta > best_delta:
            best_delta = delta
            best_op = ("add", candidate)

    for candidate in _candidate_remove_neighbors(graph, node, group_set)[:96]:
        if not graph.has_edge(node, candidate):
            continue
        graph.remove_edge(node, candidate)
        new_score = _score_node_to_profile(graph, node, target_degree, target_ncc, alpha, beta, max_degree)
        graph.add_edge(node, candidate)
        delta = current_score - new_score
        if delta > best_delta:
            best_delta = delta
            best_op = ("remove", candidate)

    if best_op is None:
        return False
    op_type, candidate = best_op
    if op_type == "add":
        graph.add_edge(node, candidate)
    elif graph.has_edge(node, candidate):
        graph.remove_edge(node, candidate)
    return True


def anonymize_ange_modified_ncc(
    graph: nx.Graph,
    k: int,
    seed: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    passes: int = 2,
    max_node_iterations: int = 24,
    fast_graph_threshold: int = 1000,
) -> nx.Graph:
    """
    Implémentation du pseudo-code ANGE_MODIFIE_NCC (Algorithme.pdf).
    """
    if k < 2:
        raise ValueError("k must be >= 2")

    cfg = AngeModifieConfig(k=k, alpha=float(alpha), beta=float(beta), passes=max(1, int(passes)))
    rng = random.Random(seed)
    modified = graph.copy()
    if modified.number_of_nodes() == 0:
        return modified

    fast_mode = modified.number_of_nodes() >= int(fast_graph_threshold)
    active_passes = 1 if fast_mode else cfg.passes
    active_iterations = 0 if fast_mode else max_node_iterations

    for _ in range(active_passes):
        clusters = clusterisation_ncc(modified, cfg.k, cfg.alpha, cfg.beta, fast_mode=fast_mode)
        max_degree = max(float(max((modified.degree[node] for node in modified.nodes()), default=1)), 1.0)

        for cluster in clusters:
            target_degree, target_ncc, ordered_nodes = _build_cluster_profile(modified, cluster)
            if not ordered_nodes:
                continue
            _harmonize_cluster_internal_template(modified, ordered_nodes)
            _harmonize_cluster_external_template(modified, ordered_nodes)

            for node in ordered_nodes:
                for _iter in range(active_iterations):
                    score = _score_node_to_profile(
                        modified,
                        node,
                        target_degree,
                        target_ncc,
                        cfg.alpha,
                        cfg.beta,
                        max_degree,
                    )
                    if score <= 1e-12:
                        break
                    changed = _apply_best_ange_operation(
                        modified,
                        node,
                        ordered_nodes,
                        target_degree,
                        target_ncc,
                        cfg.alpha,
                        cfg.beta,
                        max_degree,
                        rng,
                    )
                    if not changed:
                        break

    return nx.Graph(modified)
