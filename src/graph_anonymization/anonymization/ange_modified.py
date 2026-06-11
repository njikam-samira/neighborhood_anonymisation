from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import random

import networkx as nx
import numpy as np

from .ncc import NCCCode, calculate_all_ncc, calculate_ncc, max_ncc, ncc_distance

try:
    from graph_anonymization.metrics.structural_metrics import calculate_edge_intersection
except Exception:  # pragma: no cover - optional import path in some legacy contexts
    calculate_edge_intersection = None


@dataclass(frozen=True)
class AngeModifieConfig:
    k: int
    alpha: float = 1.0
    beta: float = 0.2
    passes: int = 2
    removal_penalty: float = 0.5
    preserve_original_edges: bool = True


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def _normalized_edge_set(graph: nx.Graph) -> Set[Tuple[int, int]]:
    return {_edge_key(int(u), int(v)) for u, v in graph.edges() if int(u) != int(v)}


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _clusterisation_ncc_fast(graph: nx.Graph, k: int) -> List[List[int]]:
    """
    Fast fallback used only on large graphs.

    We keep a lighter NCC representation (simplified=True) to avoid excessive
    runtime/memory pressure while preserving deterministic behavior.
    """
    nodes = sorted(int(node) for node in graph.nodes())
    if not nodes:
        return []
    if k <= 1:
        return [[node] for node in nodes]

    degrees = {node: int(graph.degree[node]) for node in nodes}
    signatures = calculate_all_ncc(graph, simplified=True, max_components=4, max_degree_entries=0)
    ordered = sorted(nodes, key=lambda node: (-degrees[node], signatures[node], node))
    groups = [ordered[i : i + k] for i in range(0, len(ordered), k)]
    if len(groups) > 1 and len(groups[-1]) < k:
        groups[-2].extend(groups[-1])
        groups.pop()
    return [sorted(group) for group in groups if group]


def _pair_degree_distance(node_a: int, node_b: int, degrees: Dict[int, int], max_degree: float) -> float:
    return abs(float(degrees[node_a]) - float(degrees[node_b])) / max_degree


def _pair_ncc_distance(code_a: NCCCode, code_b: NCCCode) -> float:
    return float(ncc_distance(code_a, code_b, max_value=max_ncc(code_a, code_b)))


def _cluster_cost(
    candidate: int,
    cluster_nodes: Sequence[int],
    degrees: Dict[int, int],
    signatures: Dict[int, NCCCode],
    alpha: float,
    beta: float,
    max_degree: float,
) -> float:
    # cost(u, C) = alpha * avg degree distance + beta * avg NCC distance
    deg_dist = _safe_mean(
        [_pair_degree_distance(candidate, ref, degrees, max_degree) for ref in cluster_nodes]
    )
    ncc_dist = _safe_mean(
        [_pair_ncc_distance(signatures[candidate], signatures[ref]) for ref in cluster_nodes]
    )
    return float((alpha * deg_dist) + (beta * ncc_dist))


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
    signatures = calculate_all_ncc(graph, simplified=False)
    max_degree = max(float(max(degrees.values()) if degrees else 1), 1.0)

    remaining = set(nodes)
    clusters: List[List[int]] = []

    # Build full clusters greedily with cluster-level cost (not seed-only cost).
    while len(remaining) >= k:
        seed = max(remaining, key=lambda node: (degrees[node], -node))
        remaining.remove(seed)
        cluster = [seed]

        while len(cluster) < k and remaining:
            candidate = min(
                remaining,
                key=lambda node: (
                    _cluster_cost(node, cluster, degrees, signatures, alpha, beta, max_degree),
                    -degrees[node],  # tie-break: degree descending
                    node,  # tie-break: id ascending
                ),
            )
            cluster.append(candidate)
            remaining.remove(candidate)

        clusters.append(sorted(cluster))

    # Attach leftovers to closest existing clusters while keeping deterministic tie-breaks.
    leftovers = sorted(remaining, key=lambda node: (-degrees[node], node))
    for node in leftovers:
        if not clusters:
            clusters.append([node])
            continue
        cluster_index = min(
            range(len(clusters)),
            key=lambda idx: (
                _cluster_cost(node, clusters[idx], degrees, signatures, alpha, beta, max_degree),
                -len(clusters[idx]),
                idx,
            ),
        )
        clusters[cluster_index].append(node)

    return [sorted(cluster) for cluster in clusters if cluster]


def _internal_template_edges(group: Sequence[int]) -> Set[Tuple[int, int]]:
    ordered = sorted(int(node) for node in group)
    size = len(ordered)
    if size < 2:
        return set()

    ring_radius = 1 if size < 6 else 2
    expected: Set[Tuple[int, int]] = set()
    for i, src in enumerate(ordered):
        for step in range(1, ring_radius + 1):
            dst = ordered[(i + step) % size]
            if src != dst:
                expected.add(_edge_key(src, dst))
    return expected


def _harmonize_cluster_internal_template(graph: nx.Graph, cluster_nodes: Sequence[int]) -> None:
    """
    Add-only internal harmonization:
    - compute expected template edges;
    - add missing edges;
    - never remove existing edges.
    """
    if len(cluster_nodes) < 2:
        return
    desired_internal = _internal_template_edges(cluster_nodes)
    current_internal = {
        _edge_key(int(u), int(v))
        for u, v in graph.subgraph(cluster_nodes).edges()
        if int(u) != int(v)
    }
    for u, v in sorted(desired_internal - current_internal):
        graph.add_edge(u, v)


def _harmonize_cluster_external_template(graph: nx.Graph, cluster_nodes: Sequence[int]) -> None:
    """
    Add-only external harmonization:
    - identify representative external neighbors for the cluster;
    - add missing links to these representatives;
    - never remove existing external neighbors.
    """
    if len(cluster_nodes) < 2:
        return

    group_nodes = sorted(int(node) for node in cluster_nodes)
    group_set = set(group_nodes)

    external_counter: Dict[int, int] = {}
    external_sizes: List[int] = []
    for node in group_nodes:
        ext_neighbors = [int(nbr) for nbr in graph.neighbors(node) if int(nbr) not in group_set]
        external_sizes.append(len(ext_neighbors))
        for nbr in ext_neighbors:
            external_counter[nbr] = external_counter.get(nbr, 0) + 1

    target_external = int(round(float(np.median(external_sizes)))) if external_sizes else 0
    target_external = max(target_external, 0)
    ranked_candidates = sorted(
        external_counter.items(),
        key=lambda item: (-item[1], int(graph.degree[item[0]]), int(item[0])),
    )
    template_external = [node for node, _ in ranked_candidates[:target_external]]
    template_set = set(template_external)

    for node in group_nodes:
        for nbr in sorted(template_set):
            if nbr != node and not graph.has_edge(node, nbr):
                graph.add_edge(node, nbr)


def _build_cluster_profile(
    graph: nx.Graph,
    cluster: Sequence[int],
    *,
    use_simplified_ncc: bool,
) -> Tuple[int, NCCCode, List[int]]:
    ordered = sorted(int(node) for node in cluster)
    if not ordered:
        return 0, tuple(), []

    degrees = [int(graph.degree[node]) for node in ordered]
    target_degree = int(round(float(np.median(degrees))))

    signatures = {
        node: calculate_ncc(
            graph,
            node,
            simplified=use_simplified_ncc,
            max_components=4 if use_simplified_ncc else None,
            max_degree_entries=0 if use_simplified_ncc else 6,
        )
        for node in ordered
    }

    # Medoid NCC in the cluster for a more stable structural target.
    target_node = min(
        ordered,
        key=lambda node: (
            _safe_mean(
                [
                    _pair_ncc_distance(signatures[node], signatures[other])
                    for other in ordered
                    if other != node
                ]
            ),
            -int(graph.degree[node]),
            int(node),
        ),
    )
    target_ncc = signatures[target_node]
    return target_degree, target_ncc, ordered


def _score_node_to_profile(
    graph: nx.Graph,
    node: int,
    target_degree: int,
    target_ncc: NCCCode,
    alpha: float,
    beta: float,
    max_degree: float,
    *,
    use_simplified_ncc: bool,
) -> float:
    node_degree = int(graph.degree[node])
    node_ncc = calculate_ncc(
        graph,
        int(node),
        simplified=use_simplified_ncc,
        max_components=4 if use_simplified_ncc else None,
        max_degree_entries=0 if use_simplified_ncc else 6,
    )

    degree_gap = abs(float(node_degree) - float(target_degree)) / max_degree
    ncc_gap = _pair_ncc_distance(node_ncc, target_ncc)
    return float((alpha * degree_gap) + (beta * ncc_gap))


def _candidate_add_neighbors(
    graph: nx.Graph,
    node: int,
    group_set: Set[int],
    rng: random.Random,
    max_candidates: int = 96,
) -> List[int]:
    """
    Candidate priority for edge addition:
    1) same-cluster non-neighbors
    2) distance-2 nodes
    3) distance-3 nodes
    4) bounded random fallback (<=16)
    """
    node = int(node)
    neighbors = {int(nbr) for nbr in graph.neighbors(node)}

    def rank(nodes: Sequence[int]) -> List[int]:
        unique = sorted(set(int(x) for x in nodes))
        return sorted(unique, key=lambda x: (-int(graph.degree[x]), int(x)))

    priority1 = rank([v for v in group_set if v != node and v not in neighbors])

    distances = nx.single_source_shortest_path_length(graph, source=node, cutoff=3)
    priority2 = rank([int(v) for v, d in distances.items() if d == 2 and int(v) not in neighbors and int(v) != node])
    priority3 = rank([int(v) for v, d in distances.items() if d == 3 and int(v) not in neighbors and int(v) != node])

    ordered: List[int] = []
    seen: Set[int] = set()

    def append_candidates(candidates: Sequence[int]) -> None:
        for candidate in candidates:
            if candidate in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate)
            if len(ordered) >= max_candidates:
                return

    append_candidates(priority1)
    if len(ordered) < max_candidates:
        append_candidates(priority2)
    if len(ordered) < max_candidates:
        append_candidates(priority3)

    if len(ordered) < max_candidates:
        remaining_pool = [
            int(v)
            for v in graph.nodes()
            if int(v) != node and int(v) not in neighbors and int(v) not in seen
        ]
        fallback_size = min(16, len(remaining_pool), max_candidates - len(ordered))
        if fallback_size > 0:
            random_candidates = rng.sample(remaining_pool, k=fallback_size)
            append_candidates(random_candidates)

    return ordered[:max_candidates]


def _candidate_remove_neighbors(graph: nx.Graph, node: int, group_set: Set[int]) -> List[int]:
    neighbors = [int(nbr) for nbr in graph.neighbors(node)]
    outside = [nbr for nbr in neighbors if nbr not in group_set]
    inside = [nbr for nbr in neighbors if nbr in group_set]
    outside.sort(key=lambda nbr: (int(graph.degree[nbr]), int(nbr)))
    inside.sort(key=lambda nbr: (int(graph.degree[nbr]), int(nbr)))
    return outside + inside


def _apply_best_ange_operation(
    graph: nx.Graph,
    node: int,
    cluster_nodes: Sequence[int],
    target_degree: int,
    target_ncc: NCCCode,
    alpha: float,
    beta: float,
    max_degree: float,
    rng: random.Random,
    *,
    use_simplified_ncc: bool,
    removal_penalty: float,
    protected_edges: Set[Tuple[int, int]] | None,
) -> bool:
    """
    Add-biased local operator.

    A removal is accepted only if it clearly outperforms the best addition:
    gain_remove > gain_add * (1 + removal_penalty)
    """
    group_set = set(int(v) for v in cluster_nodes)
    current_score = _score_node_to_profile(
        graph,
        node,
        target_degree,
        target_ncc,
        alpha,
        beta,
        max_degree,
        use_simplified_ncc=use_simplified_ncc,
    )

    best_add_gain = 0.0
    best_add_candidate: int | None = None
    for candidate in _candidate_add_neighbors(graph, node, group_set, rng):
        if graph.has_edge(node, candidate):
            continue
        graph.add_edge(node, candidate)
        new_score = _score_node_to_profile(
            graph,
            node,
            target_degree,
            target_ncc,
            alpha,
            beta,
            max_degree,
            use_simplified_ncc=use_simplified_ncc,
        )
        graph.remove_edge(node, candidate)
        gain = current_score - new_score
        if gain > best_add_gain:
            best_add_gain = gain
            best_add_candidate = int(candidate)

    best_remove_gain = 0.0
    best_remove_candidate: int | None = None
    for candidate in _candidate_remove_neighbors(graph, node, group_set)[:96]:
        edge = _edge_key(node, candidate)
        if protected_edges and edge in protected_edges:
            continue
        if not graph.has_edge(node, candidate):
            continue
        graph.remove_edge(node, candidate)
        new_score = _score_node_to_profile(
            graph,
            node,
            target_degree,
            target_ncc,
            alpha,
            beta,
            max_degree,
            use_simplified_ncc=use_simplified_ncc,
        )
        graph.add_edge(node, candidate)
        gain = current_score - new_score
        if gain > best_remove_gain:
            best_remove_gain = gain
            best_remove_candidate = int(candidate)

    choose_removal = False
    if best_remove_candidate is not None:
        if best_add_candidate is None:
            choose_removal = best_remove_gain > 0.0
        else:
            choose_removal = best_remove_gain > (best_add_gain * (1.0 + float(removal_penalty)))

    if choose_removal and best_remove_candidate is not None:
        if graph.has_edge(node, best_remove_candidate):
            graph.remove_edge(node, best_remove_candidate)
            return True

    if best_add_candidate is not None and best_add_gain > 0.0:
        graph.add_edge(node, best_add_candidate)
        return True

    return False


def diagnose_anonymization(original_graph: nx.Graph, anonymized_graph: nx.Graph) -> Dict[str, float]:
    original_edges = _normalized_edge_set(original_graph)
    anonymized_edges = _normalized_edge_set(anonymized_graph)
    preserved = len(original_edges.intersection(anonymized_edges))

    if calculate_edge_intersection is not None:
        edge_intersection = float(calculate_edge_intersection(original_graph, anonymized_graph))
    else:
        edge_intersection = float(preserved / len(original_edges)) if original_edges else 0.0

    return {
        "num_nodes_original": float(original_graph.number_of_nodes()),
        "num_nodes_anonymized": float(anonymized_graph.number_of_nodes()),
        "num_edges_original": float(original_graph.number_of_edges()),
        "num_edges_anonymized": float(anonymized_graph.number_of_edges()),
        "num_original_edges_preserved": float(preserved),
        "edge_intersection": float(edge_intersection),
        "num_edges_added": float(len(anonymized_edges - original_edges)),
        "num_edges_removed": float(len(original_edges - anonymized_edges)),
    }


def anonymize_ange_modified_ncc(
    graph: nx.Graph,
    k: int,
    seed: int,
    alpha: float = 1.0,
    beta: float = 0.2,
    passes: int = 2,
    max_node_iterations: int = 24,
    fast_graph_threshold: int = 1000,
    removal_penalty: float = 0.5,
    preserve_original_edges: bool = True,
) -> nx.Graph:
    """
    Improved ANGE+NCC anonymization with:
    - full NCC when feasible;
    - cluster-level assignment cost;
    - add-only harmonization;
    - add-biased local search;
    - optional restoration of original edges.
    """
    if k < 2:
        raise ValueError("k must be >= 2")

    cfg = AngeModifieConfig(
        k=k,
        alpha=float(alpha),
        beta=float(beta),
        passes=max(1, int(passes)),
        removal_penalty=float(removal_penalty),
        preserve_original_edges=bool(preserve_original_edges),
    )

    rng = random.Random(seed)
    modified = nx.Graph(graph)
    if modified.number_of_nodes() == 0:
        return modified

    original_edges = _normalized_edge_set(modified) if cfg.preserve_original_edges else set()

    fast_mode = modified.number_of_nodes() >= int(fast_graph_threshold)
    use_simplified_ncc = bool(fast_mode)
    active_passes = 1 if fast_mode else cfg.passes
    active_iterations = 0 if fast_mode else int(max_node_iterations)

    for _ in range(active_passes):
        clusters = clusterisation_ncc(
            modified,
            cfg.k,
            cfg.alpha,
            cfg.beta,
            fast_mode=fast_mode,
        )
        max_degree = max(float(max((modified.degree[node] for node in modified.nodes()), default=1)), 1.0)
        protected_edges = original_edges if cfg.preserve_original_edges else None

        for cluster in clusters:
            target_degree, target_ncc, ordered_nodes = _build_cluster_profile(
                modified,
                cluster,
                use_simplified_ncc=use_simplified_ncc,
            )
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
                        use_simplified_ncc=use_simplified_ncc,
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
                        use_simplified_ncc=use_simplified_ncc,
                        removal_penalty=cfg.removal_penalty,
                        protected_edges=protected_edges,
                    )
                    if not changed:
                        break

    # Optional defensive restore: keep all original edges in final graph.
    if cfg.preserve_original_edges:
        for u, v in sorted(original_edges):
            modified.add_edge(u, v)

    modified.remove_edges_from(nx.selfloop_edges(modified))
    return nx.Graph(modified)
