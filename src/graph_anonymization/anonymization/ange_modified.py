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
    gamma: float = 0.2
    delta: float = 0.1
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


def _ncc_code_weight(code: NCCCode) -> float:
    total = 0.0
    for size, edges, deg_seq in code:
        total += float(abs(size) + abs(edges) + sum(abs(int(x)) for x in deg_seq))
    return max(1.0, total)


def _pair_ncc_distance_with_weights(
    node_a: int,
    node_b: int,
    signatures: Dict[int, NCCCode],
    signature_weights: Dict[int, float],
) -> float:
    return float(
        ncc_distance(
            signatures[node_a],
            signatures[node_b],
            max_value=max(
                1.0,
                float(signature_weights.get(node_a, 1.0)),
                float(signature_weights.get(node_b, 1.0)),
            ),
        )
    )


def _ncc_vector_params(signatures: Dict[int, NCCCode]) -> Tuple[int, int]:
    max_width = max(
        [len(component[2]) for code in signatures.values() for component in code] + [0]
    )
    max_components = max([len(code) for code in signatures.values()] + [0])
    return int(max_width), int(max_components)


def _ncc_code_vector(code: NCCCode, degree_width: int, max_components_count: int) -> np.ndarray:
    values: List[float] = []
    component_width = 2 + int(degree_width)
    for idx in range(int(max_components_count)):
        if idx >= len(code):
            values.extend([0.0] * component_width)
            continue
        size, edges, deg_seq = code[idx]
        degree_values = list(deg_seq[:degree_width])
        if len(degree_values) < degree_width:
            degree_values.extend([0] * (degree_width - len(degree_values)))
        values.extend([float(size), float(edges)] + [float(value) for value in degree_values])
    return np.array(values, dtype=np.float64)


def _pair_ncc_distance_from_vectors(
    node_a: int,
    node_b: int,
    ncc_vectors: Dict[int, np.ndarray],
    signature_weights: Dict[int, float],
) -> float:
    return float(
        np.abs(ncc_vectors[node_a] - ncc_vectors[node_b]).sum()
        / max(
            1.0,
            float(signature_weights.get(node_a, 1.0)),
            float(signature_weights.get(node_b, 1.0)),
        )
    )


def _common_neighbors_count(graph: nx.Graph, node_a: int, node_b: int) -> int:
    if graph.number_of_nodes() == 0 or node_a not in graph or node_b not in graph:
        return 0
    try:
        neighbors_a = set(graph.neighbors(node_a))
        neighbors_b = set(graph.neighbors(node_b))
    except nx.NetworkXError:
        return 0
    return int(len(neighbors_a.intersection(neighbors_b)))


def _common_neighbors_count_from_sets(
    neighbor_sets: Dict[int, Set[int]],
    node_a: int,
    node_b: int,
) -> int:
    neighbors_a = neighbor_sets.get(int(node_a))
    neighbors_b = neighbor_sets.get(int(node_b))
    if neighbors_a is None or neighbors_b is None:
        return 0
    if len(neighbors_a) > len(neighbors_b):
        neighbors_a, neighbors_b = neighbors_b, neighbors_a
    return int(sum(1 for neighbor in neighbors_a if neighbor in neighbors_b))


def _cluster_common_neighbors_distance(
    graph: nx.Graph,
    candidate: int,
    cluster_nodes: Sequence[int],
    max_common_neighbors: float,
    neighbor_sets: Dict[int, Set[int]] | None = None,
) -> float:
    cn_values = []
    for ref in cluster_nodes:
        if candidate == ref:
            continue
        if neighbor_sets is None:
            cn_values.append(_common_neighbors_count(graph, candidate, ref))
        else:
            cn_values.append(_common_neighbors_count_from_sets(neighbor_sets, candidate, ref))
    if not cn_values:
        return 0.0
    avg_cn = _safe_mean([float(value) for value in cn_values])
    return float(1.0 - min(avg_cn / max(float(max_common_neighbors), 1.0), 1.0))


def _compute_community_map(graph: nx.Graph, fast_mode: bool = False) -> Dict[int, int]:
    nodes = sorted(int(node) for node in graph.nodes())
    if not nodes:
        return {}
    if fast_mode:
        return {node: 0 for node in nodes}

    try:
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
    except Exception:
        return {node: 0 for node in nodes}

    ordered_communities = sorted(
        [sorted(int(node) for node in community) for community in communities],
        key=lambda community: (community[0] if community else -1, len(community)),
    )
    community_map: Dict[int, int] = {}
    for community_id, community in enumerate(ordered_communities):
        for node in community:
            community_map[int(node)] = int(community_id)
    return {node: community_map.get(node, 0) for node in nodes}


def _cluster_community_distance(
    candidate: int,
    cluster_nodes: Sequence[int],
    community_map: Dict[int, int],
) -> float:
    if not cluster_nodes:
        return 0.0
    candidate_community = community_map.get(candidate, -1)
    community_counts: Dict[int, int] = {}
    for ref in cluster_nodes:
        ref_community = community_map.get(ref, -1)
        community_counts[ref_community] = community_counts.get(ref_community, 0) + 1
    majority_community = min(
        community_counts,
        key=lambda community: (-community_counts[community], community),
    )
    if candidate_community == majority_community:
        return 0.0

    mismatches = [
        1.0 if community_map.get(ref, -1) != candidate_community else 0.0
        for ref in cluster_nodes
    ]
    return float(min(max(_safe_mean(mismatches), 0.0), 1.0))


def _cluster_cost(
    candidate: int,
    cluster_nodes: Sequence[int],
    graph: nx.Graph,
    degrees: Dict[int, int],
    signatures: Dict[int, NCCCode],
    community_map: Dict[int, int],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    max_degree: float,
    max_common_neighbors: float,
    neighbor_sets: Dict[int, Set[int]] | None = None,
    signature_weights: Dict[int, float] | None = None,
    ncc_vectors: Dict[int, np.ndarray] | None = None,
) -> float:
    # cost(u, C) combines degree similarity, NCC similarity,
    # common-neighbor preservation, and community consistency.
    deg_dist = _safe_mean(
        [_pair_degree_distance(candidate, ref, degrees, max_degree) for ref in cluster_nodes]
    )
    if signature_weights is not None and ncc_vectors is not None:
        ncc_dist = _safe_mean(
            [
                _pair_ncc_distance_from_vectors(candidate, ref, ncc_vectors, signature_weights)
                for ref in cluster_nodes
            ]
        )
    elif signature_weights is None:
        ncc_dist = _safe_mean(
            [_pair_ncc_distance(signatures[candidate], signatures[ref]) for ref in cluster_nodes]
        )
    else:
        ncc_dist = _safe_mean(
            [
                _pair_ncc_distance_with_weights(candidate, ref, signatures, signature_weights)
                for ref in cluster_nodes
            ]
        )
    common_neighbors_dist = _cluster_common_neighbors_distance(
        graph,
        candidate,
        cluster_nodes,
        max_common_neighbors,
        neighbor_sets,
    )
    community_dist = _cluster_community_distance(candidate, cluster_nodes, community_map)
    return float(
        (alpha * deg_dist)
        + (beta * ncc_dist)
        + (gamma * common_neighbors_dist)
        + (delta * community_dist)
    )


def _select_best_cluster_candidate(
    candidates: Set[int],
    cluster_nodes: Sequence[int],
    graph: nx.Graph,
    degrees: Dict[int, int],
    signatures: Dict[int, NCCCode],
    community_map: Dict[int, int],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    max_degree: float,
    max_common_neighbors: float,
    neighbor_sets: Dict[int, Set[int]] | None = None,
    signature_weights: Dict[int, float] | None = None,
    ncc_vectors: Dict[int, np.ndarray] | None = None,
    chunk_size: int = 128,
) -> int:
    ordered_candidates = sorted(int(node) for node in candidates)
    if not ordered_candidates:
        raise ValueError("No candidates available for cluster selection.")

    refs = [int(node) for node in cluster_nodes]
    if not refs:
        return min(ordered_candidates, key=lambda node: (-degrees[node], node))

    ref_count = float(len(refs))
    ref_degrees = np.array([float(degrees[ref]) for ref in refs], dtype=np.float64)
    ref_communities = np.array([int(community_map.get(ref, -1)) for ref in refs], dtype=np.int64)

    community_counts: Dict[int, int] = {}
    for community in ref_communities.tolist():
        community_counts[int(community)] = community_counts.get(int(community), 0) + 1
    majority_community = min(
        community_counts,
        key=lambda community: (-community_counts[community], community),
    )

    ref_vectors: np.ndarray | None = None
    ref_weights: np.ndarray | None = None
    if signature_weights is not None and ncc_vectors is not None:
        ref_vectors = np.vstack([ncc_vectors[ref] for ref in refs])
        ref_weights = np.array(
            [float(signature_weights.get(ref, 1.0)) for ref in refs],
            dtype=np.float64,
        )

    best_node = ordered_candidates[0]
    best_key = (float("inf"), -degrees[best_node], best_node)
    chunk = max(1, int(chunk_size))

    for start in range(0, len(ordered_candidates), chunk):
        candidate_nodes = ordered_candidates[start : start + chunk]
        candidate_degrees = np.array(
            [float(degrees[node]) for node in candidate_nodes],
            dtype=np.float64,
        )
        degree_dist = np.abs(candidate_degrees[:, None] - ref_degrees[None, :]).mean(axis=1)
        degree_dist = degree_dist / max(float(max_degree), 1.0)

        if ref_vectors is not None and ref_weights is not None and signature_weights is not None and ncc_vectors is not None:
            candidate_vectors = np.vstack([ncc_vectors[node] for node in candidate_nodes])
            candidate_weights = np.array(
                [float(signature_weights.get(node, 1.0)) for node in candidate_nodes],
                dtype=np.float64,
            )
            ncc_l1 = np.abs(candidate_vectors[:, None, :] - ref_vectors[None, :, :]).sum(axis=2)
            denom = np.maximum(
                np.maximum(candidate_weights[:, None], ref_weights[None, :]),
                1.0,
            )
            ncc_dist = (ncc_l1 / denom).mean(axis=1)
        else:
            ncc_dist = np.array(
                [
                    _safe_mean(
                        [
                            _pair_ncc_distance(signatures[candidate], signatures[ref])
                            for ref in refs
                        ]
                    )
                    for candidate in candidate_nodes
                ],
                dtype=np.float64,
            )

        common_values: List[float] = []
        for candidate in candidate_nodes:
            if neighbor_sets is None:
                cn_total = sum(_common_neighbors_count(graph, candidate, ref) for ref in refs)
            else:
                cn_total = sum(
                    _common_neighbors_count_from_sets(neighbor_sets, candidate, ref)
                    for ref in refs
                )
            avg_cn = float(cn_total) / ref_count
            common_values.append(
                float(1.0 - min(avg_cn / max(float(max_common_neighbors), 1.0), 1.0))
            )
        common_neighbors_dist = np.array(common_values, dtype=np.float64)

        community_values: List[float] = []
        for candidate in candidate_nodes:
            candidate_community = int(community_map.get(candidate, -1))
            if candidate_community == majority_community:
                community_values.append(0.0)
            else:
                community_values.append(
                    float(np.mean(ref_communities != candidate_community))
                )
        community_dist = np.array(community_values, dtype=np.float64)

        costs = (
            (float(alpha) * degree_dist)
            + (float(beta) * ncc_dist)
            + (float(gamma) * common_neighbors_dist)
            + (float(delta) * community_dist)
        )

        for candidate, cost in zip(candidate_nodes, costs):
            key = (float(cost), -degrees[candidate], candidate)
            if key < best_key:
                best_key = key
                best_node = int(candidate)

    return best_node


def clusterisation_ncc(
    graph: nx.Graph,
    k: int,
    alpha: float,
    beta: float,
    gamma: float = 0.2,
    delta: float = 0.1,
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
    signature_weights = {node: _ncc_code_weight(signatures[node]) for node in nodes}
    ncc_degree_width, ncc_max_components = _ncc_vector_params(signatures)
    ncc_vectors = {
        node: _ncc_code_vector(signatures[node], ncc_degree_width, ncc_max_components)
        for node in nodes
    }
    neighbor_sets = {node: set(int(neighbor) for neighbor in graph.neighbors(node)) for node in nodes}
    max_degree = max(float(max(degrees.values()) if degrees else 1), 1.0)
    max_common_neighbors = max(float(max(degrees.values()) if degrees else 1), 1.0)
    community_map = _compute_community_map(graph, fast_mode=fast_mode)

    remaining = set(nodes)
    clusters: List[List[int]] = []

    # Build full clusters greedily with cluster-level cost (not seed-only cost).
    while len(remaining) >= k:
        seed = max(remaining, key=lambda node: (degrees[node], -node))
        remaining.remove(seed)
        cluster = [seed]

        while len(cluster) < k and remaining:
            candidate = _select_best_cluster_candidate(
                remaining,
                cluster,
                graph,
                degrees,
                signatures,
                community_map,
                alpha,
                beta,
                gamma,
                delta,
                max_degree,
                max_common_neighbors,
                neighbor_sets,
                signature_weights,
                ncc_vectors,
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
                _cluster_cost(
                    node,
                    clusters[idx],
                    graph,
                    degrees,
                    signatures,
                    community_map,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    max_degree,
                    max_common_neighbors,
                    neighbor_sets,
                    signature_weights,
                    ncc_vectors,
                ),
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

    def safe_connected_components_count(graph: nx.Graph) -> float:
        if graph.number_of_nodes() == 0:
            return 0.0
        try:
            return float(nx.number_connected_components(graph))
        except Exception:
            return 0.0

    def safe_average_clustering(graph: nx.Graph) -> float:
        if graph.number_of_nodes() == 0:
            return 0.0
        try:
            return float(nx.average_clustering(graph))
        except Exception:
            return 0.0

    return {
        "num_nodes_original": float(original_graph.number_of_nodes()),
        "num_nodes_anonymized": float(anonymized_graph.number_of_nodes()),
        "num_edges_original": float(original_graph.number_of_edges()),
        "num_edges_anonymized": float(anonymized_graph.number_of_edges()),
        "num_connected_components_original": safe_connected_components_count(original_graph),
        "num_connected_components_anonymized": safe_connected_components_count(anonymized_graph),
        "average_clustering_original": safe_average_clustering(original_graph),
        "average_clustering_anonymized": safe_average_clustering(anonymized_graph),
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
    gamma: float = 0.2,
    delta: float = 0.1,
) -> nx.Graph:
    """
    Improved ANGE+NCC anonymization with:
    - full NCC when feasible;
    - cluster-level assignment cost;
    - common-neighbor and community-aware clustering;
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
        gamma=float(gamma),
        delta=float(delta),
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
            gamma=cfg.gamma,
            delta=cfg.delta,
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
