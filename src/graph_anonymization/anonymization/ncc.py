from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

# Each component is encoded as:
# (component_size, num_internal_edges, sorted_degree_sequence_inside_component)
NCCComponent = Tuple[int, int, Tuple[int, ...]]
NCCCode = Tuple[NCCComponent, ...]


def _component_signature(
    neighborhood: nx.Graph,
    component_nodes: Iterable[int],
    *,
    simplified: bool,
    max_degree_entries: int,
) -> NCCComponent:
    nodes = sorted(int(node) for node in component_nodes)
    sub = neighborhood.subgraph(nodes)
    size = int(sub.number_of_nodes())
    edges = int(sub.number_of_edges())

    if simplified:
        degree_signature: Tuple[int, ...] = ()
    else:
        deg_values = sorted((int(sub.degree[node]) for node in sub.nodes()), reverse=True)
        degree_signature = tuple(deg_values[: max(1, int(max_degree_entries))])

    return (size, edges, degree_signature)


def calculate_ncc(
    graph: nx.Graph,
    node: int,
    *,
    simplified: bool = False,
    max_components: int | None = None,
    max_degree_entries: int = 6,
) -> NCCCode:
    """
    Compute a Neighborhood Component Code (NCC) for one node.

    The code is deterministic and comparable across nodes.
    """
    if int(node) not in graph:
        return tuple()

    neighbors = [int(nbr) for nbr in graph.neighbors(int(node))]
    if not neighbors:
        return tuple()

    neighborhood = graph.subgraph(neighbors).copy()
    if neighborhood.number_of_nodes() == 0:
        return tuple()

    components: List[NCCComponent] = []
    for comp_nodes in nx.connected_components(neighborhood):
        comp_sig = _component_signature(
            neighborhood,
            comp_nodes,
            simplified=bool(simplified),
            max_degree_entries=int(max_degree_entries),
        )
        components.append(comp_sig)

    components.sort(key=lambda item: (-item[0], -item[1], tuple(-x for x in item[2])))
    if max_components is not None:
        components = components[: max(1, int(max_components))]
    return tuple(components)


def calculate_all_ncc(
    graph: nx.Graph,
    *,
    simplified: bool = False,
    max_components: int | None = None,
    max_degree_entries: int = 6,
) -> Dict[int, NCCCode]:
    """Compute NCC for all nodes in a graph."""
    codes: Dict[int, NCCCode] = {}
    for node in graph.nodes():
        node_id = int(node)
        codes[node_id] = calculate_ncc(
            graph,
            node_id,
            simplified=bool(simplified),
            max_components=max_components,
            max_degree_entries=max_degree_entries,
        )
    return codes


def max_ncc(*codes: NCCCode) -> float:
    """
    Return a scale proxy for NCC values used to normalize distances.
    """
    if not codes:
        return 1.0

    def code_weight(code: NCCCode) -> float:
        total = 0.0
        for size, edges, deg_seq in code:
            total += float(abs(size) + abs(edges) + sum(abs(int(x)) for x in deg_seq))
        return total

    return max(1.0, max(code_weight(code) for code in codes))


def _component_vector(component: NCCComponent, degree_width: int) -> List[float]:
    size, edges, deg_seq = component
    deg = list(deg_seq[:degree_width])
    if len(deg) < degree_width:
        deg.extend([0] * (degree_width - len(deg)))
    return [float(size), float(edges)] + [float(x) for x in deg]


def ncc_distance(left: NCCCode, right: NCCCode, *, max_value: float | None = None) -> float:
    """
    L1 distance between two NCC codes, normalized by a robust scale.
    """
    if left == right:
        return 0.0

    degree_width = max(
        [len(comp[2]) for comp in left] + [len(comp[2]) for comp in right] + [0]
    )
    max_components = max(len(left), len(right))

    l1 = 0.0
    for idx in range(max_components):
        left_comp: NCCComponent = left[idx] if idx < len(left) else (0, 0, tuple())
        right_comp: NCCComponent = right[idx] if idx < len(right) else (0, 0, tuple())
        lv = _component_vector(left_comp, degree_width=degree_width)
        rv = _component_vector(right_comp, degree_width=degree_width)
        l1 += sum(abs(a - b) for a, b in zip(lv, rv))

    denom = float(max_value) if max_value is not None else float(max_ncc(left, right))
    denom = max(1.0, denom)
    return float(l1 / denom)
