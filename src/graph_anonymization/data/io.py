from __future__ import annotations

from pathlib import Path

import networkx as nx


def load_graph_from_pairs(file_path: Path, node_type: type = int) -> nx.Graph:
    """Load an undirected graph from a .pairs edge-list file."""
    graph = nx.Graph()
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            graph.add_edge(node_type(parts[0]), node_type(parts[1]))
    return graph


def save_graph_to_pairs(graph: nx.Graph, file_path: Path) -> None:
    """Save a graph to a .pairs edge-list file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
    with file_path.open("w", encoding="utf-8") as handle:
        for u, v in edges:
            handle.write(f"{u} {v}\n")


def prepare_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected graph without self loops."""
    clean = nx.Graph(graph)
    clean.remove_edges_from(nx.selfloop_edges(clean))
    return clean
