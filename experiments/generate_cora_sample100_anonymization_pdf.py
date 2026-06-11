from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.anonymization.ange_original import anonymize_ange_original
from graph_anonymization.anonymization.zhou_pei import anonymize_zhou_pei
from graph_anonymization.benchmarks.hikda_benchmark import anonymize_1hikda
from graph_anonymization.data.io import load_graph_from_pairs, prepare_simple_graph


Anonymizer = Callable[[nx.Graph, int, int], nx.Graph]


def align_to_reference_nodes(graph: nx.Graph, reference_nodes: List[int]) -> nx.Graph:
    """Keep only reference nodes and remove possible extra nodes created by some pipelines."""
    ref_set = set(int(node) for node in reference_nodes)
    aligned = nx.Graph()
    aligned.add_nodes_from(reference_nodes)
    for u, v in graph.edges():
        uu = int(u)
        vv = int(v)
        if uu == vv:
            continue
        if uu in ref_set and vv in ref_set:
            aligned.add_edge(uu, vv)
    return prepare_simple_graph(aligned)


def sample_subgraph(graph: nx.Graph, sample_size: int, seed: int) -> nx.Graph:
    """Build a reproducible 100-node sample with good local connectivity."""
    graph = prepare_simple_graph(graph)
    nodes = list(graph.nodes())
    if len(nodes) <= sample_size:
        return graph.copy()

    rng = random.Random(seed)
    by_degree = sorted(nodes, key=lambda node: (-graph.degree[node], int(node)))

    selected: List[int] = []
    seen = set()
    queue: List[int] = []

    def push(node: int) -> None:
        if node not in seen:
            queue.append(node)
            seen.add(node)

    push(int(by_degree[0]))
    while queue and len(selected) < sample_size:
        current = queue.pop(0)
        selected.append(current)
        neighbors = sorted(
            (int(nbr) for nbr in graph.neighbors(current)),
            key=lambda node: (-graph.degree[node], node),
        )
        for nbr in neighbors:
            if len(selected) + len(queue) >= sample_size:
                break
            push(nbr)

    if len(selected) < sample_size:
        remaining = [int(node) for node in by_degree if int(node) not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    sampled = graph.subgraph(selected).copy()
    return prepare_simple_graph(sampled)


def draw_graph_panel(ax: plt.Axes, graph: nx.Graph, pos: Dict[int, Tuple[float, float]], title: str) -> None:
    ax.set_facecolor("white")
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Graph vide", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        return

    edge_pos = {node: pos[node] for node in graph.nodes() if node in pos}
    if len(edge_pos) != graph.number_of_nodes():
        edge_pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(graph, pos=edge_pos, ax=ax, edge_color="gray", width=0.5, alpha=0.35)
    nx.draw_networkx_nodes(graph, pos=edge_pos, ax=ax, node_color="#1f77b4", node_size=35)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def save_pdf(
    output_pdf: Path,
    dataset_name: str,
    k_value: int,
    sample_size: int,
    seed: int,
    results: Dict[str, nx.Graph],
    errors: Dict[str, str],
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    original = results["Original"]
    layout_pos = nx.spring_layout(original, seed=seed) if original.number_of_nodes() else {}

    with PdfPages(output_pdf) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.suptitle("Cora Sample (100 nodes) - Visual Comparison", fontsize=20, fontweight="bold")
        text = textwrap.dedent(
            f"""
            Dataset: {dataset_name}
            Sample size: {sample_size} nodes
            k-anonymity parameter: {k_value}
            Seed: {seed}

            Methods:
            - Ange Original
            - Ange Modifie NCC
            - Zhou-Pei
            - 1HiKDA
            """
        ).strip()
        cover.text(0.08, 0.82, text, fontsize=12, va="top")
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="white")
        axes_flat = list(axes.flatten())
        ordered_names = [
            "Original",
            "Ange Original",
            "Ange Modifie NCC",
            "Zhou-Pei",
            "1HiKDA",
        ]

        for idx, method_name in enumerate(ordered_names):
            ax = axes_flat[idx]
            graph = results.get(method_name, nx.Graph())
            err = errors.get(method_name, "")
            base_title = f"{method_name} (n={graph.number_of_nodes()}, m={graph.number_of_edges()})"
            if err:
                draw_graph_panel(ax, nx.Graph(), {}, f"{method_name} - erreur")
                ax.text(0.03, 0.03, err[:180], ha="left", va="bottom", fontsize=8, transform=ax.transAxes)
            else:
                draw_graph_panel(ax, graph, layout_pos, base_title)

        axes_flat[-1].axis("off")
        fig.suptitle("Original vs Anonymized Graphs", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for method_name in ordered_names:
            graph = results.get(method_name, nx.Graph())
            err = errors.get(method_name, "")
            page = plt.figure(figsize=(12, 9), facecolor="white")
            ax = page.add_subplot(111)
            if err:
                draw_graph_panel(ax, nx.Graph(), {}, f"{method_name} - erreur")
                ax.text(0.03, 0.03, err, ha="left", va="bottom", fontsize=10, transform=ax.transAxes)
            else:
                title = f"{method_name} ({'sampled graph'}, n={graph.number_of_nodes()}, m={graph.number_of_edges()})"
                draw_graph_panel(ax, graph, layout_pos, title)
            pdf.savefig(page, bbox_inches="tight")
            plt.close(page)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDF visualization for Cora sample anonymization.")
    parser.add_argument("--input-pairs", default="data/cora.pairs")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-pdf",
        default="plots_graph_pdf/cora_sample100_anonymization_comparison.pdf",
    )
    args = parser.parse_args()

    input_pairs = (PROJECT_ROOT / args.input_pairs).resolve()
    if not input_pairs.exists():
        raise FileNotFoundError(f"Input graph not found: {input_pairs}")

    original_graph = prepare_simple_graph(load_graph_from_pairs(input_pairs, node_type=int))
    sampled_graph = sample_subgraph(original_graph, sample_size=int(args.sample_size), seed=int(args.seed))
    relabel_map = {node: idx for idx, node in enumerate(sorted(sampled_graph.nodes()))}
    sampled_graph = nx.relabel_nodes(sampled_graph, relabel_map, copy=True)
    reference_nodes = sorted(int(node) for node in sampled_graph.nodes())

    methods: Dict[str, Anonymizer] = {
        "Ange Original": anonymize_ange_original,
        "Ange Modifie NCC": anonymize_ange_modified_ncc,
        "Zhou-Pei": anonymize_zhou_pei,
        "1HiKDA": anonymize_1hikda,
    }

    results: Dict[str, nx.Graph] = {"Original": sampled_graph.copy()}
    errors: Dict[str, str] = {}

    for method_name, fn in methods.items():
        try:
            anonymized = fn(sampled_graph.copy(), int(args.k), int(args.seed))
            results[method_name] = align_to_reference_nodes(nx.Graph(anonymized), reference_nodes=reference_nodes)
            errors[method_name] = ""
        except Exception as exc:  # pragma: no cover - runtime safety
            results[method_name] = nx.Graph()
            errors[method_name] = f"{type(exc).__name__}: {exc}"

    output_pdf = (PROJECT_ROOT / args.output_pdf).resolve()
    save_pdf(
        output_pdf=output_pdf,
        dataset_name="Cora",
        k_value=int(args.k),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        results=results,
        errors=errors,
    )

    print("PDF generated:")
    print(output_pdf)
    print("\nGraph stats:")
    for name in ["Original", "Ange Original", "Ange Modifie NCC", "Zhou-Pei", "1HiKDA"]:
        g = results.get(name, nx.Graph())
        err = errors.get(name, "")
        if err:
            print(f"- {name}: ERROR -> {err}")
        else:
            print(f"- {name}: n={g.number_of_nodes()}, m={g.number_of_edges()}")


if __name__ == "__main__":
    main()
