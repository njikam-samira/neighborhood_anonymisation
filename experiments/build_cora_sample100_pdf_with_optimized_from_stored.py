from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.data.io import load_graph_from_pairs, prepare_simple_graph


def sample_subgraph(graph: nx.Graph, sample_size: int, seed: int) -> nx.Graph:
    # Keep the exact same sampling strategy as the previous script.
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


def align_to_reference_nodes(graph: nx.Graph, reference_nodes: List[int]) -> nx.Graph:
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


def draw_graph_png(graph: nx.Graph, output_png: Path, title: str, seed: int) -> Dict[str, int]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=seed) if graph.number_of_nodes() else {}
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.set_facecolor("white")
    if graph.number_of_nodes() > 0:
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="gray", width=0.5, alpha=0.35)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color="#1f77b4", node_size=35)
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    return {"n": int(graph.number_of_nodes()), "m": int(graph.number_of_edges())}


def build_optimized_page_pdf(
    optimized_png: Path,
    output_pdf: Path,
    *,
    k: int,
    sample_size: int,
    seed: int,
    stats: Dict[str, int],
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    image = plt.imread(optimized_png)

    with PdfPages(output_pdf) as pdf:
        fig_cover = plt.figure(figsize=(11.69, 8.27))
        fig_cover.suptitle("Cora sample 100 - Methode optimisee", fontsize=18, fontweight="bold")
        txt = (
            f"Methode: AngeModifieNCCOptimise\n"
            f"Sample size: {sample_size}\n"
            f"k: {k}\n"
            f"seed: {seed}\n\n"
            f"Graphe affiche: n={stats['n']}, m={stats['m']}\n"
            "Cette page est calculee maintenant, les 4 autres methodes sont reprises du PDF stocke."
        )
        fig_cover.text(0.08, 0.82, txt, va="top", fontsize=12)
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("AngeModifieNCCOptimise (sampled graph)", fontsize=16, pad=12)
        ax.imshow(image)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def merge_pdfs(
    stored_pdf: Path,
    optimized_pdf: Path,
    output_pdf: Path,
    *,
    stored_pages_to_keep: List[int],
) -> None:
    reader_stored = PdfReader(str(stored_pdf))
    reader_opt = PdfReader(str(optimized_pdf))
    writer = PdfWriter()

    for page_idx in stored_pages_to_keep:
        if 0 <= page_idx < len(reader_stored.pages):
            writer.add_page(reader_stored.pages[page_idx])

    for page in reader_opt.pages:
        writer.add_page(page)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final PDF with stored 4 methods + optimized method.")
    parser.add_argument("--input-pairs", default="data/cora.pairs")
    parser.add_argument("--stored-pdf", default="plots_graph_pdf/cora_sample100_anonymization_comparison.pdf")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimized-png",
        default="plots_graph_pdf/cora_sample100_ange_modifie_ncc_optimise.png",
    )
    parser.add_argument(
        "--optimized-pages-pdf",
        default="plots_graph_pdf/cora_sample100_ange_modifie_ncc_optimise_pages.pdf",
    )
    parser.add_argument(
        "--output-pdf",
        default="plots_graph_pdf/cora_sample100_all_methods_with_optimise.pdf",
    )
    args = parser.parse_args()

    input_pairs = (PROJECT_ROOT / args.input_pairs).resolve()
    stored_pdf = (PROJECT_ROOT / args.stored_pdf).resolve()
    optimized_png = (PROJECT_ROOT / args.optimized_png).resolve()
    optimized_pages_pdf = (PROJECT_ROOT / args.optimized_pages_pdf).resolve()
    output_pdf = (PROJECT_ROOT / args.output_pdf).resolve()

    if not input_pairs.exists():
        raise FileNotFoundError(f"Input pairs not found: {input_pairs}")
    if not stored_pdf.exists():
        raise FileNotFoundError(f"Stored comparison PDF not found: {stored_pdf}")

    original_graph = prepare_simple_graph(load_graph_from_pairs(input_pairs, node_type=int))
    sampled_graph = sample_subgraph(original_graph, sample_size=int(args.sample_size), seed=int(args.seed))
    relabel_map = {node: idx for idx, node in enumerate(sorted(sampled_graph.nodes()))}
    sampled_graph = nx.relabel_nodes(sampled_graph, relabel_map, copy=True)
    reference_nodes = sorted(int(node) for node in sampled_graph.nodes())

    optimized_graph = anonymize_ange_modified_ncc(
        sampled_graph.copy(),
        k=int(args.k),
        seed=int(args.seed),
        alpha=1.0,
        beta=0.2,
        passes=2,
        max_node_iterations=24,
        fast_graph_threshold=1000,
        removal_penalty=0.5,
        preserve_original_edges=True,
    )
    optimized_graph = align_to_reference_nodes(nx.Graph(optimized_graph), reference_nodes=reference_nodes)

    title = (
        f"AngeModifieNCCOptimise (sampled graph, "
        f"n={optimized_graph.number_of_nodes()}, m={optimized_graph.number_of_edges()})"
    )
    stats = draw_graph_png(optimized_graph, optimized_png, title=title, seed=int(args.seed))
    build_optimized_page_pdf(
        optimized_png,
        optimized_pages_pdf,
        k=int(args.k),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        stats=stats,
    )

    # The stored PDF was generated previously with:
    # page 0: cover, page 1: grid, pages 2..6: Original + 4 methods (individual pages).
    # We keep pages 2..6 to reuse the already stored visuals exactly as requested.
    merge_pdfs(
        stored_pdf=stored_pdf,
        optimized_pdf=optimized_pages_pdf,
        output_pdf=output_pdf,
        stored_pages_to_keep=[2, 3, 4, 5, 6],
    )

    print("Final PDF generated:")
    print(output_pdf)
    print("\nReused stored PDF (pages 2..6):")
    print(stored_pdf)
    print("\nOptimized method artifacts:")
    print(optimized_png)
    print(optimized_pages_pdf)
    print(
        f"\nOptimized graph stats: n={optimized_graph.number_of_nodes()}, "
        f"m={optimized_graph.number_of_edges()}, k={args.k}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
