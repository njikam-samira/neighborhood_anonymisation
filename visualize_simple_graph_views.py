from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".pairs",
    ".edgelist",
    ".txt",
    ".csv",
    ".gml",
    ".graphml",
    ".pkl",
    ".pickle",
    ".gpickle",
    ".npz",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    output_prefix: str
    source_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    output_suffix: str
    display_name: str
    search_tokens: Tuple[str, ...]


DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(key="cora", output_prefix="cora", source_tokens=("cora",)),
    DatasetSpec(key="citeseer", output_prefix="citeseer", source_tokens=("citeseer",)),
    DatasetSpec(key="polblog", output_prefix="polblog", source_tokens=("polblog", "polblogs")),
)

METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec(key="original", output_suffix="original", display_name="Original Graph", search_tokens=("original",)),
    MethodSpec(
        key="ange_original",
        output_suffix="ange_original",
        display_name="Ange Original Anonymized",
        search_tokens=("ange_original",),
    ),
    MethodSpec(
        key="ange_modifie_ncc",
        output_suffix="ange_modifie_ncc",
        display_name="Ange Modifie NCC Anonymized",
        search_tokens=("ange_modifie_ncc",),
    ),
    MethodSpec(
        key="zhou_pei",
        output_suffix="zhou_pei",
        display_name="Zhou-Pei Anonymized",
        search_tokens=("zhou_pei",),
    ),
    MethodSpec(
        key="1hikda",
        output_suffix="1hikda",
        display_name="1HiKDA Anonymized",
        search_tokens=("1hikda",),
    ),
)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find_supported_files(search_roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
    return files


def load_graph_from_file(path: Path) -> nx.Graph:
    suffix = path.suffix.lower()

    if suffix in {".pairs", ".edgelist", ".txt"}:
        graph = nx.Graph()
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[\s,;]+", line)
                if len(parts) < 2:
                    continue
                u = _try_cast_node(parts[0])
                v = _try_cast_node(parts[1])
                graph.add_edge(u, v)
        return graph

    if suffix == ".csv":
        graph = nx.Graph()
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                if row[0].strip().lower() in {"source", "src", "from", "u"}:
                    continue
                u = _try_cast_node(row[0])
                v = _try_cast_node(row[1])
                graph.add_edge(u, v)
        return graph

    if suffix == ".gml":
        return nx.Graph(nx.read_gml(path))

    if suffix == ".graphml":
        return nx.Graph(nx.read_graphml(path))

    if suffix in {".pkl", ".pickle", ".gpickle"}:
        try:
            graph = nx.read_gpickle(path)
            return nx.Graph(graph)
        except Exception:
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            if isinstance(obj, nx.Graph):
                return nx.Graph(obj)
            raise ValueError(f"Unsupported pickle object for graph: {type(obj)}")

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        graph = nx.Graph()
        if "edges" in data:
            edges = np.asarray(data["edges"])
        elif "edge_index" in data:
            edge_index = np.asarray(data["edge_index"])
            if edge_index.ndim == 2 and edge_index.shape[0] == 2:
                edges = edge_index.T
            else:
                raise ValueError("edge_index format not supported")
        else:
            raise ValueError("No 'edges' or 'edge_index' key in npz")
        for u, v in edges:
            graph.add_edge(_try_cast_node(u), _try_cast_node(v))
        return graph

    raise ValueError(f"Unsupported graph format: {path.suffix}")


def _try_cast_node(value: object) -> object:
    text = str(value).strip()
    if text == "":
        return text
    try:
        return int(text)
    except ValueError:
        return text


def to_simple_undirected(graph: nx.Graph) -> nx.Graph:
    simple = nx.Graph(graph)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    return simple


def choose_display_nodes(graph: nx.Graph, max_nodes: int = 120) -> List[object]:
    if graph.number_of_nodes() <= max_nodes:
        return list(graph.nodes())
    ranked = sorted(graph.degree(), key=lambda item: (-item[1], str(item[0])))
    return [node for node, _ in ranked[:max_nodes]]


def make_display_graph(graph: nx.Graph, reference_nodes: List[object], max_nodes: int = 120) -> Tuple[nx.Graph, str]:
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy(), "full graph"

    present_nodes = [node for node in reference_nodes if node in graph]
    if not present_nodes:
        return nx.Graph(), "sampled graph"
    return graph.subgraph(present_nodes).copy(), "sampled graph"


def find_original_path(project_root: Path, dataset: DatasetSpec) -> Optional[Path]:
    candidates: List[Path] = []
    for token in dataset.source_tokens:
        candidates.extend(
            [
                project_root / "data" / f"{token}.pairs",
                project_root / "kdld" / "data" / f"{token}.pairs",
                project_root / "secGraph" / "data" / f"{token}.pairs",
            ]
        )
    candidates.append(project_root / "data" / f"{dataset.key}.pairs")
    for path in candidates:
        if path.exists():
            return path
    return None


def find_anonymized_path(
    all_supported_files: Sequence[Path],
    dataset: DatasetSpec,
    method: MethodSpec,
    k_value: int,
) -> Optional[Path]:
    dataset_tokens = [normalize_token(token) for token in dataset.source_tokens]
    method_tokens = [normalize_token(token) for token in method.search_tokens]
    k_token = normalize_token(f"k{k_value}")

    scored: List[Tuple[int, Path]] = []
    for path in all_supported_files:
        norm = normalize_token(str(path))
        if not any(token in norm for token in dataset_tokens):
            continue
        if not any(token in norm for token in method_tokens):
            continue
        if k_token not in norm:
            continue

        score = 0
        lower = str(path).lower()
        if "pairs" in lower:
            score += 50
        if "legacy_benchmarks" in lower:
            score += 40
        if "recalc" in lower:
            score += 30
        if "global4" in lower:
            score += 20
        if "legacy_experiments" in lower:
            score += 10
        if "attacks" in lower or "stage1" in lower:
            score -= 80
        scored.append((score, path))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return scored[0][1]


def draw_single_graph(
    graph: nx.Graph,
    pos: Dict[object, np.ndarray],
    title: str,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.set_facecolor("white")
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="gray", width=0.5, alpha=0.35)
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color="#1f77b4", node_size=35)
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)


def draw_comparison_grid(
    dataset_label: str,
    rendered_graphs: Dict[str, nx.Graph],
    positions: Dict[str, Dict[object, np.ndarray]],
    title_map: Dict[str, str],
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    axes_flat = list(axes.flatten())

    ordered = [m.key for m in METHODS]
    for idx, method_key in enumerate(ordered):
        ax = axes_flat[idx]
        ax.set_facecolor("white")
        graph = rendered_graphs.get(method_key)
        pos = positions.get(method_key)
        if graph is None or graph.number_of_nodes() == 0 or pos is None:
            ax.text(0.5, 0.5, "Graph not found", ha="center", va="center", fontsize=14)
            ax.axis("off")
            ax.set_title(title_map.get(method_key, method_key), fontsize=13)
            continue
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="gray", width=0.5, alpha=0.35)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color="#1f77b4", node_size=18)
        ax.axis("off")
        ax.set_title(title_map.get(method_key, method_key), fontsize=13)

    axes_flat[-1].axis("off")
    fig.suptitle(f"{dataset_label} - Comparison Grid", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize available original/anonymized graph views.")
    parser.add_argument("--k", type=int, default=10, help="k value used to select anonymized graph files.")
    parser.add_argument("--seed", type=int, default=42, help="seed for spring layout.")
    parser.add_argument("--max-nodes", type=int, default=120, help="max displayed nodes before sampling.")
    parser.add_argument("--output-dir", default="plots_graph_views", help="output directory for PNG and CSV.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        project_root / "results",
        project_root / "resultat",
        project_root / "1HIkDA_anonymity",
        project_root / "data",
        project_root / "kdld" / "data",
        project_root / "secGraph" / "output",
        project_root / "umga" / "data",
    ]

    all_supported_files = find_supported_files(search_roots)

    summary_rows: List[Dict[str, object]] = []
    generated_images: List[Path] = []
    missing_graphs: List[str] = []

    for dataset in DATASETS:
        original_path = find_original_path(project_root, dataset)
        original_graph: Optional[nx.Graph] = None
        reference_nodes: List[object] = []

        if original_path is not None:
            original_graph = to_simple_undirected(load_graph_from_file(original_path))
            reference_nodes = choose_display_nodes(original_graph, max_nodes=int(args.max_nodes))

        rendered_graphs: Dict[str, nx.Graph] = {}
        rendered_positions: Dict[str, Dict[object, np.ndarray]] = {}
        title_map: Dict[str, str] = {}

        for method in METHODS:
            graph_found = False
            error_message = ""
            source_path: Optional[Path] = None
            full_graph: Optional[nx.Graph] = None
            display_graph = nx.Graph()
            view_type = "sampled graph"

            try:
                if method.key == "original":
                    source_path = original_path
                    if source_path is not None:
                        full_graph = original_graph
                else:
                    source_path = find_anonymized_path(
                        all_supported_files=all_supported_files,
                        dataset=dataset,
                        method=method,
                        k_value=int(args.k),
                    )
                    if source_path is not None:
                        full_graph = to_simple_undirected(load_graph_from_file(source_path))

                if full_graph is None:
                    graph_found = False
                    error_message = "Graph file not found"
                    missing_graphs.append(f"{dataset.output_prefix}/{method.output_suffix}")
                else:
                    graph_found = True
                    if not reference_nodes:
                        reference_nodes = choose_display_nodes(full_graph, max_nodes=int(args.max_nodes))
                    display_graph, view_type = make_display_graph(
                        graph=full_graph,
                        reference_nodes=reference_nodes,
                        max_nodes=int(args.max_nodes),
                    )

                if graph_found:
                    if method.key == "original":
                        base_for_layout = display_graph
                        base_pos = nx.spring_layout(base_for_layout, seed=int(args.seed)) if base_for_layout.number_of_nodes() else {}
                        pos = base_pos
                        rendered_positions[method.key] = pos
                    else:
                        original_pos = rendered_positions.get("original", {})
                        if original_pos:
                            pos = {node: original_pos[node] for node in display_graph.nodes() if node in original_pos}
                            if len(pos) != display_graph.number_of_nodes():
                                pos = nx.spring_layout(display_graph, seed=int(args.seed))
                        else:
                            pos = nx.spring_layout(display_graph, seed=int(args.seed))
                        rendered_positions[method.key] = pos

                    title = (
                        f"{method.display_name} "
                        f"({view_type}, n={display_graph.number_of_nodes()}, m={display_graph.number_of_edges()})"
                    )
                    output_png = output_dir / f"{dataset.output_prefix}_{method.output_suffix}.png"
                    draw_single_graph(display_graph, pos, title, output_png)
                    generated_images.append(output_png)
                    output_png_text = str(output_png)

                    rendered_graphs[method.key] = display_graph
                    title_map[method.key] = method.display_name
                else:
                    output_png_text = ""

                summary_rows.append(
                    {
                        "dataset": dataset.output_prefix,
                        "method": method.output_suffix,
                        "graph_found": graph_found,
                        "num_nodes_full": int(full_graph.number_of_nodes()) if full_graph is not None else "",
                        "num_edges_full": int(full_graph.number_of_edges()) if full_graph is not None else "",
                        "num_nodes_displayed": int(display_graph.number_of_nodes()) if graph_found else "",
                        "num_edges_displayed": int(display_graph.number_of_edges()) if graph_found else "",
                        "view_type": view_type if graph_found else "",
                        "output_png": output_png_text,
                        "error_message": error_message,
                    }
                )
            except Exception as exc:
                missing_graphs.append(f"{dataset.output_prefix}/{method.output_suffix}")
                summary_rows.append(
                    {
                        "dataset": dataset.output_prefix,
                        "method": method.output_suffix,
                        "graph_found": False,
                        "num_nodes_full": "",
                        "num_edges_full": "",
                        "num_nodes_displayed": "",
                        "num_edges_displayed": "",
                        "view_type": "",
                        "output_png": "",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )

        grid_png = output_dir / f"{dataset.output_prefix}_comparison_grid.png"
        draw_comparison_grid(
            dataset_label=dataset.output_prefix,
            rendered_graphs=rendered_graphs,
            positions=rendered_positions,
            title_map=title_map,
            output_png=grid_png,
        )
        generated_images.append(grid_png)

    summary_csv = output_dir / "graph_views_summary.csv"
    fieldnames = [
        "dataset",
        "method",
        "graph_found",
        "num_nodes_full",
        "num_edges_full",
        "num_nodes_displayed",
        "num_edges_displayed",
        "view_type",
        "output_png",
        "error_message",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Images generees ===")
    for path in sorted(set(generated_images)):
        print(f"- {path}")

    print("\n=== Graphes manquants ===")
    if missing_graphs:
        for item in sorted(set(missing_graphs)):
            print(f"- {item}")
    else:
        print("- Aucun")

    print("\n=== CSV resume ===")
    print(f"- {summary_csv}")

    print("\n=== Commande de relance ===")
    print("python visualize_simple_graph_views.py")


if __name__ == "__main__":
    main()
