from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from ..anonymization.ange_modified import anonymize_ange_modified_ncc
from ..anonymization.cluster_formation_mae import cluster_formation_MAE
from ..anonymization.reconstruction_optimize_mean import graph_reconstruction_optimize_mean


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SECGRAPH_JAR = PROJECT_ROOT / "secGraph" / "secGraphCLI.jar"

METHOD_LABELS: Dict[str, str] = {
    "ange_original": "Ange original",
    "ange_modifie_ncc": "Ange modifie NCC",
    "zhou_pei": "Zhou & Pei",
}


def load_graph_from_pairs(file_path: Path) -> nx.Graph:
    graph = nx.Graph()
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            graph.add_edge(int(parts[0]), int(parts[1]))
    return graph


def save_graph_to_pairs(graph: nx.Graph, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
    with file_path.open("w", encoding="utf-8") as handle:
        for u, v in edges:
            handle.write(f"{u} {v}\n")


def create_weight_matrix(graph: nx.Graph, seed: int) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros((0, 0), dtype=np.int16)
    max_id = max(int(node) for node in graph.nodes())
    cw = np.zeros((max_id + 1, max_id + 1), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for u, v in graph.edges():
        weight = int(rng.integers(1, 100))
        cw[int(u), int(v)] = weight
        cw[int(v), int(u)] = weight
    return cw


def anonymize_ange_original(graph: nx.Graph, k: int, seed: int) -> nx.Graph:
    if graph.number_of_nodes() > 1000:
        return anonymize_ange_original_scalable(graph, k=k, seed=seed)

    profile = [{"id": int(node), "degree": int(graph.degree[node])} for node in graph.nodes()]
    profile.sort(key=lambda item: item["degree"], reverse=True)
    clusters, cluster_degrees, _ = cluster_formation_MAE(profile, k)
    cw = create_weight_matrix(graph, seed=seed)
    _, _, _, _, _, updated_graph = graph_reconstruction_optimize_mean(
        clusters=clusters,
        cluster_degrees=cluster_degrees,
        CW=cw,
        G=graph.copy(),
    )
    return nx.Graph(updated_graph)


def anonymize_ange_original_scalable(graph: nx.Graph, k: int, seed: int, max_rounds: int = 3) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()

    profile = [{"id": int(node), "degree": int(graph.degree[node])} for node in graph.nodes()]
    profile.sort(key=lambda item: item["degree"], reverse=True)
    clusters, cluster_degrees, _ = cluster_formation_MAE(profile, k)

    target_degree: Dict[int, int] = {}
    for idx, cluster in enumerate(clusters):
        target = int(cluster_degrees[idx])
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


def compute_structural_signatures(graph: nx.Graph) -> Dict[int, np.ndarray]:
    features: Dict[int, np.ndarray] = {}
    triangles = nx.triangles(graph)
    clustering = nx.clustering(graph)
    for node in graph.nodes():
        degree = int(graph.degree[node])
        neighbors = list(graph.neighbors(node))
        avg_neighbor_degree = float(np.mean([graph.degree[n] for n in neighbors])) if neighbors else 0.0
        neighborhood = graph.subgraph(neighbors).copy()
        component_sizes = (
            sorted((len(c) for c in nx.connected_components(neighborhood)), reverse=True)
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


def average_path_length_reachable(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    ordered_nodes = sorted(int(node) for node in graph.nodes())
    index = {node: idx for idx, node in enumerate(ordered_nodes)}
    total_distance = 0.0
    pair_count = 0
    for source, distances in nx.all_pairs_shortest_path_length(graph):
        src_idx = index[int(source)]
        for target, dist in distances.items():
            if index[int(target)] > src_idx:
                total_distance += float(dist)
                pair_count += 1
    if pair_count == 0:
        return float("inf")
    return total_distance / float(pair_count)


def degree_mae(original: nx.Graph, anonymized: nx.Graph) -> float:
    if original.number_of_nodes() == 0:
        return 0.0
    errors = []
    for node in original.nodes():
        od = float(original.degree[node])
        ad = float(anonymized.degree[node]) if anonymized.has_node(node) else 0.0
        errors.append(abs(od - ad))
    return float(np.mean(errors))


def compute_graph_stats(graph: nx.Graph) -> Dict[str, float]:
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "clustering": float(nx.average_clustering(graph)),
        "apl": float(average_path_length_reachable(graph)),
    }


def compute_metrics(original: nx.Graph, anonymized: nx.Graph, original_stats: Dict[str, float] | None = None) -> Dict[str, float]:
    if original_stats is None:
        original_stats = compute_graph_stats(original)
    anonymized_stats = compute_graph_stats(anonymized)

    on = original_stats["nodes"]
    an = anonymized_stats["nodes"]
    oe = original_stats["edges"]
    ae = anonymized_stats["edges"]
    od = original_stats["density"]
    ad = anonymized_stats["density"]
    occ = original_stats["clustering"]
    acc = anonymized_stats["clustering"]
    oapl = original_stats["apl"]
    aapl = anonymized_stats["apl"]

    return {
        "original_nodes": on,
        "anonymized_nodes": an,
        "node_variation": an - on,
        "original_edges": oe,
        "anonymized_edges": ae,
        "edge_variation": ae - oe,
        "original_density": od,
        "anonymized_density": ad,
        "density_variation": ad - od,
        "mae": degree_mae(original, anonymized),
        "original_clustering": occ,
        "anonymized_clustering": acc,
        "clustering_variation": acc - occ,
        "original_apl": oapl,
        "anonymized_apl": aapl,
        "apl_variation": aapl - oapl,
    }


def parse_mapping_file(mapping_file: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not mapping_file.exists():
        return mapping
    with mapping_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                left = int(parts[0])
                right = int(parts[1])
            except ValueError:
                continue
            mapping[left] = right
    return mapping


def run_secgraph_ns_attack(
    original_graph: nx.Graph,
    anonymized_graph: nx.Graph,
    original_pairs_path: Path,
    anonymized_pairs_path: Path,
    output_dir: Path,
    run_seed: int,
    theta: float = 0.5,
) -> Dict[str, float | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    common_nodes = sorted(set(int(n) for n in original_graph.nodes()) & set(int(n) for n in anonymized_graph.nodes()))
    if len(common_nodes) < 2:
        return {
            "deanon_success_pct": 0.0,
            "deanon_correct_mappings": 0.0,
            "deanon_eval_nodes": 0.0,
            "deanon_mapped_nonseed": 0.0,
            "attack_status": "skipped_not_enough_common_nodes",
            "attack_stdout": "",
            "attack_stderr": "",
        }

    rng = random.Random(run_seed)
    seed_count = max(5, int(round(0.01 * len(common_nodes))))
    seed_count = min(seed_count, max(1, len(common_nodes) - 1), 50)
    seed_nodes = set(rng.sample(common_nodes, seed_count))

    seed_file = output_dir / "seed.txt"
    mapping_output = output_dir / "ns_mapping_output.txt"

    if mapping_output.exists() and mapping_output.stat().st_size > 0 and seed_file.exists():
        mapping = parse_mapping_file(mapping_output)
        eval_nodes = set(common_nodes) - seed_nodes
        mapped_nonseed = {l: r for l, r in mapping.items() if l in eval_nodes}
        correct = sum(1 for l, r in mapped_nonseed.items() if r == l and r in eval_nodes)
        success = 100.0 * float(correct) / float(len(eval_nodes)) if eval_nodes else 0.0
        return {
            "deanon_success_pct": success,
            "deanon_correct_mappings": float(correct),
            "deanon_eval_nodes": float(len(eval_nodes)),
            "deanon_mapped_nonseed": float(len(mapped_nonseed)),
            "attack_status": "reused_existing_mapping",
            "attack_stdout": "",
            "attack_stderr": "",
        }

    with seed_file.open("w", encoding="utf-8") as handle:
        for node in sorted(seed_nodes):
            handle.write(f"{node} {node}\n")

    command = [
        "java",
        "-Xmx8g",
        "-jar",
        str(SECGRAPH_JAR),
        "-m",
        "d",
        "-a",
        "NS",
        "-gA",
        str(anonymized_pairs_path),
        "-gB",
        str(original_pairs_path),
        "-seed",
        str(seed_file),
        "-theta",
        str(theta),
        "-gO",
        str(mapping_output),
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=75, check=False)
    except subprocess.TimeoutExpired:
        return {
            "deanon_success_pct": float("nan"),
            "deanon_correct_mappings": float("nan"),
            "deanon_eval_nodes": float(len(common_nodes) - len(seed_nodes)),
            "deanon_mapped_nonseed": float("nan"),
            "attack_status": "timeout",
            "attack_stdout": "",
            "attack_stderr": "Timeout",
        }

    mapping = parse_mapping_file(mapping_output)
    eval_nodes = set(common_nodes) - seed_nodes
    mapped_nonseed = {l: r for l, r in mapping.items() if l in eval_nodes}
    correct = sum(1 for l, r in mapped_nonseed.items() if r == l and r in eval_nodes)
    success = 100.0 * float(correct) / float(len(eval_nodes)) if eval_nodes else 0.0
    status = "ok" if result.returncode == 0 else f"error_code_{result.returncode}"
    return {
        "deanon_success_pct": success,
        "deanon_correct_mappings": float(correct),
        "deanon_eval_nodes": float(len(eval_nodes)),
        "deanon_mapped_nonseed": float(len(mapped_nonseed)),
        "attack_status": status,
        "attack_stdout": result.stdout.strip()[:500],
        "attack_stderr": result.stderr.strip()[:500],
    }


def safe_float(value: float | str) -> float:
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_results_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_pdf_report(results: List[Dict[str, object]], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = pdf_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir.resolve())

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    datasets = sorted(set(str(row["dataset"]) for row in results))
    methods = sorted(set(str(row["method"]) for row in results))
    k_values = sorted(set(int(row["k"]) for row in results))
    method_labels = sorted(set(str(row["method_label"]) for row in results))

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Rapport Comparatif d'Anonymisation", fontsize=18, fontweight="bold")
        txt = (
            f"Datasets: {', '.join(datasets)}\n"
            f"Valeurs de k: {', '.join(str(k) for k in k_values)}\n"
            f"Methodes: {', '.join(method_labels)}\n\n"
            "Metriques structurelles:\n"
            "- Variation de noeuds\n"
            "- Variation d'aretes\n"
            "- Variation de densite\n\n"
            "Metriques d'utilite/confidentialite:\n"
            "- MAE\n"
            "- Variation coefficient de clustering\n"
            "- Variation APL\n"
            "- Pourcentage de reussite de desanonymisation (SecGraph NS)\n"
        )
        fig.text(0.05, 0.72, txt, fontsize=12, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for dataset in datasets:
            rows = [row for row in results if str(row["dataset"]) == dataset]
            rows.sort(key=lambda row: (int(row["k"]), str(row["method"])))
            cols = [
                "k",
                "method_label",
                "node_variation",
                "edge_variation",
                "density_variation",
                "mae",
                "clustering_variation",
                "apl_variation",
                "deanon_success_pct",
            ]
            table_data = []
            for row in rows:
                table_data.append(
                    [
                        int(row["k"]),
                        str(row["method_label"]),
                        f"{safe_float(row['node_variation']):.1f}",
                        f"{safe_float(row['edge_variation']):.1f}",
                        f"{safe_float(row['density_variation']):.5f}",
                        f"{safe_float(row['mae']):.4f}",
                        f"{safe_float(row['clustering_variation']):.5f}",
                        f"{safe_float(row['apl_variation']):.4f}",
                        f"{safe_float(row['deanon_success_pct']):.2f}",
                    ]
                )

            fig_table = plt.figure(figsize=(11.69, 8.27))
            fig_table.suptitle(f"Resultats detailles - {dataset}", fontsize=16, fontweight="bold")
            ax = fig_table.add_subplot(111)
            ax.axis("off")
            table = ax.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            pdf.savefig(fig_table)
            plt.close(fig_table)

            metric_specs = [
                ("Variation noeuds", "node_variation"),
                ("Variation aretes", "edge_variation"),
                ("Variation densite", "density_variation"),
                ("MAE", "mae"),
                ("Variation clustering", "clustering_variation"),
                ("Variation APL", "apl_variation"),
                ("Reussite attaque (%)", "deanon_success_pct"),
            ]

            fig_plot, axes = plt.subplots(2, 4, figsize=(14, 8))
            fig_plot.suptitle(f"Tendance des metriques selon k - {dataset}", fontsize=16, fontweight="bold")
            axes_flat = axes.flatten()

            for idx, (title, metric_name) in enumerate(metric_specs):
                axis = axes_flat[idx]
                for method in methods:
                    mrows = [r for r in rows if str(r["method"]) == method and str(r["status"]) == "ok"]
                    if not mrows:
                        continue
                    mrows.sort(key=lambda row: int(row["k"]))
                    x = [int(r["k"]) for r in mrows]
                    y = [safe_float(r[metric_name]) for r in mrows]
                    axis.plot(x, y, marker="o", linewidth=2, label=str(METHOD_LABELS.get(method, method)))
                axis.set_title(title)
                axis.set_xlabel("k")
                axis.grid(True, alpha=0.3)

            axes_flat[-1].axis("off")
            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig_plot.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
            fig_plot.tight_layout(rect=[0, 0.05, 1, 0.95])
            pdf.savefig(fig_plot)
            plt.close(fig_plot)

        valid = [row for row in results if str(row["status"]) == "ok"]
        by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in valid:
            by_method[str(row["method"])].append(row)

        fig_summary = plt.figure(figsize=(11.69, 8.27))
        fig_summary.suptitle("Synthese Globale", fontsize=16, fontweight="bold")
        ax = fig_summary.add_subplot(111)
        ax.axis("off")

        lines = []
        for method in sorted(by_method):
            rows = by_method[method]
            avg_mae = np.nanmean([safe_float(row["mae"]) for row in rows]) if rows else float("nan")
            avg_attack = np.nanmean([safe_float(row["deanon_success_pct"]) for row in rows]) if rows else float("nan")
            avg_apl_var = np.nanmean([abs(safe_float(row["apl_variation"])) for row in rows]) if rows else float("nan")
            avg_cc_var = np.nanmean([abs(safe_float(row["clustering_variation"])) for row in rows]) if rows else float("nan")
            lines.append(
                f"{METHOD_LABELS.get(method, method)}: MAE={avg_mae:.4f}, "
                f"|APL var|={avg_apl_var:.4f}, |Clustering var|={avg_cc_var:.5f}, "
                f"Succes attaque={avg_attack:.2f}%"
            )
        if not lines:
            lines = ["Aucun run valide pour la synthese."]

        ax.text(0.03, 0.92, "\n".join(lines), fontsize=12, va="top")
        ax.text(
            0.03,
            0.2,
            "Guide d'interpretation:\n"
            "- Plus le taux de reussite de l'attaque est bas, meilleure est la confidentialite.\n"
            "- Plus MAE, |APL variation| et |clustering variation| sont bas, meilleure est l'utilite.",
            fontsize=11,
            va="top",
        )
        pdf.savefig(fig_summary)
        plt.close(fig_summary)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark complet Cora/Citeseer pour Ange original, Ange modifie NCC et Zhou-Pei; "
            "attaque SecGraph NS, metriques et rapport PDF."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"], help="Datasets (data/<dataset>.pairs)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 8, 10, 15, 100], help="Valeurs de k")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ange_original", "ange_modifie_ncc", "zhou_pei"],
        choices=["ange_original", "ange_modifie_ncc", "zhou_pei"],
        help="Methodes a evaluer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed de base")
    parser.add_argument("--experiments-dir", default="results/experiments", help="Repertoire de sortie")
    parser.add_argument("--skip-existing", action="store_true", help="Reutiliser les graphes anonymises existants")
    parser.add_argument("--no-report", action="store_true", help="Ne pas generer le PDF")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    method_functions = {
        "ange_original": anonymize_ange_original,
        "ange_modifie_ncc": anonymize_ange_modified_ncc,
        "zhou_pei": anonymize_zhou_pei,
    }

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.is_absolute():
        experiments_dir = PROJECT_ROOT / experiments_dir
    pairs_dir = experiments_dir / "pairs"
    attacks_dir = experiments_dir / "attacks"
    reports_dir = experiments_dir / "reports"
    results_csv = experiments_dir / "results_metrics.csv"
    report_pdf = reports_dir / "comparison_report.pdf"

    if not SECGRAPH_JAR.exists():
        raise FileNotFoundError(f"SecGraph jar introuvable: {SECGRAPH_JAR}")

    results: List[Dict[str, object]] = []

    for dataset in args.datasets:
        original_pairs = DATA_DIR / f"{dataset}.pairs"
        if not original_pairs.exists():
            raise FileNotFoundError(f"Dataset introuvable: {original_pairs}")
        print(f"\n[Dataset] {dataset}")
        original_graph = load_graph_from_pairs(original_pairs)
        original_stats = compute_graph_stats(original_graph)
        print(f"Graphe original: nodes={original_graph.number_of_nodes()}, edges={original_graph.number_of_edges()}")

        for k in args.k_values:
            for method in args.methods:
                method_label = METHOD_LABELS.get(method, method)
                print(f"  - Methode={method_label}, k={k}")
                run_seed = int(args.seed + (hash((dataset, method, k)) % 1_000_000))
                run_id = f"{dataset}_{method}_k{k}"
                anonymized_pairs = pairs_dir / dataset / method / f"{run_id}.pairs"

                row: Dict[str, object] = {
                    "dataset": dataset,
                    "k": int(k),
                    "method": method,
                    "method_label": method_label,
                    "status": "ok",
                    "error_message": "",
                    "runtime_seconds": float("nan"),
                    "attack_status": "",
                    "attack_stdout": "",
                    "attack_stderr": "",
                    "deanon_success_pct": float("nan"),
                    "deanon_correct_mappings": float("nan"),
                    "deanon_eval_nodes": float("nan"),
                    "deanon_mapped_nonseed": float("nan"),
                    "original_nodes": float("nan"),
                    "anonymized_nodes": float("nan"),
                    "node_variation": float("nan"),
                    "original_edges": float("nan"),
                    "anonymized_edges": float("nan"),
                    "edge_variation": float("nan"),
                    "original_density": float("nan"),
                    "anonymized_density": float("nan"),
                    "density_variation": float("nan"),
                    "mae": float("nan"),
                    "original_clustering": float("nan"),
                    "anonymized_clustering": float("nan"),
                    "clustering_variation": float("nan"),
                    "original_apl": float("nan"),
                    "anonymized_apl": float("nan"),
                    "apl_variation": float("nan"),
                }

                start_time = time.perf_counter()
                try:
                    if anonymized_pairs.exists() and args.skip_existing:
                        anonymized_graph = load_graph_from_pairs(anonymized_pairs)
                    else:
                        anonymizer = method_functions[method]
                        anonymized_graph = anonymizer(original_graph.copy(), k=int(k), seed=run_seed)
                        save_graph_to_pairs(anonymized_graph, anonymized_pairs)

                    metrics = compute_metrics(original_graph, anonymized_graph, original_stats=original_stats)
                    row.update(metrics)

                    attack_output_dir = attacks_dir / run_id
                    attack_metrics = run_secgraph_ns_attack(
                        original_graph=original_graph,
                        anonymized_graph=anonymized_graph,
                        original_pairs_path=original_pairs,
                        anonymized_pairs_path=anonymized_pairs,
                        output_dir=attack_output_dir,
                        run_seed=run_seed,
                        theta=0.5,
                    )
                    row.update(attack_metrics)
                except Exception as exc:  # pylint: disable=broad-except
                    row["status"] = "error"
                    row["error_message"] = str(exc)
                    print(f"    Erreur: {exc}")

                row["runtime_seconds"] = float(time.perf_counter() - start_time)
                results.append(row)
                write_results_csv(results, results_csv)
                print(
                    f"    status={row['status']}, nodes_var={row['node_variation']}, "
                    f"edges_var={row['edge_variation']}, MAE={row['mae']}, "
                    f"attack_success={row['deanon_success_pct']}"
                )

    write_results_csv(results, results_csv)
    print(f"\nCSV ecrit: {results_csv}")
    if not args.no_report:
        generate_pdf_report(results, report_pdf)
        print(f"PDF ecrit: {report_pdf}")
    else:
        print("Generation PDF ignoree (--no-report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
