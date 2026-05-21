from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..anonymization.hikda.utils_G1HI import modify_graph_to_break_1_neighborhood
from ..anonymization.hikda.utils_WRKDA import WRKDA_main, edge_sig_CI, verify_k_anonymity


def load_graph_from_pairs(path: Path) -> nx.Graph:
    g = nx.Graph()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            g.add_edge(int(parts[0]), int(parts[1]))
    return g


def save_graph_to_pairs(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
    with path.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def anonymize_1hikda(graph: nx.Graph, k: int, seed: int) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)
    g1, _ = modify_graph_to_break_1_neighborhood(graph)
    edge_significant = edge_sig_CI(g1)
    g2 = WRKDA_main(graph, g1, k, edge_significant, lam=0.01)
    return nx.Graph(g2)


def anonymize_1hikda_from_stage1(
    original_graph: nx.Graph,
    stage1_graph: nx.Graph,
    edge_significant: List[tuple[int, int]],
    k: int,
    seed: int,
) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)
    g2 = WRKDA_main(original_graph, stage1_graph, k, edge_significant, lam=0.01)
    return nx.Graph(g2)


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


def compute_metrics(original: nx.Graph, anonymized: nx.Graph, original_stats: Dict[str, float]) -> Dict[str, float]:
    anon_stats = compute_graph_stats(anonymized)
    return {
        "original_nodes": original_stats["nodes"],
        "anonymized_nodes": anon_stats["nodes"],
        "node_variation": anon_stats["nodes"] - original_stats["nodes"],
        "original_edges": original_stats["edges"],
        "anonymized_edges": anon_stats["edges"],
        "edge_variation": anon_stats["edges"] - original_stats["edges"],
        "original_density": original_stats["density"],
        "anonymized_density": anon_stats["density"],
        "density_variation": anon_stats["density"] - original_stats["density"],
        "mae": degree_mae(original, anonymized),
        "original_clustering": original_stats["clustering"],
        "anonymized_clustering": anon_stats["clustering"],
        "clustering_variation": anon_stats["clustering"] - original_stats["clustering"],
        "original_apl": original_stats["apl"],
        "anonymized_apl": anon_stats["apl"],
        "apl_variation": anon_stats["apl"] - original_stats["apl"],
    }


def parse_mapping_file(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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
    original_pairs: Path,
    anonymized_pairs: Path,
    secgraph_jar: Path,
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

    seed_file = output_dir / "seed.txt"
    mapping_output = output_dir / "ns_mapping_output.txt"

    if mapping_output.exists() and mapping_output.stat().st_size > 0 and seed_file.exists():
        stored_seed_nodes: set[int] = set()
        with seed_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                try:
                    stored_seed_nodes.add(int(parts[0]))
                except ValueError:
                    continue
        seed_nodes = stored_seed_nodes
        mapping = parse_mapping_file(mapping_output)
        eval_nodes = set(common_nodes) - seed_nodes
        mapped_nonseed = {left: right for left, right in mapping.items() if left in eval_nodes}
        correct = sum(1 for left, right in mapped_nonseed.items() if right == left and right in eval_nodes)
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

    rng = random.Random(run_seed)
    seed_count = max(5, int(round(0.01 * len(common_nodes))))
    seed_count = min(seed_count, max(1, len(common_nodes) - 1), 50)
    seed_nodes = set(rng.sample(common_nodes, seed_count))

    with seed_file.open("w", encoding="utf-8") as f:
        for node in sorted(seed_nodes):
            f.write(f"{node} {node}\n")
    command = [
        "java",
        "-Xmx8g",
        "-jar",
        str(secgraph_jar),
        "-m",
        "d",
        "-a",
        "NS",
        "-gA",
        str(anonymized_pairs),
        "-gB",
        str(original_pairs),
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
    mapped_nonseed = {left: right for left, right in mapping.items() if left in eval_nodes}
    correct = sum(1 for left, right in mapped_nonseed.items() if right == left and right in eval_nodes)
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


def write_results_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: float | str) -> float:
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def generate_pdf_report(rows: List[Dict[str, object]], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = pdf_path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    datasets = sorted(set(str(r["dataset"]) for r in rows))
    k_values = sorted(set(int(r["k"]) for r in rows))

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Evaluation 1HIkDA - Cora/Citeseer", fontsize=18, fontweight="bold")
        txt = (
            f"Datasets: {', '.join(datasets)}\n"
            f"k: {', '.join(str(k) for k in k_values)}\n\n"
            "Metriques structurelles:\n"
            "- Variation de noeuds\n- Variation d'aretes\n- Variation de densite\n\n"
            "Metriques utilite/confidentialite:\n"
            "- MAE\n- Variation coefficient de clustering\n- Variation APL\n"
            "- Pourcentage de reussite attaque NS (SecGraph)\n"
        )
        fig.text(0.05, 0.76, txt, fontsize=12, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for dataset in datasets:
            drows = [r for r in rows if str(r["dataset"]) == dataset]
            drows.sort(key=lambda r: int(r["k"]))

            columns = [
                "k",
                "k_degree_anonymous",
                "node_variation",
                "edge_variation",
                "density_variation",
                "mae",
                "clustering_variation",
                "apl_variation",
                "deanon_success_pct",
            ]
            cell_text = []
            for r in drows:
                cell_text.append(
                    [
                        int(r["k"]),
                        str(r["k_degree_anonymous"]),
                        f"{safe_float(r['node_variation']):.1f}",
                        f"{safe_float(r['edge_variation']):.1f}",
                        f"{safe_float(r['density_variation']):.5f}",
                        f"{safe_float(r['mae']):.4f}",
                        f"{safe_float(r['clustering_variation']):.5f}",
                        f"{safe_float(r['apl_variation']):.4f}",
                        f"{safe_float(r['deanon_success_pct']):.2f}",
                    ]
                )

            fig_table = plt.figure(figsize=(11.69, 8.27))
            fig_table.suptitle(f"Resultats detailles - {dataset}", fontsize=16, fontweight="bold")
            ax = fig_table.add_subplot(111)
            ax.axis("off")
            table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            pdf.savefig(fig_table)
            plt.close(fig_table)

            metrics = [
                ("Variation noeuds", "node_variation"),
                ("Variation aretes", "edge_variation"),
                ("Variation densite", "density_variation"),
                ("MAE", "mae"),
                ("Variation clustering", "clustering_variation"),
                ("Variation APL", "apl_variation"),
                ("Succes attaque (%)", "deanon_success_pct"),
            ]
            fig_plot, axes = plt.subplots(2, 4, figsize=(14, 8))
            fig_plot.suptitle(f"Tendance des metriques - {dataset}", fontsize=16, fontweight="bold")
            axes_flat = axes.flatten()
            x = [int(r["k"]) for r in drows]
            for i, (title, metric_name) in enumerate(metrics):
                ax = axes_flat[i]
                y = [safe_float(r[metric_name]) for r in drows]
                ax.plot(x, y, marker="o", linewidth=2)
                ax.set_title(title)
                ax.set_xlabel("k")
                ax.grid(True, alpha=0.3)
            axes_flat[-1].axis("off")
            fig_plot.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_plot)
            plt.close(fig_plot)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 1HIkDA sur Cora/Citeseer + attaque SecGraph NS + metriques + PDF."
    )
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 8, 10, 15, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str((PROJECT_ROOT / "data").resolve()))
    parser.add_argument("--secgraph-jar", default=str((PROJECT_ROOT / "secGraph" / "secGraphCLI.jar").resolve()))
    parser.add_argument("--experiments-dir", default="results/experiments_1hikda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    secgraph_jar = Path(args.secgraph_jar)
    if not secgraph_jar.exists():
        raise FileNotFoundError(f"SecGraph jar introuvable: {secgraph_jar}")

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.is_absolute():
        experiments_dir = PROJECT_ROOT / experiments_dir
    pairs_dir = experiments_dir / "pairs"
    attacks_dir = experiments_dir / "attacks"
    stage1_dir = experiments_dir / "stage1"
    reports_dir = experiments_dir / "reports"
    results_csv = experiments_dir / "results_metrics.csv"
    report_pdf = reports_dir / "comparison_report.pdf"

    rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        original_pairs = data_dir / f"{dataset}.pairs"
        if not original_pairs.exists():
            raise FileNotFoundError(f"Dataset introuvable: {original_pairs}")

        print(f"\n[Dataset] {dataset}")
        original_graph = load_graph_from_pairs(original_pairs)
        original_stats = compute_graph_stats(original_graph)
        print(f"Original: nodes={original_graph.number_of_nodes()}, edges={original_graph.number_of_edges()}")

        stage1_pairs = stage1_dir / f"{dataset}_g1hi.pairs"
        stage1_edges_file = stage1_dir / f"{dataset}_edge_sig_ci.txt"
        dataset_seed = int(args.seed + (hash(dataset) % 1_000_000))

        if stage1_pairs.exists() and stage1_edges_file.exists() and args.skip_existing:
            stage1_graph = load_graph_from_pairs(stage1_pairs)
            edge_significant: List[tuple[int, int]] = []
            with stage1_edges_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    edge_significant.append((int(parts[0]), int(parts[1])))
        else:
            random.seed(dataset_seed)
            np.random.seed(dataset_seed)
            stage1_graph, _ = modify_graph_to_break_1_neighborhood(original_graph)
            save_graph_to_pairs(stage1_graph, stage1_pairs)
            edge_significant = edge_sig_CI(stage1_graph)
            stage1_edges_file.parent.mkdir(parents=True, exist_ok=True)
            with stage1_edges_file.open("w", encoding="utf-8") as f:
                for u, v in edge_significant:
                    f.write(f"{int(u)} {int(v)}\n")

        for k in args.k_values:
            print(f"  - k={k}")
            run_seed = int(args.seed + (hash((dataset, k)) % 1_000_000))
            run_id = f"{dataset}_1HIkDA_k{k}"
            anonymized_pairs = pairs_dir / dataset / f"{run_id}.pairs"

            row: Dict[str, object] = {
                "dataset": dataset,
                "k": int(k),
                "method": "1HIkDA",
                "status": "ok",
                "error_message": "",
                "runtime_seconds": float("nan"),
                "k_degree_anonymous": "",
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

            t0 = time.perf_counter()
            try:
                if anonymized_pairs.exists() and args.skip_existing:
                    anonymized_graph = load_graph_from_pairs(anonymized_pairs)
                else:
                    anonymized_graph = anonymize_1hikda_from_stage1(
                        original_graph=original_graph.copy(),
                        stage1_graph=stage1_graph.copy(),
                        edge_significant=edge_significant,
                        k=int(k),
                        seed=run_seed,
                    )
                    save_graph_to_pairs(anonymized_graph, anonymized_pairs)

                row["k_degree_anonymous"] = str(verify_k_anonymity(anonymized_graph, int(k)))
                row.update(compute_metrics(original_graph, anonymized_graph, original_stats))
                row.update(
                    run_secgraph_ns_attack(
                        original_graph=original_graph,
                        anonymized_graph=anonymized_graph,
                        original_pairs=original_pairs,
                        anonymized_pairs=anonymized_pairs,
                        secgraph_jar=secgraph_jar,
                        output_dir=attacks_dir / run_id,
                        run_seed=run_seed,
                        theta=0.5,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                row["status"] = "error"
                row["error_message"] = str(exc)
                print(f"    Erreur: {exc}")

            row["runtime_seconds"] = float(time.perf_counter() - t0)
            rows.append(row)
            write_results_csv(rows, results_csv)
            print(
                f"    status={row['status']}, k_anon={row['k_degree_anonymous']}, "
                f"nodes_var={row['node_variation']}, edges_var={row['edge_variation']}, "
                f"MAE={row['mae']}, attack={row['deanon_success_pct']}"
            )

    write_results_csv(rows, results_csv)
    print(f"\nCSV ecrit: {results_csv}")
    if not args.no_report:
        generate_pdf_report(rows, report_pdf)
        print(f"PDF ecrit: {report_pdf}")
    else:
        print("PDF ignore (--no-report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
