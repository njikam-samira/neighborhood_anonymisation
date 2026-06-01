from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.benchmarks.full_benchmark import (
    average_path_length_reachable,
    degree_mae,
    load_graph_from_pairs,
    run_secgraph_ns_attack,
    save_graph_to_pairs,
)
from graph_anonymization.metrics.structural_metrics import calculate_il


DATASET = "cora"
K_VALUES = [2, 5, 8, 10, 15, 50]
SEED = 42

METHOD_NEW = "AngeModifieNCCOptimise"
OLD_RESULTS_CANDIDATES = [
    PROJECT_ROOT / "results" / "legacy_benchmarks" / "experiments_global4" / "results_metrics_4methods_cora.csv",
    PROJECT_ROOT / "results" / "legacy_benchmarks" / "experiments_global4" / "results_metrics.csv",
    PROJECT_ROOT / "results" / "legacy_experiments" / "experiments" / "results_metrics.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "results" / "cora_ange_modifie_ncc_optimise_comparison"
PAIRS_DIR = OUTPUT_DIR / "pairs" / DATASET / METHOD_NEW
ATTACKS_DIR = OUTPUT_DIR / "attacks"
CSV_NEW_ONLY = OUTPUT_DIR / "results_cora_ange_modifie_ncc_optimise_only.csv"
CSV_COMPARISON = OUTPUT_DIR / "results_cora_comparison_with_existing_methods.csv"
PDF_COMPARISON = OUTPUT_DIR / "results_cora_comparison_with_existing_methods.pdf"
RUN_LOG = OUTPUT_DIR / "run_log.txt"

METRIC_COLUMNS = [
    "node_variation",
    "edge_variation",
    "density_variation",
    "mae",
    "clustering_variation",
    "apl_variation",
    "attack_success_pct",
    "information_loss",
]


def _safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(parsed) or math.isinf(parsed):
        return float("nan")
    return float(parsed)


def _as_output_value(value: object) -> object:
    val = _safe_float(value)
    if math.isnan(val):
        return "N/A"
    return val


def _degree_profile(graph: nx.Graph, reference_nodes: Sequence[int]) -> List[Dict[str, int]]:
    profile: List[Dict[str, int]] = []
    for node in reference_nodes:
        degree = int(graph.degree[node]) if graph.has_node(node) else 0
        profile.append({"id": int(node), "degree": degree})
    return profile


def _graph_density(graph: nx.Graph) -> float:
    return float(nx.density(graph))


def _graph_clustering(graph: nx.Graph) -> float:
    return float(nx.average_clustering(graph))


def _find_existing_results_csv() -> Optional[Path]:
    for candidate in OLD_RESULTS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _normalize_old_method_name(value: str) -> str:
    key = str(value).strip()
    mapping = {
        "ange_original": "Ange_Original",
        "ange_modifie_ncc": "Ange_Modifie_NCC",
        "zhou_pei": "Zhou_Pei",
        "1HIkDA": "1HiKDA",
        "1HiKDA": "1HiKDA",
    }
    return mapping.get(key, key)


def _load_existing_rows(csv_path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns or "k" not in df.columns or "method" not in df.columns:
        return []

    data = df.copy()
    data["dataset"] = data["dataset"].astype(str).str.lower()
    data = data[data["dataset"] == DATASET]
    data = data[data["k"].isin(K_VALUES)]

    rows: List[Dict[str, object]] = []
    for _, row in data.iterrows():
        method_name = _normalize_old_method_name(str(row.get("method", "")))
        out: Dict[str, object] = {
            "dataset": DATASET,
            "k": int(row["k"]),
            "method": method_name,
            "status": str(row.get("status", "ok")),
            "source": f"existing:{csv_path}",
            "node_variation": _as_output_value(row.get("node_variation")),
            "edge_variation": _as_output_value(row.get("edge_variation")),
            "density_variation": _as_output_value(row.get("density_variation")),
            "mae": _as_output_value(row.get("mae")),
            "clustering_variation": _as_output_value(row.get("clustering_variation")),
            "apl_variation": _as_output_value(row.get("apl_variation")),
            "attack_success_pct": _as_output_value(row.get("deanon_success_pct")),
            # No guaranteed info-loss column in legacy files -> required N/A fallback.
            "information_loss": "N/A",
        }
        rows.append(out)
    return rows


def _run_optimized_only(original_graph: nx.Graph, original_pairs: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    original_nodes = sorted(int(node) for node in original_graph.nodes())
    original_profile = _degree_profile(original_graph, original_nodes)
    original_num_nodes = float(original_graph.number_of_nodes())
    original_num_edges = float(original_graph.number_of_edges())
    original_density = _graph_density(original_graph)
    original_clustering = _graph_clustering(original_graph)
    original_apl = float(average_path_length_reachable(original_graph))

    for k in K_VALUES:
        start = time.perf_counter()
        run_seed = int(SEED + k)
        pair_path = PAIRS_DIR / f"cora_ange_modifie_ncc_optimise_k{k}.pairs"

        row: Dict[str, object] = {
            "dataset": DATASET,
            "k": int(k),
            "method": METHOD_NEW,
            "status": "ok",
            "source": "computed:new",
            "node_variation": "N/A",
            "edge_variation": "N/A",
            "density_variation": "N/A",
            "mae": "N/A",
            "clustering_variation": "N/A",
            "apl_variation": "N/A",
            "attack_success_pct": "N/A",
            "information_loss": "N/A",
            "runtime_seconds": "N/A",
            "error_message": "",
            "attack_status": "N/A",
        }

        try:
            anonymized = anonymize_ange_modified_ncc(
                original_graph.copy(),
                k=int(k),
                seed=run_seed,
                alpha=1.0,
                beta=0.2,
                passes=2,
                max_node_iterations=24,
                fast_graph_threshold=1000,
                removal_penalty=0.5,
                preserve_original_edges=True,
            )
            anonymized = nx.Graph(anonymized)
            anonymized.remove_edges_from(nx.selfloop_edges(anonymized))

            save_graph_to_pairs(anonymized, pair_path)

            anon_num_nodes = float(anonymized.number_of_nodes())
            anon_num_edges = float(anonymized.number_of_edges())
            anon_density = _graph_density(anonymized)
            anon_clustering = _graph_clustering(anonymized)
            anon_apl = float(average_path_length_reachable(anonymized))
            anon_profile = _degree_profile(anonymized, original_nodes)

            row["node_variation"] = anon_num_nodes - original_num_nodes
            row["edge_variation"] = anon_num_edges - original_num_edges
            row["density_variation"] = anon_density - original_density
            row["mae"] = float(degree_mae(original_graph, anonymized))
            row["clustering_variation"] = anon_clustering - original_clustering
            row["apl_variation"] = anon_apl - original_apl
            row["information_loss"] = float(calculate_il(original_profile, anon_profile))

            if shutil.which("java") is not None:
                attack = run_secgraph_ns_attack(
                    original_graph=original_graph,
                    anonymized_graph=anonymized,
                    original_pairs_path=original_pairs,
                    anonymized_pairs_path=pair_path,
                    output_dir=ATTACKS_DIR / f"cora_ange_modifie_ncc_optimise_k{k}",
                    run_seed=run_seed,
                    theta=0.5,
                )
                row["attack_success_pct"] = _as_output_value(attack.get("deanon_success_pct"))
                row["attack_status"] = str(attack.get("attack_status", "N/A"))
            else:
                row["attack_success_pct"] = "N/A"
                row["attack_status"] = "java_not_found"

        except Exception as exc:  # pragma: no cover
            row["status"] = "error"
            row["error_message"] = f"{type(exc).__name__}: {exc}"

        row["runtime_seconds"] = float(time.perf_counter() - start)
        rows.append(row)

    return rows


def _combine_rows(
    optimized_rows: Sequence[Dict[str, object]],
    existing_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    key_to_row: Dict[tuple[str, int, str], Dict[str, object]] = {}
    for row in list(existing_rows) + list(optimized_rows):
        key = (str(row["dataset"]).lower(), int(row["k"]), str(row["method"]))
        key_to_row[key] = dict(row)

    all_methods = sorted(
        {str(row["method"]) for row in optimized_rows}.union({str(row["method"]) for row in existing_rows})
    )

    output: List[Dict[str, object]] = []
    for k in K_VALUES:
        for method in all_methods:
            key = (DATASET, int(k), method)
            if key in key_to_row:
                row = dict(key_to_row[key])
            else:
                row = {
                    "dataset": DATASET,
                    "k": int(k),
                    "method": method,
                    "status": "missing",
                    "source": "N/A",
                    "node_variation": "N/A",
                    "edge_variation": "N/A",
                    "density_variation": "N/A",
                    "mae": "N/A",
                    "clustering_variation": "N/A",
                    "apl_variation": "N/A",
                    "attack_success_pct": "N/A",
                    "information_loss": "N/A",
                    "runtime_seconds": "N/A",
                    "error_message": "",
                    "attack_status": "N/A",
                }

            for col in METRIC_COLUMNS:
                if col not in row:
                    row[col] = "N/A"
            output.append(row)
    return output


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    columns = [
        "dataset",
        "k",
        "method",
        "status",
        "source",
        "node_variation",
        "edge_variation",
        "density_variation",
        "mae",
        "clustering_variation",
        "apl_variation",
        "attack_success_pct",
        "information_loss",
        "runtime_seconds",
        "attack_status",
        "error_message",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "N/A") for key in columns})


def _metric_values(rows: Sequence[Dict[str, object]], method: str, metric: str) -> tuple[List[int], List[float]]:
    x: List[int] = []
    y: List[float] = []
    filtered = [row for row in rows if str(row["method"]) == method]
    filtered.sort(key=lambda row: int(row["k"]))
    for row in filtered:
        val = _safe_float(row.get(metric))
        if math.isnan(val):
            continue
        x.append(int(row["k"]))
        y.append(float(val))
    return x, y


def _render_pdf(rows: Sequence[Dict[str, object]], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = pdf_path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

    methods = sorted({str(row["method"]) for row in rows})

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Comparaison Cora - AngeModifieNCCOptimise vs methodes existantes", fontsize=16, fontweight="bold")
        text = (
            "Dataset: Cora\n"
            f"k values: {', '.join(str(k) for k in K_VALUES)}\n"
            "Seule methode executee: AngeModifieNCCOptimise\n"
            "Autres methodes: chargees depuis resultats existants (sans recalcul)\n\n"
            "Metriques (uniquement):\n"
            "1. Variation de noeuds\n"
            "2. Variation d'aretes\n"
            "3. Variation de densite\n"
            "4. MAE\n"
            "5. Variation coefficient de clustering\n"
            "6. Variation APL\n"
            "7. Pourcentage de reussite de l'attaque\n"
            "8. Information Loss\n"
        )
        fig.text(0.07, 0.88, text, va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        df = pd.DataFrame(rows)
        df = df.sort_values(["k", "method"]).reset_index(drop=True)
        table_cols = ["k", "method"] + METRIC_COLUMNS
        display_df = df[table_cols].copy()
        for col in METRIC_COLUMNS:
            display_df[col] = display_df[col].map(lambda v: "N/A" if math.isnan(_safe_float(v)) else f"{_safe_float(v):.6f}")

        fig_table, ax_table = plt.subplots(figsize=(16, 9))
        ax_table.axis("off")
        ax_table.set_title("Tableau comparatif des metriques", fontsize=14, fontweight="bold", pad=12)
        table = ax_table.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.18)
        pdf.savefig(fig_table, bbox_inches="tight")
        plt.close(fig_table)

        plot_specs = [
            ("Variation de noeuds", "node_variation"),
            ("Variation d'aretes", "edge_variation"),
            ("Variation de densite", "density_variation"),
            ("MAE", "mae"),
            ("Variation clustering", "clustering_variation"),
            ("Variation APL", "apl_variation"),
            ("Reussite attaque (%)", "attack_success_pct"),
            ("Information Loss", "information_loss"),
        ]

        fig_plot, axes = plt.subplots(2, 4, figsize=(16, 9))
        axes_flat = list(axes.flatten())
        for idx, (title, metric) in enumerate(plot_specs):
            ax = axes_flat[idx]
            for method in methods:
                x, y = _metric_values(rows, method=method, metric=metric)
                if not x:
                    continue
                ax.plot(x, y, marker="o", linewidth=2, label=method)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig_plot.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig_plot.suptitle("Tendance par methode", fontsize=14, fontweight="bold")
        fig_plot.tight_layout(rect=[0, 0.06, 1, 0.95])
        pdf.savefig(fig_plot, bbox_inches="tight")
        plt.close(fig_plot)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)

    original_pairs = PROJECT_ROOT / "data" / f"{DATASET}.pairs"
    if not original_pairs.exists():
        raise FileNotFoundError(f"Dataset introuvable: {original_pairs}")

    existing_csv = _find_existing_results_csv()
    existing_rows = _load_existing_rows(existing_csv) if existing_csv else []

    original_graph = load_graph_from_pairs(original_pairs)
    optimized_rows = _run_optimized_only(original_graph=original_graph, original_pairs=original_pairs)
    combined_rows = _combine_rows(optimized_rows=optimized_rows, existing_rows=existing_rows)

    _write_csv(CSV_NEW_ONLY, optimized_rows)
    _write_csv(CSV_COMPARISON, combined_rows)
    _render_pdf(combined_rows, PDF_COMPARISON)

    methods_found = sorted({str(row["method"]) for row in existing_rows})
    with RUN_LOG.open("w", encoding="utf-8") as handle:
        handle.write("dataset=cora\n")
        handle.write(f"k_values={K_VALUES}\n")
        handle.write(f"seed={SEED}\n")
        handle.write(f"existing_source={existing_csv}\n")
        handle.write(f"existing_methods_found={methods_found}\n")
        handle.write(f"metrics={METRIC_COLUMNS}\n")
        handle.write(f"new_only_csv={CSV_NEW_ONLY}\n")
        handle.write(f"comparison_csv={CSV_COMPARISON}\n")
        handle.write(f"comparison_pdf={PDF_COMPARISON}\n")

    print("Experiment done.")
    print(f"Existing source: {existing_csv}")
    print(f"Old methods found: {methods_found}")
    print(f"k tested: {K_VALUES}")
    print(f"New-method CSV: {CSV_NEW_ONLY}")
    print(f"Comparison CSV: {CSV_COMPARISON}")
    print(f"Comparison PDF: {PDF_COMPARISON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
