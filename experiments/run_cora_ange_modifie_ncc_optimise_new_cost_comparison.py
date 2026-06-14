from __future__ import annotations

import csv
import hashlib
import inspect
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
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
METHOD_NEW = "AngeModifieNCCOptimise"
K_VALUES = [2, 5, 8, 10, 15, 50]
SEED = 42
ALPHA = 0.3
BETA = 0.4
GAMMA = 0.2
DELTA = 0.1
PASSES = 1
MAX_NODE_ITERATIONS = 0
FULL_COST_GRAPH_THRESHOLD = 10000
ATTACK_TIMEOUT_SECONDS = 300

OUTPUT_DIR = PROJECT_ROOT / "results" / "cora_ange_modifie_ncc_optimise_comparison"
PAIRS_DIR = OUTPUT_DIR / "pairs" / DATASET / METHOD_NEW
ATTACKS_DIR = OUTPUT_DIR / "attacks"
CSV_NEW = OUTPUT_DIR / "cora_ange_modifie_ncc_optimise_metrics.csv"
CSV_COMPARISON = OUTPUT_DIR / "cora_ange_modifie_ncc_optimise_comparison.csv"
PDF_COMPARISON = OUTPUT_DIR / "cora_ange_modifie_ncc_optimise_comparison.pdf"
RUN_LOG = OUTPUT_DIR / "cora_ange_modifie_ncc_optimise_run_log.txt"

EXISTING_RESULTS = PROJECT_ROOT / "results" / "legacy_benchmarks" / "experiments_global4" / "results_metrics_4methods_cora.csv"

OUTPUT_COLUMNS = [
    "dataset",
    "method",
    "k",
    "variation_nodes",
    "variation_edges",
    "variation_density",
    "degree_mae",
    "variation_clustering",
    "variation_apl",
    "attack_success_rate",
    "information_loss",
    "source",
]

METRIC_COLUMNS = [
    "variation_nodes",
    "variation_edges",
    "variation_density",
    "degree_mae",
    "variation_clustering",
    "variation_apl",
    "attack_success_rate",
    "information_loss",
]


def _require_new_signature() -> None:
    signature = inspect.signature(anonymize_ange_modified_ncc)
    missing = [name for name in ("gamma", "delta") if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            "Nouvelle version indisponible: la signature de "
            f"anonymize_ange_modified_ncc ne contient pas {missing}."
        )


def _safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(parsed) or math.isinf(parsed):
        return float("nan")
    return float(parsed)


def _output_value(value: object) -> object:
    parsed = _safe_float(value)
    return "N/A" if math.isnan(parsed) else parsed


def _file_sha1_short(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _degree_profile(graph: nx.Graph, reference_nodes: Sequence[int]) -> List[Dict[str, int]]:
    return [
        {"id": int(node), "degree": int(graph.degree[node]) if graph.has_node(node) else 0}
        for node in reference_nodes
    ]


def _density(graph: nx.Graph) -> float:
    return float(nx.density(graph))


def _clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(graph))


def _normalize_method(value: object) -> str:
    method = str(value).strip()
    mapping = {
        "ange_original": "Ange_Original",
        "ange_modifie_ncc": "Ange_Modifie_NCC",
        "zhou_pei": "Zhou_Pei",
        "1HIkDA": "1HiKDA",
        "1HiKDA": "1HiKDA",
    }
    return mapping.get(method, method)


def _load_existing_results() -> tuple[List[Dict[str, object]], List[str]]:
    if not EXISTING_RESULTS.exists():
        return [], []

    df = pd.read_csv(EXISTING_RESULTS)
    df = df[df["dataset"].astype(str).str.lower() == DATASET]
    methods = sorted(_normalize_method(method) for method in df["method"].dropna().unique())

    existing_by_key: Dict[tuple[str, int], Dict[str, object]] = {}
    for _, row in df.iterrows():
        method = _normalize_method(row.get("method"))
        k_value = int(row.get("k"))
        if k_value not in K_VALUES:
            continue
        existing_by_key[(method, k_value)] = {
            "dataset": DATASET,
            "method": method,
            "k": k_value,
            "variation_nodes": _output_value(row.get("node_variation")),
            "variation_edges": _output_value(row.get("edge_variation")),
            "variation_density": _output_value(row.get("density_variation")),
            "degree_mae": _output_value(row.get("mae")),
            "variation_clustering": _output_value(row.get("clustering_variation")),
            "variation_apl": _output_value(row.get("apl_variation")),
            "attack_success_rate": _output_value(row.get("deanon_success_pct")),
            "information_loss": "N/A",
            "source": "existing_results",
        }

    rows: List[Dict[str, object]] = []
    for method in methods:
        for k_value in K_VALUES:
            rows.append(
                existing_by_key.get(
                    (method, k_value),
                    {
                        "dataset": DATASET,
                        "method": method,
                        "k": k_value,
                        "variation_nodes": "N/A",
                        "variation_edges": "N/A",
                        "variation_density": "N/A",
                        "degree_mae": "N/A",
                        "variation_clustering": "N/A",
                        "variation_apl": "N/A",
                        "attack_success_rate": "N/A",
                        "information_loss": "N/A",
                        "source": "existing_results",
                    },
                )
            )
    return rows, methods


def _run_new_method(original_graph: nx.Graph, original_pairs: Path) -> List[Dict[str, object]]:
    original_nodes = sorted(int(node) for node in original_graph.nodes())
    original_profile = _degree_profile(original_graph, original_nodes)
    original_nodes_count = float(original_graph.number_of_nodes())
    original_edges_count = float(original_graph.number_of_edges())
    original_density = _density(original_graph)
    original_clustering = _clustering(original_graph)
    original_apl = float(average_path_length_reachable(original_graph))

    rows: List[Dict[str, object]] = []
    for k_value in K_VALUES:
        run_seed = SEED + int(k_value)
        anonymized_pairs = PAIRS_DIR / f"cora_ange_modifie_ncc_optimise_k{k_value}.pairs"
        start = time.perf_counter()

        anonymized = anonymize_ange_modified_ncc(
            original_graph.copy(),
            k=int(k_value),
            seed=run_seed,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            delta=DELTA,
            passes=PASSES,
            max_node_iterations=MAX_NODE_ITERATIONS,
            fast_graph_threshold=FULL_COST_GRAPH_THRESHOLD,
            removal_penalty=0.5,
            preserve_original_edges=True,
        )
        anonymized = nx.Graph(anonymized)
        anonymized.remove_edges_from(nx.selfloop_edges(anonymized))
        save_graph_to_pairs(anonymized, anonymized_pairs)

        anonymized_profile = _degree_profile(anonymized, original_nodes)
        attack_success_rate: object = "N/A"
        attack_status = "java_not_found"
        if shutil.which("java") is not None:
            attack = run_secgraph_ns_attack(
                original_graph=original_graph,
                anonymized_graph=anonymized,
                original_pairs_path=original_pairs,
                anonymized_pairs_path=anonymized_pairs,
                output_dir=ATTACKS_DIR
                / (
                    f"cora_ange_modifie_ncc_optimise_k{k_value}_"
                    f"{_file_sha1_short(anonymized_pairs)}_t{ATTACK_TIMEOUT_SECONDS}"
                ),
                run_seed=run_seed,
                timeout_seconds=ATTACK_TIMEOUT_SECONDS,
            )
            attack_success_rate = _output_value(attack.get("deanon_success_pct"))
            attack_status = str(attack.get("attack_status", "N/A"))

        row = {
            "dataset": DATASET,
            "method": METHOD_NEW,
            "k": int(k_value),
            "variation_nodes": float(anonymized.number_of_nodes()) - original_nodes_count,
            "variation_edges": float(anonymized.number_of_edges()) - original_edges_count,
            "variation_density": _density(anonymized) - original_density,
            "degree_mae": float(degree_mae(original_graph, anonymized)),
            "variation_clustering": _clustering(anonymized) - original_clustering,
            "variation_apl": float(average_path_length_reachable(anonymized)) - original_apl,
            "attack_success_rate": attack_success_rate,
            "information_loss": float(calculate_il(original_profile, anonymized_profile)),
            "source": "new_run",
        }
        rows.append(row)
        print(
            f"k={k_value} done in {time.perf_counter() - start:.2f}s "
            f"attack_status={attack_status}"
        )

    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "N/A") for column in OUTPUT_COLUMNS})


def _metric_series(rows: Sequence[Dict[str, object]], method: str, metric: str) -> tuple[List[int], List[float]]:
    filtered = [row for row in rows if row["method"] == method]
    filtered.sort(key=lambda row: int(row["k"]))
    x_values: List[int] = []
    y_values: List[float] = []
    for row in filtered:
        value = _safe_float(row.get(metric))
        if math.isnan(value):
            continue
        x_values.append(int(row["k"]))
        y_values.append(value)
    return x_values, y_values


def _format_cell(value: object) -> str:
    parsed = _safe_float(value)
    if math.isnan(parsed):
        return "N/A"
    return f"{parsed:.6f}"


def _write_pdf(rows: Sequence[Dict[str, object]], methods_found: Sequence[str]) -> None:
    PDF_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = OUTPUT_DIR / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

    with PdfPages(PDF_COMPARISON) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Comparaison Cora - AngeModifieNCCOptimise", fontsize=16, fontweight="bold")
        fig.text(
            0.07,
            0.85,
            (
                "Dataset : Cora\n"
                f"Valeurs de k : {', '.join(str(k) for k in K_VALUES)}\n"
                "Seule methode executee : AngeModifieNCCOptimise\n"
                "Fonction de cout : degre + NCC + voisins communs + communaute\n"
                f"Parametres : alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, delta={DELTA}\n"
                "Les autres methodes sont chargees depuis les resultats existants.\n\n"
                "Metriques incluses uniquement : variation de noeuds, variation d'aretes, "
                "variation de densite, MAE, variation de clustering, variation APL, "
                "reussite de l'attaque, information loss."
            ),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        df = pd.DataFrame(rows)[OUTPUT_COLUMNS].copy()
        df = df.sort_values(["k", "method"]).reset_index(drop=True)
        display_columns = ["method", "k"] + METRIC_COLUMNS
        display_df = df[display_columns].copy()
        for metric in METRIC_COLUMNS:
            display_df[metric] = display_df[metric].map(_format_cell)

        fig_table, ax_table = plt.subplots(figsize=(16, 9))
        ax_table.axis("off")
        ax_table.set_title("Tableau comparatif", fontsize=14, fontweight="bold", pad=12)
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

        metric_titles = [
            ("Variation de noeuds", "variation_nodes"),
            ("Variation d'aretes", "variation_edges"),
            ("Variation de densite", "variation_density"),
            ("MAE degres", "degree_mae"),
            ("Variation clustering", "variation_clustering"),
            ("Variation APL", "variation_apl"),
            ("Reussite attaque", "attack_success_rate"),
            ("Information Loss", "information_loss"),
        ]
        methods = sorted({str(row["method"]) for row in rows})
        fig_plot, axes = plt.subplots(2, 4, figsize=(16, 9))
        axes_flat = list(axes.flatten())
        for ax, (title, metric) in zip(axes_flat, metric_titles):
            for method in methods:
                x_values, y_values = _metric_series(rows, method, metric)
                if x_values:
                    ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=method)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig_plot.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig_plot.suptitle("Evolution des 8 metriques demandees", fontsize=14, fontweight="bold")
        fig_plot.tight_layout(rect=[0, 0.07, 1, 0.95])
        pdf.savefig(fig_plot, bbox_inches="tight")
        plt.close(fig_plot)

        new_rows = [row for row in rows if row["method"] == METHOD_NEW]
        best_attack = min(new_rows, key=lambda row: _safe_float(row["attack_success_rate"]))
        best_loss = min(new_rows, key=lambda row: _safe_float(row["information_loss"]))
        fig_text = plt.figure(figsize=(11.69, 8.27))
        fig_text.suptitle("Interpretation courte", fontsize=15, fontweight="bold")
        fig_text.text(
            0.07,
            0.86,
            (
                "Cette experimentation isole la nouvelle version de AngeModifieNCCOptimise. "
                "Les valeurs historiques des autres methodes servent uniquement de contexte comparatif.\n\n"
                f"Meilleure resistance observee pour la nouvelle methode : k={int(best_attack['k'])} "
                f"avec un taux de reussite d'attaque de {_format_cell(best_attack['attack_success_rate'])}.\n"
                f"Information loss minimale observee : k={int(best_loss['k'])} "
                f"avec une valeur de {_format_cell(best_loss['information_loss'])}.\n\n"
                "Les valeurs N/A indiquent une metrique absente dans les resultats historiques ou "
                "un k non disponible pour une ancienne methode. Aucune ancienne methode n'a ete recalculee."
            ),
            va="top",
            fontsize=11,
            wrap=True,
        )
        pdf.savefig(fig_text, bbox_inches="tight")
        plt.close(fig_text)

    print(f"PDF written: {PDF_COMPARISON}")


def main() -> int:
    _require_new_signature()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)

    original_pairs = PROJECT_ROOT / "data" / "cora.pairs"
    if not original_pairs.exists():
        raise FileNotFoundError(f"Dataset Cora introuvable: {original_pairs}")

    original_graph = load_graph_from_pairs(original_pairs)
    existing_rows, methods_found = _load_existing_results()
    new_rows = _run_new_method(original_graph, original_pairs)
    comparison_rows = new_rows + existing_rows

    _write_csv(CSV_NEW, new_rows)
    _write_csv(CSV_COMPARISON, comparison_rows)
    _write_pdf(comparison_rows, methods_found)

    with RUN_LOG.open("w", encoding="utf-8") as handle:
        handle.write("dataset=cora\n")
        handle.write(f"method_executed={METHOD_NEW}\n")
        handle.write(f"k_values={K_VALUES}\n")
        handle.write(f"alpha={ALPHA}\n")
        handle.write(f"beta={BETA}\n")
        handle.write(f"gamma={GAMMA}\n")
        handle.write(f"delta={DELTA}\n")
        handle.write(f"passes={PASSES}\n")
        handle.write(f"max_node_iterations={MAX_NODE_ITERATIONS}\n")
        handle.write(f"fast_graph_threshold={FULL_COST_GRAPH_THRESHOLD}\n")
        handle.write(f"attack_timeout_seconds={ATTACK_TIMEOUT_SECONDS}\n")
        handle.write(f"existing_results={EXISTING_RESULTS}\n")
        handle.write(f"existing_methods_found={methods_found}\n")
        handle.write(f"metrics={METRIC_COLUMNS}\n")
        handle.write(f"new_csv={CSV_NEW}\n")
        handle.write(f"comparison_csv={CSV_COMPARISON}\n")
        handle.write(f"comparison_pdf={PDF_COMPARISON}\n")

    print("DONE")
    print(f"Old methods found: {methods_found}")
    print(f"New metrics CSV: {CSV_NEW}")
    print(f"Comparison CSV: {CSV_COMPARISON}")
    print(f"Comparison PDF: {PDF_COMPARISON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
