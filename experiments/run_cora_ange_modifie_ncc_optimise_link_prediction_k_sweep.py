from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.data.io import load_graph_from_pairs, prepare_simple_graph
from graph_anonymization.evaluation.link_prediction import compare_link_prediction_utility


DATASET = "cora"
METHOD = "AngeModifieNCCOptimise"
K_VALUES_DEFAULT = "2,5,10,50"
SEEDS_DEFAULT = "42"
TEST_FRAC_DEFAULT = 0.1
ALPHA = 0.3
BETA = 0.4
GAMMA = 0.2
DELTA = 0.1
PASSES = 1
MAX_NODE_ITERATIONS = 0
FULL_COST_GRAPH_THRESHOLD = 10000

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "resultat" / "cora_ange_modifie_ncc_optimise_link_prediction_k_sweep"
)
DEFAULT_EXISTING_DETAILED = (
    PROJECT_ROOT
    / "resultat"
    / "cora_link_prediction_k_sweep"
    / "results_cora_link_prediction_k_sweep.csv"
)
DEFAULT_EXISTING_SUMMARY = (
    PROJECT_ROOT
    / "resultat"
    / "cora_link_prediction_k_sweep"
    / "results_cora_link_prediction_k_sweep_summary.csv"
)

NUMERIC_COLUMNS = [
    "auc",
    "average_precision",
    "precision_at_50",
    "precision_at_100",
    "precision_at_500",
    "auc_loss",
    "ap_loss",
    "precision_at_100_loss",
    "auc_utility_ratio",
    "ap_utility_ratio",
    "precision_at_100_utility_ratio",
    "global_utility_score",
]


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def require_new_signature() -> None:
    signature = inspect.signature(anonymize_ange_modified_ncc)
    missing = [name for name in ("gamma", "delta") if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            "Nouvelle version indisponible: anonymize_ange_modified_ncc "
            f"ne contient pas {missing}."
        )


def add_global_utility_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    ratio_cols = ["auc_utility_ratio", "ap_utility_ratio", "precision_at_100_utility_ratio"]
    for col in ratio_cols:
        if col not in result.columns:
            result[col] = np.nan
    result["global_utility_score"] = result[ratio_cols].mean(axis=1, skipna=True)
    return result


def build_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    existing_numeric_cols = [col for col in NUMERIC_COLUMNS if col in detailed_df.columns]
    summary = (
        detailed_df.groupby(["dataset", "k", "method"], as_index=False)[existing_numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["k", "method"])
        .reset_index(drop=True)
    )
    return summary


def build_anonymizer_for_k(k_value: int) -> Dict[str, Callable]:
    def wrapper(graph, seed: int, _k: int = int(k_value)):
        return anonymize_ange_modified_ncc(
            graph,
            k=_k,
            seed=int(seed),
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

    return {METHOD: wrapper}


def ordered_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "dataset",
        "method",
        "k",
        "seed",
        "test_frac",
        "num_nodes",
        "num_edges_original",
        "num_edges_train",
        "num_test_pos_edges",
        "auc",
        "average_precision",
        "precision_at_50",
        "precision_at_100",
        "precision_at_500",
        "auc_loss",
        "ap_loss",
        "precision_at_100_loss",
        "auc_utility_ratio",
        "ap_utility_ratio",
        "precision_at_100_utility_ratio",
        "global_utility_score",
        "method_error",
        "source",
    ]
    return df[[col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]]


def load_existing_results(
    path: Path,
    k_values: Sequence[int],
    *,
    detailed: bool,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == DATASET]
    if "k" in df.columns:
        df = df[df["k"].astype(int).isin([int(k) for k in k_values])]
    if "method" in df.columns:
        df = df[df["method"].astype(str) != METHOD]
    if detailed and seeds is not None and "seed" in df.columns:
        df = df[df["seed"].astype(int).isin([int(seed) for seed in seeds])]
    df = add_global_utility_score(df)
    df["source"] = "existing_results"
    if detailed and "seed" not in df.columns:
        df["seed"] = np.nan
    return ordered_columns(df.reset_index(drop=True))


def plot_metric_by_k(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return

    plt.figure(figsize=(10, 6))
    for method in sorted(summary_df["method"].astype(str).unique()):
        subset = summary_df[summary_df["method"].astype(str) == method].sort_values("k")
        values = pd.to_numeric(subset[metric], errors="coerce")
        if values.notna().any():
            plt.plot(subset["k"], values, marker="o", linewidth=2, label=method)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _fmt(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(parsed):
        return "N/A"
    return f"{parsed:.4f}"


def render_text_page(pdf: PdfPages, title: str, paragraphs: Sequence[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    y = 0.86
    for paragraph in paragraphs:
        wrapped = textwrap.fill(str(paragraph), width=115)
        fig.text(0.07, y, wrapped, va="top", fontsize=10.5)
        y -= 0.055 * (wrapped.count("\n") + 1) + 0.025
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_table_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(summary_df: pd.DataFrame, k_values: Sequence[int]) -> List[str]:
    lines = [
        "Le protocole utilise Adamic-Adar sur les mêmes valeurs de k. "
        f"Seule la méthode {METHOD} est recalculée; les autres méthodes sont reprises depuis les CSV existants.",
    ]
    new_rows = summary_df[summary_df["method"].astype(str) == METHOD].sort_values("k")
    if not new_rows.empty:
        for metric in ["auc", "average_precision", "precision_at_100", "global_utility_score"]:
            values = pd.to_numeric(new_rows[metric], errors="coerce")
            if values.notna().any():
                best_idx = values.idxmax()
                best_row = new_rows.loc[best_idx]
                lines.append(
                    f"Pour {METHOD}, meilleur {metric} observé à k={int(best_row['k'])} "
                    f"avec {_fmt(best_row[metric])}."
                )
    for k_value in k_values:
        subset = summary_df[summary_df["k"].astype(int) == int(k_value)].copy()
        subset["global_utility_score"] = pd.to_numeric(subset["global_utility_score"], errors="coerce")
        anonymized = subset[subset["method"].astype(str) != "original"]
        if anonymized["global_utility_score"].notna().any():
            best = anonymized.loc[anonymized["global_utility_score"].idxmax()]
            lines.append(
                f"À k={int(k_value)}, meilleure utilité globale comparative: "
                f"{best['method']} ({_fmt(best['global_utility_score'])})."
            )
    return lines


def generate_pdf(
    comparison_summary: pd.DataFrame,
    output_pdf: Path,
    plot_paths: Dict[str, Path],
    k_values: Sequence[int],
    existing_methods: Sequence[str],
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        render_text_page(
            pdf,
            "Cora - Prédiction de liens Adamic-Adar",
            [
                f"Dataset: Cora. Valeurs de k: {', '.join(str(k) for k in k_values)}.",
                f"Méthode exécutée maintenant: {METHOD}. Paramètres: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, delta={DELTA}. Aucun theta n'est utilisé.",
                "Les autres méthodes ne sont pas recalculées: elles sont chargées depuis les résultats existants du k-sweep Cora.",
                "Méthodes historiques trouvées: " + (", ".join(existing_methods) if existing_methods else "aucune"),
            ],
        )

        table_cols = [
            "k",
            "method",
            "auc",
            "average_precision",
            "precision_at_100",
            "auc_utility_ratio",
            "ap_utility_ratio",
            "precision_at_100_utility_ratio",
            "global_utility_score",
            "source",
        ]
        display_df = comparison_summary[[col for col in table_cols if col in comparison_summary.columns]].copy()
        for col in [
            "auc",
            "average_precision",
            "precision_at_100",
            "auc_utility_ratio",
            "ap_utility_ratio",
            "precision_at_100_utility_ratio",
            "global_utility_score",
        ]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(_fmt)
        render_table_page(pdf, "Résumé comparatif moyen par méthode et par k", display_df)

        render_image_page(pdf, "AUC selon k et méthode", plot_paths["auc"])
        render_image_page(pdf, "Average Precision selon k et méthode", plot_paths["average_precision"])
        render_image_page(pdf, "Precision@100 selon k et méthode", plot_paths["precision_at_100"])
        render_image_page(pdf, "Score global d'utilité selon k et méthode", plot_paths["global_utility_score"])

        render_text_page(
            pdf,
            "Interprétation automatique",
            build_interpretation(comparison_summary, k_values),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cora link prediction k-sweep for the current AngeModifieNCCOptimise method only."
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "cora.pairs"))
    parser.add_argument("--k-values", default=K_VALUES_DEFAULT)
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC_DEFAULT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--existing-detailed", default=str(DEFAULT_EXISTING_DETAILED))
    parser.add_argument("--existing-summary", default=str(DEFAULT_EXISTING_SUMMARY))
    args = parser.parse_args()

    if str(args.dataset).lower() != DATASET:
        raise ValueError("Ce script est limité au dataset Cora.")
    require_new_signature()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    graph_path = Path(args.input)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graphe Cora introuvable: {graph_path}")

    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)
    graph = prepare_simple_graph(load_graph_from_pairs(graph_path, node_type=int))

    new_frames: List[pd.DataFrame] = []
    for k_value in k_values:
        start = time.perf_counter()
        df_k = compare_link_prediction_utility(
            G_original=graph,
            anonymization_functions=build_anonymizer_for_k(int(k_value)),
            test_frac=float(args.test_frac),
            seeds=seeds,
            k_values=(50, 100, 500),
        )
        df_k["dataset"] = DATASET
        df_k["k"] = int(k_value)
        new_frames.append(df_k)
        print(f"k={int(k_value)} done in {time.perf_counter() - start:.2f}s")

    new_detailed_all = add_global_utility_score(pd.concat(new_frames, ignore_index=True))
    new_detailed_all["source"] = "new_run"
    new_method_detailed = new_detailed_all[new_detailed_all["method"].astype(str) == METHOD].copy()
    new_summary = build_summary(new_method_detailed)
    new_summary["source"] = "new_run"

    existing_detailed = load_existing_results(
        Path(args.existing_detailed),
        k_values,
        detailed=True,
        seeds=seeds,
    )
    if existing_detailed.empty:
        existing_summary = load_existing_results(Path(args.existing_summary), k_values, detailed=False)
    else:
        existing_summary = build_summary(existing_detailed)
        existing_summary["source"] = "existing_results"
    existing_methods = sorted(
        method
        for method in existing_summary.get("method", pd.Series(dtype=str)).astype(str).unique()
        if method != METHOD
    )

    comparison_detailed = pd.concat([new_method_detailed, existing_detailed], ignore_index=True)
    comparison_summary = pd.concat([new_summary, existing_summary], ignore_index=True)
    comparison_detailed = ordered_columns(comparison_detailed.sort_values(["k", "method", "seed"]).reset_index(drop=True))
    comparison_summary = ordered_columns(comparison_summary.sort_values(["k", "method"]).reset_index(drop=True))

    new_csv = output_dir / "results_cora_ange_modifie_ncc_optimise_link_prediction_new.csv"
    new_summary_csv = output_dir / "results_cora_ange_modifie_ncc_optimise_link_prediction_new_summary.csv"
    comparison_csv = output_dir / "results_cora_ange_modifie_ncc_optimise_link_prediction_comparison.csv"
    comparison_summary_csv = output_dir / "results_cora_ange_modifie_ncc_optimise_link_prediction_comparison_summary.csv"
    report_pdf = output_dir / "results_cora_ange_modifie_ncc_optimise_link_prediction_comparison_report.pdf"
    run_log = output_dir / "run_cora_ange_modifie_ncc_optimise_link_prediction.log"

    ordered_columns(new_method_detailed).to_csv(new_csv, index=False)
    ordered_columns(new_summary).to_csv(new_summary_csv, index=False)
    comparison_detailed.to_csv(comparison_csv, index=False)
    comparison_summary.to_csv(comparison_summary_csv, index=False)

    plot_paths = {
        "auc": output_dir / "auc_by_k_and_method.png",
        "average_precision": output_dir / "average_precision_by_k_and_method.png",
        "precision_at_100": output_dir / "precision_at_100_by_k_and_method.png",
        "global_utility_score": output_dir / "utility_score_by_k_and_method.png",
    }
    plot_metric_by_k(comparison_summary, "auc", plot_paths["auc"], "AUC selon k et méthode")
    plot_metric_by_k(
        comparison_summary,
        "average_precision",
        plot_paths["average_precision"],
        "Average Precision selon k et méthode",
    )
    plot_metric_by_k(
        comparison_summary,
        "precision_at_100",
        plot_paths["precision_at_100"],
        "Precision@100 selon k et méthode",
    )
    plot_metric_by_k(
        comparison_summary,
        "global_utility_score",
        plot_paths["global_utility_score"],
        "Score global d'utilité selon k et méthode",
    )
    generate_pdf(comparison_summary, report_pdf, plot_paths, k_values, existing_methods)

    with run_log.open("w", encoding="utf-8") as handle:
        handle.write("dataset=cora\n")
        handle.write(f"method_executed={METHOD}\n")
        handle.write(f"k_values={k_values}\n")
        handle.write(f"seeds={seeds}\n")
        handle.write(f"test_frac={float(args.test_frac)}\n")
        handle.write("protocol=Adamic-Adar link prediction\n")
        handle.write(f"alpha={ALPHA}\n")
        handle.write(f"beta={BETA}\n")
        handle.write(f"gamma={GAMMA}\n")
        handle.write(f"delta={DELTA}\n")
        handle.write(f"passes={PASSES}\n")
        handle.write(f"max_node_iterations={MAX_NODE_ITERATIONS}\n")
        handle.write(f"fast_graph_threshold={FULL_COST_GRAPH_THRESHOLD}\n")
        handle.write(f"existing_detailed={Path(args.existing_detailed)}\n")
        handle.write(f"existing_summary={Path(args.existing_summary)}\n")
        handle.write(f"existing_methods_found={existing_methods}\n")
        handle.write(f"new_csv={new_csv}\n")
        handle.write(f"new_summary_csv={new_summary_csv}\n")
        handle.write(f"comparison_csv={comparison_csv}\n")
        handle.write(f"comparison_summary_csv={comparison_summary_csv}\n")
        handle.write(f"report_pdf={report_pdf}\n")

    print("DONE")
    print(f"Existing methods found: {existing_methods}")
    print(f"New CSV: {new_csv}")
    print(f"New summary CSV: {new_summary_csv}")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Comparison summary CSV: {comparison_summary_csv}")
    print(f"PDF: {report_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
