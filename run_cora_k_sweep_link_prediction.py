from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.anonymization.ange_original import anonymize_ange_original
from graph_anonymization.anonymization.zhou_pei import anonymize_zhou_pei
from graph_anonymization.benchmarks.hikda_benchmark import anonymize_1hikda
from graph_anonymization.data.io import load_graph_from_pairs, prepare_simple_graph
from graph_anonymization.evaluation.link_prediction import compare_link_prediction_utility


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def build_anonymizers_for_k(k: int, hikda_max_nodes: int = 3000) -> Dict[str, Callable]:
    def ange_original_wrapper(graph, seed: int, _k: int = k):
        return anonymize_ange_original(graph, k=_k, seed=seed)

    def ange_modified_wrapper(graph, seed: int, _k: int = k):
        return anonymize_ange_modified_ncc(graph, k=_k, seed=seed)

    def zhou_pei_wrapper(graph, seed: int, _k: int = k):
        return anonymize_zhou_pei(graph, k=_k, seed=seed)

    def hikda_wrapper(graph, seed: int, _k: int = k, _max_nodes: int = hikda_max_nodes):
        if graph.number_of_nodes() > _max_nodes:
            raise RuntimeError(
                f"1HiKDA skipped for graph with {graph.number_of_nodes()} nodes "
                f"(limit={_max_nodes})."
            )
        return anonymize_1hikda(graph, k=_k, seed=seed)

    return {
        "Ange_Original": ange_original_wrapper,
        "Ange_Modifie_NCC": ange_modified_wrapper,
        "Zhou_Pei": zhou_pei_wrapper,
        "1HiKDA": hikda_wrapper,
    }


def add_global_utility_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    ratio_cols = [
        "auc_utility_ratio",
        "ap_utility_ratio",
        "precision_at_100_utility_ratio",
    ]
    for col in ratio_cols:
        if col not in result.columns:
            result[col] = np.nan
    result["global_utility_score"] = result[ratio_cols].mean(axis=1, skipna=True)
    return result


def build_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
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
    existing_numeric_cols = [col for col in numeric_cols if col in detailed_df.columns]
    summary = (
        detailed_df.groupby(["dataset", "k", "method"], as_index=False)[existing_numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["k", "method"])
        .reset_index(drop=True)
    )
    return summary


def plot_metric_by_k(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return

    plt.figure(figsize=(10, 6))
    methods = [m for m in sorted(summary_df["method"].unique())]
    for method in methods:
        subset = summary_df[summary_df["method"] == method].sort_values("k")
        plt.plot(subset["k"], subset[metric], marker="o", linewidth=2, label=method)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def render_table_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(
            lambda x: "NaN"
            if pd.isna(x)
            else (f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
        )
    table = ax.table(
        cellText=rendered.values,
        colLabels=rendered.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    y = 0.96
    for line in lines:
        wrapped = textwrap.wrap(line, width=100) or [""]
        for part in wrapped:
            ax.text(0.02, y, part, va="top", ha="left", fontsize=10, transform=ax.transAxes)
            y -= 0.032
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.96

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(summary_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("Interpretation automatique par valeur de k.")

    for k in sorted(summary_df["k"].unique()):
        sdf = summary_df[summary_df["k"] == k].copy()
        anon = sdf[sdf["method"] != "original"].copy()
        if anon.empty:
            lines.append(f"k={k}: aucune methode anonymisee valide.")
            continue

        anon_valid = anon.dropna(subset=["global_utility_score"]).copy()
        if anon_valid.empty:
            lines.append(f"k={k}: toutes les methodes anonymisees ont des metriques invalides.")
            continue

        best = anon_valid.sort_values("global_utility_score", ascending=False).iloc[0]
        worst = anon_valid.sort_values("global_utility_score", ascending=True).iloc[0]

        lines.append(
            f"k={k}: meilleure utilite={best['method']} "
            f"(score={best['global_utility_score']:.4f}), "
            f"plus degradante={worst['method']} (score={worst['global_utility_score']:.4f})."
        )

        for _, row in anon_valid.sort_values("global_utility_score", ascending=False).iterrows():
            auc = row.get("auc", np.nan)
            ap = row.get("average_precision", np.nan)
            p100 = row.get("precision_at_100", np.nan)
            ratio = row.get("global_utility_score", np.nan)
            note = "conserve bien l'utilite" if pd.notna(ratio) and ratio >= 0.95 else "degrade l'utilite"
            if pd.notna(auc) and abs(auc - 0.5) <= 0.03:
                note += "; AUC proche de 0.5 (comportement proche de l'aleatoire)"
            lines.append(
                f"- {row['method']}: AUC={auc:.4f}, AP={ap:.4f}, P@100={p100:.4f}, "
                f"score={ratio:.4f} -> {note}."
            )

    if {"k", "method", "auc", "average_precision", "precision_at_100"}.issubset(summary_df.columns):
        lines.append("Tendance generale quand k augmente:")
        for metric in ["auc", "average_precision", "precision_at_100"]:
            metric_df = summary_df[summary_df["method"] != "original"].groupby("k")[metric].mean(numeric_only=True)
            if len(metric_df) >= 2:
                trend = metric_df.iloc[-1] - metric_df.iloc[0]
                direction = "augmente" if trend > 0 else "diminue"
                lines.append(f"- {metric}: {direction} en moyenne ({trend:+.4f} entre k min et k max).")

    return lines


def generate_report(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    report_pdf: Path,
    plot_paths: Dict[str, Path],
    dataset: str,
    k_values: List[int],
) -> None:
    report_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_pdf) as pdf:
        render_text_page(
            pdf,
            f"{dataset} - Link Prediction k-sweep",
            [
                f"Dataset: {dataset}",
                f"k values: {', '.join(str(k) for k in k_values)}",
                "Seeds: 42, 123, 2024",
                "test_frac: 0.1",
                "Protocole: split des liens avant anonymisation, anonymisation appliquee uniquement sur G_train.",
            ],
        )

        render_table_page(
            pdf,
            "Resume moyen par methode et par k",
            summary_df[
                [
                    "dataset",
                    "k",
                    "method",
                    "auc",
                    "average_precision",
                    "precision_at_100",
                    "auc_utility_ratio",
                    "ap_utility_ratio",
                    "precision_at_100_utility_ratio",
                    "global_utility_score",
                ]
            ],
        )

        render_table_page(
            pdf,
            "Resultats detailles",
            detailed_df[
                [
                    "dataset",
                    "method",
                    "k",
                    "seed",
                    "auc",
                    "average_precision",
                    "precision_at_50",
                    "precision_at_100",
                    "precision_at_500",
                    "global_utility_score",
                    "method_error",
                ]
            ],
        )

        for title, key in [
            ("AUC by k and method", "auc"),
            ("Average Precision by k and method", "average_precision"),
            ("Precision@100 by k and method", "precision_at_100"),
            ("Global utility score by k and method", "global_utility_score"),
        ]:
            img_path = plot_paths[key]
            if not img_path.exists():
                continue
            image = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.imshow(image)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        render_text_page(pdf, "Interpretation automatique", build_interpretation(summary_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cora link prediction utility k-sweep.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--input", default="data/cora.pairs")
    parser.add_argument("--k-values", default="2,5,10,50")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--output-dir", default="resultat/cora_link_prediction_k_sweep")
    parser.add_argument(
        "--hikda-max-nodes",
        type=int,
        default=3000,
        help="Skip 1HiKDA above this node count and log method_error.",
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input graph not found: {input_path}")

    graph = prepare_simple_graph(load_graph_from_pairs(input_path, node_type=int))

    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)

    all_results: List[pd.DataFrame] = []
    for k in k_values:
        anonymizers = build_anonymizers_for_k(k, hikda_max_nodes=int(args.hikda_max_nodes))
        df_k = compare_link_prediction_utility(
            G_original=graph,
            anonymization_functions=anonymizers,
            test_frac=float(args.test_frac),
            seeds=seeds,
            k_values=(50, 100, 500),
        )
        df_k["dataset"] = dataset
        df_k["k"] = int(k)
        all_results.append(df_k)

    detailed_df = pd.concat(all_results, ignore_index=True)
    detailed_df = add_global_utility_score(detailed_df)

    # Column order requested by the user.
    requested_cols = [
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
    ]
    detailed_df = detailed_df[[c for c in requested_cols if c in detailed_df.columns]]

    summary_df = build_summary(detailed_df)

    slug = dataset.lower().replace(" ", "_")
    detailed_csv = output_dir / f"results_{slug}_link_prediction_k_sweep.csv"
    summary_csv = output_dir / f"results_{slug}_link_prediction_k_sweep_summary.csv"
    report_pdf = output_dir / f"results_{slug}_link_prediction_k_sweep_report.pdf"

    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_paths = {
        "auc": output_dir / "auc_by_k_and_method.png",
        "average_precision": output_dir / "average_precision_by_k_and_method.png",
        "precision_at_100": output_dir / "precision_at_100_by_k_and_method.png",
        "global_utility_score": output_dir / "utility_score_by_k_and_method.png",
    }
    plot_metric_by_k(summary_df, "auc", plot_paths["auc"], "AUC by k and method")
    plot_metric_by_k(
        summary_df,
        "average_precision",
        plot_paths["average_precision"],
        "Average Precision by k and method",
    )
    plot_metric_by_k(
        summary_df,
        "precision_at_100",
        plot_paths["precision_at_100"],
        "Precision@100 by k and method",
    )
    plot_metric_by_k(
        summary_df,
        "global_utility_score",
        plot_paths["global_utility_score"],
        "Global utility score by k and method",
    )

    generate_report(detailed_df, summary_df, report_pdf, plot_paths, dataset=dataset, k_values=k_values)

    ranking_df = summary_df[summary_df["method"] != "original"][
        ["k", "method", "global_utility_score"]
    ].copy()
    ranking_df = ranking_df.sort_values(["k", "global_utility_score"], ascending=[True, False])
    ranking_df["rank_within_k"] = ranking_df.groupby("k")["global_utility_score"].rank(
        method="first", ascending=False
    )
    ranking_df = ranking_df.sort_values(["k", "rank_within_k"])

    print("\n=== Tableau detaille ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(detailed_df)

    print("\n=== Resume moyen par methode et par k ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(summary_df)

    print("\n=== Classement des methodes par k ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(ranking_df)

    print("\n=== Interpretation automatique ===")
    for line in build_interpretation(summary_df):
        print(line)

    print("\n=== Fichiers generes ===")
    print(f"- {detailed_csv}")
    print(f"- {summary_csv}")
    print(f"- {report_pdf}")
    for path in plot_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
