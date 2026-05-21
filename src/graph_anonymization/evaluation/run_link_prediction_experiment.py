from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..anonymization.ange_modified import anonymize_ange_modified_ncc
from ..anonymization.ange_original import anonymize_ange_original
from ..anonymization.zhou_pei import anonymize_zhou_pei
from .link_prediction import compare_link_prediction_utility, prepare_graph


def load_graph_from_pairs(path: Path, nodetype=int) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            graph.add_edge(nodetype(parts[0]), nodetype(parts[1]))
    return prepare_graph(graph)


def discover_project_anonymizers(k: int = 10) -> Dict[str, Callable[[nx.Graph, int], nx.Graph]]:
    """
    Cherche les fonctions d'anonymisation deja presentes dans le projet.
    Retourne des wrappers homogenes: f(graph, seed=...) -> graph_anonymise.
    """
    def _ange_original_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_ange_original(graph, k=_k, seed=seed)

    def _ange_ncc_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_ange_modified_ncc(graph, k=_k, seed=seed)

    def _zhou_pei_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_zhou_pei(graph, k=_k, seed=seed)

    return {
        "Ange_Original": _ange_original_wrapper,
        "Ange_Modifie_NCC": _ange_ncc_wrapper,
        "Zhou_Pei": _zhou_pei_wrapper,
    }


def plot_metric(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if metric not in summary_df.columns or summary_df.empty:
        return
    plot_df = summary_df.copy()
    if "method" in plot_df.columns:
        plot_df = plot_df.sort_values("method")

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["method"], plot_df[metric], color="#1f77b4")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def auto_interpretation(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "Aucun resultat disponible."

    if "method" not in summary_df.columns:
        return "Resume incomplet: colonne 'method' absente."

    anon = summary_df[summary_df["method"] != "original"].copy()
    if anon.empty:
        return (
            "Seule la reference originale a ete evaluee. "
            "Ajoutez des fonctions d'anonymisation pour comparer l'utilite applicative."
        )

    metric = "auc_utility_ratio"
    if metric not in anon.columns:
        return "Impossible de comparer les methodes: auc_utility_ratio absent."

    anon = anon.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if anon.empty:
        return "Les methodes anonymisees ont produit des metriques invalides (NaN/inf)."

    best_idx = anon[metric].idxmax()
    best_method = str(anon.loc[best_idx, "method"])
    best_ratio = float(anon.loc[best_idx, metric])
    best_ap = float(anon.loc[best_idx, "ap_utility_ratio"]) if "ap_utility_ratio" in anon.columns else float("nan")
    best_p100 = (
        float(anon.loc[best_idx, "precision_at_100_utility_ratio"])
        if "precision_at_100_utility_ratio" in anon.columns
        else float("nan")
    )

    level = "faible"
    if best_ratio >= 0.9:
        level = "elevee"
    elif best_ratio >= 0.7:
        level = "moderee"

    return (
        f"La meilleure methode anonymisee est {best_method} "
        f"(AUC utility ratio={best_ratio:.4f}, AP ratio={best_ap:.4f}, "
        f"Precision@100 ratio={best_p100:.4f}). "
        f"Globalement, l'utilite applicative conservee est {level}."
    )


def parse_int_list(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    ]
    existing_cols = [col for col in metric_cols if col in df.columns]
    if not existing_cols:
        return pd.DataFrame(columns=["method"])
    return (
        df.groupby("method", as_index=False)[existing_cols]
        .mean(numeric_only=True)
        .sort_values("method")
        .reset_index(drop=True)
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment d'utilite applicative par prediction de liens (Adamic-Adar).")
    parser.add_argument("--input", type=str, default="data/citeseer.pairs", help="Fichier edge list (.pairs).")
    parser.add_argument("--k", type=int, default=10, help="Valeur k pour les anonymiseurs detectes.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Proportion d'aretes retirees pour le test.")
    parser.add_argument("--seeds", type=str, default="42,123,2024", help="Liste de seeds separees par des virgules.")
    parser.add_argument("--k-values", type=str, default="50,100,500", help="Liste des k pour Precision@k.")
    parser.add_argument("--output-csv", type=str, default="results/link_prediction/results_link_prediction.csv")
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default="results/link_prediction/results_link_prediction_summary.csv",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input graph not found: {input_path}")

    seeds = parse_int_list(args.seeds)
    k_values = parse_int_list(args.k_values)
    G_original = load_graph_from_pairs(input_path, nodetype=int)

    anonymization_functions = discover_project_anonymizers(k=int(args.k))
    if not anonymization_functions:
        print(
            "Aucune fonction d'anonymisation detectee automatiquement. "
            "TODO: fournir des wrappers personnalises dans discover_project_anonymizers()."
        )

    df_results = compare_link_prediction_utility(
        G_original=G_original,
        anonymization_functions=anonymization_functions,
        test_frac=float(args.test_frac),
        seeds=seeds,
        k_values=k_values,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    summary_df = build_summary(df_results)
    output_summary_csv = Path(args.output_summary_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary_csv, index=False)

    plot_dir = output_csv.parent

    plot_metric(summary_df, "auc", plot_dir / "auc_by_method.png", "AUC moyen par methode")
    plot_metric(
        summary_df,
        "average_precision",
        plot_dir / "average_precision_by_method.png",
        "Average Precision moyen par methode",
    )
    plot_metric(
        summary_df,
        "precision_at_100",
        plot_dir / "precision_at_100_by_method.png",
        "Precision@100 moyenne par methode",
    )

    generated_files = [
        output_csv,
        output_summary_csv,
        plot_dir / "auc_by_method.png",
        plot_dir / "average_precision_by_method.png",
        plot_dir / "precision_at_100_by_method.png",
    ]
    existing_files = [str(path) for path in generated_files if path.exists()]

    print("\n=== DataFrame complet des resultats ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df_results)

    print("\n=== Resume moyen par methode ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(summary_df)

    print("\n=== Fichiers generes ===")
    for file_path in existing_files:
        print(f"- {file_path}")

    print("\n=== Interpretation automatique ===")
    print(auto_interpretation(summary_df))


if __name__ == "__main__":
    main()
