from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ..benchmarks.hikda_benchmark import anonymize_1hikda
from .link_prediction import compare_link_prediction_utility, prepare_graph
from .link_prediction_report import generate_pdf_report
from .run_link_prediction_experiment import auto_interpretation, build_summary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "results" / "link_prediction_1hikda"


def load_graph_from_pairs(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            graph.add_edge(int(parts[0]), int(parts[1]))
    return prepare_graph(graph)


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def load_1hikda_anonymizer(k: int) -> Dict[str, Callable[[nx.Graph, int], nx.Graph]]:
    def anonymize_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_1hikda(graph, k=_k, seed=seed)

    return {"1HiKDA": anonymize_wrapper}


def plot_metric(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if metric not in summary_df.columns or summary_df.empty:
        return

    plot_df = summary_df.sort_values("method")
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["method"], plot_df[metric], color="#2c7fb8")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prediction de liens Adamic-Adar pour 1HiKDA.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "citeseer.pairs"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--k-values", default="50,100,500")
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = Path(args.input)
    if not graph_path.exists():
        raise FileNotFoundError(f"Input graph not found: {graph_path}")

    G_original = load_graph_from_pairs(graph_path)
    anonymization_functions = load_1hikda_anonymizer(k=int(args.k))

    df_results = compare_link_prediction_utility(
        G_original=G_original,
        anonymization_functions=anonymization_functions,
        test_frac=float(args.test_frac),
        seeds=parse_int_list(args.seeds),
        k_values=parse_int_list(args.k_values),
    )

    detailed_csv = output_dir / "results_link_prediction_1hikda.csv"
    summary_csv = output_dir / "results_link_prediction_1hikda_summary.csv"
    report_pdf = output_dir / "results_link_prediction_1hikda_report.pdf"

    df_results.to_csv(detailed_csv, index=False)
    summary_df = build_summary(df_results)
    summary_df.to_csv(summary_csv, index=False)

    plot_metric(summary_df, "auc", output_dir / "auc_by_method.png", "AUC moyen par methode")
    plot_metric(
        summary_df,
        "average_precision",
        output_dir / "average_precision_by_method.png",
        "Average Precision moyenne par methode",
    )
    plot_metric(
        summary_df,
        "precision_at_100",
        output_dir / "precision_at_100_by_method.png",
        "Precision@100 moyenne par methode",
    )

    generate_pdf_report(detailed_csv, summary_csv, report_pdf, image_dir=output_dir)

    generated = [
        detailed_csv,
        summary_csv,
        report_pdf,
        output_dir / "auc_by_method.png",
        output_dir / "average_precision_by_method.png",
        output_dir / "precision_at_100_by_method.png",
    ]

    print("\n=== Resultats complets 1HiKDA ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df_results)

    print("\n=== Resume moyen 1HiKDA ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(summary_df)

    print("\n=== Fichiers generes ===")
    for path in generated:
        if path.exists():
            print(f"- {path}")

    print("\n=== Interpretation automatique ===")
    print(auto_interpretation(summary_df))


if __name__ == "__main__":
    main()
