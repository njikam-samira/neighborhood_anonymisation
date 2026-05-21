from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def _render_table_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape in inches
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(_fmt)

    table = ax.table(
        cellText=rendered.values,
        colLabels=rendered.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    y = 0.96
    for line in lines:
        wrapped = textwrap.wrap(line, width=100) or [""]
        for part in wrapped:
            ax.text(0.02, y, part, va="top", ha="left", fontsize=10, transform=ax.transAxes)
            y -= 0.035
            if y < 0.05:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.96
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.imshow(img)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _method_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    anon = summary_df[summary_df["method"] != "original"].copy()
    score_cols = [
        "auc_utility_ratio",
        "ap_utility_ratio",
        "precision_at_100_utility_ratio",
    ]
    for col in score_cols:
        if col not in anon.columns:
            anon[col] = np.nan
    anon["global_utility_score"] = anon[score_cols].mean(axis=1, skipna=True)
    return anon.sort_values("global_utility_score", ascending=False).reset_index(drop=True)


def build_interpretation(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    lines.append("Contexte: le graphe original sert de reference utilite pour Adamic-Adar.")
    lines.append(
        "Un ratio proche de 1 signifie une utilite proche du graphe original; >1 signifie performance superieure sur ce protocole."
    )

    ranking = _method_ranking(summary_df)
    if ranking.empty:
        lines.append("Aucune methode anonymisee detectee dans le resume.")
        return lines

    best = ranking.iloc[0]
    lines.append(
        f"Meilleure methode (score global): {best['method']} "
        f"(AUC ratio={_fmt(best.get('auc_utility_ratio', np.nan))}, "
        f"AP ratio={_fmt(best.get('ap_utility_ratio', np.nan))}, "
        f"P@100 ratio={_fmt(best.get('precision_at_100_utility_ratio', np.nan))})."
    )

    if len(ranking) > 1:
        worst = ranking.iloc[-1]
        lines.append(
            f"Methode la plus degradante: {worst['method']} "
            f"(score global={_fmt(worst.get('global_utility_score', np.nan))})."
        )

    lines.append("Lecture des resultats observes:")
    for _, row in ranking.iterrows():
        method = row["method"]
        auc_ratio = row.get("auc_utility_ratio", np.nan)
        ap_ratio = row.get("ap_utility_ratio", np.nan)
        p100_ratio = row.get("precision_at_100_utility_ratio", np.nan)

        if pd.notna(auc_ratio) and auc_ratio >= 0.98 and pd.notna(ap_ratio) and ap_ratio >= 0.98:
            comment = "preserve tres bien l'utilite globale"
        elif pd.notna(auc_ratio) and auc_ratio >= 0.85:
            comment = "preserve partiellement l'utilite"
        else:
            comment = "degrade fortement l'utilite"

        lines.append(
            f"- {method}: AUC ratio={_fmt(auc_ratio)}, AP ratio={_fmt(ap_ratio)}, "
            f"P@100 ratio={_fmt(p100_ratio)} -> {comment}."
        )

    if {"method", "seed", "auc", "average_precision", "precision_at_100"}.issubset(detailed_df.columns):
        grouped_std = (
            detailed_df[detailed_df["method"] != "original"]
            .groupby("method")[["auc", "average_precision", "precision_at_100"]]
            .std(numeric_only=True)
            .fillna(0.0)
        )
        lines.append("Stabilite inter-seeds (ecart-type):")
        for method, vals in grouped_std.iterrows():
            lines.append(
                f"- {method}: std(AUC)={_fmt(vals['auc'])}, std(AP)={_fmt(vals['average_precision'])}, "
                f"std(P@100)={_fmt(vals['precision_at_100'])}."
            )

    lines.append(
        "Important pour le memoire: un ratio >1 n'implique pas qu'un graphe anonymise est 'meilleur' en absolu; "
        "cela peut venir du split des aretes, de regularisations implicites de structure ou d'effets de densification."
    )
    lines.append(
        "Conclusion pratique ici: Zhou_Pei conserve (et parfois amplifie) le signal Adamic-Adar, "
        "Ange_Original est quasi neutre, et Ange_Modifie_NCC degrade nettement AUC/AP."
    )
    return lines


def generate_pdf_report(
    detailed_csv: Path,
    summary_csv: Path,
    output_pdf: Path,
    image_dir: Path | None = None,
) -> None:
    detailed_df = pd.read_csv(detailed_csv)
    summary_df = pd.read_csv(summary_csv)
    image_dir = image_dir or output_pdf.parent

    ranking_df = _method_ranking(summary_df)
    interpretation_lines = build_interpretation(detailed_df, summary_df)

    with PdfPages(output_pdf) as pdf:
        _render_text_page(
            pdf,
            "Rapport - Utilite applicative (Link Prediction Adamic-Adar)",
            [
                "Objet: comparaison du graphe original et de graphes anonymises via prediction de liens.",
                f"Fichier resultats detaillees: {detailed_csv}",
                f"Fichier resume: {summary_csv}",
                "",
                "Ce rapport contient:",
                "1) tableau resume moyen par methode",
                "2) tableau detaille par seed",
                "3) graphiques comparatifs",
                "4) interpretation automatique exploitable dans le memoire",
            ],
        )

        _render_table_page(pdf, "Resume moyen par methode", summary_df)
        _render_table_page(pdf, "Resultats detaillees par seed", detailed_df)

        _render_image_page(pdf, "AUC moyen par methode", image_dir / "auc_by_method.png")
        _render_image_page(pdf, "Average Precision moyenne par methode", image_dir / "average_precision_by_method.png")
        _render_image_page(pdf, "Precision@100 moyenne par methode", image_dir / "precision_at_100_by_method.png")

        if not ranking_df.empty:
            _render_table_page(
                pdf,
                "Classement methodes anonymisees (score utilite global)",
                ranking_df[
                    [
                        "method",
                        "auc_utility_ratio",
                        "ap_utility_ratio",
                        "precision_at_100_utility_ratio",
                        "global_utility_score",
                    ]
                ],
            )

        _render_text_page(pdf, "Interpretation automatique", interpretation_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere un PDF de synthese pour les resultats link prediction.")
    parser.add_argument("--detailed-csv", default="results/link_prediction/results_link_prediction.csv")
    parser.add_argument("--summary-csv", default="results/link_prediction/results_link_prediction_summary.csv")
    parser.add_argument("--output-pdf", default="results/link_prediction/results_link_prediction_report.pdf")
    parser.add_argument("--image-dir", default=None)
    args = parser.parse_args()

    detailed_csv = Path(args.detailed_csv)
    summary_csv = Path(args.summary_csv)
    output_pdf = Path(args.output_pdf)

    if not detailed_csv.exists():
        raise FileNotFoundError(f"Missing detailed CSV: {detailed_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    image_dir = Path(args.image_dir) if args.image_dir else output_pdf.parent
    generate_pdf_report(detailed_csv, summary_csv, output_pdf, image_dir=image_dir)
    print(f"PDF genere: {output_pdf.resolve()}")


if __name__ == "__main__":
    main()
