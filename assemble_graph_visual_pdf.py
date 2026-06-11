from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf")

DATASETS: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("cora", "Cora", ("cora",)),
    ("citeseer", "CiteSeer", ("citeseer",)),
    ("polblog", "PolBlog", ("polblog", "polblogs")),
)

METHODS: Sequence[Tuple[str, str, str, Sequence[str]]] = (
    ("original", "Original", "original", ("original",)),
    ("ange_original", "Ange Original", "ange_original", ("ange", "original")),
    ("ange_modifie_ncc", "Ange Modifie NCC", "ange_modifie_ncc", ("ange", "modifie", "ncc")),
    ("zhou_pei", "Zhou-Pei", "zhou_pei", ("zhou", "pei")),
    ("1hikda", "1HiKDA", "1hikda", ("1hikda", "1hikda")),
)


def normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in {"_", "-", "/", "\\"})


def list_candidate_images(search_dirs: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = []
    for base in search_dirs:
        if not base.exists() or not base.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidates.extend(base.rglob(f"*{ext}"))
    return [path for path in candidates if path.is_file()]


def score_candidate(
    path: Path,
    dataset_tokens: Sequence[str],
    method_tokens: Sequence[str],
    expected_stem: str,
) -> int:
    norm_path = normalize(str(path))
    stem = normalize(path.stem)

    score = 0

    if "comparison_grid" in stem:
        score -= 1000

    if stem == normalize(expected_stem):
        score += 1000

    if any(token in norm_path for token in dataset_tokens):
        score += 200
    if all(token in norm_path for token in method_tokens):
        score += 220

    if "plots_graph_views" in norm_path:
        score += 300
    if "plots_graph_pdf" in norm_path:
        score += 80

    if path.suffix.lower() == ".png":
        score += 50
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        score += 30
    elif path.suffix.lower() == ".pdf":
        score += 10

    score -= len(str(path)) // 25
    return score


def find_best_image(
    all_images: Sequence[Path],
    dataset_tokens: Sequence[str],
    method_tokens: Sequence[str],
    expected_stem: str,
) -> Optional[Path]:
    ranked: List[Tuple[int, Path]] = []
    for path in all_images:
        score = score_candidate(path, dataset_tokens, method_tokens, expected_stem)
        if score > 0:
            ranked.append((score, path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return ranked[0][1]


def render_cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.5,
        0.78,
        "Comparaison visuelle des graphes originaux et anonymises",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    description = (
        "Ce rapport regroupe les visualisations deja generees des graphes originaux "
        "et anonymises pour Cora, CiteSeer et PolBlog."
    )
    fig.text(0.5, 0.65, description, ha="center", va="center", fontsize=12, wrap=True)
    fig.text(0.5, 0.12, "Page 1", ha="center", va="center", fontsize=10)
    plt.axis("off")
    pdf.savefig(fig, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _draw_missing(ax: plt.Axes, method_title: str, message: str = "Image manquante") -> None:
    ax.set_title(method_title, fontsize=10)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")


def render_dataset_page(
    pdf: PdfPages,
    dataset_label: str,
    method_to_image: Dict[str, Optional[Path]],
) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 5, figsize=(16.54, 11.69), facecolor="white")
    fig.suptitle(dataset_label, fontsize=18, weight="bold")

    errors: Dict[str, str] = {}

    for idx, (method_key, method_title, _, _) in enumerate(METHODS):
        ax = axes[idx]
        ax.set_facecolor("white")
        path = method_to_image.get(method_key)

        if path is None:
            _draw_missing(ax, method_title, "Image manquante")
            errors[method_key] = "Image manquante"
            continue

        if path.suffix.lower() == ".pdf":
            _draw_missing(ax, method_title, "Image PDF non rasterisee")
            errors[method_key] = "Image PDF non rasterisee"
            continue

        try:
            image = plt.imread(path)
            ax.imshow(image)
            ax.set_title(method_title, fontsize=10)
            ax.axis("off")
            errors[method_key] = ""
        except Exception as exc:
            _draw_missing(ax, method_title, "Image illisible")
            errors[method_key] = f"Image illisible: {type(exc).__name__}: {exc}"

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble existing graph images into one comparison PDF.")
    parser.add_argument(
        "--search-dirs",
        nargs="*",
        default=["plots_graph_views", "plots_graph_pdf", "plots", "figures", "resultat", "results"],
        help="Directories to search for existing images.",
    )
    parser.add_argument(
        "--output-pdf",
        default="plots_graph_pdf/visual_comparison_original_vs_anonymized.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="plots_graph_pdf/pdf_image_manifest.csv",
        help="Output manifest CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    search_dirs = [project_root / rel for rel in args.search_dirs]
    all_images = list_candidate_images(search_dirs)

    output_pdf = project_root / args.output_pdf
    manifest_csv = project_root / args.manifest_csv
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    used_images: List[Path] = []
    missing_images: List[str] = []

    with PdfPages(output_pdf) as pdf:
        render_cover_page(pdf)

        for dataset_idx, (dataset_key, dataset_label, dataset_tokens_raw) in enumerate(DATASETS, start=2):
            dataset_tokens = tuple(normalize(token) for token in dataset_tokens_raw)
            method_to_image: Dict[str, Optional[Path]] = {}

            for method_key, _, method_suffix, method_tokens_raw in METHODS:
                method_tokens = tuple(normalize(token) for token in method_tokens_raw)
                expected_stem = f"{dataset_key}_{method_suffix}"
                image_path = find_best_image(
                    all_images=all_images,
                    dataset_tokens=dataset_tokens,
                    method_tokens=method_tokens,
                    expected_stem=expected_stem,
                )
                method_to_image[method_key] = image_path

            page_errors = render_dataset_page(pdf, dataset_label=dataset_label, method_to_image=method_to_image)

            for method_key, method_title, method_suffix, _ in METHODS:
                image_path = method_to_image.get(method_key)
                error_message = page_errors.get(method_key, "")
                image_found = image_path is not None and error_message == ""

                if image_found and image_path is not None:
                    used_images.append(image_path)
                else:
                    missing_images.append(f"{dataset_key}/{method_suffix}")

                manifest_rows.append(
                    {
                        "dataset": dataset_label,
                        "method": method_title,
                        "image_found": "True" if image_found else "False",
                        "image_path": str(image_path) if image_path is not None else "",
                        "page": str(dataset_idx),
                        "error_message": error_message,
                    }
                )

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "image_found",
                "image_path",
                "page",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n=== PDF final ===")
    print(output_pdf)

    print("\n=== CSV manifest ===")
    print(manifest_csv)

    print("\n=== Images utilisees ===")
    for path in sorted(set(used_images)):
        print(path)

    print("\n=== Images manquantes ===")
    if missing_images:
        for item in sorted(set(missing_images)):
            print(item)
    else:
        print("Aucune")


if __name__ == "__main__":
    main()
