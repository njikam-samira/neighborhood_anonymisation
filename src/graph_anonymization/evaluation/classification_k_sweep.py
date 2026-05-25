from __future__ import annotations

import argparse
import copy
import random
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, SAGEConv

try:
    from torch_geometric.nn import Node2Vec
except Exception:  # pragma: no cover - optional backend on some HPC environments
    Node2Vec = None

from graph_anonymization.anonymization.ange_modified import anonymize_ange_modified_ncc
from graph_anonymization.anonymization.ange_original import anonymize_ange_original
from graph_anonymization.anonymization.zhou_pei import anonymize_zhou_pei
from graph_anonymization.benchmarks.hikda_benchmark import anonymize_1hikda
from graph_anonymization.data.io import load_graph_from_pairs, prepare_simple_graph


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def parse_methods_list(text: str) -> List[str]:
    aliases = {
        "original": "original",
        "ange_original": "Ange_Original",
        "ange_modifie_ncc": "Ange_Modifie_NCC",
        "ange_modifie": "Ange_Modifie_NCC",
        "zhou_pei": "Zhou_Pei",
        "1hikda": "1HiKDA",
        "1hikda_anonymity": "1HiKDA",
    }
    methods: List[str] = []
    for token in text.split(","):
        key = token.strip()
        if not key:
            continue
        canonical = aliases.get(key.lower(), key)
        methods.append(canonical)
    return methods


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(dataset: str) -> str:
    mapping = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }
    key = str(dataset).strip().lower()
    return mapping.get(key, dataset)


def model_file_slug(model: str) -> str:
    mapping = {
        "node2vec_logreg": "node2vec",
        "graphsage": "graphsage",
        "gat": "gat",
    }
    return mapping.get(str(model).lower(), str(model).lower())


def load_planetoid_data(dataset: str, root_dir: Path) -> Data:
    name = normalize_dataset_name(dataset)
    ds = Planetoid(root=str(root_dir), name=name)
    data = ds[0]
    data.x = data.x.float()
    data.y = data.y.long()
    return data


def load_base_graph(pair_path: Path, fallback_data: Data) -> nx.Graph:
    if pair_path.exists():
        graph = prepare_simple_graph(load_graph_from_pairs(pair_path, node_type=int))
    else:
        graph = nx.Graph()
        edge_index = fallback_data.edge_index.detach().cpu().numpy()
        for src, dst in edge_index.T:
            graph.add_edge(int(src), int(dst))
        graph = prepare_simple_graph(graph)

    graph.add_nodes_from(range(int(fallback_data.num_nodes)))
    return prepare_simple_graph(graph)


def sanitize_graph_nodes(graph: nx.Graph, num_nodes: int) -> nx.Graph:
    clean = nx.Graph()
    clean.add_nodes_from(range(int(num_nodes)))
    for u, v in graph.edges():
        try:
            uu = int(u)
            vv = int(v)
        except (TypeError, ValueError):
            continue
        if uu == vv:
            continue
        if 0 <= uu < num_nodes and 0 <= vv < num_nodes:
            clean.add_edge(uu, vv)
    return prepare_simple_graph(clean)


def graph_to_edge_index(graph: nx.Graph, num_nodes: int, device: torch.device) -> torch.Tensor:
    edges: List[tuple[int, int]] = []
    for u, v in graph.edges():
        uu = int(u)
        vv = int(v)
        if uu == vv:
            continue
        if not (0 <= uu < num_nodes and 0 <= vv < num_nodes):
            continue
        edges.append((uu, vv))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    arr = np.asarray(edges, dtype=np.int64)
    rev = arr[:, [1, 0]]
    all_edges = np.vstack([arr, rev])
    all_edges = np.unique(all_edges, axis=0)
    edge_index = torch.from_numpy(all_edges.T).long().to(device)
    return edge_index


def make_eval_data(base_data: Data, graph: nx.Graph, device: torch.device) -> Data:
    num_nodes = int(base_data.num_nodes)
    edge_index = graph_to_edge_index(graph=graph, num_nodes=num_nodes, device=device)

    data = Data(
        x=base_data.x.detach().clone().to(device),
        y=base_data.y.detach().clone().to(device),
        edge_index=edge_index,
    )

    if getattr(base_data, "train_mask", None) is not None:
        data.train_mask = base_data.train_mask.detach().clone().to(device)
    if getattr(base_data, "val_mask", None) is not None:
        data.val_mask = base_data.val_mask.detach().clone().to(device)
    if getattr(base_data, "test_mask", None) is not None:
        data.test_mask = base_data.test_mask.detach().clone().to(device)

    return data


def _safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or np.isclose(den, 0.0):
        return float("nan")
    if num is None or not np.isfinite(num):
        return float("nan")
    return float(num / den)


def _safe_sub(left: float, right: float) -> float:
    if left is None or right is None:
        return float("nan")
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(left - right)


def metric_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def _mask_or_full(mask: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=int(heads),
            dropout=float(dropout),
        )
        self.conv2 = GATConv(
            hidden_channels * int(heads),
            num_classes,
            heads=1,
            concat=False,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_and_predict_gnn(
    model: nn.Module,
    data: Data,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping: bool,
    patience: int = 25,
) -> np.ndarray:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    train_mask = _mask_or_full(getattr(data, "train_mask", None), data.num_nodes, data.x.device)
    val_mask = _mask_or_full(getattr(data, "val_mask", None), data.num_nodes, data.x.device)

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            logits_val = model(data.x, data.edge_index)
            val_loss = float(F.cross_entropy(logits_val[val_mask], data.y[val_mask]).item())
            model.train()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if early_stopping and no_improve >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1).detach().cpu().numpy()
    return pred


def evaluate_graphsage(
    data: Data,
    seed: int,
    hidden_channels: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    early_stopping: bool,
) -> Dict[str, float]:
    set_global_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = GraphSAGEClassifier(
        in_channels=int(data.x.shape[1]),
        hidden_channels=int(hidden_channels),
        num_classes=num_classes,
        dropout=float(dropout),
    ).to(data.x.device)

    y_pred = train_and_predict_gnn(
        model=model,
        data=data,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        early_stopping=bool(early_stopping),
    )

    test_mask = _mask_or_full(getattr(data, "test_mask", None), data.num_nodes, data.x.device)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)
    y_true = data.y.detach().cpu().numpy()[test_idx]
    y_pred_test = y_pred[test_idx]
    return metric_scores(y_true, y_pred_test)


def evaluate_gat(
    data: Data,
    seed: int,
    hidden_channels: int,
    heads: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    early_stopping: bool,
) -> Dict[str, float]:
    set_global_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = GATClassifier(
        in_channels=int(data.x.shape[1]),
        hidden_channels=int(hidden_channels),
        num_classes=num_classes,
        heads=int(heads),
        dropout=float(dropout),
    ).to(data.x.device)

    y_pred = train_and_predict_gnn(
        model=model,
        data=data,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        early_stopping=bool(early_stopping),
    )

    test_mask = _mask_or_full(getattr(data, "test_mask", None), data.num_nodes, data.x.device)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)
    y_true = data.y.detach().cpu().numpy()[test_idx]
    y_pred_test = y_pred[test_idx]
    return metric_scores(y_true, y_pred_test)


def evaluate_node2vec_logreg(
    data: Data,
    seed: int,
    embedding_dim: int,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
    p: float,
    q: float,
    epochs: int,
) -> Dict[str, float]:
    if Node2Vec is None:
        raise RuntimeError(
            "torch_geometric.nn.Node2Vec indisponible (backend manquant: pyg-lib ou torch-cluster)."
        )

    set_global_seed(seed)
    device = data.x.device

    model = Node2Vec(
        data.edge_index,
        embedding_dim=int(embedding_dim),
        walk_length=int(walk_length),
        context_size=int(context_size),
        walks_per_node=int(walks_per_node),
        p=float(p),
        q=float(q),
        num_negative_samples=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=min(256, int(data.num_nodes)), shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(int(epochs)):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model().detach().cpu().numpy()

    train_mask = _mask_or_full(getattr(data, "train_mask", None), data.num_nodes, data.x.device)
    test_mask = _mask_or_full(getattr(data, "test_mask", None), data.num_nodes, data.x.device)

    train_idx = train_mask.detach().cpu().numpy().astype(bool)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)

    x_train = z[train_idx]
    x_test = z[test_idx]
    y_train = data.y.detach().cpu().numpy()[train_idx]
    y_test = data.y.detach().cpu().numpy()[test_idx]

    clf = LogisticRegression(max_iter=5000, random_state=int(seed), multi_class="auto")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return metric_scores(y_test, y_pred)


def build_anonymizers_for_k(k: int, hikda_max_nodes: int = 3000) -> Dict[str, Callable[[nx.Graph, int], nx.Graph]]:
    def ange_original_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_ange_original(graph, k=_k, seed=seed)

    def ange_modified_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_ange_modified_ncc(graph, k=_k, seed=seed)

    def zhou_pei_wrapper(graph: nx.Graph, seed: int, _k: int = k) -> nx.Graph:
        return anonymize_zhou_pei(graph, k=_k, seed=seed)

    def hikda_wrapper(graph: nx.Graph, seed: int, _k: int = k, _max_nodes: int = hikda_max_nodes) -> nx.Graph:
        if graph.number_of_nodes() > _max_nodes:
            raise RuntimeError(
                f"1HiKDA ignore sur graphes > {_max_nodes} noeuds (actuel={graph.number_of_nodes()})."
            )
        return anonymize_1hikda(graph, k=_k, seed=seed)

    return {
        "Ange_Original": ange_original_wrapper,
        "Ange_Modifie_NCC": ange_modified_wrapper,
        "Zhou_Pei": zhou_pei_wrapper,
        "1HiKDA": hikda_wrapper,
    }


def add_global_utility_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratio_cols = [
        "accuracy_utility_ratio",
        "macro_f1_utility_ratio",
        "micro_f1_utility_ratio",
    ]
    for col in ratio_cols:
        if col not in out.columns:
            out[col] = np.nan
    out["global_utility_score"] = out[ratio_cols].mean(axis=1, skipna=True)
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "accuracy",
        "macro_f1",
        "micro_f1",
        "accuracy_loss",
        "macro_f1_loss",
        "micro_f1_loss",
        "accuracy_utility_ratio",
        "macro_f1_utility_ratio",
        "micro_f1_utility_ratio",
        "global_utility_score",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    summary = (
        df.groupby(["dataset", "model", "k", "method"], as_index=False)[cols]
        .mean(numeric_only=True)
        .sort_values(["k", "method"])
        .reset_index(drop=True)
    )
    return summary


def plot_metric_by_k(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for method in sorted(summary_df["method"].unique()):
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


def render_table_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    table_df = df.copy()
    for col in table_df.columns:
        table_df[col] = table_df[col].map(
            lambda x: "NaN" if pd.isna(x) else (f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
        )

    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    report_pdf: Path,
    plot_paths: Dict[str, Path],
    dataset: str,
    model: str,
    k_values: List[int],
) -> None:
    report_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_pdf) as pdf:
        render_text_page(
            pdf,
            f"{dataset} - {model} classification k-sweep",
            [
                f"Dataset: {dataset}",
                f"Modele: {model}",
                f"k values: {', '.join(str(k) for k in k_values)}",
                "Seeds: 42, 123, 2024",
                "Methodes: original, Ange_Original, Ange_Modifie_NCC, Zhou_Pei, 1HiKDA",
            ],
        )

        cols_summary = [
            c
            for c in [
                "dataset",
                "model",
                "k",
                "method",
                "accuracy",
                "macro_f1",
                "micro_f1",
                "accuracy_utility_ratio",
                "macro_f1_utility_ratio",
                "micro_f1_utility_ratio",
                "global_utility_score",
            ]
            if c in summary_df.columns
        ]
        render_table_page(pdf, "Resume moyen", summary_df[cols_summary])

        cols_detailed = [
            c
            for c in [
                "dataset",
                "model",
                "method",
                "k",
                "seed",
                "accuracy",
                "macro_f1",
                "micro_f1",
                "global_utility_score",
                "method_error",
            ]
            if c in detailed_df.columns
        ]
        render_table_page(pdf, "Resultats detailles", detailed_df[cols_detailed])

        for title, key in [
            ("Accuracy by k and method", "accuracy"),
            ("Macro-F1 by k and method", "macro_f1"),
            ("Micro-F1 by k and method", "micro_f1"),
            ("Global utility score by k and method", "global_utility_score"),
        ]:
            path = plot_paths[key]
            if not path.exists():
                continue
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.imshow(image)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run_k_sweep(args: argparse.Namespace, model: str) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = str(args.dataset)
    data_root = Path(args.data_root)
    planetoid_root = Path(args.planetoid_root)
    pair_path = data_root / f"{dataset.lower()}.pairs"

    try:
        data_base = load_planetoid_data(dataset=dataset, root_dir=planetoid_root)
    except Exception as exc:
        raise RuntimeError(
            "Chargement Planetoid impossible. Verifiez l'acces internet du noeud "
            "ou prechargez le dataset dans --planetoid-root."
        ) from exc
    base_graph = load_base_graph(pair_path=pair_path, fallback_data=data_base)

    method_order = parse_methods_list(args.methods)
    available = build_anonymizers_for_k(k=2, hikda_max_nodes=int(args.hikda_max_nodes))
    method_order = [m for m in method_order if m == "original" or m in available]
    if not method_order:
        raise ValueError("Aucune methode valide dans --methods.")

    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: List[Dict[str, Any]] = []
    for k in k_values:
        anonymizers = build_anonymizers_for_k(k=int(k), hikda_max_nodes=int(args.hikda_max_nodes))

        for seed in seeds:
            set_global_seed(int(seed))

            seed_rows: Dict[str, Dict[str, Any]] = {}

            for method in method_order:
                method_error = ""
                metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "micro_f1": float("nan")}

                try:
                    if method == "original":
                        eval_graph = base_graph.copy()
                    else:
                        eval_graph = anonymizers[method](base_graph.copy(), seed=int(seed))

                    eval_graph = sanitize_graph_nodes(eval_graph, num_nodes=int(data_base.num_nodes))
                    data_eval = make_eval_data(data_base, eval_graph, device=device)

                    if model == "node2vec_logreg":
                        metrics = evaluate_node2vec_logreg(
                            data_eval,
                            seed=int(seed),
                            embedding_dim=int(args.embedding_dim),
                            walk_length=int(args.walk_length),
                            context_size=int(args.context_size),
                            walks_per_node=int(args.walks_per_node),
                            p=float(args.p),
                            q=float(args.q),
                            epochs=int(args.epochs),
                        )
                    elif model == "graphsage":
                        metrics = evaluate_graphsage(
                            data_eval,
                            seed=int(seed),
                            hidden_channels=int(args.hidden_channels),
                            dropout=float(args.dropout),
                            learning_rate=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            epochs=int(args.epochs),
                            early_stopping=bool(args.early_stopping),
                        )
                    elif model == "gat":
                        metrics = evaluate_gat(
                            data_eval,
                            seed=int(seed),
                            hidden_channels=int(args.hidden_channels),
                            heads=int(args.heads),
                            dropout=float(args.dropout),
                            learning_rate=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            epochs=int(args.epochs),
                            early_stopping=bool(args.early_stopping),
                        )
                    else:
                        raise ValueError(f"Modele non supporte: {model}")

                    num_edges_anon = int(eval_graph.number_of_edges())

                except Exception as exc:
                    method_error = f"{type(exc).__name__}: {exc}"
                    num_edges_anon = int(base_graph.number_of_edges()) if method == "original" else -1

                row = {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "k": int(k),
                    "seed": int(seed),
                    "num_nodes": int(base_graph.number_of_nodes()),
                    "num_edges_original": int(base_graph.number_of_edges()),
                    "num_edges_anonymized": int(num_edges_anon),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "macro_f1": float(metrics.get("macro_f1", float("nan"))),
                    "micro_f1": float(metrics.get("micro_f1", float("nan"))),
                    "method_error": method_error,
                }
                seed_rows[method] = row

            original = seed_rows.get("original")
            if original is None:
                raise RuntimeError("La methode 'original' doit etre presente dans --methods.")

            base_acc = float(original["accuracy"])
            base_macro = float(original["macro_f1"])
            base_micro = float(original["micro_f1"])

            for method in method_order:
                row = seed_rows[method]
                row["accuracy_loss"] = _safe_sub(base_acc, float(row["accuracy"]))
                row["macro_f1_loss"] = _safe_sub(base_macro, float(row["macro_f1"]))
                row["micro_f1_loss"] = _safe_sub(base_micro, float(row["micro_f1"]))
                row["accuracy_utility_ratio"] = _safe_div(float(row["accuracy"]), base_acc)
                row["macro_f1_utility_ratio"] = _safe_div(float(row["macro_f1"]), base_macro)
                row["micro_f1_utility_ratio"] = _safe_div(float(row["micro_f1"]), base_micro)
                rows.append(row)

    detailed_df = pd.DataFrame(rows)
    detailed_df = add_global_utility_score(detailed_df)

    col_order = [
        "dataset",
        "model",
        "method",
        "k",
        "seed",
        "num_nodes",
        "num_edges_original",
        "num_edges_anonymized",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "accuracy_loss",
        "macro_f1_loss",
        "micro_f1_loss",
        "accuracy_utility_ratio",
        "macro_f1_utility_ratio",
        "micro_f1_utility_ratio",
        "global_utility_score",
        "method_error",
    ]
    detailed_df = detailed_df[[c for c in col_order if c in detailed_df.columns]]

    summary_df = build_summary(detailed_df)

    slug = dataset.lower().replace(" ", "_")
    model_slug = model_file_slug(model)
    detailed_csv = output_dir / f"results_{slug}_{model_slug}_classification.csv"
    summary_csv = output_dir / f"results_{slug}_{model_slug}_classification_summary.csv"
    report_pdf = output_dir / f"results_{slug}_{model_slug}_classification_report.pdf"

    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_paths = {
        "accuracy": plots_dir / "accuracy_by_k_and_method.png",
        "macro_f1": plots_dir / "macro_f1_by_k_and_method.png",
        "micro_f1": plots_dir / "micro_f1_by_k_and_method.png",
        "global_utility_score": plots_dir / "utility_score_by_k_and_method.png",
    }

    plot_metric_by_k(summary_df, "accuracy", plot_paths["accuracy"], "Accuracy by k and method")
    plot_metric_by_k(summary_df, "macro_f1", plot_paths["macro_f1"], "Macro-F1 by k and method")
    plot_metric_by_k(summary_df, "micro_f1", plot_paths["micro_f1"], "Micro-F1 by k and method")
    plot_metric_by_k(
        summary_df,
        "global_utility_score",
        plot_paths["global_utility_score"],
        "Global utility score by k and method",
    )

    if args.save_report:
        generate_report(
            detailed_df=detailed_df,
            summary_df=summary_df,
            report_pdf=report_pdf,
            plot_paths=plot_paths,
            dataset=dataset,
            model=model,
            k_values=k_values,
        )

    ranking = summary_df[summary_df["method"] != "original"][["k", "method", "global_utility_score"]].copy()
    ranking = ranking.sort_values(["k", "global_utility_score"], ascending=[True, False])
    ranking["rank_within_k"] = ranking.groupby("k")["global_utility_score"].rank(
        method="first", ascending=False
    )

    print("\n=== Tableau detaille ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(detailed_df)

    print("\n=== Resume moyen ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(summary_df)

    print("\n=== Classement par k ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(ranking)

    print("\n=== Fichiers generes ===")
    print(f"- {detailed_csv}")
    print(f"- {summary_csv}")
    if args.save_report:
        print(f"- {report_pdf}")
    for path in plot_paths.values():
        print(f"- {path}")


def build_parser(default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{default_model} classification k-sweep")

    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--k-values", default="2,5,10,50")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--methods", default="original,Ange_Original,Ange_Modifie_NCC,Zhou_Pei,1HiKDA")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plots-dir", default="")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--planetoid-root", default="data/planetoid")
    parser.add_argument("--hikda-max-nodes", type=int, default=3000)
    parser.add_argument("--save-report", action="store_true")

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping", action="store_true")

    parser.set_defaults(model=default_model)
    return parser


def main(default_model: str) -> None:
    parser = build_parser(default_model=default_model)
    args = parser.parse_args()
    run_k_sweep(args=args, model=str(args.model))


if __name__ == "__main__":
    main(default_model="node2vec_logreg")
