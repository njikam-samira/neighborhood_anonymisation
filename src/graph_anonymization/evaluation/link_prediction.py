from __future__ import annotations

import inspect
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


Edge = Tuple[Any, Any]


def prepare_graph(G: nx.Graph) -> nx.Graph:
    """
    Convertit le graphe en graphe simple non oriente et supprime les self-loops.
    """
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def normalize_edge(u: Any, v: Any) -> Edge:
    """
    Retourne une arete non orientee sous forme ordonnee.
    """
    try:
        return tuple(sorted((u, v)))
    except TypeError:
        return tuple(sorted((u, v), key=lambda x: str(x)))


def _protected_spanning_edges(G: nx.Graph) -> set[Edge]:
    """
    Protege un arbre couvrant par composante connexe.
    """
    protected: set[Edge] = set()
    for component_nodes in nx.connected_components(G):
        sub = G.subgraph(component_nodes)
        if sub.number_of_nodes() <= 1:
            continue
        root = next(iter(sub.nodes()))
        for u, v in nx.bfs_edges(sub, source=root):
            protected.add(normalize_edge(u, v))
    return protected


def train_test_edge_split(
    G: nx.Graph,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[nx.Graph, List[Edge]]:
    """
    Retourne G_train et les aretes positives de test en preservant un arbre couvrant
    par composante connexe autant que possible.
    """
    if not (0.0 <= test_frac < 1.0):
        raise ValueError("test_frac must be in [0, 1).")

    G_clean = prepare_graph(G)
    all_edges = sorted({normalize_edge(u, v) for u, v in G_clean.edges()})
    if not all_edges:
        return G_clean.copy(), []

    protected_edges = _protected_spanning_edges(G_clean)
    removable_edges = [edge for edge in all_edges if edge not in protected_edges]

    target_test_size = int(round(float(test_frac) * len(all_edges)))
    target_test_size = max(0, min(target_test_size, len(removable_edges)))

    rng = random.Random(seed)
    rng.shuffle(removable_edges)
    test_pos_edges = sorted(removable_edges[:target_test_size])

    G_train = G_clean.copy()
    G_train.remove_edges_from(test_pos_edges)
    return G_train, test_pos_edges


def sample_negative_edges(
    G_original: nx.Graph,
    n_samples: int,
    seed: int = 42,
) -> List[Edge]:
    """
    Echantillonne n_samples paires de noeuds non connectees dans G_original.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0.")
    if n_samples == 0:
        return []

    G_clean = prepare_graph(G_original)
    nodes = list(G_clean.nodes())
    if len(nodes) < 2:
        return []

    edge_set = {normalize_edge(u, v) for u, v in G_clean.edges()}
    total_pairs = (len(nodes) * (len(nodes) - 1)) // 2
    max_non_edges = total_pairs - len(edge_set)
    if n_samples > max_non_edges:
        raise ValueError(
            f"Cannot sample {n_samples} negatives: only {max_non_edges} non-edges available."
        )

    rng = np.random.default_rng(seed)
    negatives: set[Edge] = set()

    max_trials = max(1000, 50 * n_samples)
    trials = 0
    while len(negatives) < n_samples and trials < max_trials:
        u, v = rng.choice(nodes, size=2, replace=False)
        edge = normalize_edge(u, v)
        if edge not in edge_set:
            negatives.add(edge)
        trials += 1

    if len(negatives) < n_samples:
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                edge = normalize_edge(u, v)
                if edge in edge_set or edge in negatives:
                    continue
                negatives.add(edge)
                if len(negatives) >= n_samples:
                    break
            if len(negatives) >= n_samples:
                break

    return sorted(negatives)


def adamic_adar_scores(G: nx.Graph, candidate_edges: Sequence[Edge]) -> List[float]:
    """
    Calcule les scores Adamic-Adar pour candidate_edges.
    Si noeud absent, paire invalide, ou cas impossible: score = 0.
    """
    G_clean = prepare_graph(G)
    scores = np.zeros(len(candidate_edges), dtype=float)

    valid_edges: List[Edge] = []
    valid_indices: List[int] = []
    for idx, (u, v) in enumerate(candidate_edges):
        if u == v:
            continue
        if not G_clean.has_node(u) or not G_clean.has_node(v):
            continue
        valid_edges.append((u, v))
        valid_indices.append(idx)

    if not valid_edges:
        return scores.tolist()

    try:
        for (u, v, score), idx in zip(nx.adamic_adar_index(G_clean, valid_edges), valid_indices):
            if score is not None and np.isfinite(score):
                scores[idx] = float(score)
    except Exception:
        return scores.tolist()

    return scores.tolist()


def precision_at_k(y_true: Sequence[int], y_scores: Sequence[float], k: int = 100) -> float:
    """
    Precision@k avec tri decroissant des scores.
    """
    if k <= 0:
        return float("nan")
    if len(y_true) == 0:
        return float("nan")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    if y_true_arr.shape[0] != y_scores_arr.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")

    k_eff = min(int(k), y_true_arr.shape[0])
    if k_eff <= 0:
        return float("nan")

    ranked = np.argsort(-y_scores_arr, kind="mergesort")[:k_eff]
    return float(np.mean(y_true_arr[ranked]))


def evaluate_adamic_adar(
    G_score: nx.Graph,
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
    k_values: Sequence[int] = (50, 100, 500),
) -> Dict[str, float]:
    """
    Evalue Adamic-Adar avec AUC, Average Precision et Precision@k.
    """
    pos = list(positive_edges)
    neg = list(negative_edges)
    candidates = pos + neg

    if not candidates:
        results: Dict[str, float] = {"auc": float("nan"), "average_precision": float("nan")}
        for k in k_values:
            results[f"precision_at_{int(k)}"] = float("nan")
        return results

    y_true = np.array([1] * len(pos) + [0] * len(neg), dtype=int)
    y_scores = np.array(adamic_adar_scores(G_score, candidates), dtype=float)

    if np.unique(y_true).shape[0] < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_scores))
    ap = float(average_precision_score(y_true, y_scores))

    results = {"auc": auc, "average_precision": ap}
    for k in k_values:
        kk = int(k)
        results[f"precision_at_{kk}"] = precision_at_k(y_true, y_scores, k=kk)
    return results


def _safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or math.isclose(den, 0.0, abs_tol=1e-15):
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


def _invoke_anonymizer(func: Callable[..., nx.Graph], G_train: nx.Graph, seed: int) -> nx.Graph:
    """
    Invoque une fonction d'anonymisation de maniere souple.
    Signature attendue idealement: f(graph, seed=...).
    """
    graph_copy = G_train.copy()

    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return func(graph_copy)

    params = signature.parameters
    param_names = set(params.keys())
    positional_params = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: Dict[str, Any] = {}
    args: List[Any] = []

    if positional_params or has_var_pos:
        args.append(graph_copy)
    elif "graph" in param_names:
        kwargs["graph"] = graph_copy
    elif "G" in param_names:
        kwargs["G"] = graph_copy
    elif has_var_kw:
        kwargs["graph"] = graph_copy
    else:
        raise TypeError("Unable to pass graph argument to anonymization function.")

    if "seed" in param_names:
        kwargs["seed"] = seed
    elif has_var_kw:
        kwargs["seed"] = seed

    return func(*args, **kwargs)


def compare_link_prediction_utility(
    G_original: nx.Graph,
    anonymization_functions: Dict[str, Callable[..., nx.Graph]] | None = None,
    test_frac: float = 0.1,
    seeds: Sequence[int] = (42, 123, 2024),
    k_values: Sequence[int] = (50, 100, 500),
) -> pd.DataFrame:
    """
    Compare l'utilite applicative (link prediction Adamic-Adar) entre le graphe
    original et des graphes anonymises (produits depuis G_train).
    """
    G_clean = prepare_graph(G_original)
    methods = anonymization_functions or {}
    k_values = [int(k) for k in k_values]

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        G_train, test_pos_edges = train_test_edge_split(G_clean, test_frac=test_frac, seed=int(seed))
        neg_edges = sample_negative_edges(G_clean, n_samples=len(test_pos_edges), seed=int(seed))

        original_metrics = evaluate_adamic_adar(
            G_score=G_train,
            positive_edges=test_pos_edges,
            negative_edges=neg_edges,
            k_values=k_values,
        )
        original_p100 = float(original_metrics.get("precision_at_100", float("nan")))

        base_row: Dict[str, Any] = {
            "method": "original",
            "seed": int(seed),
            "test_frac": float(test_frac),
            "num_nodes": int(G_clean.number_of_nodes()),
            "num_edges_original": int(G_clean.number_of_edges()),
            "num_edges_train": int(G_train.number_of_edges()),
            "num_test_pos_edges": int(len(test_pos_edges)),
            "auc": float(original_metrics.get("auc", float("nan"))),
            "average_precision": float(original_metrics.get("average_precision", float("nan"))),
            "precision_at_50": float(original_metrics.get("precision_at_50", float("nan"))),
            "precision_at_100": original_p100,
            "precision_at_500": float(original_metrics.get("precision_at_500", float("nan"))),
            "auc_loss": 0.0,
            "ap_loss": 0.0,
            "precision_at_100_loss": 0.0,
            "auc_utility_ratio": 1.0,
            "ap_utility_ratio": 1.0,
            "precision_at_100_utility_ratio": 1.0,
            "method_error": "",
        }
        rows.append(base_row)

        for method_name, anonymize_func in methods.items():
            method_error = ""
            try:
                anonymized_train = _invoke_anonymizer(anonymize_func, G_train=G_train, seed=int(seed))
                anonymized_train = prepare_graph(anonymized_train)
                metrics = evaluate_adamic_adar(
                    G_score=anonymized_train,
                    positive_edges=test_pos_edges,
                    negative_edges=neg_edges,
                    k_values=k_values,
                )
            except Exception as exc:
                anonymized_train = nx.Graph()
                metrics = {"auc": float("nan"), "average_precision": float("nan")}
                for k in k_values:
                    metrics[f"precision_at_{int(k)}"] = float("nan")
                method_error = f"{type(exc).__name__}: {exc}"

            method_auc = float(metrics.get("auc", float("nan")))
            method_ap = float(metrics.get("average_precision", float("nan")))
            method_p100 = float(metrics.get("precision_at_100", float("nan")))
            original_auc = float(original_metrics.get("auc", float("nan")))
            original_ap = float(original_metrics.get("average_precision", float("nan")))

            row = {
                "method": str(method_name),
                "seed": int(seed),
                "test_frac": float(test_frac),
                "num_nodes": int(G_clean.number_of_nodes()),
                "num_edges_original": int(G_clean.number_of_edges()),
                "num_edges_train": int(G_train.number_of_edges()),
                "num_test_pos_edges": int(len(test_pos_edges)),
                "auc": method_auc,
                "average_precision": method_ap,
                "precision_at_50": float(metrics.get("precision_at_50", float("nan"))),
                "precision_at_100": method_p100,
                "precision_at_500": float(metrics.get("precision_at_500", float("nan"))),
                "auc_loss": _safe_sub(original_auc, method_auc),
                "ap_loss": _safe_sub(original_ap, method_ap),
                "precision_at_100_loss": _safe_sub(original_p100, method_p100),
                "auc_utility_ratio": _safe_div(method_auc, original_auc),
                "ap_utility_ratio": _safe_div(method_ap, original_ap),
                "precision_at_100_utility_ratio": _safe_div(method_p100, original_p100),
                "method_error": method_error,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    expected_order = [
        "method",
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
        "method_error",
    ]
    present_cols = [col for col in expected_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in present_cols]
    return df[present_cols + extra_cols]
