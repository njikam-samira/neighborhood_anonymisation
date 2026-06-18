from __future__ import annotations

import argparse
import copy
import pickle
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


PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
DISPLAY_DATASET_NAMES = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "enron": "Enron",
    "polblogs": "PolBlogs",
    "wiki-vote": "Wiki-Vote",
}
DATASET_FILE_STEMS = {
    "cora": ["cora", "Cora"],
    "citeseer": ["citeseer", "Citeseer", "CiteSeer"],
    "pubmed": ["pubmed", "Pubmed", "PubMed"],
    "enron": ["Enron", "enron"],
    "polblogs": ["polblogs", "Polblogs", "PolBlogs"],
    "wiki-vote": ["Wiki-Vote", "wiki-vote", "Wiki_Vote", "wiki_vote", "WikiVote", "wikivote"],
}
NODE_FEATURE_ATTR_CANDIDATES = ("x", "features", "feature", "attrs", "attr", "embedding")
NODE_LABEL_ATTR_CANDIDATES = ("y", "label", "class", "target", "community", "group")
GRAPH_FEATURE_KEY_CANDIDATES = ("x", "features", "feature_matrix", "node_features")
GRAPH_LABEL_KEY_CANDIDATES = ("y", "labels", "node_labels", "class_map")
MASK_KEY_CANDIDATES = ("train_mask", "val_mask", "test_mask")


def canonical_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "wikivote": "wiki-vote",
        "wiki-vote": "wiki-vote",
        "wiki-vote.": "wiki-vote",
        "cite-seer": "citeseer",
    }
    return aliases.get(key, key)


def normalize_dataset_name(dataset: str) -> str:
    mapping = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }
    key = canonical_dataset_name(dataset)
    return mapping.get(key, dataset)


def display_dataset_name(dataset: str) -> str:
    return DISPLAY_DATASET_NAMES.get(canonical_dataset_name(dataset), str(dataset))


def dataset_output_slug(dataset: str) -> str:
    return canonical_dataset_name(dataset).replace("-", "_")


def resolve_dataset_artifact(root_dir: Path, dataset: str, suffix: str) -> Path | None:
    dataset_key = canonical_dataset_name(dataset)
    for stem in DATASET_FILE_STEMS.get(dataset_key, [dataset_key]):
        candidate = root_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except UnicodeDecodeError:
        return False
    except OSError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def _node_sort_key(node: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(node):020d}")
    except (TypeError, ValueError):
        return (1, str(node))


def _as_numeric_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        array = array.reshape(-1)
    if array.size == 0:
        return None
    array = np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return array


def _coerce_label_values(values: Sequence[Any]) -> np.ndarray:
    if not values:
        raise ValueError("Aucun label disponible pour la classification.")

    numeric_labels: List[float] = []
    numeric_ok = True
    for value in values:
        try:
            numeric_labels.append(float(value))
        except (TypeError, ValueError):
            numeric_ok = False
            break

    if numeric_ok:
        arr = np.asarray(numeric_labels, dtype=np.float32)
        if arr.ndim == 1 and np.all(np.isfinite(arr)):
            unique = sorted({float(v) for v in arr.tolist()})
            mapping = {label: idx for idx, label in enumerate(unique)}
            return np.asarray([mapping[float(v)] for v in arr.tolist()], dtype=np.int64)

    normalized = [str(value) for value in values]
    classes = sorted(set(normalized))
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([mapping[label] for label in normalized], dtype=np.int64)


def _extract_matrix_from_container(
    container: Dict[str, Any],
    keys: Sequence[str],
    ordered_nodes: Sequence[Any],
    num_nodes: int,
) -> np.ndarray | None:
    for key in keys:
        if key not in container:
            continue
        raw = container[key]
        if isinstance(raw, dict):
            rows: List[np.ndarray] = []
            for node in ordered_nodes:
                vec = _as_numeric_vector(raw.get(node))
                if vec is None:
                    rows = []
                    break
                rows.append(vec)
            if rows and len({row.shape[0] for row in rows}) == 1:
                return np.vstack(rows).astype(np.float32)
            continue

        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            try:
                arr = np.asarray(raw)
            except Exception:
                continue
        if arr.ndim == 1 and arr.shape[0] == num_nodes:
            return arr.astype(np.float32).reshape(num_nodes, 1)
        if arr.ndim == 2 and arr.shape[0] == num_nodes:
            return arr.astype(np.float32)
    return None


def _extract_labels_from_container(
    container: Dict[str, Any],
    keys: Sequence[str],
    ordered_nodes: Sequence[Any],
    num_nodes: int,
) -> np.ndarray | None:
    for key in keys:
        if key not in container:
            continue
        raw = container[key]
        if isinstance(raw, dict):
            values = [raw.get(node) for node in ordered_nodes]
            if any(value is None for value in values):
                continue
            return _coerce_label_values(values)

        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[0] == num_nodes:
            return np.asarray(np.argmax(arr, axis=1), dtype=np.int64)
        if arr.ndim == 1 and arr.shape[0] == num_nodes:
            return _coerce_label_values(arr.tolist())
    return None


def _extract_mask_from_container(
    container: Dict[str, Any],
    key: str,
    ordered_nodes: Sequence[Any],
    num_nodes: int,
) -> torch.Tensor | None:
    if key not in container:
        return None
    raw = container[key]
    if isinstance(raw, dict):
        values = [bool(raw.get(node, False)) for node in ordered_nodes]
        return torch.as_tensor(values, dtype=torch.bool)
    if torch.is_tensor(raw):
        arr = raw.detach().cpu().numpy()
    else:
        arr = np.asarray(raw)
    if arr.ndim == 1 and arr.shape[0] == num_nodes:
        return torch.as_tensor(arr.astype(bool), dtype=torch.bool)
    return None


def _extract_features_from_node_attrs(graph: nx.Graph, num_nodes: int) -> np.ndarray | None:
    rows: List[np.ndarray] = []
    for node in range(num_nodes):
        node_attrs = dict(graph.nodes[node])
        vector = None
        for key in NODE_FEATURE_ATTR_CANDIDATES:
            vector = _as_numeric_vector(node_attrs.get(key))
            if vector is not None:
                break
        if vector is None:
            return None
        rows.append(vector)
    if len({row.shape[0] for row in rows}) != 1:
        return None
    return np.vstack(rows).astype(np.float32)


def _extract_labels_from_node_attrs(graph: nx.Graph, num_nodes: int) -> np.ndarray | None:
    values: List[Any] = []
    for node in range(num_nodes):
        node_attrs = dict(graph.nodes[node])
        label = None
        for key in NODE_LABEL_ATTR_CANDIDATES:
            if key in node_attrs:
                label = node_attrs[key]
                break
        if label is None:
            return None
        values.append(label)
    return _coerce_label_values(values)


def _extract_masks_from_node_attrs(graph: nx.Graph, num_nodes: int) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for key in MASK_KEY_CANDIDATES:
        values: List[bool] = []
        present = False
        for node in range(num_nodes):
            node_attrs = dict(graph.nodes[node])
            if key in node_attrs:
                present = True
            values.append(bool(node_attrs.get(key, False)))
        if present:
            masks[key] = torch.as_tensor(values, dtype=torch.bool)
    return masks


def _build_structural_features(graph: nx.Graph) -> np.ndarray:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return np.zeros((0, 5), dtype=np.float32)

    degrees = np.asarray([float(graph.degree[node]) for node in range(num_nodes)], dtype=np.float32)
    degree_norm = degrees / max(float(np.max(degrees)) if degrees.size else 1.0, 1.0)

    clustering = np.asarray([float(nx.clustering(graph, node)) for node in range(num_nodes)], dtype=np.float32)
    core_numbers = nx.core_number(graph) if graph.number_of_edges() > 0 else {node: 0 for node in graph.nodes()}
    core = np.asarray([float(core_numbers.get(node, 0.0)) for node in range(num_nodes)], dtype=np.float32)
    core_norm = core / max(float(np.max(core)) if core.size else 1.0, 1.0)

    avg_neighbor_degree = nx.average_neighbor_degree(graph) if graph.number_of_edges() > 0 else {}
    neighbor = np.asarray([float(avg_neighbor_degree.get(node, 0.0)) for node in range(num_nodes)], dtype=np.float32)
    neighbor_norm = neighbor / max(float(np.max(neighbor)) if neighbor.size else 1.0, 1.0)

    triangles = np.asarray([float(nx.triangles(graph, node)) for node in range(num_nodes)], dtype=np.float32)
    triangle_norm = triangles / max(float(np.max(triangles)) if triangles.size else 1.0, 1.0)

    return np.column_stack(
        [
            degree_norm,
            clustering,
            core_norm,
            neighbor_norm,
            triangle_norm,
        ]
    ).astype(np.float32)


def _build_default_split_masks(num_nodes: int, labels: np.ndarray, seed: int = 42) -> Dict[str, torch.Tensor]:
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    rng = np.random.default_rng(int(seed))

    for label in sorted(set(int(value) for value in labels.tolist())):
        indices = np.where(labels == label)[0]
        indices = indices.copy()
        rng.shuffle(indices)

        if indices.size == 1:
            train_mask[indices[0]] = True
            continue
        if indices.size == 2:
            train_mask[indices[0]] = True
            test_mask[indices[1]] = True
            continue

        train_count = max(1, int(round(0.6 * indices.size)))
        val_count = max(1, int(round(0.2 * indices.size)))
        if train_count + val_count >= indices.size:
            overflow = train_count + val_count - (indices.size - 1)
            if overflow > 0:
                val_count = max(1, val_count - overflow)
        if train_count + val_count >= indices.size:
            train_count = max(1, indices.size - 2)
            val_count = 1
        test_count = indices.size - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if val_count > 1:
                val_count -= 1
            else:
                train_count = max(1, train_count - 1)

        train_idx = indices[:train_count]
        val_idx = indices[train_count:train_count + val_count]
        test_idx = indices[train_count + val_count:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    if not val_mask.any():
        first_train = int(np.flatnonzero(train_mask)[0])
        train_mask[first_train] = False
        val_mask[first_train] = True
    if not test_mask.any():
        fallback_source = val_mask if val_mask.any() else train_mask
        first_idx = int(np.flatnonzero(fallback_source)[0])
        fallback_source[first_idx] = False
        test_mask[first_idx] = True
    if not train_mask.any():
        first_idx = int(np.flatnonzero(~train_mask)[0])
        train_mask[first_idx] = True
        if val_mask[first_idx]:
            val_mask[first_idx] = False
        elif test_mask[first_idx]:
            test_mask[first_idx] = False

    return {
        "train_mask": torch.as_tensor(train_mask, dtype=torch.bool),
        "val_mask": torch.as_tensor(val_mask, dtype=torch.bool),
        "test_mask": torch.as_tensor(test_mask, dtype=torch.bool),
    }


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    edges = [(int(u), int(v)) for u, v in graph.edges() if int(u) != int(v)]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.asarray(edges, dtype=np.int64)
    rev = arr[:, [1, 0]]
    all_edges = np.vstack([arr, rev])
    all_edges = np.unique(all_edges, axis=0)
    return torch.from_numpy(all_edges.T).long()


def load_planetoid_data(dataset: str, root_dir: Path) -> Data:
    name = normalize_dataset_name(dataset)
    ds = Planetoid(root=str(root_dir), name=name)
    data = ds[0]
    data.x = data.x.float()
    data.y = data.y.long()
    return data


def _load_custom_dataset_bundle(dataset: str, data_root: Path) -> tuple[Data, nx.Graph, Dict[Any, int]]:
    gpickle_path = resolve_dataset_artifact(data_root, dataset, ".gpickle")
    if gpickle_path is None:
        raise FileNotFoundError(
            f"Aucun fichier .gpickle trouve pour le dataset '{dataset}' dans {data_root}."
        )
    if is_git_lfs_pointer(gpickle_path):
        raise RuntimeError(
            f"Le fichier {gpickle_path} est un pointeur Git LFS non resolu. "
            "Recuperez le contenu binaire reel avant de lancer la classification."
        )

    with gpickle_path.open("rb") as handle:
        loaded = pickle.load(handle)

    if isinstance(loaded, Data):
        data = copy.deepcopy(loaded)
        data.x = data.x.float()
        data.y = data.y.long()
        graph = nx.Graph()
        edge_index = data.edge_index.detach().cpu().numpy()
        for src, dst in edge_index.T:
            graph.add_edge(int(src), int(dst))
        graph.add_nodes_from(range(int(data.num_nodes)))
        identity_mapping = {int(node): int(node) for node in range(int(data.num_nodes))}
        return data, prepare_simple_graph(graph), identity_mapping

    graph_obj = loaded
    graph_metadata: Dict[str, Any] = {}
    if isinstance(loaded, dict):
        if isinstance(loaded.get("data"), Data):
            data = copy.deepcopy(loaded["data"])
            data.x = data.x.float()
            data.y = data.y.long()
            graph = nx.Graph()
            edge_index = data.edge_index.detach().cpu().numpy()
            for src, dst in edge_index.T:
                graph.add_edge(int(src), int(dst))
            graph.add_nodes_from(range(int(data.num_nodes)))
            identity_mapping = {int(node): int(node) for node in range(int(data.num_nodes))}
            return data, prepare_simple_graph(graph), identity_mapping
        for key in ("graph", "G", "nx_graph"):
            if isinstance(loaded.get(key), nx.Graph):
                graph_obj = loaded[key]
                graph_metadata = {meta_key: meta_value for meta_key, meta_value in loaded.items() if meta_key != key}
                break

    if not isinstance(graph_obj, nx.Graph):
        raise TypeError(
            f"Format .gpickle non supporte pour le dataset '{dataset}': {type(graph_obj)!r}"
        )

    raw_graph = nx.Graph(graph_obj)
    raw_graph.remove_edges_from(nx.selfloop_edges(raw_graph))
    ordered_nodes = sorted(raw_graph.nodes(), key=_node_sort_key)
    node_mapping = {node: idx for idx, node in enumerate(ordered_nodes)}
    graph = nx.relabel_nodes(raw_graph, node_mapping, copy=True)
    graph = prepare_simple_graph(graph)
    num_nodes = graph.number_of_nodes()

    feature_matrix = _extract_matrix_from_container(graph_metadata, GRAPH_FEATURE_KEY_CANDIDATES, ordered_nodes, num_nodes)
    if feature_matrix is None:
        feature_matrix = _extract_matrix_from_container(raw_graph.graph, GRAPH_FEATURE_KEY_CANDIDATES, ordered_nodes, num_nodes)
    if feature_matrix is None:
        feature_matrix = _extract_features_from_node_attrs(graph, num_nodes)
    if feature_matrix is None:
        feature_matrix = _build_structural_features(graph)

    labels = _extract_labels_from_container(graph_metadata, GRAPH_LABEL_KEY_CANDIDATES, ordered_nodes, num_nodes)
    if labels is None:
        labels = _extract_labels_from_container(raw_graph.graph, GRAPH_LABEL_KEY_CANDIDATES, ordered_nodes, num_nodes)
    if labels is None:
        labels = _extract_labels_from_node_attrs(graph, num_nodes)
    if labels is None:
        raise ValueError(
            f"Aucun label exploitable trouve dans {gpickle_path} pour le dataset '{dataset}'."
        )

    masks: Dict[str, torch.Tensor] = {}
    for mask_name in MASK_KEY_CANDIDATES:
        mask = _extract_mask_from_container(graph_metadata, mask_name, ordered_nodes, num_nodes)
        if mask is None:
            mask = _extract_mask_from_container(raw_graph.graph, mask_name, ordered_nodes, num_nodes)
        if mask is not None:
            masks[mask_name] = mask
    node_masks = _extract_masks_from_node_attrs(graph, num_nodes)
    masks.update({key: value for key, value in node_masks.items() if key not in masks})
    if not {"train_mask", "val_mask", "test_mask"}.issubset(masks):
        masks.update(_build_default_split_masks(num_nodes=num_nodes, labels=labels, seed=42))

    data = Data(
        x=torch.as_tensor(feature_matrix, dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=_edge_index_from_graph(graph),
    )
    data.train_mask = masks["train_mask"].clone().to(dtype=torch.bool)
    data.val_mask = masks["val_mask"].clone().to(dtype=torch.bool)
    data.test_mask = masks["test_mask"].clone().to(dtype=torch.bool)
    return data, graph, node_mapping


def _load_graph_from_pairs_with_mapping(pair_path: Path, node_mapping: Dict[Any, int]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_mapping)))
    with pair_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                left: Any = int(parts[0])
            except ValueError:
                left = parts[0]
            try:
                right: Any = int(parts[1])
            except ValueError:
                right = parts[1]
            if left not in node_mapping or right not in node_mapping:
                continue
            graph.add_edge(int(node_mapping[left]), int(node_mapping[right]))
    return prepare_simple_graph(graph)


def load_dataset_bundle(dataset: str, data_root: Path, planetoid_root: Path) -> tuple[Data, nx.Graph, Path | None]:
    dataset_key = canonical_dataset_name(dataset)
    pair_path = resolve_dataset_artifact(data_root, dataset_key, ".pairs")

    if dataset_key in PLANETOID_DATASETS:
        data = load_planetoid_data(dataset=dataset_key, root_dir=planetoid_root)
        base_graph = load_base_graph(pair_path=pair_path, fallback_data=data)
        return data, base_graph, pair_path

    data, graph_from_dataset, node_mapping = _load_custom_dataset_bundle(dataset=dataset_key, data_root=data_root)
    if pair_path is not None and pair_path.exists():
        pair_graph = _load_graph_from_pairs_with_mapping(pair_path=pair_path, node_mapping=node_mapping)
        if pair_graph.number_of_edges() > 0 and pair_graph.number_of_nodes() == graph_from_dataset.number_of_nodes():
            graph_from_dataset = pair_graph

    graph_from_dataset.add_nodes_from(range(int(data.num_nodes)))
    return data, prepare_simple_graph(graph_from_dataset), pair_path


def model_file_slug(model: str) -> str:
    mapping = {
        "node2vec_logreg": "node2vec",
        "graphsage": "graphsage",
        "gat": "gat",
    }
    return mapping.get(str(model).lower(), str(model).lower())


def load_base_graph(pair_path: Path | None, fallback_data: Data) -> nx.Graph:
    if pair_path is not None and pair_path.exists():
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
        return anonymize_ange_modified_ncc(
            graph,
            k=_k,
            seed=seed,
            alpha=0.3,
            beta=0.4,
            gamma=0.2,
            delta=0.1,
            passes=1,
            max_node_iterations=0,
            fast_graph_threshold=30000,
        )

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
    dataset_label = display_dataset_name(dataset)
    with PdfPages(report_pdf) as pdf:
        render_text_page(
            pdf,
            f"{dataset_label} - {model} classification k-sweep",
            [
                f"Dataset: {dataset_label}",
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

    dataset = canonical_dataset_name(str(args.dataset))
    data_root = Path(args.data_root)
    planetoid_root = Path(args.planetoid_root)
    data_base, base_graph, pair_path = load_dataset_bundle(
        dataset=dataset,
        data_root=data_root,
        planetoid_root=planetoid_root,
    )
    if pair_path is None and dataset in PLANETOID_DATASETS:
        print(
            f"[INFO] Aucun fichier .pairs trouve pour {display_dataset_name(dataset)} ; "
            "fallback Planetoid utilise."
        )

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

    slug = dataset_output_slug(dataset)
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
