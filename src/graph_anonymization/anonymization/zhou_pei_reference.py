#!/usr/bin/env python3
"""
Zhou & Pei k-neighborhood anonymity — version k-anonymisation only.

Cette implémentation reprend uniquement la partie k-anonymity de l'article :
    B. Zhou, J. Pei, "The k-anonymity and l-diversity approaches for privacy
    preservation in social networks against neighborhood attacks", KAIS, 2011.

Ce que fait le programme :
1) lit un graphe non orienté depuis un fichier edge-list / .pairs ;
2) extrait Neighbor_G(u), le sous-graphe induit par les voisins directs de u ;
3) calcule un NCC pratique : codes canoniques des composantes connexes du voisinage ;
4) applique la logique SeedVertex / CandidateSet de l'Algorithm 1 ;
5) anonymise par ajout d'arêtes uniquement ;
6) vérifie si chaque classe de voisinage contient au moins k sommets ;
7) écrit le graphe anonymisé.

Important : le papier utilise le minimum DFS code de gSpan. Ici, pour rester autonome
et exécutable avec NetworkX, on utilise :
- un codage canonique exact par permutations pour les petites composantes ;
- une signature déterministe pour les grandes composantes ;
- une option --exact-verify avec VF2 pour contrôler les classes d'isomorphisme.

Aucune l-diversity n'est implémentée ici.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
import argparse
import json
import random
import sys
import time

import networkx as nx

Node = Hashable
Code = Tuple[Any, ...]


# ---------------------------------------------------------------------------
# Labels optionnels : pour un graphe non labellisé, tous les sommets valent "*".
# ---------------------------------------------------------------------------
@dataclass
class LabelHierarchy:
    parent: Dict[Any, Any] = field(default_factory=dict)
    root: Any = "*"

    def ancestors(self, label: Any) -> List[Any]:
        if label is None:
            label = self.root
        out = [label]
        seen = {label}
        while out[-1] in self.parent:
            nxt = self.parent[out[-1]]
            if nxt in seen:
                break
            out.append(nxt)
            seen.add(nxt)
            if nxt == self.root:
                break
        if self.root not in out:
            out.append(self.root)
        return out

    def lca(self, a: Any, b: Any) -> Any:
        aa = self.ancestors(a)
        bb = set(self.ancestors(b))
        for x in aa:
            if x in bb:
                return x
        return self.root

    def leaf_count(self, label: Any) -> int:
        children: Dict[Any, List[Any]] = defaultdict(list)
        for child, par in self.parent.items():
            children[par].append(child)
        if label not in children:
            return 1
        stack = list(children[label])
        count = 0
        while stack:
            x = stack.pop()
            if x in children:
                stack.extend(children[x])
            else:
                count += 1
        return max(count, 1)

    def ncp(self, label: Any) -> float:
        return self.leaf_count(label) / max(self.leaf_count(self.root), 1)


@dataclass
class ZhouPeiKConfig:
    k: int = 5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    label_attr: str = "label"
    default_label: Any = "*"
    seed: int = 42
    relabel_output: bool = False
    max_canonical_component_size: int = 8
    max_repair_rounds: int = 200
    use_template_repair: bool = True
    verbose: bool = True


class ZhouPeiKAnonymizer:
    def __init__(self, config: ZhouPeiKConfig, hierarchy: Optional[LabelHierarchy] = None):
        if config.k < 2:
            raise ValueError("k doit être >= 2")
        self.cfg = config
        self.hierarchy = hierarchy or LabelHierarchy(root=config.default_label)
        self.rng = random.Random(config.seed)
        self.added_edges: set[Tuple[Node, Node]] = set()

    # ------------------------------------------------------------------
    # Fonctions de base : labels, voisinages, composantes, NCC.
    # ------------------------------------------------------------------
    def _edge_key(self, u: Node, v: Node) -> Tuple[Node, Node]:
        return (u, v) if str(u) <= str(v) else (v, u)

    def add_edge(self, G: nx.Graph, u: Node, v: Node) -> None:
        if u == v:
            return
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            self.added_edges.add(self._edge_key(u, v))

    def label(self, G: nx.Graph, v: Node) -> Any:
        return G.nodes[v].get(self.cfg.label_attr, self.cfg.default_label)

    def set_label(self, G: nx.Graph, v: Node, value: Any) -> None:
        G.nodes[v][self.cfg.label_attr] = value

    def common_label(self, G: nx.Graph, nodes: Iterable[Node]) -> Any:
        nodes = list(nodes)
        if not nodes:
            return self.cfg.default_label
        cur = self.label(G, nodes[0])
        for v in nodes[1:]:
            cur = self.hierarchy.lca(cur, self.label(G, v))
        return cur

    def neighborhood_nodes(self, G: nx.Graph, u: Node) -> List[Node]:
        return list(G.neighbors(u))

    def neighborhood_subgraph(self, G: nx.Graph, u: Node) -> nx.Graph:
        return G.subgraph(self.neighborhood_nodes(G, u)).copy()

    def neighborhood_size_key(self, G: nx.Graph, u: Node) -> Tuple[int, int]:
        H = self.neighborhood_subgraph(G, u)
        return (H.number_of_nodes(), H.number_of_edges())

    def neighborhood_components(self, G: nx.Graph, u: Node) -> List[nx.Graph]:
        H = self.neighborhood_subgraph(G, u)
        comps = []
        for nodes in nx.connected_components(H):
            comps.append(H.subgraph(nodes).copy())
        comps.sort(key=lambda C: (C.number_of_nodes(), C.number_of_edges(), self.canonical_component_code(C)))
        return comps

    def canonical_component_code(self, C: nx.Graph) -> Code:
        """Code canonique pratique d'une composante de voisinage.

        Pour petite composante : exact par permutations.
        Pour grande composante : signature déterministe. L'option --exact-verify permet
        ensuite une vérification VF2 plus stricte.
        """
        nodes = list(C.nodes())
        n = len(nodes)
        if n == 0:
            return tuple()

        if n > self.cfg.max_canonical_component_size:
            labels = tuple(sorted(str(self.label(C, x)) for x in nodes))
            degs = tuple(sorted(dict(C.degree()).values()))
            # Une petite amélioration : distribution des degrés des voisins.
            nd = []
            for x in nodes:
                nd.append(tuple(sorted(C.degree(y) for y in C.neighbors(x))))
            return ("LARGE", n, C.number_of_edges(), labels, degs, tuple(sorted(nd)))

        best: Optional[Code] = None
        for perm in permutations(nodes):
            node_labels = tuple(str(self.label(C, node)) for node in perm)
            edge_bits = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_bits.append(1 if C.has_edge(perm[i], perm[j]) else 0)
            code = (node_labels, tuple(edge_bits))
            if best is None or code < best:
                best = code
        return best if best is not None else tuple()

    def neighborhood_code(self, G: nx.Graph, u: Node) -> Tuple[Code, ...]:
        return tuple(self.canonical_component_code(C) for C in self.neighborhood_components(G, u))

    def neighborhoods_isomorphic_exact(self, G: nx.Graph, u: Node, v: Node) -> bool:
        Hu = self.neighborhood_subgraph(G, u)
        Hv = self.neighborhood_subgraph(G, v)
        nm = nx.algorithms.isomorphism.categorical_node_match(
            self.cfg.label_attr, self.cfg.default_label
        )
        return nx.is_isomorphic(Hu, Hv, node_match=nm)

    # ------------------------------------------------------------------
    # Coût d'anonymisation de deux composantes / deux voisinages.
    # ------------------------------------------------------------------
    def _mapping_cost_same_size(self, G: nx.Graph, A: nx.Graph, B: nx.Graph) -> Tuple[float, Dict[Node, Node]]:
        a_nodes = list(A.nodes())
        b_nodes = list(B.nodes())
        if len(a_nodes) != len(b_nodes):
            return float("inf"), {}

        n = len(a_nodes)
        if n == 0:
            return 0.0, {}

        if n > self.cfg.max_canonical_component_size:
            # Heuristique pour éviter l'explosion factorielle.
            a_sorted = sorted(a_nodes, key=lambda x: (-A.degree(x), str(self.label(A, x))))
            b_sorted = sorted(b_nodes, key=lambda x: (-B.degree(x), str(self.label(B, x))))
            mapping = dict(zip(a_sorted, b_sorted))
            edge_adds = 0
            label_cost = 0.0
            for x, y in mapping.items():
                label_cost += self.hierarchy.ncp(self.hierarchy.lca(self.label(G, x), self.label(G, y)))
            for x, y in combinations(a_nodes, 2):
                bx, by = mapping[x], mapping[y]
                if A.has_edge(x, y) != B.has_edge(bx, by):
                    edge_adds += 1
            return self.cfg.alpha * label_cost + self.cfg.beta * edge_adds, mapping

        best_cost = float("inf")
        best_map: Dict[Node, Node] = {}
        for perm in permutations(b_nodes):
            mapping = dict(zip(a_nodes, perm))
            label_cost = 0.0
            edge_adds = 0
            for x, y in mapping.items():
                label_cost += self.hierarchy.ncp(self.hierarchy.lca(self.label(G, x), self.label(G, y)))
            for x, y in combinations(a_nodes, 2):
                bx, by = mapping[x], mapping[y]
                if A.has_edge(x, y) != B.has_edge(bx, by):
                    edge_adds += 1
            cost = self.cfg.alpha * label_cost + self.cfg.beta * edge_adds
            if cost < best_cost:
                best_cost = cost
                best_map = mapping
        return best_cost, best_map

    def component_pair_cost(self, G: nx.Graph, A: nx.Graph, B: nx.Graph) -> float:
        if A.number_of_nodes() == B.number_of_nodes():
            cost, _ = self._mapping_cost_same_size(G, A, B)
            return cost
        # Approximation du coût des sommets externes nécessaires pour rendre les tailles compatibles.
        return (
            self.cfg.gamma * abs(A.number_of_nodes() - B.number_of_nodes())
            + self.cfg.beta * abs(A.number_of_edges() - B.number_of_edges())
        )

    def pair_cost(self, G: nx.Graph, u: Node, v: Node) -> float:
        Cu = self.neighborhood_components(G, u)
        Cv = self.neighborhood_components(G, v)
        used_v: set[int] = set()
        total = 0.0

        # On part des plus grandes composantes, comme dans l'esprit du papier.
        order_u = sorted(range(len(Cu)), key=lambda i: (-Cu[i].number_of_nodes(), -Cu[i].number_of_edges()))
        for i in order_u:
            A = Cu[i]
            best_cost = float("inf")
            best_j = None
            for j, B in enumerate(Cv):
                if j in used_v:
                    continue
                c = self.component_pair_cost(G, A, B)
                if c < best_cost:
                    best_cost, best_j = c, j
            if best_j is None:
                total += self.cfg.gamma * A.number_of_nodes() + self.cfg.beta * A.number_of_edges()
            else:
                used_v.add(best_j)
                total += best_cost

        for j, B in enumerate(Cv):
            if j not in used_v:
                total += self.cfg.gamma * B.number_of_nodes() + self.cfg.beta * B.number_of_edges()
        return total

    # ------------------------------------------------------------------
    # Anonymisation de deux voisinages et d'un groupe.
    # ------------------------------------------------------------------
    def _generalize_mapping_labels(self, G: nx.Graph, mapping: Dict[Node, Node]) -> None:
        for a, b in mapping.items():
            lca = self.hierarchy.lca(self.label(G, a), self.label(G, b))
            self.set_label(G, a, lca)
            self.set_label(G, b, lca)

    def _copy_missing_edges_between_mapped_components(self, G: nx.Graph, A: nx.Graph, B: nx.Graph, mapping: Dict[Node, Node]) -> None:
        inv = {b: a for a, b in mapping.items()}
        for x, y in A.edges():
            bx, by = mapping[x], mapping[y]
            self.add_edge(G, bx, by)
        for x, y in B.edges():
            ax, ay = inv[x], inv[y]
            self.add_edge(G, ax, ay)

    def _share_component_with_center(self, G: nx.Graph, center: Node, component_nodes: Iterable[Node]) -> None:
        for x in component_nodes:
            if x != center:
                self.add_edge(G, center, x)

    def anonymize_two_neighborhoods(self, G: nx.Graph, u: Node, v: Node) -> None:
        Cu = self.neighborhood_components(G, u)
        Cv = self.neighborhood_components(G, v)
        used_u: set[int] = set()
        used_v: set[int] = set()

        # 1) Composantes parfaitement identiques selon le code.
        for i, A in enumerate(Cu):
            codeA = self.canonical_component_code(A)
            for j, B in enumerate(Cv):
                if i in used_u or j in used_v:
                    continue
                if codeA == self.canonical_component_code(B):
                    used_u.add(i)
                    used_v.add(j)
                    break

        # 2) Appariement glouton des composantes restantes de même taille.
        candidates: List[Tuple[float, int, int, Dict[Node, Node]]] = []
        for i, A in enumerate(Cu):
            if i in used_u:
                continue
            for j, B in enumerate(Cv):
                if j in used_v:
                    continue
                if A.number_of_nodes() == B.number_of_nodes():
                    c, mapping = self._mapping_cost_same_size(G, A, B)
                    candidates.append((c, i, j, mapping))
        candidates.sort(key=lambda x: x[0])

        for _, i, j, mapping in candidates:
            if i in used_u or j in used_v:
                continue
            A, B = Cu[i], Cv[j]
            self._generalize_mapping_labels(G, mapping)
            self._copy_missing_edges_between_mapped_components(G, A, B, mapping)
            used_u.add(i)
            used_v.add(j)

        # 3) Composantes restantes : ajout dans l'autre voisinage.
        for i, A in enumerate(Cu):
            if i not in used_u:
                self._share_component_with_center(G, v, A.nodes())
        for j, B in enumerate(Cv):
            if j not in used_v:
                self._share_component_with_center(G, u, B.nodes())

    def anonymize_group_pairwise(self, G: nx.Graph, group: Sequence[Node]) -> None:
        if len(group) < 2:
            return
        seed = group[0]
        for x in group[1:]:
            self.anonymize_two_neighborhoods(G, seed, x)
        # Petite propagation interne pour rapprocher tous les voisinages du groupe.
        for a, b in combinations(group, 2):
            self.anonymize_two_neighborhoods(G, a, b)

    def anonymize_group_template(self, G: nx.Graph, group: Sequence[Node]) -> None:
        """Réparation de sécurité par template commun.

        Cette étape n'est pas la partie principale du papier. Elle sert à garantir que le
        groupe choisi devient réellement k-anonyme en ajoutant seulement des arêtes.
        En pire cas, cette logique converge vers une clique, ce qui reste compatible avec
        l'hypothèse d'ajout d'arêtes uniquement.
        """
        group = list(dict.fromkeys(group))
        if len(group) < 2:
            return

        # Généraliser les labels des centres pour qu'ils soient interchangeables.
        lca = self.common_label(G, group)
        for x in group:
            self.set_label(G, x, lca)

        # Tous les sommets du groupe deviennent clique.
        for a, b in combinations(group, 2):
            self.add_edge(G, a, b)

        # Tous les centres partagent le même voisinage externe.
        external: set[Node] = set()
        for x in group:
            external.update(G.neighbors(x))
        external.difference_update(group)

        for center in group:
            for x in external:
                self.add_edge(G, center, x)

    # ------------------------------------------------------------------
    # Classes d'équivalence et vérification.
    # ------------------------------------------------------------------
    def equivalence_classes_by_code(self, G: nx.Graph) -> List[List[Node]]:
        buckets: Dict[Tuple[Code, ...], List[Node]] = defaultdict(list)
        for v in G.nodes():
            buckets[self.neighborhood_code(G, v)].append(v)
        return list(buckets.values())

    def equivalence_classes_exact(self, G: nx.Graph) -> List[List[Node]]:
        # On pré-bucket par code pour réduire les comparaisons, puis on split par VF2.
        final_classes: List[List[Node]] = []
        for bucket in self.equivalence_classes_by_code(G):
            reps: List[Node] = []
            groups: List[List[Node]] = []
            for v in bucket:
                placed = False
                for idx, r in enumerate(reps):
                    if self.neighborhoods_isomorphic_exact(G, v, r):
                        groups[idx].append(v)
                        placed = True
                        break
                if not placed:
                    reps.append(v)
                    groups.append([v])
            final_classes.extend(groups)
        return final_classes

    def validate_k_anonymity(self, G: nx.Graph, exact: bool = False) -> Tuple[bool, List[List[Node]]]:
        classes = self.equivalence_classes_exact(G) if exact else self.equivalence_classes_by_code(G)
        return all(len(c) >= self.cfg.k for c in classes), classes

    # ------------------------------------------------------------------
    # Algorithme principal Zhou & Pei k-anonymity.
    # ------------------------------------------------------------------
    def anonymize(self, G_in: nx.Graph) -> nx.Graph:
        G = G_in.copy()
        self.added_edges.clear()

        for v in G.nodes():
            if self.cfg.label_attr not in G.nodes[v]:
                G.nodes[v][self.cfg.label_attr] = self.cfg.default_label

        if G.number_of_nodes() < self.cfg.k:
            raise ValueError("Le graphe contient moins de k sommets : impossible d'obtenir une k-anonymité non triviale.")

        status: Dict[Node, str] = {v: "unanonymized" for v in G.nodes()}
        iteration = 0

        while True:
            vertex_list = [v for v, s in status.items() if s == "unanonymized"]
            if not vertex_list:
                break

            # Papier : ordre décroissant de |V(Neighbor)|, puis |E(Neighbor)|.
            vertex_list.sort(key=lambda x: self.neighborhood_size_key(G, x), reverse=True)
            seed = vertex_list[0]
            remaining = vertex_list[1:]

            if len(vertex_list) < self.cfg.k:
                # Les derniers sommets seront réparés à la passe finale.
                for v in vertex_list:
                    status[v] = "anonymized"
                break

            if len(vertex_list) >= 2 * self.cfg.k:
                scored = [(self.pair_cost(G, seed, v), v) for v in remaining]
                scored.sort(key=lambda x: x[0])
                candidate_set = [v for _, v in scored[: self.cfg.k - 1]]
            else:
                candidate_set = remaining

            group = [seed] + candidate_set
            self.anonymize_group_pairwise(G, group)

            # On applique un template léger sur le groupe pour sécuriser l'isomorphisme final.
            if self.cfg.use_template_repair and len(group) >= self.cfg.k:
                self.anonymize_group_template(G, group)

            for v in group:
                status[v] = "anonymized"

            iteration += 1
            if self.cfg.verbose and iteration % 25 == 0:
                print(f"[info] groupes traités={iteration}, arêtes ajoutées={len(self.added_edges)}", file=sys.stderr)

        self.repair_until_k_anonymous(G)

        if self.cfg.relabel_output:
            mapping = self.random_mapping(list(G.nodes()))
            G = nx.relabel_nodes(G, mapping, copy=True)
        return G

    def repair_until_k_anonymous(self, G: nx.Graph) -> None:
        """Fusionne les classes faibles jusqu'à ce que chaque classe ait au moins k sommets.

        Cette réparation est volontairement conservatrice : elle ajoute des arêtes seulement.
        """
        for round_idx in range(self.cfg.max_repair_rounds):
            ok, classes = self.validate_k_anonymity(G, exact=False)
            if ok:
                return

            weak = sorted([c for c in classes if len(c) < self.cfg.k], key=len)
            if not weak:
                return
            base = list(weak[0])
            seed = base[0]
            need = self.cfg.k - len(base)
            candidates = [v for v in G.nodes() if v not in base]
            if not candidates:
                return

            scored = [(self.pair_cost(G, seed, v), v) for v in candidates]
            scored.sort(key=lambda x: x[0])
            group = base + [v for _, v in scored[:need]]
            self.anonymize_group_pairwise(G, group)
            if self.cfg.use_template_repair:
                self.anonymize_group_template(G, group)

            if self.cfg.verbose and (round_idx + 1) % 25 == 0:
                print(f"[repair] round={round_idx + 1}, weak_classes={len(weak)}, arêtes ajoutées={len(self.added_edges)}", file=sys.stderr)

    def random_mapping(self, nodes: List[Node]) -> Dict[Node, Node]:
        shuffled = nodes[:]
        self.rng.shuffle(shuffled)
        return dict(zip(nodes, shuffled))


# ---------------------------------------------------------------------------
# I/O fichiers .pairs / edge-list.
# ---------------------------------------------------------------------------
def parse_node_token(token: str) -> str:
    # On garde en string pour éviter les surprises entre datasets numériques et textuels.
    return token


def read_edge_list(path: str, delimiter: Optional[str] = None, comment_prefix: str = "#") -> nx.Graph:
    G = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith(comment_prefix):
                continue
            parts = line.split(delimiter)
            parts = [p for p in parts if p != ""]
            if len(parts) < 2:
                continue
            u, v = parse_node_token(parts[0]), parse_node_token(parts[1])
            if u != v:
                G.add_edge(u, v)
    return G


def read_labels(path: str, G: nx.Graph, label_attr: str = "label", delimiter: Optional[str] = None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.split(delimiter) if p != ""]
            if len(parts) < 2:
                continue
            node, label = parse_node_token(parts[0]), parts[1]
            if node in G:
                G.nodes[node][label_attr] = label


def write_edge_list(G: nx.Graph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for u, v in sorted(G.edges(), key=lambda e: (str(e[0]), str(e[1]))):
            f.write(f"{u} {v}\n")


def write_added_edges(edges: Iterable[Tuple[Node, Node]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for u, v in sorted(edges, key=lambda e: (str(e[0]), str(e[1]))):
            f.write(f"{u} {v}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zhou & Pei k-neighborhood anonymization, k-anonymity only.")
    p.add_argument("--input", "-i", required=True, help="Fichier edge-list / .pairs d'entrée")
    p.add_argument("--output", "-o", required=True, help="Fichier edge-list / .pairs anonymisé")
    p.add_argument("--k", type=int, required=True, help="Paramètre k")
    p.add_argument("--labels", default=None, help="Optionnel : fichier 'node label'")
    p.add_argument("--label-attr", default="label")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-canonical-component-size", type=int, default=8)
    p.add_argument("--max-repair-rounds", type=int, default=200)
    p.add_argument("--no-template-repair", action="store_true", help="Désactive la réparation template qui garantit mieux l'isomorphisme")
    p.add_argument("--exact-verify", action="store_true", help="Vérification finale par VF2, plus lente")
    p.add_argument("--relabel-output", action="store_true", help="Relabeling aléatoire des sommets en sortie")
    p.add_argument("--added-edges-output", default=None, help="Optionnel : écrit la liste des arêtes ajoutées")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()

    G = read_edge_list(args.input)
    if args.labels:
        read_labels(args.labels, G, label_attr=args.label_attr)

    cfg = ZhouPeiKConfig(
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        seed=args.seed,
        label_attr=args.label_attr,
        relabel_output=args.relabel_output,
        max_canonical_component_size=args.max_canonical_component_size,
        max_repair_rounds=args.max_repair_rounds,
        use_template_repair=not args.no_template_repair,
        verbose=not args.quiet,
    )

    anonymizer = ZhouPeiKAnonymizer(cfg)
    n0, m0 = G.number_of_nodes(), G.number_of_edges()
    Gp = anonymizer.anonymize(G)
    ok_fast, classes_fast = anonymizer.validate_k_anonymity(Gp, exact=False)
    ok_exact = None
    classes_exact = None
    if args.exact_verify:
        ok_exact, classes_exact = anonymizer.validate_k_anonymity(Gp, exact=True)

    write_edge_list(Gp, args.output)
    if args.added_edges_output:
        write_added_edges(anonymizer.added_edges, args.added_edges_output)

    summary = {
        "input_nodes": n0,
        "input_edges": m0,
        "output_nodes": Gp.number_of_nodes(),
        "output_edges": Gp.number_of_edges(),
        "added_edges_count": Gp.number_of_edges() - m0,
        "k": args.k,
        "k_anonymous_fast_code_check": ok_fast,
        "num_equivalence_classes_fast": len(classes_fast),
        "min_class_size_fast": min(len(c) for c in classes_fast) if classes_fast else 0,
        "exact_verify_enabled": bool(args.exact_verify),
        "k_anonymous_exact_vf2": ok_exact,
        "num_equivalence_classes_exact": len(classes_exact) if classes_exact is not None else None,
        "min_class_size_exact": min(len(c) for c in classes_exact) if classes_exact else None,
        "runtime_sec": round(time.time() - t0, 3),
        "output": args.output,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.exact_verify and ok_exact is False:
        return 2
    if not ok_fast:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
