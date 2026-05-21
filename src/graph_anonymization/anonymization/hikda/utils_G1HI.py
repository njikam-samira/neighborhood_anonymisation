#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import copy
import itertools
import random
import math

def modify_graph_to_break_1_neighborhood(
    g,
    max_sample_nodes=250,
    max_sample_pairs=5000,
    max_iterations=700,
):
    
    modifications = []  # record all modified edges (including both additions and deletions)
    G=copy.deepcopy(g)
    nodes = set(G.nodes)
    n=len(nodes)
    
    iteration = 0
    while nodes and iteration < max_iterations:
        iteration += 1
        max_triangles = 0
        all_triangle_count = 0 
        best_edge = None
        best_U = set()
        
        # For large graphs, use a sampled candidate set to keep runtime tractable.
        if len(nodes) > max_sample_nodes:
            candidate_nodes = set(random.sample(list(nodes), max_sample_nodes))
        else:
            candidate_nodes = set(nodes)

        neighbors = {node: set(G.neighbors(node)).intersection(nodes) for node in candidate_nodes}
        candidate_list = list(candidate_nodes)

        if len(candidate_nodes) > 260:
            pair_iter = set()
            while len(pair_iter) < max_sample_pairs:
                u, v = random.sample(candidate_list, 2)
                if u > v:
                    u, v = v, u
                pair_iter.add((u, v))
        else:
            pair_iter = itertools.combinations(candidate_nodes, 2)

        for u, v in pair_iter:
            n_u = neighbors.get(u, set())
            n_v = neighbors.get(v, set())
            triangle_count = len(n_u.intersection(n_v))
            all_triangle_count += triangle_count
            if triangle_count > max_triangles:
                max_triangles = triangle_count
                best_edge = (u, v)
                best_U = n_u.intersection(n_v).union({u, v})

        if all_triangle_count == 0: 
            u, v = random.sample(list(nodes), 2)
            best_edge = (u, v)
            n_u = set(G.neighbors(u)).intersection(nodes)
            n_v = set(G.neighbors(v)).intersection(nodes)
            best_U = n_u.intersection(n_v).union({u, v})

        if best_edge:
            modifications.append(best_edge)
            u, v = best_edge
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            else:
                G.add_edge(u, v)
            nodes -= best_U
        else:
            break

        if len(nodes) == 1:
            uu = list(nodes)[0]
            remaining_nodes = set(G.nodes) - nodes
            vv = random.choice(list(remaining_nodes))
            modifications.append((uu,vv))
            if G.has_edge(uu, vv):
                G.remove_edge(uu, vv)
            else:
                G.add_edge(uu, vv)
            nodes -= {uu}
            break

    # If there are remaining nodes because max_iterations was reached,
    # apply light random toggles to finish stage-1 perturbation quickly.
    if nodes:
        remain = list(nodes)
        random.shuffle(remain)
        for uu in remain[: min(len(remain), max_sample_nodes)]:
            others = list(set(G.nodes) - {uu})
            if not others:
                continue
            vv = random.choice(others)
            modifications.append((uu, vv))
            if G.has_edge(uu, vv):
                G.remove_edge(uu, vv)
            else:
                G.add_edge(uu, vv)

    return G,modifications

