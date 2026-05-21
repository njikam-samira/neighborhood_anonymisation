from torch_geometric.datasets import Planetoid
import torch
import numpy as np
from cluster_formation_RMSE import cluster_formation
from graph_reconstruction import graph_reconstruction
from graph_reconstruction_optimize_mean import graph_reconstruction_optimize_mean
from graph_reconstruction_not_optimize import graph_reconstruction_not_optimize
from classify_gat import classify_gat
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import collections
from typing import List, Dict
from torch_geometric.data import Data
import os
from metrique import calculate_apl 
from metrique import calculate_il
from metrique import calculate_clustering_coefficient
from metrique import calculate_edge_intersection


def load_cora_graph():
    # Charge le dataset Cora et retourne la séquence P et le graphe NetworkX.
    # Charger le dataset Cora
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    # Convertir en graphe NetworkX
    G = to_networkx(data, to_undirected=True)
    # Générer la séquence P (id, degree)
    P = [{'id': int(node), 'degree': G.degree[node]} for node in G.nodes()]
    # tri de P par degré décroissant
    P = sorted(P, key=lambda x: x['degree'], reverse=True)
    return data, P, G

def create_CW(G, num_nodes, save_path="cw.npy"):
    # Vérifier si une matrice existe déjà
    if os.path.exists(save_path):
        print(f"Chargement de la matrice CW depuis {save_path}")
        CW = np.load(save_path)
        if CW.shape != (num_nodes, num_nodes):
            raise ValueError(f"La matrice chargée a une forme {CW.shape}, mais {num_nodes}x{num_nodes} était attendu.")
        return CW

    CW = np.zeros((num_nodes, num_nodes))  # Initialiser la matrice de poids à 0
    for u, v in G.edges():
        weight = np.random.randint(1, 100)  # Poids aléatoire entre 1 et 99
        CW[u][v] = weight
        CW[v][u] = weight  
        
    # Sauvegarder la matrice
    np.save(save_path, CW)
    print(f"Matrice CW sauvegardée dans {save_path}")
    return CW
"""""
def adapt_cluster_format(clusters: List[List[Dict]], cluster_degrees: List[int]) -> List[Dict]:
    return [
        {
            'nodes': cluster, 
            'target_degree': degree
        }
        for cluster, degree in zip(clusters, cluster_degrees)
    ]

def prepare_final_data(clusters: List[Dict], CW: np.ndarray, original_features: torch.Tensor) -> Data:
    G = nx.from_numpy_array(CW)
    clustering = nx.clustering(G)
    max_degree = max(node['degree'] for cluster in clusters for node in cluster['nodes']) or 1
    neighbor_deg = {
        n: np.mean([G.degree(nb) for nb in G.neighbors(n)]) / max_degree 
        if n in G and G.degree(n) > 0 else 0 
        for n in range(CW.shape[0])
    }
    
    num_nodes_total = sum(len(c['nodes']) for c in clusters)
    num_features = original_features.size(1) + 5
    X = torch.zeros((num_nodes_total, num_features), dtype=torch.float32)
    y = torch.zeros(num_nodes_total, dtype=torch.long)
    node_id_to_idx = {}

    idx = 0
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster['nodes']:
            if node['id'] < original_features.size(0):
                X[idx, :original_features.size(1)] = original_features[node['id']]
            X[idx, original_features.size(1):] = torch.tensor([
                node['degree'] / max_degree,
                cluster['target_degree'] / max_degree,
                np.sum(CW[:, node['id']]) / (len(CW) - 1),
                clustering[node['id']],
                neighbor_deg[node['id']]
            ], dtype=torch.float32)
            y[idx] = cluster_id
            node_id_to_idx[node['id']] = idx
            
            idx += 1
    

    edge_indices = []
    for i in node_id_to_idx:
        edge_indices.extend([[node_id_to_idx[i], node_id_to_idx[j]] for j in np.where(CW[i] > 0)[0] if j in node_id_to_idx])

    return Data(
        x=X,
        y=y,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    ), node_id_to_idx

def prepare_final_data2(clusters: List[Dict], CW: np.ndarray, original_features: torch.Tensor) -> Data:
    G = nx.from_numpy_array(CW)
    clustering = nx.clustering(G)
    max_degree = max(node['degree'] for cluster in clusters for node in cluster['nodes']) or 1
    neighbor_deg = {
        n: np.mean([G.degree(nb) for nb in G.neighbors(n)]) / max_degree 
        if n in G and G.degree(n) > 0 else 0 
        for n in range(CW.shape[0])
    }
    
    num_nodes_total = sum(len(c['nodes']) for c in clusters)
    num_features = 5  # Uniquement les 5 attributs structurels
    X = torch.zeros((num_nodes_total, num_features), dtype=torch.float32)
    y = torch.zeros(num_nodes_total, dtype=torch.long)
    node_id_to_idx = {}

    idx = 0
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster['nodes']:
            X[idx] = torch.tensor([
                node['degree'] / max_degree,
                cluster['target_degree'] / max_degree,
                np.sum(CW[:, node['id']]) / (len(CW) - 1),
                clustering[node['id']],
                neighbor_deg[node['id']]
            ], dtype=torch.float32)
            y[idx] = cluster_id
            node_id_to_idx[node['id']] = idx
            idx += 1
    
    edge_indices = []
    for i in node_id_to_idx:
        edge_indices.extend([[node_id_to_idx[i], node_id_to_idx[j]] for j in np.where(CW[i] > 0)[0] if j in node_id_to_idx])

    return Data(
        x=X,
        y=y,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    ), node_id_to_idx
"""""
# Définir la taille minimale des clusters (k)
k = 5
# Charger le dataset Cora
print("Chargement du dataset Cora...")
data, P, G = load_cora_graph()
print(f"Nombre de nœuds dans G: {len(G.nodes())}")
print(f"Nombre de nœuds dans P: {len(P)}")
print(f"Nombre d'arêtes: {G.number_of_edges()}")

# Formation des clusters
clusters, cluster_degrees, P_new = cluster_formation(P, k)
print("Séquence anonymisée (id, degree) :")
print(f"\nPnew: {P_new}")
print(f"\nListe des clusters: {clusters}")
print(f"\nListe des targets: {cluster_degrees}")
"""""
# Calcul de l'information loss
il = calculate_il(P, P_new)
print(f"\nCoût total des changements de degrés: {il}")
"""""
# Créer la matrice de poids CW pour le dataset Cora
CW = create_CW(G, len(G.nodes()))

# Reconstruire le graphe
#security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated = graph_reconstruction(clusters, cluster_degrees, CW, G)
security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated = graph_reconstruction_optimize_mean(clusters, cluster_degrees, CW, G)

# Afficher les informations sur G_updated
print(f"Nombre de nœuds après reconstruction: {len(G_updated.nodes())}")
print(f"Nombre d’arêtes après reconstruction: {G_updated.number_of_edges()}")

#Calcul de l'APL
#origine_apl = calculate_apl(G)
#print(f"APL origine: {origine_apl:.4f}")
"""""
origine_apl = 5.314579685360838
print(f"APL origine: {origine_apl:.4f}")

final_apl = calculate_apl(G_updated)
print(f"APL final: {final_apl:.4f}")
print(f"Changement d'APL: {abs(final_apl - origine_apl):.4f}")

# Calcul du coefficient de clustering et de EI
cc_original = calculate_clustering_coefficient(G)
cc_final = calculate_clustering_coefficient(G_updated)
ei = calculate_edge_intersection(G, G_updated)

print(f"Clustering Coefficient original: {cc_original}")
print(f"Clustering Coefficient final: {cc_final}")
print(f"Edge Intersection (EI): {ei}")

formatted_clusters = adapt_cluster_format(C_list_updated, cluster_degrees)

final_data, node_id_to_idx = prepare_final_data(formatted_clusters, CW_updated, data.x) 

final_clusters, accuracy, recall, f1 = classify_gat(clusters=formatted_clusters, data=final_data, k=k, epochs=800)

print("Nombre de noeuds dans le dataset final:", final_data.x.shape)

print("Features min/max:", final_data.x.min().item(), final_data.x.max().item())  

print("nombre de clusters dans mes données: ", len(formatted_clusters))
print(f"Valeurs uniques dans data.y: {torch.unique(final_data.y)}")
            
# Sauvegarde
torch.save(final_data, 'final_data.pt')
"""""
"""""
# Chargement
loaded_data = torch.load('final_data.pt', weights_only=False)
print(loaded_data)

torch.set_printoptions(profile="full") 
# Affiche les 5 premières lignes de data.x

for i in range(5):
    print(f"Nœud {i} : {loaded_data.x[i]}")
"""""