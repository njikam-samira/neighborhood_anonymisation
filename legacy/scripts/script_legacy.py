from torch_geometric.datasets import Planetoid
import torch
import numpy as np
from cluster_formation_RMSE import cluster_formation
from graph_reconstruction import graph_reconstruction
from graph_reconstruction_not_optimize import graph_reconstruction_not_optimize
from graph_reconstruction_optimize_mean import graph_reconstruction_optimize_mean
from classify_gat import classify_gat
from main import create_CW, adapt_cluster_format, prepare_final_data, load_cora_graph
import csv
from metrique import calculate_apl 
from metrique import calculate_il
from metrique import calculate_clustering_coefficient
from metrique import calculate_edge_intersection

# Liste des valeurs de k à tester
k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Définir les pipelines
pipelines = [
    {
        'name': 'glouton',
        'reconstruction_func': graph_reconstruction
    },
    {
        'name': 'moyen',
        'reconstruction_func': graph_reconstruction_optimize_mean
    }
]

# Fichier CSV pour sauvegarder les résultats
output_file = "resultats.csv"

# Charger le dataset Cora
print("Chargement du dataset Cora...")
data, P, G = load_cora_graph()
print(f"Nombre de nœuds dans G: {len(G.nodes())}")
print(f"Nombre de nœuds dans P: {len(P)}")
print(f"Nombre d'arêtes: {G.number_of_edges()}")

# Calculer les données du graphe original 
apl_original = calculate_apl(G)
print(f"APL original: {apl_original:.4f}")
cc_original = calculate_clustering_coefficient(G)
print(f"Clustering Coefficient original: {cc_original}")

# Créer la matrice de poids CW pour le dataset Cora
CW = create_CW(G, len(G.nodes()))

# Initialiser le CSV avec les en-têtes
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "pipeline",
        "k",
        "num_clusters",
        "num_nodes",
        "num_noisy_nodes",
        "num_edges",
        "apl_original",
        "cc_original",
        "accuracy",
        "recall",
        "f1",
        "apl",
        "apl_diff" 
        "il",
        "cc",
        "cc_diff"
        "EI"
    ])

# Boucler sur les valeurs de k, puis sur les pipelines
for k in k_values:
    print(f"\n=== Tests pour k={k} ===")
    
    # Étape 1 : Cluster formation
    try:
        clusters, cluster_degrees, P_new = cluster_formation(P, k)
    except Exception as e:
        print(f"Erreur lors de la formation des clusters pour k={k}: {e}")
        for pipeline in pipelines:
            pipeline_name = pipeline['name']
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    pipeline_name,
                    k,
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                ])
        continue  # Passe au prochain k si la formation des clusters échoue

    # Tester chaque pipeline pour ce k
    for pipeline in pipelines:
        pipeline_name = pipeline['name']
        reconstruction_func = pipeline['reconstruction_func']
        print(f"\nTest avec le pipeline {pipeline_name}")
        
        try:
            # Étape 2 : Graph reconstruction
            security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated = reconstruction_func(
                clusters, cluster_degrees, CW, G
            )
            # Adapter les clusters pour classification
            formatted_clusters = adapt_cluster_format(C_list_updated, cluster_degrees)
            
            # Préparer les données
            final_data, node_id_to_idx = prepare_final_data(formatted_clusters, CW_updated, data.x)
            
            # Étape 3 : Classification
            preds, accuracy, recall, f1 = classify_gat(
                clusters=formatted_clusters,
                data=final_data,
                k=k,
                epochs=800
            )

            # Calculer APL et IL
            il = calculate_il(P, P_new)
            apl = calculate_apl(G_updated)
            apl_diff = abs(apl - apl_original)

            # Calcul du coefficient de clustering et de EI
            cc_final = calculate_clustering_coefficient(G_updated)
            cc_diff = abs(cc_original - cc_final)
            ei = calculate_edge_intersection(G, G_updated)

            # Collecter les métriques
            num_clusters = len(C_list_updated)
            num_nodes = len(G_updated.nodes())
            num_noisy_nodes = len(noisy_nodes)
            num_edges = len(G_updated.number_of_edges())
            
            # Afficher les métriques
            print(f"Nombre de clusters: {num_clusters}")
            print(f"Nombre de noeuds: {num_nodes}")
            print(f"Nombre de noeuds bruyants: {num_noisy_nodes}")
            print(f"Nombre d'arêtes: {num_edges}")
            print(f"Métriques: Accuracy={accuracy*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")
            print(f"APL: {apl:.4f}, IL: {il:.4f}, CC: {cc_final:.4f}, EI: {ei:.4f}")
            
            # Sauvegarde des métriques dans le CSV
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    pipeline_name,
                    k,
                    num_clusters,
                    num_nodes,
                    num_noisy_nodes,
                    num_edges,
                    apl_original,
                    cc_original,
                    accuracy,
                    recall,
                    f1,
                    apl,
                    apl_diff, 
                    il,
                    cc_final,
                    cc_diff,
                    ei
                ])
            
        except Exception as e:
            print(f"Erreur pour pipeline {pipeline_name}, k={k}: {e}")
            with open(output_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    pipeline_name,
                    k,
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error",
                    "error"
                ])

print(f"\nRésultats sauvegardés dans {output_file}")