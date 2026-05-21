import os
import networkx as nx
import numpy as np
import csv
import time
import copy
import pickle
from pathlib import Path
from cluster_formation_RMSE import cluster_formation_RMSE
from cluster_formation_MAE import cluster_formation_MAE
from graph_reconstruction import graph_reconstruction
from graph_reconstruction_optimize_mean import graph_reconstruction_optimize_mean

def create_CW(G, save_path):
    # Supprimer le fichier existant pour forcer une recréation
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Fichier {save_path} supprimé pour recréation.")
    
    np.random.seed(42)
    
    # Trouver la taille nécessaire en fonction des IDs existants
    max_id = max(G.nodes()) + 1 
    CW = np.zeros((max_id, max_id))
    
    for u, v in G.edges():
        weight = np.random.randint(1, 100)  # Poids aléatoire entre 1 et 99
        CW[u][v] = weight
        CW[v][u] = weight
    
    np.save(save_path, CW)
    print(f"Matrice CW de taille {max_id}x{max_id} sauvegardée dans {save_path}")
    return CW

# Chemins
DATA_DIR = "data"
SECGRAPH_OUTPUT_DIR = "secGraph/output"
TIME_FILE = "execution_times.csv"
MATRIX_DIR = "matrices"

# Créer le dossier pour les matrices s'il n'existe pas
Path(MATRIX_DIR).mkdir(parents=True, exist_ok=True)

# Définition des pipelines 
pipelines = [
    {"name": "MAEglouton", "cluster_func": cluster_formation_MAE, "reconstruct_func": graph_reconstruction},
    {"name": "MAEmean", "cluster_func": cluster_formation_MAE, "reconstruct_func": graph_reconstruction_optimize_mean},
    {"name": "RMSEglouton", "cluster_func": cluster_formation_RMSE, "reconstruct_func": graph_reconstruction},
    {"name": "RMSEmean", "cluster_func": cluster_formation_RMSE, "reconstruct_func": graph_reconstruction_optimize_mean},
]

# Lister les fichiers .gpickle
gpickle_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".gpickle")]

if not gpickle_files:
    print(f"Aucun fichier .gpickle trouvé dans {DATA_DIR}.")
    exit()

# Initialiser le fichier CSV pour les temps d'exécution
if not os.path.exists(TIME_FILE):
    with open(TIME_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "k", "Method", "Execution_Time"])

# Liste des valeurs de k à tester
K_VALUES = list(range(5, 101, 5)) 

for gpickle_file in gpickle_files:
    dataset_name = os.path.splitext(gpickle_file)[0]
    input_path = os.path.join(DATA_DIR, gpickle_file).replace(os.sep, '/')
    
    # Charger le graphe networkx
    with open(input_path, 'rb') as f:
        G_original = pickle.load(f)
    P_original = [{'id': int(node), 'degree': G_original.degree[node]} for node in G_original.nodes()]
    P_original = sorted(P_original, key=lambda x: x['degree'], reverse=True)

    # Créer la matrice de poids CW pour le dataset
    cw_path = os.path.join(MATRIX_DIR, f"cw{dataset_name}.npy").replace(os.sep, '/')
    # CW_original = create_CW(G_original, len(G_original.nodes()), cw_path)
    CW_original = create_CW(G_original, cw_path)

    for k in K_VALUES:
        for pipeline in pipelines:
            method_name = pipeline["name"]
            cluster_func = pipeline["cluster_func"]
            reconstruct_func = pipeline["reconstruct_func"]
            
            print(f"\n=== Anonymisation de {dataset_name} avec k={k}, méthode={method_name} ===")
            start_time = time.time()

            try:
                # Copier le graphe original
                with open(input_path, 'rb') as f:
                    G = pickle.load(f)
                P = copy.deepcopy(P_original)
                CW = np.copy(CW_original)

                # Étape 1 : Clusterisation
                clusters, cluster_degrees, P_new = cluster_func(P, k)

                # Étape 2 : Reconstruction
                security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated = reconstruct_func(
                    clusters, cluster_degrees, CW, G
                )

                execution_time = time.time() - start_time

                # Sauvegarder le graphe anonymisé en format .pairs
                output_dir = os.path.join(SECGRAPH_OUTPUT_DIR, dataset_name, method_name).replace(os.sep, '/')
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_pairs = os.path.join(output_dir, f"{dataset_name}{method_name}{k}.pairs").replace(os.sep, '/')
                
                with open(output_pairs, 'w') as f:
                    for edge in G_updated.edges():
                        f.write(f"{edge[0]} {edge[1]}\n")
                
                # Enregistrer le temps d'exécution
                with open(TIME_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([dataset_name, k, method_name, execution_time])

                print(f"Anonymisation terminée : Graphe sauvegardé dans {output_pairs}")

            except Exception as e:
                print(f"Erreur pour k={k}, méthode={method_name} sur {dataset_name}: {e}")
                with open(TIME_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([dataset_name, k, method_name, "error"])

print(f"\nTemps d'exécution sauvegardés dans {TIME_FILE}")