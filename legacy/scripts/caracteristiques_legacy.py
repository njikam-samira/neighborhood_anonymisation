import os
import networkx as nx
from pathlib import Path

# Chemins
DATA_DIR = "data"

# Vérifier si le répertoire existe
if not os.path.exists(DATA_DIR):
    print(f"Le répertoire {DATA_DIR} n'existe pas.")
    exit()

# Parcourir tous les fichiers .pairs dans DATA_DIR
for file in os.listdir(DATA_DIR):
    if file.endswith(".pairs"):
        file_path = os.path.join(DATA_DIR, file).replace(os.sep, '/')
        print(f"\nAnalyse du fichier : {file_path}")

        # Charger le graphe à partir du fichier .pairs
        G = nx.Graph()
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    u, v = map(int, line.strip().split())
                    G.add_edge(u, v)
                except ValueError:
                    print(f"Erreur de format dans la ligne : {line.strip()}")
                    continue

        # Calculer les caractéristiques
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Afficher les résultats
        print(f"Nombre de nœuds : {num_nodes}")
        print(f"Nombre d'arêtes : {num_edges}")