from torch_geometric.datasets import Planetoid
import os

# Charger le dataset Cora depuis Planetoid
dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0]

# Extraire les arêtes sous forme de paires de nœuds
edges = data.edge_index.t().tolist()  # Liste de paires [node1, node2]

# Créer un fichier avec l'extension .pairs
output_file = 'pubmed.pairs'
with open(output_file, 'w') as f:
    for edge in edges:
        f.write(f"{edge[0]} {edge[1]}\n")  # Format PAIRS : deux entiers séparés par un espace

print(f"Fichier PAIRS généré : {output_file}")
print(f"Nombre d'arêtes : {len(edges)}")