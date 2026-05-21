import os
import re
from pathlib import Path

# Chemins de base
UMGA_DATA_DIR = "umga/data"
SECGRAPH_OUTPUT_DIR = "secGraph/output"

# Parcourir tous les sous-dossiers dans umga/data/
for dataset_name in os.listdir(UMGA_DATA_DIR):
    dataset_dir = os.path.join(UMGA_DATA_DIR, dataset_name).replace(os.sep, '/')
    if not os.path.isdir(dataset_dir):
        continue
    
    # Créer le dossier correspondant dans secgraph/output/nomgraphe/umga/
    secgraph_dataset_dir = os.path.join(SECGRAPH_OUTPUT_DIR, dataset_name, "umga").replace(os.sep, '/')
    if not os.path.exists(secgraph_dataset_dir):
        os.makedirs(secgraph_dataset_dir)
    
    # Parcourir tous les fichiers dans umga/data/nomgraphe/
    for graph_file in os.listdir(dataset_dir):
        if not graph_file.endswith(".gml") or "-k" not in graph_file:
            continue  # Ignore les fichiers qui ne sont pas des graphes anonymes (ex. cora.gml original)
        
        # Extraire la valeur de k du nom du fichier (ex. cora-k5-G-NC.gml -> k=5)
        match = re.search(r'-k(\d+)-', graph_file)
        if not match:
            print(f"Impossible d'extraire k de {graph_file}. Passage au suivant.")
            continue
        k = int(match.group(1))
        
        # Chemin du fichier .gml d'entrée
        input_gml = os.path.join(dataset_dir, graph_file).replace(os.sep, '/')
        
        # Nom du fichier de sortie (ex. coraUMGA5.pairs)
        output_pairs = os.path.join(secgraph_dataset_dir, f"{dataset_name}UMGA{k}.pairs").replace(os.sep, '/')
        
        # Parser le fichier .gml pour extraire les arêtes
        edges = []
        current_edge = {}
        in_edge_block = False
        
        try:
            with open(input_gml, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Le fichier {input_gml} est vide.")
                    continue
                
                # Remettre le curseur au début pour parsing
                f.seek(0)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Détecter le début d'un edge
                    if line.startswith('edge') or 'edge [' in line:
                        in_edge_block = True
                        current_edge = {}
                    # Détecter la fin d'un edge
                    elif in_edge_block and ']' in line:
                        in_edge_block = False
                        if 'source' in current_edge and 'target' in current_edge:
                            edges.append((current_edge['source'], current_edge['target']))
                    # Extraire les champs dans un bloc edge
                    elif in_edge_block:
                        parts = re.split(r'\s+', line, 1)
                        if len(parts) >= 2:
                            key, value = parts[0], parts[1].strip()
                            if key in ['source', 'target']:
                                try:
                                    current_edge[key] = str(int(value))  # Convertir en entier puis en string pour consistance
                                except ValueError:
                                    current_edge[key] = value
        
            # Écrire les arêtes dans le fichier .pairs
            if edges:
                with open(output_pairs, 'w') as f:
                    for source, target in edges:
                        f.write(f"{source} {target}\n")
                print(f"Conversion terminée : {input_gml} -> {output_pairs}")
            else:
                print(f"Aucune arête à écrire pour {input_gml}. Fichier {output_pairs} non créé ou vide.")
        
        except Exception as e:
            print(f"Erreur lors de la conversion de {input_gml} en {output_pairs}: {e}")
            continue

print("Conversion de tous les graphes anonymes terminée.")