import numpy as np
from typing import List, Dict, Tuple

def calculate_cluster_cost(cluster: List[Dict[int, int]]) -> Tuple[float, float, int, int]:
    if not cluster:
        return 0, 0, 0, 0
    degrees = [node['degree'] for node in cluster]
    mean_degree = round(sum(degrees) / len(degrees))
    median_degree = round(np.median(degrees))
    cost_mean = sum((degree - mean_degree)**2 for degree in degrees)
    cost_median = sum((degree - median_degree)**2 for degree in degrees)
    return cost_mean, cost_median, mean_degree, median_degree

def cluster_formation_RMSE(P: List[Dict[int, int]], k: int) -> Tuple[List[List[Dict[int, int]]], List[int], List[Dict[int, int]]]:
    """
    Regroupe les nœuds en clusters avec des degrés cibles associés à chaque cluster
    
    Args:
        P : Liste de dictionnaires {'id': int, 'degree': int} triés par degré décroissant.
        k : Taille minimale de chaque cluster.

    Returns:
        Tuple contenant :
        - Liste des clusters (chaque cluster est une liste de dictionnaires {'id', 'degree'}).
        - Liste des degrés associés à chaque cluster (moyenne ou médiane arrondie).
        - Liste de dictionnaires {'id', 'degree'} où chaque nœud a un degré ajusté.
    """
    # Vérification des paramètres
    if k < 1:
        raise ValueError("k doit être positif")
    if len(P) < k:
        raise ValueError("La séquence P doit contenir au moins k nœuds")

    # Initialisation des variables
    result = []  # Liste pour stocker la séquence anonymisée [{'id', 'degree'}]
    clusters = []  # Liste pour stocker les clusters
    cluster_degrees = []  # Liste pour stocker les degrés associés
    i = 0

    # Boucle principale pour former les clusters
    while i < len(P):
        # Prendre au moins k nœuds pour former un cluster
        cluster = P[i:i + k]
        if len(cluster) < k:
            # Si moins de k nœuds restent, les fusionner dans le dernier cluster
            cluster = P[i:]
            cost_mean, cost_median, mean_degree, median_degree = calculate_cluster_cost(cluster)
            # Choisir le degré qui minimise le coût
            target_degree = mean_degree if cost_mean < cost_median else median_degree
            clusters.append(cluster)
            cluster_degrees.append(target_degree)
            for node in cluster:
                result.append({'id': node['id'], 'degree': target_degree})
            break

        # Initialiser le degré cible pour le cluster
        cost_mean, cost_median, mean_degree, median_degree = calculate_cluster_cost(cluster)
        target_degree = mean_degree if cost_mean < cost_median else median_degree

        # Comparer les coûts pour étendre le cluster
        j = i + k
        while j < len(P):
            cost_mean, cost_median, _, _ = calculate_cluster_cost(cluster)
            current_cost = min(cost_mean, cost_median)

            # Option 1 : Créer un nouveau cluster avec les k nœuds suivants
            new_cluster = P[j:j + k] if j + k <= len(P) else []
            C_new = float('inf')
            if len(new_cluster) >= k:
                cost_mean_new, cost_median_new, _, _ = calculate_cluster_cost(new_cluster)
                C_new = current_cost + min(cost_mean_new, cost_median_new)

            # Option 2 : Fusionner le nœud suivant et créer un nouveau cluster
            C_merge = float('inf')
            if j + 1 < len(P):
                extended_cluster = cluster + [P[j]]
                next_new_cluster = P[j+1:j+1+k] if j+1+k <= len(P) else []
                if len(next_new_cluster) >= k:
                    cost_mean_ext, cost_median_ext, _, _ = calculate_cluster_cost(extended_cluster)
                    cost_mean_next, cost_median_next, _, _ = calculate_cluster_cost(next_new_cluster)
                    C_merge = min(cost_mean_ext, cost_median_ext) + min(cost_mean_next, cost_median_next)

            # Choisir l'option avec le coût le plus faible
            if C_merge < C_new:
                cluster = extended_cluster
                cost_mean, cost_median, mean_degree, median_degree = calculate_cluster_cost(cluster)
                target_degree = mean_degree if cost_mean < cost_median else median_degree
                j += 1
            else:
                break

        # Ajouter le cluster à la liste des clusters
        clusters.append(cluster)
        cluster_degrees.append(target_degree)

        # Ajouter les nœuds du cluster au résultat avec le degré choisi
        for node in cluster:
            result.append({'id': node['id'], 'degree': target_degree})

        i = j

    if len(clusters) > 1 and len(clusters[-1]) < k:
        last_cluster = clusters.pop()
        cluster_degrees.pop()
        clusters[-1].extend(last_cluster)
        cost_mean, cost_median, mean_degree, median_degree = calculate_cluster_cost(clusters[-1])
        cluster_degrees[-1] = mean_degree if cost_mean < cost_median else median_degree
        # Mettre à jour result
        result = []
        for cluster, target_degree in zip(clusters, cluster_degrees):
            for node in cluster:
                result.append({'id': node['id'], 'degree': target_degree})

    # Affichage des clusters
    """""
    print('Clusters formés :')
    for i, cluster in enumerate(clusters):
        cluster_int = f'Cluster {i + 1}: ['
        cluster_int += ' '.join(f"({node['id']},{node['degree']})" for node in cluster)
        cluster_int += ']'
        print(cluster_int)
    """""
    # Affichage des degrés des clusters
    #print('Degrés des clusters :')
    #print(cluster_degrees)
    #print(f"correspondance entre le nombre de cluste {len(clusters)} et le nombre de degre cible {len(cluster_degrees)}")

    return clusters, cluster_degrees, result