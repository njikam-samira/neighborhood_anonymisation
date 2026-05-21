import numpy as np
import networkx as nx

# Calcul du poid minimum et le dégré moyen dans le graphe 
def calculate_network_stats(CW):
    non_nul_weights = CW[CW > 0]
    min_weight_global = np.min(non_nul_weights) if len(non_nul_weights) > 0 else 1
    avg_degree = np.mean([sum(row > 0) for row in CW])
    return min_weight_global, avg_degree

# Trouver le cluster dont le degré est le plus proche du dégré moyen
def find_target_cluster(cluster_degrees, avg_degree):
    avg_cluster = np.argmin([abs(deg - avg_degree) for deg in cluster_degrees])
    return cluster_degrees[avg_cluster], avg_cluster

# Ajustement des dégrés des noeuds 
def optimize_noisy_nodes(S, S_less, CW_updated, C_list_updated, G_updated, target_degree, target_cluster, cluster_degrees):
    noisy_nodes = []
    next_id = CW_updated.shape[0]  # Prochain ID disponible pour les nouveaux nœuds
    min_weight_global = np.min(CW_updated[CW_updated > 0]) if np.any(CW_updated > 0) else 1  # Poids minimum global
    
    while S > 0:
        if len(S_less) >= target_degree:
            # Créer un nœud bruyant
            noisy_node = {'id': next_id, 'degree': 0, 'cluster': target_cluster}  # Degré initial à 0
            #print("Création d'un nœud bruyant de degré cible:", target_degree)
            # Redimensionnement de la matrice de poids si nécessaire
            if next_id + 1 > CW_updated.shape[0]:  # Si le nouvel ID dépasse la taille actuelle
                new_size = next_id + 1
                CW_updated = np.pad(CW_updated, ((0, new_size - CW_updated.shape[0]), (0, new_size - CW_updated.shape[1])), mode='constant')
            # Ajouter le nœud bruyant à G
            G_updated.add_node(next_id)
            C_list_updated[target_cluster].append(noisy_node)
            
            j = 0
            while noisy_node['degree'] < target_degree:
                less_node = S_less[j]  # Prendre le j-ème nœud dans S_less
                # Ajouter l’arête dans CW_updated
                CW_updated[noisy_node['id'], less_node['id']] = min_weight_global
                CW_updated[less_node['id'], noisy_node['id']] = min_weight_global
                # Ajouter l’arête dans G_updated
                G_updated.add_edge(noisy_node['id'], less_node['id'])
                less_node['required_edge'] -= 1  # Décrémenter le besoin d’arêtes

                # Mettre à jour le degré dans C_list_updated
                for cluster in C_list_updated:
                    for node in cluster:
                        if node['id'] == less_node['id']:
                            node['degree'] += 1
                        if node['id'] == noisy_node['id']:
                            node['degree'] += 1
                j += 1

            noisy_nodes.append(noisy_node)

            # Supprimer les nœuds de S_less dont required_edge == 0
            S_less[:] = [node for node in S_less if node['required_edge'] > 0]
            # Mettre à jour S et passer au prochain nœud
            S -= target_degree
            next_id += 1
        else:
            break
    
    if S > 0:
        #print("S_less:", S_less)
        #print("Reste:", S)
        while S > 0:
            # Si S_less est vide, on ne peut plus ajouter de nœuds bruyants
            if len(S_less) == 0:
                #print("Erreur : S_less est vide, impossible de satisfaire les arêtes manquantes restantes.")
                break
            
            # Ajuster target_degree pour qu'il ne dépasse pas len(S_less)
            target_degree, target_cluster = find_target_cluster(cluster_degrees, len(S_less))
            target_degree = min(target_degree, len(S_less))  # Ne pas dépasser le nombre de nœuds disponibles
            
            noisy_node = {'id': next_id, 'degree': 0, 'cluster': target_cluster}
            #print("Création d'un nœud bruyant de degré cible:", target_degree)
            if next_id + 1 > CW_updated.shape[0]:
                new_size = next_id + 1
                CW_updated = np.pad(CW_updated, ((0, new_size - CW_updated.shape[0]), (0, new_size - CW_updated.shape[1])), mode='constant')
            G_updated.add_node(next_id)
            C_list_updated[target_cluster].append(noisy_node)
            
            j = 0
            while noisy_node['degree'] < target_degree and j < len(S_less):
                less_node = S_less[j]
                CW_updated[noisy_node['id'], less_node['id']] = min_weight_global
                CW_updated[less_node['id'], noisy_node['id']] = min_weight_global
                G_updated.add_edge(noisy_node['id'], less_node['id'])
                less_node['required_edge'] -= 1

                for cluster in C_list_updated:
                    for node in cluster:
                        if node['id'] == less_node['id']:
                            node['degree'] += 1
                        if node['id'] == noisy_node['id']:
                            node['degree'] += 1
                j += 1

            noisy_nodes.append(noisy_node)
            S_less[:] = [node for node in S_less if node['required_edge'] > 0]
            S -= min(target_degree, j)  # Réduire S du nombre d'arêtes effectivement ajoutées
            next_id += 1
    #print("S_less:", S_less)
    #print("Reste:", S)
    return CW_updated, C_list_updated, noisy_nodes, G_updated


def graph_reconstruction_optimize_mean(clusters, cluster_degrees, CW, G):
    """
    Reconstruit le graphe en ajustant les degrés des nœuds pour correspondre aux degrés cibles des clusters.

    Args:
        clusters: Liste des clusters (chaque cluster est une liste de tuples (id, degree)).
        cluster_degrees: Liste des degrés cibles associés à chaque cluster.
        CW: Matrice des poids des connexions (CW[i][j] = poids entre nœuds i et j).
        G: Graphe NetworkX non orienté (pour obtenir les degrés actuels).

    Returns:
        security_table, reconstructed_graph, CW_updated, C_list_updated_updated, noisy_nodes, G_updated
    """
    # Initialisation
    security_table = []  # Tableau des nœuds sécurisés
    reconstructed_graph = []  # Liste des arêtes et poids du graphe reconstruit
    CW_updated = CW.copy()  # Copie de la matrice des poids pour mise à jour
    G_updated = G.copy() 
    C_list_updated = [cluster.copy() for cluster in clusters]  # Copie de la liste des clusters pour mise à jour
    S_more = []  # Stockera les dictionnaires {'id', 'degree', 'parent'}
    S_less = []  # Stockera les dictionnaires {'id', 'degree', 'required_edge'}
    S = 0

    # Vérification initiale de la cohérence entre G et CW
    for node_id in G_updated.nodes():
        degree_G = G_updated.degree(node_id)
        degree_CW = np.count_nonzero(CW_updated[node_id])
        if degree_G != degree_CW:
            print(f"Incohérence initiale pour le nœud {node_id}: degré dans G = {degree_G}, degré dans CW = {degree_CW}")

    min_weight_global, avg_degree = calculate_network_stats(CW)
    target_degree, target_cluster = find_target_cluster(cluster_degrees, avg_degree)
    noisy_nodes = []  # Stockera les dictionnaires {'id', 'dedree', 'parent'}

    # Ensemble des IDs des nœuds des clusters déjà traités
    treated_nodes = set()

    # Parcourir chaque cluster dans C_list_updated
    for c, cluster in enumerate(C_list_updated):
        treated_nodes.update(node['id'] for node in cluster)
        D = cluster_degrees[c]  # Degré cible du cluster
        # Initialiser les listes less et more
        less = [node for node in cluster if node['degree'] < D]  # Nœuds avec degré inférieur
        more = [node for node in cluster if node['degree'] > D]  # Nœuds avec degré supérieur
        
        # Neutralisation des nœuds dans le cluster
        while less or more:
            #less = [node for node in cluster if node['degree'] < D]  # Nœuds avec degré inférieur
            #more = [node for node in cluster if node['degree'] > D]  # Nœuds avec degré supérieur
            if more:
                # Trouver le nœud avec le degré minimum dans 'more'
                min_more_idx = np.argmin([node['degree'] for node in more])
                node_more = more[min_more_idx]  # Nœud correspondant
                node_id = node_more['id']  # Identifiant du nœud

                if node_more['degree'] < D:
                    more.pop(min_more_idx)
                    less.append(node_more)

                elif node_more['degree'] == D:
                    more.pop(min_more_idx)
                else :
                    # Identifier l'arête connectée au poids minimum
                    weights = CW_updated[node_id].astype(float).copy()  # Convertir en float
                    weights[weights == 0] = np.inf  # Ignorer les connexions inexistantes

                    # Trier les indices par poids croissant
                    sorted_indices = np.argsort(weights)
                    edge_removed = False

                    # Chercher la première arête valide (extrémité non traitée)
                    for min_conn_idx in sorted_indices:
                        if weights[min_conn_idx] == np.inf:  # Plus d’arêtes disponibles
                            break
                        min_weight = weights[min_conn_idx]

                        # Supprimer l’arête dans les deux directions
                        CW_updated[node_id, min_conn_idx] = 0
                        CW_updated[min_conn_idx, node_id] = 0
                        # Supprimer l’arête dans G_updated
                        if G_updated.has_edge(node_id, min_conn_idx):
                            G_updated.remove_edge(node_id, min_conn_idx)

                        security_table.append({
                            'noeud1': node_id,
                            'poids': min_weight,
                            'noeud2': min_conn_idx,
                        })

                        node_more['degree'] -= 1
                        for cluster in C_list_updated:
                            for node in cluster:
                                if node['id'] == node_more['id']:
                                    node['degree'] = node_more['degree']
                                if node['id'] == min_conn_idx:
                                    node['degree'] -= 1

                        if node_more['degree'] == D:
                            more.pop(min_more_idx)
                        edge_removed = True
                        break

                    if not edge_removed:
                        print(f"Cluster {c}: Aucune arête supprimable pour {node_id} (toutes les extrémités sont dans des clusters traités)")
                        more.pop(min_more_idx)  # On retire quand même pour éviter une boucle infinie

            if less:
                # Trouver le nœud avec le degré minimum dans 'less'
                min_less_idx = np.argmin([node['degree'] for node in less])
                node_less = less[min_less_idx]  # Nœud correspondant
                node_id = node_less['id']  # Identifiant du nœud

                if node_less['degree'] > D:
                    less.pop(min_less_idx)
                    more.append(node_less)

                elif node_less['degree'] == D:
                    less.pop(min_less_idx)
                else:
                    neighbors = list(G_updated.neighbors(node_less['id']))
                    # Trouver une arête disponible dans la table de sécurité
                    if len(security_table) > 0:
                        edge_found = None
                        for i, edge in enumerate(security_table):
                            if edge['noeud2'] != node_less['id'] and edge['noeud2'] not in neighbors:
                                if edge['noeud2'] not in treated_nodes:
                                    edge_found = edge
                                    security_table.pop(i)
                                    break
                        
                        if edge_found is not None:
                            # Mise à jour des degrés
                            node_less['degree'] += 1

                            # Mettre à jour le degré dans C_list_updated_updated
                            for cluster in C_list_updated:
                                for node in cluster:
                                    if node['id'] == node_less['id']:
                                        node['degree'] = node_less['degree']
                                    if node['id'] == edge['noeud2']:
                                        node['degree'] += 1 

                            # Mise à jour de notre matrice de poids
                            CW_updated[node_less['id'], edge['noeud2']] = edge['poids']
                            CW_updated[edge['noeud2'], node_less['id']] = edge['poids']
                            G_updated.add_edge(node_less['id'], edge['noeud2'])

                            if node_less['degree'] == D:
                                less.pop(min_less_idx)  # Retirer node_less de 'less'
                        else:
                            #print("Aucune arête disponible dans la table de sécurité pour ce nœud")
                            if not more:
                                less.pop(min_less_idx)
                    else:
                        #print('Table de sécurité vide')
                        if not more:
                            break

    # Calculer S, S_less et S_more après neutralisation
    for c, cluster in enumerate(C_list_updated):
        D = cluster_degrees[c]
        for node in cluster:
            if node['degree'] < D:
                S += D - node['degree']
                node_less = {'id': node['id'], 'degree': node['degree'], 'required_edge': D - node['degree']}
                S_less.append(node_less)
            elif node['degree'] > D:
                S_more.append(node)
    """""
    print("Nombre d'arêtes manquantes:", S)
    print("S_less:", S_less)
    print("S_more:", S_more)
    """""
    # Ajouter des nœuds bruyants pour satisfaire les degrés manquants
    CW_updated, C_list_updated, noisy_nodes, G_updated = optimize_noisy_nodes(S, S_less, CW_updated, C_list_updated, G_updated, target_degree, target_cluster, cluster_degrees)

    # Vérification finale de la cohérence entre G_updated et CW_updated
    for node_id in G_updated.nodes():
        degree_G = G_updated.degree(node_id)
        degree_CW = np.count_nonzero(CW_updated[node_id])
        if degree_G != degree_CW:
            print(f"Incohérence finale pour le nœud {node_id}: degré dans G_updated = {degree_G}, degré dans CW_updated = {degree_CW}")

    # Vérification de la cohérence des degrés dans C_list_updated
    for c, cluster in enumerate(C_list_updated):
        for node in cluster:
            expected_degree = node['degree']
            actual_degree = np.count_nonzero(CW_updated[node['id']])
            degree_G = G_updated.degree(node['id'])
            if expected_degree != actual_degree:
                print(f"Incohérence pour le nœud {node['id']} dans le cluster {c}: "
                      f"degré attendu = {expected_degree}, degré réel (dans CW) = {actual_degree}")
            if expected_degree != degree_G:
                print(f"Incohérence pour le nœud {node['id']} dans le cluster {c}: "
                      f"degré attendu = {expected_degree}, degré dans G_updated = {degree_G}")
            else:
                #print(f"Nœud {node['id']} dans le cluster {c}: degré OK ({expected_degree})")
                continue
    
    """""
    print("Degré moyen :", avg_degree)
    print("Degré cible des noeuds bruyants :", target_degree)
    
    print("Nombre de nœuds:", CW_updated.shape[0])
    print("Nombre de nœuds dans G_updated:", len(G_updated.nodes()))
    print("Nombre d’arêtes dans G_updated:", G_updated.number_of_edges())
    nombre_noeuds = sum(len(cluster) for cluster in C_list_updated)
    print("Nombre de nœuds dans C_list:", nombre_noeuds)
    print("Nombre de nœuds bruyants optimisés:", len(noisy_nodes))
    print("Table de sécurité:", security_table)
    """""
    return security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated