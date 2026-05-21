import numpy as np
import networkx as nx

def optimize_noisy_nodes(S, S_less, C_list_degree, CW_updated, C_list_updated, G_updated):
    noisy_nodes = []
    next_id = CW_updated.shape[0]  # Prochain ID disponible pour les nouveaux nœuds
    min_weight_global = np.min(CW_updated[CW_updated > 0]) if np.any(CW_updated > 0) else 1  # Poids minimum global

    i = 0  # On commence avec le premier degré (le plus grand si trié)
    while S > 0:
        if i >= len(C_list_degree):  # Si on dépasse la liste des degrés
            #print("Plus de degrés disponibles dans C_list_degree. Arrêt.")
            break
        #print('S_less:', S_less)
        if C_list_degree[i] <= S and len(S_less) >= C_list_degree[i]:
            # Créer un nœud bruyant
            noisy_node = {'id': next_id, 'degree': 0, 'cluster': i}  # Degré initial à 0
            #print("Création d'un nœud bruyant de degré cible:", C_list_degree[i])
            # Redimensionnement de la matrice de poids si nécessaire
            if next_id + 1 > CW_updated.shape[0]:  # Si le nouvel ID dépasse la taille actuelle
                new_size = next_id + 1
                CW_updated = np.pad(CW_updated, ((0, new_size - CW_updated.shape[0]), (0, new_size - CW_updated.shape[1])), mode='constant')
            # Ajouter le nœud bruyant à G
            G_updated.add_node(next_id)
            C_list_updated[i].append(noisy_node)
            # Connecter le nœud bruyant à C_list_degree[i] nœuds dans S_less
            for j in range(C_list_degree[i]):
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

            noisy_nodes.append(noisy_node)

            # Supprimer les nœuds de S_less dont required_edge == 0
            S_less[:] = [node for node in S_less if node['required_edge'] > 0]
            # Mettre à jour S et passer au prochain nœud
            S -= C_list_degree[i]
            next_id += 1
        else:
            i += 1  # Le degré était trop grand, on passe au suivant

    #print("Reste:", S)
    return CW_updated, C_list_updated, noisy_nodes, G_updated


def graph_reconstruction(clusters, cluster_degrees, CW, G):
    """
    Reconstruit le graphe en ajustant les degrés des nœuds pour correspondre aux degrés cibles des clusters.

    Args:
        clusters: Liste des clusters (chaque cluster est une liste de tuples (id, degree)).
        cluster_degrees: Liste des degrés cibles associés à chaque cluster.
        CW: Matrice des poids des connexions (CW[i][j] = poids entre nœuds i et j).
        G: Graphe NetworkX non orienté (pour obtenir les degrés actuels).

    Returns:
        security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated
    """
    # Initialisation
    security_table = []  # Tableau des nœuds sécurisés
    reconstructed_graph = []  # Liste des arêtes et poids du graphe reconstruit
    CW_updated = CW.copy()  # Copie de la matrice des poids pour mise à jour
    G_updated = G.copy()  # Copie de G pour mise à jour
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
                else:
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

                        # Supprimer l’arête dans les deux directions dans CW_updated
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
                        more.pop(min_more_idx)  # On retire pour éviter une boucle infinie

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

                            # Mettre à jour le degré dans C_list_updated
                            for cluster in C_list_updated:
                                for node in cluster:
                                    if node['id'] == node_less['id']:
                                        node['degree'] = node_less['degree']
                                    if node['id'] == edge['noeud2']:
                                        node['degree'] += 1
                                    

                            # Mise à jour de notre matrice de poids
                            CW_updated[node_less['id'], edge['noeud2']] = edge['poids']
                            CW_updated[edge['noeud2'], node_less['id']] = edge['poids']
                            # Ajouter l’arête dans G_updated
                            G_updated.add_edge(node_less['id'], edge['noeud2'])

                            if node_less['degree'] == D:
                                less.pop(min_less_idx)  # Retirer node_less de 'less'
                        else:
                            print("Aucune arête disponible dans la table de sécurité pour ce nœud")
                            if not more:
                                less.pop(min_less_idx)
                    else:
                        print('Table de sécurité vide')
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

    print("Nombre d'arêtes manquantes:", S)
    print("S_less:", S_less)
    print("S_more:", S_more)
    """""
    # Affichage des clusters à jour
    print('Clusters formés :')
    for i, cluster in enumerate(C_list_updated):
        cluster_str = f'Cluster {i + 1}: ['
        cluster_str += ' '.join(f'({node["id"]},{node["degree"]})' for node in cluster)
        cluster_str += f': {cluster_degrees[i]}'
        cluster_str += ']'
        print(cluster_str)
    """""
    # Ajouter des nœuds bruyants pour satisfaire les degrés manquants
    CW_updated, C_list_updated, noisy_nodes, G_updated = optimize_noisy_nodes(S, S_less, cluster_degrees, CW_updated, C_list_updated, G_updated)

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
    print("Nombre de nœuds:", CW_updated.shape[0])
    print("Nombre de nœuds dans G_updated:", len(G_updated.nodes()))
    print("Nombre d’arêtes dans G_updated:", G_updated.number_of_edges())
    nombre_noeuds = sum(len(cluster) for cluster in C_list_updated)
    print("Nombre de nœuds dans C_list:", nombre_noeuds)
    print("Nombre de nœuds bruyants optimisés:", len(noisy_nodes))
    print("Table de sécurité:", security_table)
    """""
    return security_table, reconstructed_graph, CW_updated, C_list_updated, noisy_nodes, G_updated