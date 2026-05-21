#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import copy
import itertools
import random
import math
import time


def wasserstein_distance_discrete(v, u):
    
    values_v, counts_v = np.unique(v, return_counts=True)
    values_u, counts_u = np.unique(u, return_counts=True)
    
    all_values = np.union1d(values_v, values_u)
    counts_v_dict = dict(zip(values_v, counts_v))
    counts_u_dict = dict(zip(values_u, counts_u))
    counts_v = np.array([counts_v_dict.get(val, 0) for val in all_values])
    counts_u = np.array([counts_u_dict.get(val, 0) for val in all_values])
    
    prob_v = counts_v / np.sum(counts_v)
    prob_u = counts_u / np.sum(counts_u)
    # SciPy-free 1D EMD approximation on aligned discrete support.
    cdf_v = np.cumsum(prob_v)
    cdf_u = np.cumsum(prob_u)
    return np.sum(np.abs(cdf_v - cdf_u))

def WR(u,v,lam):

    ndu = np.array(u)
    ndv = np.array(v)
    L_cost=np.abs(ndu-ndv)
    L1 = wasserstein_distance_discrete(u,v) + 1/2*lam*np.sum(L_cost)
    return L1

def deg_list_generation(G0,G1,k,lam):
    '''
    Inputs:
        G0: initial graph
        G1: stage1 anonymity graph implemented by G1HI 
        k: k-degree anonymity level
        lam: = 0.01 
    Outputs:
        d2: anonymity degree sequence
        sorted_V1: index of sorted nodes
        sorted_d1: according degrees of sorted_V1
    '''
    sorted_d0_list = sorted(G0.degree(), key =lambda x:x[1], reverse = True)
    sorted_d0 = [y for x,y in sorted_d0_list] 
    
    sorted_d1_list = sorted(G1.degree(), key =lambda x:x[1], reverse = True)
    sorted_d1 = [y for x,y in sorted_d1_list] 
    sorted_V1 = [x for x,y in sorted_d1_list] 
    
    d0=deepcopy(sorted_d0)
    d1=deepcopy(sorted_d1)
    d2=[] 
    
    while len(d1)>=2*k:
        n=len(d1) 
        i=k
        WASs=[9999]
        while True:
            
            ave = round(np.mean(d1[:i]))
            temp_c = min(i+k,n)  
            d1_i=[ave]*i
            if i != temp_c:
                temp_ave = round(np.mean(d1[i:temp_c]))
                d1_i.extend([temp_ave]*(temp_c-i))
                d1_i.extend(d1[temp_c:])
            
            was = WR(d1_i,d0,lam)
            if was < WASs[-1]: 
                WASs.append(was)
                i+=1
                if i >= n-k+1: 
                    i = n
            else:  
                
                if i == n:
                    i = n-k
                else:
                    i -= 1
                break
        
        ave = round(np.mean(d1[:i]))
        d2.extend([ave]*i)
        d1 = d1[i:]  
        d0 = d0[i:]  
    ave = round(np.mean(d1))
    d2.extend([ave]*len(d1))
    return d2,sorted_V1,sorted_d1

def line_collective_influence(G, l):
    '''
    Inputs:
    G: stage 1 anonymity graph
    '''
    line_G = nx.line_graph(G)
    CI = defaultdict(float)
    for node in line_G.nodes():
        k_i_minus_1 = line_G.degree(node) - 1
        ball_boundary = nx.single_source_shortest_path_length(line_G, node, cutoff=l).keys()
        sum_k_j_minus_1 = sum(line_G.degree(n) - 1 for n in ball_boundary if n != node)
        CI[node] = k_i_minus_1 * sum_k_j_minus_1
    return CI

def edge_sig_CI(G1,l = 2):
    LCIs = line_collective_influence(G1, l) 
    sorted_LCIs = sorted(LCIs.items(), key=lambda x:x[1],reverse=False) 
    edge_significant = [e for e,v in sorted_LCIs]
    return edge_significant

def edge_sig_NC(G1):
    NCs = {}
    max_deg = max(list(G1.degree),key=lambda x:x[1])[1]
    for edge in G1.edges:
        gamma0=list(G1[edge[0]])
        gamma1=list(G1[edge[1]])
        temp = len(list(set(gamma0)|set(gamma1))) - len(list(set(gamma0)&set(gamma1)))
        nc = temp/(2*max_deg)
        NCs[edge] = nc
    sorted_NCs = sorted(NCs.items(), key=lambda x:x[1],reverse=False)
    edge_significant = [e for e,v in sorted_NCs]
    return edge_significant

def choose_edge(edge_significant, nei_uv):
    r = 0
    while r < len(edge_significant):
        p, q = edge_significant[r]
        if p not in nei_uv and q not in nei_uv:
            return p, q
        r += 1
    return None, None 

def choose_neighbor(edge_significant, u, nei_u):
    temp = []
    for v in nei_u:
        edge = tuple(sorted((u, v)))
        if edge in edge_significant:
            ind = edge_significant.index(edge)
            temp.append((edge, ind))
    if temp:
        select_edge = min(temp, key=lambda x: x[1])[0]
        return select_edge
    else:
        return None


def g_modified(g_pert,deg_pert,deg_anon,node_index,edge_sig_sort):
    '''
    Inputs:
        g_pert: stage 1 anonymity graph, which is modified in this stage;
        deg_pert: sorted_d1;
        deg_anon: k-degree anonymity sequence;
        node_index：sorted_V1；
        edge_sig_sort: edge_significant implemented by LCI or NC;
    Outputs:
        g: final anonymous graph
    '''
    g=deepcopy(g_pert)
    edge_significant = deepcopy(edge_sig_sort)
    deg_diff = list(np.array(deg_anon) - np.array(deg_pert)) 
    temp_change = list(zip(node_index,deg_diff)) 
    temp_change_non0 = [(n,v) for n,v in temp_change if v!=0] 
    change_node_deg = sorted(temp_change_non0, key=lambda x:abs(x[1]),reverse =True)
    change_node = [n for n,v in change_node_deg]
    change_deg = [v for n,v in change_node_deg]
    for i in range(len(change_node)):
        if change_deg[i] == 0:  
            continue
        j = i
        while abs(change_deg[i]) > 0:
            j+=1
            if j == len(change_node): 
                break
            if change_deg[j] == 0: 
                continue
            
            # 1. Case1: + +
            if change_deg[i] > 0 and change_deg[j] > 0:
                if change_node[j] not in list(g[change_node[i]]): 
                    g.add_edge(change_node[i],change_node[j]) 
                    change_deg[i] -= 1
                    change_deg[j] -= 1
                else:  
                    nei_ij = set(g[change_node[i]]).union(set(g[change_node[j]]))
                    p,q = choose_edge(edge_significant,nei_ij)
                    if (p,q) in g.edges():
                        g.remove_edge(p,q)
                        edge_significant.remove((p,q))
                        g.add_edges_from([(change_node[i],p),(change_node[j],q)])
                        change_deg[i] -= 1
                        change_deg[j] -= 1
                    else:
                        continue
            
            # 2. Case2: + -
            elif change_deg[i] > 0 and change_deg[j] < 0:
                nei_j = set(g[change_node[j]]).difference(set(g[change_node[i]]))
                select_edge = choose_neighbor(edge_significant,change_node[j],nei_j) 
                if select_edge:
                    g.remove_edge(select_edge[0],select_edge[1])
                    edge_significant.remove(select_edge) 
                    p=list(select_edge)
                    p.remove(change_node[j])
                    g.add_edge(p[0],change_node[i])
                    change_deg[i] -= 1
                    change_deg[j] += 1
                else:
                    continue
                
            # 3. Case3：- +
            elif change_deg[i] < 0 and change_deg[j] > 0:
                nei_i = set(g[change_node[i]]).difference(set(g[change_node[j]]))
                select_edge = choose_neighbor(edge_significant,change_node[i],nei_i)  
                if select_edge:
                    g.remove_edge(select_edge[0],select_edge[1])
                    edge_significant.remove(select_edge) 
                    p=list(select_edge)
                    p.remove(change_node[i])
                    g.add_edge(p[0],change_node[j])
                    change_deg[i] += 1
                    change_deg[j] -= 1
                else:
                    continue
                
            # 4. Case4：- -
            elif change_deg[i] < 0 and change_deg[j] < 0:
                if change_node[j] in list(g[change_node[i]]): 
                    g.remove_edge(change_node[i],change_node[j]) 
                    change_deg[i] += 1
                    change_deg[j] += 1
                else:
                    nei_i = set(g[change_node[i]]) 
                    select_edge = choose_neighbor(edge_significant,change_node[i],nei_i) 
                    if select_edge:
                        g.remove_edge(select_edge[0],select_edge[1]) 
                        edge_significant.remove(select_edge) 
                        p=list(select_edge)
                        p.remove(change_node[i])  
                        nei_j = set(g[change_node[j]]).difference(set(p)) 
                        select_edge2 = choose_neighbor(edge_significant,change_node[j],nei_j) 
                        if select_edge2:
                            g.remove_edge(select_edge2[0],select_edge2[1]) 
                            edge_significant.remove(select_edge2) 
                            q=list(select_edge2)
                            q.remove(change_node[j])
                            g.add_edge(p[0],q[0]) 
                            change_deg[i] += 1
                            change_deg[j] += 1
                        else:
                            continue
                    else:
                        continue
    remain_nd = [(n,v) for n,v in zip(change_node,change_deg) if v!=0]
    
    # 5. Case5：Only one node need to be midified
    if len(remain_nd)>0:
        for re_nd in remain_nd:
            remain_node = re_nd[0]
            remain_deg = re_nd[1]
            while remain_deg>=2:
                nei_node = set(g[remain_node])
                p,q = choose_edge(edge_significant,nei_node) 
                if (p,q) in g.edges():
                    g.remove_edge(p,q) 
                    edge_significant.remove((p,q)) 
                    g.add_edges_from([(remain_node,p),(remain_node,q)]) 
                remain_deg -= 2
            while remain_deg<=-2:
                nei_node = set(g[remain_node])
                rand_p = np.random.choice(list(nei_node))
                nei_temp = nei_node.difference(set(g[rand_p]))
                if len(nei_temp) == 0:
                    break
                rand_q = np.random.choice(list(nei_temp))
                g.remove_edges_from([(remain_node,rand_p),(remain_node,rand_q)]) 
                g.add_edge(rand_p,rand_q)
                remain_deg += 2
    return g   

def WRKDA_main(G0,G1,k,edge_significant,lam=0.01):
    
    d_anon,V_anon,d_pert = deg_list_generation(G0,G1,k,lam)   
    anon_G = g_modified(G1,d_pert,d_anon,V_anon,edge_significant) 
    return anon_G

def ASP_calculate(g):
    if nx.is_connected(g):
        average_shortest_path_length = nx.average_shortest_path_length(g)
    else:
        largest_cc = max(nx.connected_components(g), key=len)
        subgraph = g.subgraph(largest_cc)
        average_shortest_path_length = nx.average_shortest_path_length(subgraph)
    return average_shortest_path_length

def change_edges(g0,g1):
    edges_g0 = set(g0.edges())
    edges_g1 = set(g1.edges())
    red_edges = edges_g0 - edges_g1  #
    green_edges = edges_g1 - edges_g0  #
    return len(red_edges)+len(green_edges)

def verify_k_anonymity(G_anon, K):
    degree_sequence = [deg for _, deg in G_anon.degree()]
    freq = defaultdict(int)
    for deg in degree_sequence:
        freq[deg] += 1
    return all(count >= K for count in freq.values())
