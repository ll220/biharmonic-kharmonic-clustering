from dendrogram_from_girvan_newman import sort_girvan_newman_partitions, agglomerative_matrix, girvan_newman_best_partition
from functools import partial
import networkx as nx
import numpy as np
import scipy
from typing import List, Optional, Set, Tuple
from k_harmonic_distance import max_k_harmonic_distance


def create_community_node_colors(graph, communities):
    """ Return a list of colors for a graph cluster structure """
    number_of_colors = len(communities[0])
    colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors

def get_labels_from_cluster(
    graph : nx.Graph, 
    clusters : List[Set]      
) -> np.array:
    labels = np.zeros(len(graph), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for node_idx in cluster:
            labels[node_idx] = cluster_idx
    return labels

def k_harmonic_girvan_newman_labels(
    graph : nx.Graph, 
    k : int,
    num_clusters : int    
) -> np.array:
    """ Generate the cluster labels of a graph using the k-harmonic Girvan-Newman algorithm """
    partial_max_k_harmonic_distance = partial(max_k_harmonic_distance, k=k)
    cluster_iter = nx.community.girvan_newman(graph, partial_max_k_harmonic_distance)
    for _ in range(num_clusters-1):
        clusters : List[Set] = next(cluster_iter)
    labels = get_labels_from_cluster(graph, clusters)
    return labels

def k_harmonic_girvan_newman_best_labels(
    graph : nx.Graph, 
    k : int,
    num_clusters : Optional[int] = None
) -> np.array:
    """ Generate the cluster labels of a graph using the k-harmonic Girvan-Newman algorithm """
    partial_max_k_harmonic_distance = partial(max_k_harmonic_distance, k=k)
    if num_clusters is None:
        clusters = list(nx.community.girvan_newman(graph, partial_max_k_harmonic_distance))
    else:
        clusters = []
        cluster_iter = nx.community.girvan_newman(graph, partial_max_k_harmonic_distance)
        for _ in range(num_clusters-1):
            clusters.append(next(cluster_iter))
    sorted_clusters = sort_girvan_newman_partitions(clusters)
    best_cluster, _ = girvan_newman_best_partition(graph, sorted_clusters)
    labels = get_labels_from_cluster(graph, best_cluster)
    return labels

def k_harmonic_girvan_newman_agglomerative_matrix(
    graph : nx.Graph, 
    k : int,
) -> np.array:
    """ Generate the aggolemerative matrix of a graph using the k-harmonic Girvan-Newman algorithm """
    partial_max_k_harmonic_distance = partial(max_k_harmonic_distance, k=k)
    clusters = list(nx.community.girvan_newman(graph, partial_max_k_harmonic_distance))
    sorted_clusters = sort_girvan_newman_partitions(clusters)
    agg = agglomerative_matrix(graph, sorted_clusters)
    return agg
    

if __name__=="__main__":
    from generate_graphs import load_iris_graph_and_labels
    from meh import get_purity
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    
    # num_clusters = 8
    harmonic_k = 2
    graph, true_labels = load_iris_graph_and_labels(num_neighbors=25)
    pred_labels = k_harmonic_girvan_newman_best_labels(graph, harmonic_k, num_clusters=10)
    print(pred_labels)
    purity = get_purity(pred_labels, true_labels)
    print(purity)
    # agg = k_harmonic_girvan_newman_agglomerative_matrix(graph, harmonic_k)
    # dendrogram(agg, labels=true_labels)
    plt.show()


    

    

