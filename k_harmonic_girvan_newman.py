from dendrogram_from_girvan_newman import sort_girvan_newman_partitions, agglomerative_matrix, girvan_newman_best_partition
from functools import partial
import networkx as nx
import numpy as np
from typing import List, Optional, Set
from k_harmonic_distance import max_k_harmonic_distance

def get_labels_from_cluster(
    graph : nx.Graph, 
    clusters : List[Set]      
) -> np.ndarray:
    """ Convert a list of clusters into a list of labels 
    
    Args:
        graph: NetworkX graph
        clusters: A list of sets of nodes of `graph`
    Returns:
        A list of cluster labels for each node in the graph
    
    """
    labels = np.zeros(len(graph), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for node_idx in cluster:
            labels[node_idx] = cluster_idx
    return labels

def k_harmonic_girvan_newman_labels(
    graph : nx.Graph, 
    k : int,
    num_clusters : int    
) -> np.ndarray:
    """ Generate the cluster labels of a graph using the k-harmonic Girvan-Newman algorithm """
    partial_max_k_harmonic_distance = partial(max_k_harmonic_distance, k=k)
    cluster_iter = nx.community.girvan_newman(graph, partial_max_k_harmonic_distance)
    for _ in range(num_clusters-1): 
        # Each iteration of Girvan-Newman removes edges until creates a new connected component. 
        # Loop until the graph has `num_clusters` connected components.
        clusters : List[Set] = next(cluster_iter) 
    labels : np.array = get_labels_from_cluster(graph, clusters)
    return labels

def k_harmonic_girvan_newman_agglomerative_matrix(
    graph : nx.Graph, 
    k : int,
) -> np.ndarray:
    """ Generate the aggolemerative matrix of a graph using the k-harmonic Girvan-Newman algorithm """
    partial_max_k_harmonic_distance = partial(max_k_harmonic_distance, k=k)
    clusters = list(nx.community.girvan_newman(graph, partial_max_k_harmonic_distance))
    sorted_clusters = sort_girvan_newman_partitions(clusters)
    agg = agglomerative_matrix(graph, sorted_clusters)
    return agg


    

    

