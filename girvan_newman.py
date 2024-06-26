from dendrogram_from_girvan_newman import sort_girvan_newman_partitions, agglomerative_matrix, girvan_newman_best_partition
import networkx as nx
import numpy as np
from typing import List, Optional, Set


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

def girvan_newman_labels(
    graph : nx.Graph, 
    num_clusters : int    
) -> np.ndarray:
    """ Generate the cluster labels of a graph using the k-harmonic Girvan-Newman algorithm """
    cluster_iter = nx.community.girvan_newman(graph)
    for _ in range(num_clusters-1):
        clusters : List[Set] = next(cluster_iter)
    labels = get_labels_from_cluster(graph, clusters)
    return labels

def girvan_newman_agglomerative_matrix(
    graph : nx.Graph, 
    k : int,
) -> np.ndarray:
    """ Generate the aggolemerative matrix of a graph using the k-harmonic Girvan-Newman algorithm """
    clusters = list(nx.community.girvan_newman(graph))
    sorted_clusters = sort_girvan_newman_partitions(clusters)
    agg = agglomerative_matrix(graph, sorted_clusters)
    return agg
    


    

    

