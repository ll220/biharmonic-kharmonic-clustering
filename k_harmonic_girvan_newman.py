from dendrogram_from_girvan_newman import sort_girvan_newman_partitions, agglomerative_matrix, girvan_newman_best_partition
from functools import partial
import networkx as nx
import numpy as np
import scipy
from typing import List, Optional, Set, Tuple

def component_mask(graph : nx.Graph) -> np.array:
    """ Return the component mask of a graph

    The component mask of a graph is a matrix C
    where C[i,j] = 1 if vertices i and j are
    in the same connected component, and 0 otherwise.
    """
    components = [list(c) for c in nx.connected_components(graph)]
    component_indicators = np.zeros((len(graph), len(components))) # component_indicators[i,j] = 1 if vertex i in component j else 0
    for i, component in enumerate(components):
        component_indicators[component, i] = 1
    return component_indicators @ component_indicators.T

def max_k_harmonic_distance(graph : nx.Graph, k : int) -> Tuple[int,int]:
    """ Return the edge with the highest k-harmonic distance """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    k_pinv = scipy.linalg.fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    edges = list(graph.edges)
    k_harmonic_of_edges = [k_harmonic_distance(u,v) for u,v in edges]
    max_edge_idx = np.argmax(k_harmonic_of_edges)
    return edges[max_edge_idx]

def max_biharmonic_distance(graph : nx.Graph) -> Tuple[int,int]:
    """ Return the edge with the highest biharmonic distance """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    pinv_squared = pinv @ pinv
    biharmonic_distance = lambda s, t : pinv_squared[s,s] + pinv_squared[t,t] - 2*pinv_squared[s,t]
    edges = list(graph.edges)
    biharmonics_of_edges = [biharmonic_distance(u,v) for u,v in edges]
    max_edge_idx = np.argmax(biharmonics_of_edges)
    return edges[max_edge_idx]

def max_effective_resistance(graph : nx.Graph) -> Tuple[int,int]:
    """ Return the edge with the highest effective resistance """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    effective_resistance = lambda s, t : pinv[s,s] + pinv[t,t] - 2*pinv[s,t]
    edges = list(graph.edges)
    resistances_of_edges = [effective_resistance(u,v) for u,v in edges]
    max_edge_idx = np.argmax(resistances_of_edges)
    return edges[max_edge_idx]

def biharmonics_of_edges(graph : nx.Graph) -> List[float]:
    """ Return the edge with the highest biharmonic distance """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    pinv_squared = pinv @ pinv
    biharmonic_distance = lambda s, t : pinv_squared[s,s] + pinv_squared[t,t] - 2*pinv_squared[s,t]
    biharmonics_of_edges = [biharmonic_distance(u,v) for u,v in graph.edges]
    return biharmonics_of_edges

def k_harmonics_of_edges(graph : nx.Graph, k : int) -> Tuple[int,int]:
    """ Return the edge with the highest biharmonic distance """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    k_pinv = scipy.linalg.fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    k_harmonic_of_edges = [k_harmonic_distance(u,v) for u,v in graph.edges]
    return k_harmonic_of_edges

def total_resistance(graph : nx) -> float:
    """ Return the total resistance of a graph """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    return len(graph) * np.trace(pinv)

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
    """ Generate the cluster labels of a graph using the k-harmonic Girvan-Newman algorithm """
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


    

    

