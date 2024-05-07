import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans

def get_k_harmonic_position_encoding(
    G : nx.Graph,
    k : int
) -> np.array:
    """ Compute the k-harmonic positional encodings of a graph G 
    
    The k-harmonic position encoding are points p_v such that 
    the Euclidean distance between points p_u and p_v is the 
    k-harmonic distance between vertices u and v.
    
    Args:
        graph: NetworkX Graph
        k: Power of the k-harmonic distance

    Returns:
        Numpy array with row v
            corresponding to positional encoding of vertex v

    """
    laplacian = nx.laplacian_matrix(G)
    laplacian = laplacian.toarray()
    laplacian_inverse = np.linalg.pinv(laplacian, hermitian=True)
    position_encoding = fractional_matrix_power(laplacian_inverse, float(k / 2))
    position_encoding = position_encoding.real
    position_encoding = position_encoding.transpose()
    return position_encoding

def k_harmonic_k_means(
    G : nx.Graph,
    k : int,
    num_clusters : int
) -> KMeans:
    """ Compute the k-harmonic k-means clustering of a graph G """
    position_encoding = get_k_harmonic_position_encoding(G, k)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(position_encoding)
    return kmeans

def k_harmonic_k_means_labels(
    G : nx.Graph,
    k : int,
    num_clusters : int
) -> np.array:
    """ Compute the k-harmonic k-means clustering of a graph G """
    position_encoding = get_k_harmonic_position_encoding(G, k)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(position_encoding)
    return kmeans.labels_

if __name__=="__main__":
    from generate_graphs import load_iris_graph_and_labels
    from meh import get_purity
    
    num_clusters = 3
    harmonic_k = 5
    graph, true_labels = load_iris_graph_and_labels(num_neighbors=125)
    pred_labels = k_harmonic_k_means_labels(graph, harmonic_k, num_clusters)
    print(get_purity(pred_labels, true_labels, num_clusters))