import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.neighbors import kneighbors_graph

def load_iris_graph_and_labels(num_neighbors : int) -> tuple[nx.Graph, np.ndarray[int]]:
    """ Return the nearest neighbor graph and cluster labels of the Iris dataset 
    
    Args:
        num_neighbors: number of neigbors to use for nearest neighbor graph

    Returns:
        A tuple containing:
            graph: the nearest neighbor graph
            labels: a Numpy array containing the cluster label for each node as an int,
                e.g. labels = [0,0,0,1,1,1].
    """
    data, labels = load_iris(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_wine_graph_and_labels(num_neighbors : int) -> tuple[nx.Graph, np.ndarray[int]]:
    """ Return the nearest neighbor graph and cluster labels of the Wine dataset 
    
    Args:
        num_neighbors: number of neigbors to use for nearest neighbor graph

    Returns:
        A tuple containing:
            graph: the nearest neighbor graph
            labels: a Numpy array containing the cluster label for each node as an int,
                e.g. labels = [0,0,0,1,1,1].
    """
    data, labels = load_wine(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_cancer_graph_and_labels(num_neighbors : int) -> tuple[nx.Graph, np.ndarray[int]]:
    """ Return the nearest neighbor graph and cluster labels of the Breast Cancer dataset 
    
    Args:
        num_neighbors: number of neigbors to use for nearest neighbor graph

    Returns:
        A tuple containing:
            graph: the nearest neighbor graph
            labels: a Numpy array containing the cluster label for each node as an int,
                e.g. labels = [0,0,0,1,1,1].
    """
    data, labels = load_breast_cancer(return_X_y=True)
    points = np.array(data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels

def load_uci_graph_and_labels(
    num_neighbors : int, 
    id : int
) -> tuple[nx.Graph, np.ndarray[int]]:
    """ Return the nearest neighbor graph and cluster labels of a dataset from the UCI repository
    
    Args:
        num_neighbors: number of neigbors to use for nearest neighbor graph
        id: the label of the 

    Returns:
        A tuple containing:
            graph: the nearest neighbor graph
            labels: a Numpy array containing the cluster label for each node as an int,
                e.g. labels = [0,0,0,1,1,1].
    """
    from ucimlrepo import fetch_ucirepo 
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features.to_numpy()
    y = dataset.data.targets.to_numpy()[:,0]
    adjacency_matrix = kneighbors_graph(X, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, y

def load_block_stochastic_graph_and_labels(
    num_nodes_per_cluster : int,
    num_clusters : int,
    intercluster_p : float,
    intracluster_p : float
) -> tuple[nx.Graph, np.ndarray[int]]:
    """ Return a block stochatsic graph
    
    args:
        num_nodes_per_cluster
        num_clusters
        intercluster_p  : probability of edge between nodes in different clusters
        intracluster_p : probability of edge between nodes in same cluster

    """
    cluster_sizes = num_clusters * [num_nodes_per_cluster]
    P = (intracluster_p-intercluster_p)*np.eye(num_clusters) + (intercluster_p)*np.ones((num_clusters, num_clusters))
    graph = nx.stochastic_block_model(cluster_sizes, P)
    labels = np.array([
        i 
        for i in range(num_clusters)
        for _ in range(num_nodes_per_cluster)
    ])
    return graph, labels

def degree_k_tree(k : int, depth : int) -> nx.Graph:
    """ Return the degree `k` tree with `depth` level """    
    def circle_position(r : float, theta : float):
        """ Return the Euclidean coordinates from the radial coordinates """
        return (r*np.cos(2*np.pi*theta), r*np.sin(2*np.pi*theta))
    def get_node_index(k : int, level : int, i : int) -> int:
        """ Return the global node index of the `i` node in `level` """
        return (1 - k**level)//(1-k) + i

    graph = nx.Graph()
    pos = {}
    pos[0] = (0, 0) # root is at origin
    for level in range(1, depth + 1):
        num_nodes_at_level = k**level
        for i in range(k**depth):
            node = get_node_index(k, level, i)
            parent = get_node_index(k, level-1, i // k)
            pos[node] = circle_position(r=level, theta=i/num_nodes_at_level+1/(2*num_nodes_at_level))
            graph.add_edge(parent, node)
    graph.pos = pos
    return graph

def grid_graph_2d(n : int):
    """ Return a square (`n` x `n`) grid graph """
    graph = nx.grid_graph([n,n])
    map = {
        (i,j) : i*n + j
        for i in range(n)
        for j in range(n)
    }
    pos = {
        i*n+j : (i,j)
        for i in range(n)
        for j in range(n)
    }
    graph = nx.convert_node_labels_to_integers(graph)
    graph.pos = pos
    return graph
