import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
from typing import Dict, List, Tuple

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
    k_pinv = fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    edges = list(graph.edges)
    k_harmonic_of_edges = [k_harmonic_distance(u,v) for u,v in edges]
    max_edge_idx = np.argmax(k_harmonic_of_edges)
    return edges[max_edge_idx]

def max_biharmonic_distance(graph : nx.Graph) -> Tuple[int,int]:
    """ Return the edge with the highest biharmonic distance """
    return max_k_harmonic_distance(graph, k=2)

def max_effective_resistance(graph : nx.Graph) -> Tuple[int,int]:
    """ Return the edge with the highest effective resistance """
    return max_k_harmonic_distance(graph, k=1)

def biharmonics_of_edges(graph : nx.Graph, as_dict : bool=False):
    """ Return the biharmonic distance of all edges in the graph 
    
    Args:
        graph: NetworkX graph
        as_dict: If True, return the biharmonic distances as a dict of the form { (int, int) : float}.
            If False, return the biharmonic distances as a list (in the same order as graph.edges)
    """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    pinv_squared = pinv @ pinv
    biharmonic_distance = lambda s, t : pinv_squared[s,s] + pinv_squared[t,t] - 2*pinv_squared[s,t]
    if as_dict:
        biharmonics_of_edges = { (u,v) : biharmonic_distance(u,v) for u,v in graph.edges }
    else:
        biharmonics_of_edges = [ biharmonic_distance(u,v) for u,v in graph.edges ]
    return biharmonics_of_edges

def k_harmonics_of_edges(graph : nx.Graph, k : int, as_dict : bool=False) -> Tuple[int,int]:
    """ Return the k-harmonic distance of all edges in the graph 
    
    Args:
        graph: NetworkX graph
        k: power of the k-harmonic distance
        as_dict: If True, return the biharmonic distances as a dict of the form { (int, int) : float}.
            If False, return the biharmonic distances as a list (in the same order as graph.edges)
    """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    k_pinv = fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    if as_dict:
        k_harmonic_of_edges = {(u,v) : k_harmonic_distance(u,v) for u,v in graph.edges}
    else:
        k_harmonic_of_edges = [k_harmonic_distance(u,v) for u,v in graph.edges]
    return k_harmonic_of_edges

def total_resistance(graph : nx) -> float:
    """ Return the total resistance of a connected graph.
    
    Caution: This method assumes the graph is connected.
    For disconnected graphs, this function is incorrect.

    """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    return len(graph) * np.trace(pinv)

