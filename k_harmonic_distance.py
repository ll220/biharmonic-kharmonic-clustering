import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
from typing import List, Tuple

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
    k_pinv = fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    k_harmonic_of_edges = [k_harmonic_distance(u,v) for u,v in graph.edges]
    return k_harmonic_of_edges

def total_resistance(graph : nx) -> float:
    """ Return the total resistance of a graph """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    return len(graph) * np.trace(pinv)

