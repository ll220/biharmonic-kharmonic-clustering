import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
from typing import List, Tuple

def k_harmonic_betweenness_centrality(
    graph : nx.Graph, 
    k : int
) -> List[float]:
    """ Return the k-harmonic betweenness of each node in the graph 
    
    For a node v, the k-harmonic betweenness of v is 
    $$
        B^{(k)}_{v} = \sum_{u\in N(v)} (H^{(k})})^{2}_{uv}
    $$
    where $N(v)$ is the set of neighbors of v. 

    """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    k_pinv = fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    k_harmonic_node_centralities = [
        np.sum([
            k_harmonic_distance(u,v)
            for v in graph.neighbors(u)
        ])
        for u in graph.nodes
    ]
    return k_harmonic_node_centralities

def k_harmonic_closeness_centrality(
    graph : nx.Graph, 
    k : int
) -> List[float]:
    """ Return the k-harmonic closeness of each node in the graph 
    
    For a node v, the k-harmonic between of v is 
    $$
        C^{k}_{v} = (\sum_{u\in V} (H^{(k})})^{2}_{uv})^{-1}.
    $$ 
    
    """
    laplacian = nx.laplacian_matrix(graph).todense()
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    k_pinv = fractional_matrix_power(pinv, float(k))
    k_harmonic_distance = lambda s, t : k_pinv[s,s] + k_pinv[t,t] - 2*k_pinv[s,t]
    k_harmonic_node_centralities = [
        1/np.sum([
            k_harmonic_distance(u,v)
            for v in graph.nodes
        ])
        for u in graph.nodes
    ]
    return k_harmonic_node_centralities

if __name__=="__main__":
    graph = nx.path_graph(10)
    print(k_harmonic_closeness_centrality(graph, k=2))
