from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import \
    edge_current_flow_betweenness_centrality, edge_betweenness_centrality
import numpy as np
from scipy.stats import spearmanr
import tqdm
from generate_graphs import load_iris_graph_and_labels, load_wine_graph_and_labels, load_cancer_graph_and_labels
from k_harmonic_distance import biharmonics_of_edges, k_harmonics_of_edges

fset = frozenset
def fset_key_dict(d : dict):
    return { fset(key) : value for key, value in d.items() }

def add_random_edge(graph : nx.Graph):
    # add a random edge not in the graph
    n = graph.number_of_nodes()
    while True: # loop until find a graph not in the graphs
        u, v = np.random.randint(n, size=2)
        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u,v)
            return

biharmonic_dict = {
    "label" : "Biharmonic",
    "func" : partial(biharmonics_of_edges, as_dict=True),
    "spearman" : np.zeros((5,11)),
}
current_flow_dict = {
    "label" : "Current Flow",
    "func" : edge_current_flow_betweenness_centrality,
    "spearman" : np.zeros((5,11)),
}
betweeness_dict = {
    "label" : "Betweenness",
    "func" : edge_betweenness_centrality,
    "spearman" : np.zeros((5,11)),
}
k_harmonic_dict = {
    "label" : "5-Harmonic",
    "func" : partial(k_harmonics_of_edges, k=5, as_dict=True),
    "spearman" : np.zeros((5,11)),
}
resistance_dict = {
    "label" : "Effective Resistance",
    "func" : partial(k_harmonics_of_edges, k=1, as_dict=True),
    "spearman" : np.zeros((5,11)),
}
centrality_dicts = [biharmonic_dict, k_harmonic_dict, current_flow_dict, betweeness_dict, resistance_dict]


dataset_name = "Cancer"
for nn in tqdm.tqdm([25, 50, 75, 100]):
    graph, _ =  load_cancer_graph_and_labels(num_neighbors=nn)
    # run hyperparameters and data
    edge_delta = 10
    num_edges = list(range(0, 110, edge_delta))
    edges_og = list(graph.edges())
    # compute the centrality measure for 
    for centrality_dict in centrality_dicts:
        centrality_dict["centrality_og"] = fset_key_dict(centrality_dict["func"](graph))
    # main experiment
    for i in tqdm.tqdm(range(5)): # repeat the experiment 5 times
        graph_copy = graph.copy()
        for j,_ in enumerate(num_edges): # iterate over blocks of edges
            for centrality_dict in centrality_dicts: # iterate over centalities measures
                # compute the centrality in the graph with edges added
                # and compare to centrality in original graph
                centrality_new = fset_key_dict(centrality_dict["func"](graph_copy))
                centrality_dict["spearman"][i,j] = spearmanr(
                    [centrality_dict["centrality_og"][fset(edge)] for edge in edges_og],
                    [centrality_new[fset(edge)] for edge in edges_og]
                )[0]
            # add random edges to the graphs 
            for _ in range(edge_delta):
                add_random_edge(graph_copy)
    # compute aggregate statistics for the experiment
    for centrality_dict in centrality_dicts:
        centrality_dict["spearman_mean"] = np.average(centrality_dict["spearman"], axis=0)
        centrality_dict["spearman_std"]  = np.std(centrality_dict["spearman"], axis=0)
        # print(f"{centrality_dict['label']} mean: {centrality_dict['spearman_mean']}")
        # print(f"{centrality_dict['label']} std: {centrality_dict['spearman_std']}")
    # plot and save the results of the experiments
    colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    for centrality_dict, color in zip(centrality_dicts, colors):
        ax = plt.gca()
        ax.plot(num_edges, centrality_dict["spearman_mean"], label=centrality_dict["label"], color=color)
        ax.fill_between(
            num_edges,
            centrality_dict["spearman_mean"]-centrality_dict["spearman_std"],
            centrality_dict["spearman_mean"]+centrality_dict["spearman_std"],
            color=color, 
            alpha=0.2
        )
    plt.title(rf"{dataset_name} ($nn={nn}$)")
    plt.xlabel("Number Edges Added")
    plt.ylabel("Spearman Correlation")
    plt.legend(loc="lower left")
    plt.savefig(f"{dataset_name}_nn{nn}_resilience.pdf")
    plt.show()
    plt.clf()


