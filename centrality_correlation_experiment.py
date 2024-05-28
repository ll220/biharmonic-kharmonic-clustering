from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import \
    edge_current_flow_betweenness_centrality, edge_betweenness_centrality
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ConfusionMatrixDisplay
from time import time
import tqdm
from generate_graphs import load_iris_graph_and_labels, load_wine_graph_and_labels, load_cancer_graph_and_labels
from k_harmonic_distance import biharmonics_of_edges, k_harmonics_of_edges

fset = frozenset
def fset_key_dict(d : dict):
    return { fset(key) : value for key, value in d.items() }

def add_random_edge(graph : nx.Graph):
    n = graph.number_of_nodes()
    while True:
        u, v = np.random.randint(n, size=2)
        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u,v)
            return

biharmonic_dict = {
    "label" : "Biharmonic",
    "func" : partial(biharmonics_of_edges, as_dict=True),
}
current_flow_dict = {
    "label" : "Current Flow",
    "func" : edge_current_flow_betweenness_centrality,
}
betweeness_dict = {
    "label" : "Betweenness",
    "func" : edge_betweenness_centrality,
}
k_harmonic_dict = {
    "label" : "5-Harmonic",
    "func" : partial(k_harmonics_of_edges, k=5, as_dict=True),
}
resistance_dict = {
    "label" : "Effective Resistance",
    "func" : partial(k_harmonics_of_edges, k=1, as_dict=True),
}
centrality_dicts = [biharmonic_dict, k_harmonic_dict, current_flow_dict, betweeness_dict, resistance_dict]

dataset_name = "Cancer"
for nn in tqdm.tqdm([25, 50, 75, 100]):
    graph, _ =  load_cancer_graph_and_labels(num_neighbors=nn)
    edges_og = list(graph.edges())
    # compute each centrality measure
    for centrality_dict in centrality_dicts:
        centrality_dict["centrality_og"] = fset_key_dict(centrality_dict["func"](graph))
    # pairwise compare all centrality measures and store them in a matrix
    confusion_matrix = np.zeros((5,5))
    for i, centrality_dict_1 in enumerate(centrality_dicts):
        for j, centrality_dict_2 in enumerate(centrality_dicts):
            confusion_matrix[i,j] = spearmanr(
                [centrality_dict_1["centrality_og"][fset(edge)] for edge in edges_og],
                [centrality_dict_2["centrality_og"][fset(edge)] for edge in edges_og]
            )[0]
    # display and save the confusion matrix
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix, 
        display_labels=["BH", "5H", "CF", "B", "ER"]
    )
    cm_display.plot(
        xticks_rotation="vertical",
        cmap = mpl.cm.viridis
    )
    plt.gca().images[-1].colorbar.set_label("Spearman Correlation")
    plt.title(rf"{dataset_name} ($nn$={nn})")
    plt.gca().xaxis.label.set_visible(False)
    plt.gca().yaxis.label.set_visible(False)
    plt.savefig(f"{dataset_name}_nn{nn}_confusion_matrix.pdf", transparent=True)
    plt.show()
    plt.clf()



