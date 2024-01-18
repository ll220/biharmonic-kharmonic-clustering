import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph

def load_iris_graph_and_labels(num_neighbors : int) -> nx.Graph:
    data = load_iris()
    labels = data.target
    points = np.array(data.data)
    adjacency_matrix = kneighbors_graph(points, num_neighbors, mode='connectivity', include_self=False)
    graph = nx.from_numpy_array(adjacency_matrix)
    return graph, labels


if __name__=="__main__":
    graph, labels = load_iris_graph_and_labels(num_neighbors=125)
    colors = ['red', 'blue', 'green']
    node_colors = [colors[cluster] for cluster in labels]
    nx.draw(graph, with_labels=True, node_color=node_colors)
    plt.title('Iris Generated Graph k=125')
    plt.savefig('Iris Generated Graph k=125')
    plt.show()

# sizes = [75, 75, 300]
# # probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]

# sizes = [50] * 10
# # probs = [[0.1, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.1]]

# random_seed = 42
# seed = np.random.seed(random_seed)
# g = nx.random_partition_graph(sizes, 1, 0.02, seed=seed)
# # g = nx.stochastic_block_model(sizes, probs)


# # guarantee completedness
# while not nx.is_connected(g):
#     component1, component2 = random.sample(list(nx.connected_components(g)), 2)
#     node1 = random.choice(list(component1))
#     node2 = random.choice(list(component2))
    
#     g.add_edge(node1, node2)

# H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
# fh = open("./ten_graphs/size_50/100_2.txt", "wb")
# nx.write_edgelist(g, fh, data=False)
# nx.draw(g, with_labels = True)
# nx.draw(H, with_labels = True)
# # plt.savefig("./graphs/size_fifty_pngs/10_point5.png")
# plt.show()
