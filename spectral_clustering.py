import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Optional

def run_low_rank_k_harmonic_clustering(
    graph, 
    k_harmonic : int,   
    num_clusters : int,
    rank : Optional[int] = None
):
    """ Run the low-rank k-harmonic spectral clustering algorithm on `graph`
    
    Args:
        graph: NetworkX graph
        k_harmonic: Power of the k-harmonic distance
        num_clusters: Number of clusters
        rank: rank for the low-rank k-harmonic distance
            If None, set rank = num_clusters
    """
    if rank is None:
        rank = num_clusters
    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    first_k_eigenvalues = eigenvalues[1:rank+1]
    first_k_eigenvectors = eigenvectors[:,1:rank+1]
    scaled_eigenvectors = \
        np.float_power(first_k_eigenvalues, -(k_harmonic/2)).reshape(1, rank) * first_k_eigenvectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(scaled_eigenvectors)
    return kmeans.labels_

def run_spectral_clustering(
    graph : nx.Graph,
    num_clusters : int
):
    """ Run the spectral clustering algorithm on a NetworkX graph """
    laplacian = nx.laplacian_matrix(graph).toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    first_k_eigenvectors = eigenvectors[:, 1:num_clusters+1]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(first_k_eigenvectors)
    return kmeans.labels_

if __name__=="__main__":
    from generate_graphs import load_iris_graph_and_labels
    from mapping_vertices import get_purity

    K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]

    # for k_neighbors in [25, 50, 75, 100, 125]:
    #     f = open("sc_k_harmonic_combined_" + str(k_neighbors) + "_neighbors.txt", "w")
    #     G, true_clusters = load_iris_graph_and_labels(num_neighbors=k_neighbors)
    #     mean_accuracies = []

    #     for k_harmonic in K_HARMONICS:
    #         accuracies = []
    #         for _ in range(10):
    #             results = run_low_rank_k_harmonic_clustering(G, k_harmonic, num_cluster=3)
    #             purity = get_purity(results, true_clusters)
    #             accuracies.append(purity)

    #         mean_accuracy = np.mean(accuracies)
    #         mean_accuracies.append(mean_accuracy)
    #         f.write("k harmonic=" + str(k_harmonic) + " : " + str(mean_accuracy) + "\n")

    #     plt.plot(K_HARMONICS, mean_accuracies)
    #     plt.title(str(k_neighbors) + " neighbors " + "k harmonic vs mean purity")
    #     plt.savefig(str(k_neighbors) + " neighbors " + "k harmonic vs mean purity")
    #     plt.clf()
    #     f.close()

    f = open("spectral_clustering_purities.txt", "w")
    for k_neighbors in [25, 50, 75, 100, 125]:
        G, true_clusters = load_iris_graph_and_labels(num_neighbors=k_neighbors)
        
        accuracies = []
        for i in range(10):
            labels = run_spectral_clustering(G, num_clusters=3)
            purity = get_purity(labels, true_clusters)
            accuracies.append(purity)

            if (i==9):
                colors = ['red', 'blue', 'green']
                node_colors = [colors[cluster] for cluster in labels]
                nx.draw(G, with_labels=True, node_color=node_colors)
                plt.title('Clustered Graph')        
                # plt.savefig("spectral clustering_" + str(k))
                plt.show()    
                plt.clf()

        mean_accuracy = np.mean(accuracies)
        f.write(str(k_neighbors) + ":" + str(mean_accuracy) + "\n")



