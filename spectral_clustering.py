from generate_graphs import load_iris_graph_and_labels
from mapping_vertices import get_purity
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

# scaled by 1/eigenvalue takes projection to first three and scales on each eigenvector proportional to eigenvalue 
# how to argue against someone saying spectral clustering is around and does the same
# the big thing is that our embedding is independent of k number of clusters
def run_combined_k_harmonic_spectral_clustering(graph, k_harmonic):
    laplacian = nx.laplacian_matrix(graph)
    laplacian = laplacian.toarray()

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    first_k_eigenvectors = eigenvectors[:, 1:4]

    first_k_eigenvectors = first_k_eigenvectors.T

    for i in range(3):
        first_k_eigenvectors[i] = (1 / pow(eigenvalues[i + 1], k_harmonic)) * first_k_eigenvectors[i]

    first_k_eigenvectors = first_k_eigenvectors.T
    # first_k_eigenvectors = eigenvectors
    # first_k_eigenvectors = first_k_eigenvectors.T
    # print(first_k_eigenvectors)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(first_k_eigenvectors)

    return kmeans.labels_

def run_spectral_clustering(
    graph : nx.Graph,
    num_clusters : int
):
    laplacian = nx.laplacian_matrix(graph)
    laplacian = laplacian.toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    first_k_eigenvectors = eigenvectors[:, 1:num_clusters+1]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(first_k_eigenvectors)
    return kmeans.labels_

if __name__=="__main__":
    K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]

    # for k_neighbors in [25, 50, 75, 100, 125]:
    #     f = open("sc_k_harmonic_combined_" + str(k_neighbors) + "_neighbors.txt", "w")
    #     G, true_clusters = load_iris_graph_and_labels(num_neighbors=k_neighbors)
    #     mean_accuracies = []

    #     for k_harmonic in K_HARMONICS:
    #         accuracies = []
    #         for _ in range(10):
    #             results = run_combined_k_harmonic_spectral_clustering(G, k_harmonic)
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













# accuracies = []

# A = nx.adjacency_matrix(G)

# spectral_model_nn = SpectralClustering(n_clusters = 3, affinity ='nearest_neighbors')
# labels_nn = spectral_model_nn.fit_predict(A)

# print(get_purity(labels_nn, true_clusters))

# colors = ['red', 'blue', 'green']
# node_colors = [colors[cluster] for cluster in labels_nn]

# nx.draw(G, with_labels=True, node_color=node_colors)
# plt.title('Clustered Graph')        
# # plt.savefig("spectral clustering_" + str(k))
# plt.show()    
# plt.clf()

# f = open("spectral_clustering_purities.txt", "w")
# for k in [25, 50, 75, 100, 125]:
#     G, true_clusters = load_iris_graph_and_labels(num_neighbors=k)
#     # run_spectral_clustering_experiments(G, true_clusters)
#     accuracies = []

#     A = nx.adjacency_matrix(G)
#     for i in range(10):
#         spectral_model_nn = SpectralClustering(n_clusters = 3, affinity ='precomputed')
#         labels_nn = spectral_model_nn.fit_predict(A)

#         purity = get_purity(labels_nn, true_clusters)
#         accuracies.append(purity)

        
#         colors = ['red', 'blue', 'green']
#         node_colors = [colors[cluster] for cluster in labels_nn]

#         nx.draw(G, with_labels=True, node_color=node_colors)
#         plt.title('Clustered Graph')        
#         plt.savefig("spectral clustering_" + str(k))
#         # plt.show()    
#         plt.clf()

#     mean_accuracy = np.mean(accuracies)
#     f.write(str(k) + ":" + str(mean_accuracy) + "\n")


