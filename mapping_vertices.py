import random
random.seed(246)        # or any integer
import numpy as np
# np.random.seed(4812)
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys
from typing import Optional
import os

from k_harmonic_k_means import get_k_harmonic_position_encoding, k_harmonic_k_means, k_harmonic_k_means_labels
from k_harmonic_girvan_newman import k_harmonic_girvan_newman_labels, k_harmonic_girvan_newman_best_labels
from generate_graphs import load_iris_graph_and_labels


# CANNOT BE CHANGED WITHOUT CHANGING CODE
NUM_CLUSTERS = 3

K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]

def calculate_variance(cluster_labels, cluster_centroids, position_encoding):
    mean_distances = []
    max_distances = []

    for i in range(NUM_CLUSTERS):
        cluster_points = position_encoding[:,cluster_labels == i]  # Data points in the i-th cluster
        centroid = cluster_centroids[i]  # Centroid of the i-th cluster
 
        # print("centroid: ", centroid)
        # print("mean: ", np.mean(cluster_points, axis=1))

        distance_vectors = cluster_points.T - centroid.T
        # print(cluster_points)
        # print(centroid)
        # print(distance_vectors)
        distances = np.linalg.norm(distance_vectors, axis=1)
        # print(distances)
        # print(distances.shape)

        mean_distance = np.mean(distances)
        max_distance = np.amax(distances)

        mean_distances.append(mean_distance)
        max_distances.append(max_distance)

    return mean_distances, max_distances

def get_ideal_centroids(position_encoding, cluster_labels):
    cluster_centroids = np.zeros((NUM_CLUSTERS, position_encoding.shape[0]))

    for i in range(NUM_CLUSTERS):
        cluster_points = position_encoding[:,cluster_labels == i]  # Data points in the i-th cluster

        cluster_centroids[i] = np.mean(cluster_points, axis=1)

    return cluster_centroids

# def get_k_harmonic_distances(laplacian_inverse, nodes, n):
#     print(laplacian_inverse)
#     distances = np.zeros((n, n))

#     for u in range(n):
#         for v in range(u):
#             vector = np.zeros((n, 1))
#             vector[u][0] = 1
#             vector[v][0] = -1

#             intermediate = np.copy(laplacian_inverse)

#             for i in range(K-1):
#                 intermediate = np.matmul(intermediate, laplacian_inverse)

#             distance = np.matmul(vector.transpose(), intermediate)
#             distance = (np.matmul(distance, vector))[0][0]
#             distance = np.sqrt(distance)
#             print("u:", u+1, " v:", v+1, " distance:", distance)

#             distances[v][u] = distance
#     return distances

# def get_harmonic_distances(position_matrix, n):
#     print(position_matrix)
#     distances = np.zeros((n, n))

#     for u in range(n):
#         u_position = position_matrix[:,u]
#         for v in range(u):
#             v_position = position_matrix[:,v]

#             print(u_position)
#             print(v_position)

#             distance = np.linalg.norm(u_position - v_position)
#             print("u:", u+1, " v:", v+1, " distance:", distance)
#             distances[v][u] = distance
#     return distances

def get_expected_clustering(nodes):
    expected_labels = []
    for cluster in range(NUM_CLUSTERS):
        cluster = [cluster for x in range(0, int(len(nodes) / NUM_CLUSTERS))]
        expected_labels.extend(cluster)

    return np.array(expected_labels)

def get_triangle_lengths(cluster_centers):
    lengths = []

    lengths.append((np.linalg.norm(cluster_centers[1] - cluster_centers[0]), (0, 1)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[1]), (1, 2)))
    lengths.append((np.linalg.norm(cluster_centers[2] - cluster_centers[0]), (0, 2)))
    lengths.sort()

    normalized_lengths = []
    for i in range(3):
        not_normalized = lengths[i]
        normalized_lengths.append((not_normalized[0] / lengths[2][0], not_normalized[1]))

    return lengths, normalized_lengths


def plot_triangle(sorted_triangle_lengths, output_file_name, variances=None):

    # use law of cosines
    try:
        B = np.arccos((sorted_triangle_lengths[0][0]**2 + sorted_triangle_lengths[2][0]**2 - sorted_triangle_lengths[1][0]**2) / (2 * sorted_triangle_lengths[0][0] * sorted_triangle_lengths[2][0]))
    except:
        return
    
    P0 = (0, 0)
    P1 = (sorted_triangle_lengths[0][0], 0)
    P2 = (sorted_triangle_lengths[2][0] * np.cos(B), sorted_triangle_lengths[2][0] * np.sin(B))

    x_vertices = [P0[0], P1[0], P2[0], P0[0]]
    y_vertices = [P0[1], P1[1], P2[1], P0[1]]

    ax = plt.gca()

    ax.plot(x_vertices, y_vertices)

    if variances is not None:
        first_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()
        second_vertex_index = (set(sorted_triangle_lengths[0][1])).intersection(set(sorted_triangle_lengths[1][1])).pop()
        third_vertex_index = (set(sorted_triangle_lengths[1][1])).intersection(set(sorted_triangle_lengths[2][1])).pop()

        variance_1 = plt.Circle(P0, variances[first_vertex_index], fill=False)
        variance_2 = plt.Circle(P1, variances[second_vertex_index], fill=False)
        variance_3 = plt.Circle(P2, variances[third_vertex_index], fill=False)

        ax.add_patch(variance_1)
        ax.add_patch(variance_2)
        ax.add_patch(variance_3)
        ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio

    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title("Centers Triangle")
    plt.savefig(output_file_name)
    # plt.show()
    plt.clf()

def plot_k_vs_random_score_accuracy(k_harmonics, mean_accuracies, max_accuracies, mean_output_file_name, max_output_file_name):
    plt.plot(k_harmonics, mean_accuracies)
    plt.title("k vs mean adjusted random score over 10 trials")
    plt.savefig(mean_output_file_name)
    # plt.show()
    plt.clf()

    plt.plot(k_harmonics, max_accuracies)
    plt.title("k vs max adjusted random score over 10 trials")
    plt.savefig(max_output_file_name)
    plt.clf()

def plot_k_vs_purity_accuracy(k_harmonics, mean_accuracies, max_accuracies, mean_output_file_name, max_output_file_name):
    plt.plot(k_harmonics, mean_accuracies)
    plt.title("k vs mean purity over 10 trials")
    plt.savefig(mean_output_file_name)
    # plt.show()
    plt.clf()

    plt.plot(k_harmonics, max_accuracies)
    plt.title("k vs max purity over 10 trials")
    plt.savefig(max_output_file_name)
    plt.clf()

def get_purity(clustering_results, true_labels):
    num_correct = 0
    for cluster in np.unique(clustering_results):
        ground_truth = true_labels[clustering_results == cluster]
        majority_label = np.bincount(ground_truth).argmax()
        num_correct += np.count_nonzero(ground_truth == majority_label)
    purity = num_correct / clustering_results.shape[0]
    return purity


# def analysis(input_file, output_directory):
def k_means_analysis():
    # print("Now analyzing: ", input_file)
    # # assumption that all vertices are encoded as 0-n consecutive integers
    # f = open(input_file, "r")
    # file_string = f.read()
    # f.close()
    # edges = file_string.split('\n')

    # unordered_g = nx.parse_edgelist(edges, nodetype=int)

    # G = nx.Graph()
    # G.add_nodes_from(sorted(unordered_g.nodes(data=True)))
    # G.add_edges_from(unordered_g.edges(data=True))

    f = open("125_nn_results.txt", "w")

    G, true_clusters = load_iris_graph_and_labels(num_neighbors=125)
    # nodes = list(G.nodes)
    # print(nodes)

    if not nx.is_connected(G):
        print("geh")
        quit()

    # true_clusters = get_expected_clustering(nodes)
    mean_rand_score_accuracies = []
    max_rand_score_accuracies = []
    mean_purities = []
    max_purities = []

    for k in K_HARMONICS:
        f.write("k: ")
        f.write(str(k))
        f.write("\n")

        rand_score_accuracy_trials = []
        purity_trials = []
        max_rand_score_accuracy = -2
        max_purity = -2
        best_triangle = best_normalized_triangle = best_labels = best_centers = best_means = None

        position_encoding = get_k_harmonic_position_encoding(k, G)
        expected_centroids = get_ideal_centroids(position_encoding, true_clusters)
        expected_triangle, expected_normal_triangle = get_triangle_lengths(expected_centroids)
        expected_mean_distances, expected_max_distances = calculate_variance(true_clusters, expected_centroids, position_encoding)

        if k==0.1: 
            file_name = "point_1"
        elif k==0.5:
            file_name = "point_5"
        else:
            file_name = str(k)

        for _ in range(10):
            kmeans = k_harmonic_k_means(G, position_encoding, NUM_CLUSTERS, output_png=file_name)
            purity = get_purity(kmeans.labels_, true_clusters)
            rand_score_accuracy = adjusted_rand_score(true_clusters, kmeans.labels_)
            triangle, normalized_triangle = get_triangle_lengths(kmeans.cluster_centers_)

            rand_score_accuracy_trials.append(rand_score_accuracy)
            purity_trials.append(purity)

            if (purity > max_purity):
                best_kmeans = kmeans
                max_rand_score_accuracy =  rand_score_accuracy 
                max_purity = purity
                best_labels = kmeans.labels_
                best_centers = kmeans.cluster_centers_
                best_triangle = triangle
                best_normalized_triangle = normalized_triangle

        mean_rand_score_accuracy = np.mean(rand_score_accuracy_trials)
        max_rand_score_accuracy = np.max(rand_score_accuracy_trials)
        # print("Mean accuracy: ", mean_rand_score_accuracy)

        mean_purity = np.mean(purity_trials)
        max_purity = np.max(purity_trials)
        f.write("Mean purity: ")
        f.write(str(mean_purity))
        f.write("\n")

        mean_rand_score_accuracies.append(mean_rand_score_accuracy)
        max_rand_score_accuracies.append(max_rand_score_accuracy)

        mean_purities.append(mean_purity)
        max_purities.append(max_purity)

        colors = ['red', 'blue', 'green']
        node_colors = [colors[cluster] for cluster in best_kmeans.labels_]

        nx.draw(G, with_labels=True, node_color=node_colors)
        plt.title('Clustered Graph')
        plt.savefig(file_name + "_best_clustering")
        # plt.show()    
        plt.clf()

        max_accuracy_index = purity_trials.index(max(purity_trials))
        mean_distances, max_distances = calculate_variance(best_labels, best_centers, position_encoding)

        if (purity_trials[max_accuracy_index] == 1):
            mean_distances_file_name = file_name + "_mean_dist_(total_accuracy).png"
            normalized_triangle_name = file_name + "_sample_normalized_triangle_(total_accuracy).png"
            max_distances_file_name = file_name + "_max_dist_(total_accuracy).png"
        else:
            mean_distances_file_name = file_name + "_mean_dist.png"
            normalized_triangle_name = file_name + "_sample_normalized_triangle.png"
            max_distances_file_name = file_name + "_max_dist.png"

        ideal_mean_file = file_name + "ideal_triangle_mean.png"
        ideal_max_file = file_name + "ideal_triangle_max.png"
        ideal_normalized = file_name + "ideal_normalized_triangle.png"

        plot_triangle(expected_triangle, ideal_mean_file, variances=expected_mean_distances)
        plot_triangle(expected_triangle, ideal_max_file, variances=expected_max_distances)
        plot_triangle(expected_normal_triangle, ideal_normalized)

        plot_triangle(best_normalized_triangle, normalized_triangle_name)
        plot_triangle(best_triangle, mean_distances_file_name, variances=mean_distances)
        plot_triangle(best_triangle, max_distances_file_name, variances=max_distances)


    plot_k_vs_random_score_accuracy(K_HARMONICS, mean_rand_score_accuracies, max_rand_score_accuracies, "k_vs_mean_adjusted_rand_score.png", "k_vs_max_adjusted_rand_score.png")
    plot_k_vs_purity_accuracy(K_HARMONICS, mean_purities, max_purities, "k_vs_mean_purities.png", "k_vs_max_purities.png")

    f.close()

def k_hyperparameter_analysis(
    graph_and_label_func,
    graph_func_kwargs,
    clustering_algorithm,
    clustering_func_kwargs,
    dataset_name : Optional[str] = None,
    clustering_alg_name : Optional[str] = None
):
    """ Use `clustering_algorithm` to cluster the graph given by `graph_and_label_func` for various k """
    if dataset_name is None:
        dataset_name = graph_and_label_func.__name__
    if clustering_alg_name is None:
        clustering_alg_name = clustering_algorithm.__name__

    graph, true_labels = graph_and_label_func(**graph_func_kwargs)

    results = []
    for k in K_HARMONICS:
        clustering_func_kwargs["k"] = k
        pred_labels = clustering_algorithm(graph, **clustering_func_kwargs)
        purity = get_purity(pred_labels, true_labels)
        results.append({
            "dataset" : dataset_name,
            "algorithm" : clustering_alg_name,
            "purity" : purity
        } | graph_func_kwargs | clustering_func_kwargs)
    
    results_df = pd.DataFrame(results)
    with open(f"results/{dataset_name}_{clustering_alg_name}.csv", "a") as f:
        file_exists = f.tell()
        results_df.to_csv(f, mode="a", header=(not file_exists))



def main():   
    # input_directory = sys.argv[1]
    # output_directory = sys.argv[2]

    # input_files = os.listdir(input_directory)

    # for input_file in input_files:
    #     analysis(os.path.join(input_directory, input_file), output_directory)
    for num_neighbors in [25, 50, 75, 100]:
        graph_func_kwargs = {"num_neighbors" : num_neighbors}
        clustering_func_kwargs = {"num_clusters" : 10}
        funcs = [k_harmonic_k_means_labels, k_harmonic_girvan_newman_labels]
        func_names = ["k-Means", "Girvan-Newman"]
        for func, func_name in zip(funcs, func_names):
            print(f"{func_name} {num_neighbors}")
            k_hyperparameter_analysis(
                load_iris_graph_and_labels,
                graph_func_kwargs,
                func,
                clustering_func_kwargs,
                dataset_name="iris",
                clustering_alg_name=func_name
            )


if __name__ == "__main__":
    main()
