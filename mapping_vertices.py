import random
random.seed(246)        # or any integer
import numpy as np
# np.random.seed(4812)
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans


K = 2

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



# assumption that all vertices are encoded as 0-n consecutive integers
f = open("graph.txt", "r")
file_string = f.read()
edges = file_string.split(',')
# numbers = [tuple.split(' ') for tuple in tuples]
# integers = [list(map(int, x)) for x in numbers]
# edges = list(map(tuple, integers))
print((edges))
G = nx.parse_edgelist(edges, nodetype=int)

nodes = list(G.nodes)
n = len(nodes)

laplacian = nx.laplacian_matrix(G)
laplacian = laplacian.toarray()

laplacian_inverse = np.linalg.pinv(laplacian) 

position_encoding = scipy.linalg.fractional_matrix_power(laplacian_inverse, float(K / 2))
print(position_encoding)
position_encoding = position_encoding.transpose()
print(position_encoding)

# # print(laplacian_inverse)

# # check_distances = get_k_harmonic_distances(laplacian_inverse, nodes, n)
# distances = get_harmonic_distances(position_encoding, n)
# print(distances) 



k_clusters = 2
results = []

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(position_encoding)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

# position_encoding = np.zeros((n, n), dtype=int)

# # for node in range(n):
# #     vertex = np.zeros((n, 1), dtype=int)
# #     vertex[node][0] = 1
# #     print(vertex)
# #     position_vector = np.matmul(laplacian_inverse, vertex)
# #     print(position_vector)
# #     position_encoding[:, node:node+1] = position_encoding