import random
random.seed(246)        # or any integer
import numpy as np
# np.random.seed(4812)
import networkx as nx
import matplotlib.pyplot as plt

K = 2

def get_k_harmonic_distances(laplacian_inverse, nodes, n):
    distances = np.zeros((n, n))

    for u in range(n):
        for v in range(u):
            vector = np.zeros((n, 1))
            vector[u][0] = 1
            vector[v][0] = -1

            intermediate = np.copy(laplacian_inverse)

            for i in range(K):
                intermediate = np.matmul(intermediate, laplacian_inverse)

            distance = np.matmul(vector.transpose(), intermediate)
            distance = (np.matmul(distance, vector))[0][0]
            print("u:", u, " v:", v, " distance:", distance)

            distances[v][u] = distance
    return distances


# assumption that all vertices are encoded as 0-n consecutive integers
f = open("graph.txt", "r")
file_string = f.read()
tuples = file_string.split(',')
numbers = [tuple.split(' ') for tuple in tuples]
integers = [list(map(int, x)) for x in numbers]
edges = list(map(tuple, integers))

G = nx.Graph(edges)

nodes = list(G.nodes)
n = len(nodes)

laplacian = nx.laplacian_matrix(G)
laplacian = laplacian.toarray()

laplacian_inverse = np.linalg.pinv(laplacian) 
position_encoding = np.copy(laplacian_inverse)

for i in range(int(K / 2)):
    position_encoding = np.matmul(position_encoding, laplacian_inverse)

# print(laplacian_inverse)

distances = get_k_harmonic_distances(laplacian_inverse, nodes, n)
print(distances) 




# position_encoding = np.zeros((n, n), dtype=int)

# for node in range(n):
#     vertex = np.zeros((n, 1), dtype=int)
#     vertex[node][0] = 1
#     print(vertex)
#     position_vector = np.matmul(laplacian_inverse, vertex)
#     print(position_vector)
#     position_encoding[:, node:node+1] = position_encoding