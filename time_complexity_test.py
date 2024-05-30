import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import edge_current_flow_betweenness_centrality
import numpy as np
from time import time
import tqdm

from k_harmonic_distance import biharmonics_of_edges

num_samples = 5 # number of graphs for each node
ns = list(range(25, 425, 25)) # number of nodes
cf_times = []
bh_times = []
for n in tqdm.tqdm(ns):
    graphs = [nx.erdos_renyi_graph(n=n, p=0.5) for  _ in range(num_samples)]
    cf_times_curr = []
    bh_times_curr = []
    for graph in graphs:
        for centrality_func, times in zip(
            [edge_current_flow_betweenness_centrality, biharmonics_of_edges],
            [cf_times_curr, bh_times_curr]
        ):
            start_time = time()
            centrality_func(graph)
            end_time = time()
            total_time = end_time-start_time
            times.append(total_time)
    cf_times.append(cf_times_curr)
    bh_times.append(bh_times_curr)
# process and plot the times
cf_times = np.array(cf_times)
bh_times = np.array(bh_times)
for times, label, color in zip([cf_times, bh_times], ["Current-Flow", "Biharmonic"], ['#648FFF', '#785EF0']):
    times_mean = np.average(times, axis=1)
    times_std = np.std(times, axis=1)
    ax = plt.gca()
    ax.plot(ns, times_mean, label=label)
    ax.fill_between(ns, times_mean-times_std, times_mean+times_std, color=color, alpha=0.2)
plt.xlabel("Number of nodes")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("time_complexity_test.pdf")
plt.show()

    




