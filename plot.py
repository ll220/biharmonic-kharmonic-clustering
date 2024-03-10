import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional

from k_harmonic_distance import k_harmonics_of_edges

plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "serif"
}) # Use LaTeX font for captions

def plot_k_harmonic(
    graph : nx.Graph,
    k : float,
    pos : Optional[Dict] = None,
    ax = None,
    cmap : Optional[mpl.colors.Colormap] = None,
    colorbar_label : Optional[str] = None
):
    if ax is None:
        ax = plt.gca()
    if pos is None:
        pos = nx.spring_layout(graph)
    if cmap is None:
        cmap = mpl.cm.autumn
    if colorbar_label is None:
        colorbar_label = r"Squared $k$-Harmonic Distance $H_{e}^{2}$"

    edge_colors = np.round(k_harmonics_of_edges(graph, k), 6)
    nx.draw_networkx_edges(
        graph, 
        pos, 
        edge_color=edge_colors, 
        width=2.0,
        edge_cmap=cmap
    )
    nx.draw_networkx_nodes(
        graph, 
        pos, 
        node_color="#ffffff", 
        node_size=50, 
        edgecolors="#000000", 
        linewidths=2.0
    )
    # plot the color bar
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(
            mpl.colors.Normalize(np.min(edge_colors), np.max(edge_colors)),
            cmap=cmap
        ),
        location="bottom",
    )
    cb.set_label(
        label = colorbar_label,
        size=15
    )
    
    plt.axis("off") # turn off boundary around plot
    ax.set_aspect("equal")

def plot_resistance(
    graph : nx.Graph,
    ** kwargs
):
    plot_k_harmonic(graph, k=1, colorbar_label=r"Effective Resistance $R_{e}$", **kwargs)

def plot_biharmonic(
    graph : nx.Graph,
    ** kwargs
):
    plot_k_harmonic(graph, k=2, colorbar_label=r"Squared Biharmonic Distance $B^{2}_{e}$", **kwargs)

if __name__=="__main__":
    def circle_position(r : float, theta : float):
        return (r*np.cos(2*np.pi*theta), r*np.sin(2*np.pi*theta))
    def get_node_index(k : int, level : int, i : int) -> int:
        return (1 - k**level)//(1-k) + i
    def degree_k_tree(k : int, depth : int) -> nx.Graph:
        graph = nx.Graph()
        pos = {}
        num_added_nodes = 0
        pos[0] = (0, 0) # root is at origin
        for level in range(1, depth + 1):
            num_nodes_at_prev_level = k**(level-1)
            num_nodes_at_level = k**level
            for i in range(k**depth):
                node = get_node_index(k, level, i)
                parent = get_node_index(k, level-1, i // k)
                pos[node] = circle_position(r=level, theta=i/num_nodes_at_level+1/(2*num_nodes_at_level))
                graph.add_edge(parent, node)
        graph.pos = pos
        return graph
    tree = degree_k_tree(3,3)
    plot_biharmonic(tree, pos=tree.pos)
    plt.show()

