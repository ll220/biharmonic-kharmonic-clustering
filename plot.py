import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional
from k_harmonic_distance import k_harmonics_of_edges


def plot_k_harmonic(
    graph : nx.Graph,
    k : float,
    pos : Optional[Dict] = None,
    ax = None,
    cmap : mpl.colors.Colormap = mpl.cm.viridis,
    colorbar_label : Optional[str] = None,
    ** kwargs
):
    """ Plot a graph with edges colored according to squared k-harmonic distance 
    
    Args:
        graph: NetworkX graph to be plotted
        k: power of the k-harmonic distance
        pos: position of the nodes as a dict.
            If None, use networkx.spring_layout
        ax: matplotlib axis
        cmap: colormap for the edges
        colorbar_label: Label for the colorbar
            If None, use f"Squared {k}-harmonic distance"
        kwargs: kwargs that will be forwarded to the function networkx.draw_networkx_edges  
    """
    # set default arguments
    if ax is None:
        ax = plt.gca()
    if pos is None:
        pos = nx.spring_layout(graph)
    if colorbar_label is None:
        colorbar_label = f"Squared {k}-Harmonic Distance"
    
    edge_colors = np.round(k_harmonics_of_edges(graph, k), 6)
    # set min and max values for the colorbar
    if "edge_vmin" in kwargs:
        edge_vmin = kwargs["edge_vmin"]
        del kwargs["edge_vmin"]
    else:
        edge_vmin = np.min(edge_colors)
    if "edge_vmax" in kwargs: 
        edge_vmax = kwargs["edge_vmax"]
        del kwargs["edge_vmax"]
    else:
        edge_vmax = np.max(edge_colors)
    if edge_vmin ==  edge_vmax:
        # handle edge case of all edges having the same color
        edge_vmin = edge_vmin - 0.1
        edge_vmax = edge_vmax + 0.1
    # plot the graph
    nx.draw_networkx_edges(
        graph, 
        pos, 
        edge_color=edge_colors, 
        width=2.0,
        edge_cmap=cmap,
        ax=ax,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        **kwargs
    )
    nx.draw_networkx_nodes(
        graph, 
        pos, 
        node_color="white", 
        node_size=50, 
        edgecolors="black", 
        linewidths=2.0,
        ax=ax
    )
    # plot the color bar
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(
            mpl.colors.Normalize(edge_vmin, edge_vmax),
            cmap=cmap
        ),
        location="bottom",
        fraction=0.046,
        pad=0,
        ax=ax,
    )
    cb.set_label(
        label=colorbar_label,
        size=10
    )
    # set limits of the bounding box
    pos_list = np.array([val for val in pos.values()])
    x_min =  np.min(pos_list[:,0])  
    x_max =  np.max(pos_list[:,0])  
    y_min =  np.min(pos_list[:,1])  
    y_max =  np.max(pos_list[:,1])  
    tot_min = min(x_min, y_min)
    tot_max = max(x_max, y_max)
    tot_width = tot_max-tot_min
    ax.set_xlim(tot_min-0.2*tot_width, tot_max+0.2*tot_width)
    ax.set_ylim(tot_min-0.2*tot_width, tot_max+0.2*tot_width)
    ax.set_box_aspect(1)
    # turn off boundary around plot
    ax.axis("off") 

def plot_resistance(
    graph : nx.Graph,
    ** kwargs
):
    plot_k_harmonic(graph, k=1, colorbar_label=r"Effective Resistance", **kwargs)

def plot_biharmonic(
    graph : nx.Graph,
    ** kwargs
):
    plot_k_harmonic(graph, k=2, colorbar_label=r"Squared Biharmonic Distance", **kwargs)

if __name__=="__main__":
    from generate_graphs import degree_k_tree
    # Use LaTeX font for captions
    plt.rcParams.update({
        "text.usetex" : True,
        "font.family" : "serif"
    }) 
    # plot a degree-k tree
    graph = degree_k_tree(3,3)
    plot_biharmonic(graph, pos=graph.pos, edge_vmin=0)
    plt.show()

