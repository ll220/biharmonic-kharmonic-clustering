import mapping_vertices
import os
import networkx as nx
import numpy as np
import scipy.stats as st
import sys
from spectral_clustering import run_spectral_clustering


K_HARMONICS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]
# mapping_vertices.NUM_CLUSTERS = 10

def all_synthetic_graph_spectral_clustering():
    writing = open("synthetic_graph_sc_random_state.csv", "w")
    writing.write("dataset,p_q,algorithm,mean_purity,num_clusters,k,ci\n")

    graph_directories = [("./synthetic_graphs/three_graphs/size_ten", 3), 
                         ("./synthetic_graphs/three_graphs/size_twenty", 3), 
                         ("./synthetic_graphs/three_graphs/size_fifty", 3), 
                         ("./synthetic_graphs/five_graphs/size_50", 5), 
                         ("./synthetic_graphs/ten_graphs/size_50", 10)]
    
    for directory_item in graph_directories: 
        input_files = os.listdir(directory_item[0])

        for input_file in input_files:
            mapping_vertices.NUM_CLUSTERS = directory_item[1]

            print(os.path.join(directory_item[0], input_file))
            f = open(os.path.join(directory_item[0], input_file), "r")
            file_string = f.read()
            f.close()
            
            edges = file_string.split('\n')

            unordered_g = nx.parse_edgelist(edges, nodetype=int)

            G = nx.Graph()
            G.add_nodes_from(sorted(unordered_g.nodes(data=True)))
            G.add_edges_from(unordered_g.edges(data=True))

            nodes = list(G.nodes)
            print(len(nodes))

            if not nx.is_connected(G):
                print("geh")
                quit()

            true_clusters = mapping_vertices.get_expected_clustering(nodes)
            for k in K_HARMONICS:
                purities = []            

                for _ in range(10):
                    results = run_spectral_clustering(G, directory_item[1])   
                    purity = mapping_vertices.get_purity(results, true_clusters)  
                    purities.append(purity)

                print(purities)
                mean_purity = np.mean(purities)
                ci = st.t.interval(0.95, len(purities)-1, loc=mean_purity, scale=st.sem(purities))


                writing.write(directory_item[0] + ","+input_file[:-4]+",spectral_clustering,"+str(mean_purity)+","+str(directory_item[1]) + "," +str(k)+","+str(ci)+"\n")



def synthetic_graph_kmeans(input_directory_path):
    writing = open("ten_graph_50_nodes.csv", "w")
    writing.write("dataset,p_q,algorithm,mean_purity,num_clusters,k,ci\n")

    input_files = os.listdir(input_directory_path)

    for input_file in input_files:
        print(input_file)
        f = open(os.path.join(input_directory_path, input_file), "r")
        file_string = f.read()
        f.close()
        edges = file_string.split('\n')

        unordered_g = nx.parse_edgelist(edges, nodetype=int)

        G = nx.Graph()
        G.add_nodes_from(sorted(unordered_g.nodes(data=True)))
        G.add_edges_from(unordered_g.edges(data=True))

        nodes = list(G.nodes)
        print(len(nodes))

        if not nx.is_connected(G):
            print("geh")
            quit()

        true_clusters = mapping_vertices.get_expected_clustering(nodes)

        for k in K_HARMONICS:
            purities = []            

            for _ in range(10):
                kmeans = mapping_vertices.k_harmonic_k_means(G, k, mapping_vertices.NUM_CLUSTERS)   
                purity = mapping_vertices.get_purity(kmeans.labels_, true_clusters)  
                purities.append(purity)

            print(purities)
            mean_purity = np.mean(purities)
            ci = st.t.interval(0.95, len(purities)-1, loc=mean_purity, scale=st.sem(purities))


            writing.write("synthetic_10cluster_50nodes,"+input_file[:-4]+",kmeans,"+str(mean_purity)+",10,"+str(k)+","+str(ci)+"\n")

def main():
    # read_synthetic_graph_file(sys.argv[1])
    all_synthetic_graph_spectral_clustering()

if __name__ == "__main__":
    main()