import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram
import networkx as nx
from collections import deque
import json
import scipy.cluster.hierarchy as sch

def import_wiki_vote_data():
    graphFile = open("Wiki-Vote.txt", 'r').read().splitlines()[4:]
    # print(len(graphFile))
    minNode = 10000000
    maxNode = -1000000
    for each_line in graphFile:
        each_line = each_line.split('\t')
        minNode = min(minNode, int(each_line[0]), int(each_line[1]))
        maxNode = max(maxNode, int(each_line[0]), int(each_line[1]))
    adj_wiki = np.zeros((maxNode, maxNode), dtype=bool)
    for each_line in graphFile:
        each_line = each_line.split('\t')
        A = int(each_line[0]) - 1
        B = int(each_line[1]) - 1
        adj_wiki[A][B] = True
    adj_wiki = adj_wiki + adj_wiki.T
    return adj_wiki

def import_lastfm_data():
    with open('lasftm_asia/lastfm_asia_features.json', 'r') as file:
        json_file = json.load(file)
    edges = pd.read_csv("lasftm_asia/lastfm_asia_edges.csv")
    target = pd.read_csv("lasftm_asia/lastfm_asia_target.csv")

    n = np.max(edges[['node_1', 'node_2']]) + 1
    adj_las = np.zeros((n, n), dtype = bool)
    for _, row in edges.iterrows():
        start_idx = row['node_1']
        end_idx = row['node_2']
        adj_las[start_idx, end_idx] = 1
    adj_las = adj_las + adj_las.T
    return adj_las

def import_lastfm_asia_data(s):
    return import_lastfm_data()

def getInDegree(n, adj):
    return np.sum(adj[n])
def getOutDegree(n, adj):
    return np.sum(adj[:, n])
def getDegree(n, adj):
    return np.sum(adj[n])
#===============================================================================================================
def getClusters(adj):
    num_nodes = len(adj)
    visited = [False] * num_nodes
    connected_components = []

    def iterative_dfs(start_node):
        stack = [start_node]
        component = []

        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                # Add all unvisited neighbors to the stack
                for neighbor in range(num_nodes):
                    if adj[node][neighbor] == 1 and not visited[neighbor]:
                        stack.append(neighbor)

        return component

    for node in tqdm(range(num_nodes)):
        if not visited[node]:
            component = iterative_dfs(node)
            connected_components.append(component)

    return connected_components
#===============================================================================================================
def Girvan_Newman_one_level(adjacency):

    adj = adjacency.copy()

    viableNodes = np.where(np.sum(adj, axis=1) != False)[0]
    for _ in range(1):
        betweenness = np.zeros((len(adj), len(adj)), dtype=float)
        for node in tqdm(viableNodes):
            betweenness += performGirvan(node, adj) 
            
        (s, t) = np.unravel_index(np.argmax(betweenness), betweenness.shape)
        print(f"The edge {s}--->{t} has highest betweenness {np.max(betweenness)}")
        print("Hence removed this edge")
        adj[s][t] = False
        adj[t][s] = False
        
        # Clusters = getClusters(adj = adj)
        # modularity = 0
        # for cluster in Clusters:
        #     modularity += Q(cluster, adjacency)
        # print(f"Now modularity = {modularity}")

    Clusters = getClusters(adj = adj)
    graphPartition = np.zeros((len(adj), 1), dtype=int)
    
    for cluster_num in range(len(Clusters)):
        graphPartition[Clusters[cluster_num]] = min(Clusters[cluster_num])
    
    return graphPartition, Clusters
#====================================================================================================================
def performGirvan(root, adj):
    num_nodes = len(adj)
    NodeLabels = np.zeros(num_nodes, dtype=float)
    NodeCredits = np.zeros(num_nodes, dtype=float)
    EdgeCredits = np.zeros((num_nodes, num_nodes), dtype=float)
    Levels = np.full(num_nodes, -1, dtype=int)

    queue = deque([(root, 0)])
    NodeLabels[root] = 1
    Levels[root] = 0

    while queue:
        node, nodeLevel = queue.popleft()
        next_level = nodeLevel + 1
        
        # Get children
        children = np.nonzero(adj[node] & (Levels == -1))[0]

        for child in children:
            queue.append((child, next_level))
            Levels[child] = next_level
            NodeLabels[child] += 1
    
    max_level = np.max(Levels)
    indices = np.where(Levels == max_level)[0]
    NodeCredits[indices] = 1

    for level in range(max_level - 1, -1, -1):
        indices = np.where(Levels == level)[0]
        nextIndices = np.where(Levels == level + 1)[0]

        for row in indices:
            relevant_edges = adj[row, nextIndices]
            count = np.sum(NodeCredits[nextIndices] / NodeLabels[nextIndices] * relevant_edges)
            NodeCredits[row] = 1 + count
            EdgeCredits[row, nextIndices[relevant_edges]] = NodeCredits[nextIndices[relevant_edges]] / NodeLabels[nextIndices[relevant_edges]]
    
    return EdgeCredits

def showDendrogram(Clusters, adj, plot_Title = "Dendrogram of Clusters"):
    
    distance_matrix = np.zeros((len(Clusters), len(Clusters)), dtype=float)

    for i in range(len(Clusters)):
        for j in range(len(Clusters)):
            dist = np.sum(adj[Clusters[i]][:, Clusters[j]])
            if(dist == 0):
                distance_matrix[i][j] = len(adj) + 1
            else:
                distance_matrix[i][j] = 1.0 / float(dist)
        distance_matrix[i][i] = 0
        
    condensed_dist_matrix = sch.distance.squareform(distance_matrix)

    # Perform the hierarchical clustering
    Z = sch.linkage(condensed_dist_matrix, method='ward')

    # Step 3: Create the dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z, labels=[f'Cluster {i+1}' for i in range(len(Clusters))])
    plt.title(plot_Title)
    plt.xlabel('Cluster')
    plt.ylabel('Distance')
    plt.show()

def Q_int(i, adj, total_edges=None):
    if total_edges is None:
        total_edges = np.sum(adj)
    
    degree_i = np.sum(adj[i])
    
    return adj[i, i] - (degree_i ** 2) / (2 * total_edges)

def Q(X, adj, total_edges=None):   
    if total_edges is None:
        total_edges = np.sum(adj)  # Total number of edges in the full graph

    sub_adj = adj[np.ix_(X, X)]
    sigma_in = np.sum(sub_adj)
    sigma_x = np.sum(adj[X])
    
    return sigma_in / (2 * total_edges) - (sigma_x / (2 * total_edges)) ** 2

def Q_before(i, Y, adj, total_edges=None):
    if total_edges is None:
        total_edges = np.sum(adj)
        
    return Q_int(i, adj, total_edges) + Q(Y, adj, total_edges)

def Q_after(i, Y, adj, total_edges=None):
    if total_edges is None:
        total_edges = np.sum(adj)
    
    k_i_Y = np.sum(adj[i, Y]) + np.sum(adj[Y, i])
    k_i = np.sum(adj[i])
    
    sigma_in = np.sum(adj[np.ix_(Y, Y)])
    sigma_y = np.sum(adj[Y])
    
    return (sigma_in + k_i_Y) / (2 * total_edges) - ((sigma_y + k_i) / (2 * total_edges)) ** 2

def performLouvain(adj, maxIter = 1):
    viableNodes = np.where(np.sum(adj, axis=1) != 0)[0]
    V = set(viableNodes)
    C = {v: [v] for v in V}  
    
    # maxIter = 1
    total_edges = np.sum(adj)  

    for _ in range(maxIter):
        flag = False
        delQij_max = -np.inf
        i_star, j_star = None, None

        for i in tqdm(list(V)):
            for j_key in list(C):
                j = C[j_key]
                delQ = Q_after(i, np.array(j), adj, total_edges) - Q_before(i, np.array(j), adj, total_edges)
                
                if delQ > delQij_max:
                    delQij_max = delQ
                    i_star = i
                    j_star = j_key
                    flag = True
        
        if flag and i_star is not None and j_star is not None:
            V.remove(i_star)
            if i_star in C:
                del C[i_star]
            
            C[j_star].append(i_star)
            for j in C[j_star]:
                if j in V:
                    V.remove(j)
        
        if not flag:
            break

    Clusters = list(C.values())
    graphPartition = np.zeros((len(adj), 1), dtype=int)

    for cluster_num in range(len(Clusters)):
        graphPartition[Clusters[cluster_num]] = min(Clusters[cluster_num])

    return graphPartition, Clusters



if __name__ == "__main__":

    ############ Answer qn 1-4 for wiki-vote data #################################################
    # Import wiki-vote.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_wiki = import_wiki_vote_data("../data/Wiki-Vote.txt")

    # This is for question no. 1
    # graph_partition: graph_partitition is a nx1 numpy array where the rows corresponds to nodes in the network (0 to n-1) and
    #                  the elements of the array are the community ids of the corressponding nodes.
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    graph_partition_wiki, clusters_wiki_Girvan  = Girvan_Newman_one_level(nodes_connectivity_list_wiki)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # community_mat is a n x m matrix, where m is the number of levels of Girvan-Newmann algorithm and n is the number of nodes in the network.
    # Columns of the matrix corresponds to the graph_partition which is a nx1 numpy array, as before, corresponding to each level of the algorithm. 
    # community_mat_wiki = Girvan_Newman(nodes_connectivity_list_wiki)

    # This is for question no. 3
    # Visualise dendogram for the communities obtained in question no. 2.
    # Save the dendogram as a .png file in the current directory.
    showDendrogram(graph_partition_wiki, clusters_wiki_Girvan)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before. Show the resulting communities after one iteration of the algorithm.
    graph_partition_louvain_wiki, clusters_wiki_Louvain = performLouvain(nodes_connectivity_list_wiki)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import lastfm_asia_edges.csv
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_lastfm = import_lastfm_asia_data("../data/lastfm_asia_edges.csv")

    # Question 1
    graph_partition_lastfm, clusters_lastfm_Girvan = Girvan_Newman_one_level(nodes_connectivity_list_lastfm)

    # Question 2
    # community_mat_lastfm = Girvan_Newman(nodes_connectivity_list_lastfm)

    # Question 3
    showDendrogram(graph_partition_lastfm, clusters_lastfm_Girvan)

    # Question 4
    graph_partition_louvain_lastfm, clusters_lastfm_Louvain = performLouvain(nodes_connectivity_list_lastfm)

