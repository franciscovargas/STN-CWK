import pickle
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import eigs
import math
from numpy import linalg as LA

from sklearn.cluster import KMeans

# Gaussian probability density function
gaussian = lambda x, mu, sig: (
    np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) / (np.sqrt(2*np.pi) * sig)


def buildGraph(pickl, his=False):
    """
    This method used the cosine similarity measure and the
    basic z-score gaussian outlier test in order to determine
    if two nodes know each other.
    """

    dic = pickl

    dic1 = dic
    G = nx.Graph()
    dic3 = dict(dic)
    checked = []

    # Adding nodes with bios greater than 30 words.
    for key in dic:
        if((re.sub("[ ]+", "", dic[key]) != "") and len(dic[key])) > 30:
            G.add_node(key)
        else:
            del dic3[key]

    dic1 = dic3

    vect = TfidfVectorizer(min_df=1)
    coefs = list()

    joint_dict = dict()
    # Cosine similarity measure matrix
    F = vect.fit_transform(dic3.values())
    Cosine_mat = (F*F.T).A  # Symmetric matrix:
    # Traverse uper triangle for cosine similarity measures.
    for i, key in enumerate(dic3):
        for j, key1 in enumerate(dic1):
            if(i > j):
                # obtain coef for corresponding key
                tfidf = Cosine_mat[i, j]
                # Repeated nodes must be filtered
                if dic[key] == dic[key1]:

                    continue
                else:
                    coefs.append(tfidf)
                    joint_dict[str(key) + str(key1)] = tfidf

    data = [c for c in coefs if c]
    # max(data)

    mu = np.mean(data)
    std = np.std(data)
    binwidth = 0.007
    if his:
        plt.subplot(1, 2, 0)
        plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
        # PLot gaussian fit contrast
        plt.xlabel("$cos(\\theta)$")
        plt.ylabel("frecuency count of $cos(\\theta)$ values")
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, max(data), 0.001),
                 gaussian(np.arange(0, max(data), 0.001), mu, std),
                 linewidth=2)
        plt.xlabel("$cos(\\theta)$")
        plt.ylabel("fitted gaussian")
        plt.show()

    # Edge creation !
    for key in dic3:
        for key1 in dic1:
            if(key != key1):
                try:
                    x = joint_dict[str(key) + str(key1)]
                    # If cosine similarity is an outlier with 95% change
                    # Make edge between nodes that conform the similarity
                    if(x - mu > 2 * std):
                        G.add_edge(key, key1)
                except:
                    pass

    # Return the conected component with largest cardinality of nodes
    # Throw away small connected components we are interested in the big one
    # For our mini project exploration purposes
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def splitGraph(graph, results):
    graphChange1 = nx.Graph()
    graphChange2 = nx.Graph()
    i = 0
    for node in graph:
        if(results[i] == 0):
            graphChange1.add_node(node)
        else:
            graphChange2.add_node(node)
        i += 1
    # graphChange2.nodes()
    flag = 0
    flagd = 0
    for item in graph.edges():
        for node in graphChange1:
            if item[0] == node:
                flag += 1

        for node in graphChange1:
            if item[1] == node:
                flag += 1
        if flag == 2:
            graphChange1.add_edge(item[0], item[1])
        flag = 0
        for noded in graphChange2:
            if item[0] == noded:
                flagd += 1

        for noded in graphChange2:
            if item[1] == noded:
                flagd += 1
        if flagd == 2:
            graphChange2.add_edge(item[0], item[1])
        flagd = 0
    # graphChange2.edges()
    # graph.edges()
    return graphChange1, graphChange2


def communityGraph(graph):
    """
    Median method of community detection
    also used just to obtain laplacian data
    for specKMeans
    """

    lapgr = nx.laplacian_matrix(graph)

    # Get the eigenvalues and eigenvectors of the Laplacian matrix
    evals, evec = np.linalg.eigh(lapgr.todense())

    fiedler = evec[1]
    results = []
    ## "Fiedler", fiedler
    median = np.median(fiedler, axis=1)  # median of the second eigenvalue
    for i in range(0, fiedler.size):    # divide the graph nodes into two
        if(fiedler[0, i] < median):
            results.append(0)
        else:
            results.append(1)
    return results, evals, evec


def embed_nodes(g, x, y):
    return dict((g.nodes()[i], (x[i], y[i])) for i in range(len(g.nodes())))


def showEigenspace(G, V, k):
    """
    This method projects our graph G in to a sub
    eigen living in R^k , Mostly used for visualization
    purposes
    """
    V = V.T[0:k, :]
    fig = plt.figure()
    count = 0
    for i, v in enumerate(V):
        for j, u in enumerate(V):

            emb = embed_nodes(G, v.ravel().tolist()[0], u.ravel().tolist()[0])
            plt.subplot(k, k, k*k - count)
            nx.draw_networkx_nodes(G, pos=emb, node_size=50, with_labels=False, node_color=G.degree(
            ).values(), cmap=plt.get_cmap('gray'))
            nx.draw_networkx_edges(G, pos=emb)

            count += 1

    plt.show()


def specKMeans(graph, evec, k):
    """
    Spectral Kmeans algorithm:
    1. Spectral graph representation in Laplacian eigen supbspace R^k
    2. Kmeans clustering on our R^k space
    """
    nodeCluster = dict()
    evec = evec.T[1:k+1, :]

    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(evec.T)
    centroids = kmeans.cluster_centers_
    i = 0
    for node in graph:
        nodeCluster[node] = y_pred[i]
        i += 1
    return y_pred, nodeCluster, centroids


def construct(pickl, startVector, endVector, d='h'):
    his = False
    if d=='h':
        his=True
    graph = buildGraph(pickl, his)

    results, evals, evec = communityGraph(graph)
    # Hardcoded cluster number change third param to yield
    # more clusters
    kmeans, cluster, centroids = specKMeans(graph, evec, 3)

    #first, second = splitGraph(graph,results)
    franVectors = list()
    for i in range(0, len(evec[:, startVector])):

        franVectors.append(
            [evec[:, startVector].item(i), evec[:, endVector].item(i)])

    emb = embed_nodes(graph, evec[:, startVector].ravel().tolist()[
                      0], evec[:, endVector].ravel().tolist()[0])
    if d == 'e':
        showEigenspace(graph, evec, 5)

    i = 0
    for node in graph.nodes():
        graph.node[node]['category'] = kmeans[i]
        i += 1
    color_map = {0: 'b', 1: 'r', 2: 'y', 3: 'g'}

    if d=='s':
        nx.draw_spring(graph, node_color=[color_map[graph.node[node]['category']] for node in graph])
        plt.show()
    if d=='c':
        nx.draw_networkx(graph, pos=emb, node_size=100, with_labels=False, node_color=[
                         color_map[graph.node[node]['category']] for node in graph])
        plt.xlabel("second eigen vector")
        plt.ylabel("third eigen vector")
        plt.show()

    return centroids, franVectors


def fran(startVector, endVector):
    pickl = pickle.load(open("BiodictData/biodictdist4.pickle", "rb"))
    centr1, franVectors1 = construct(pickl, startVector, endVector)



if __name__ == '__main__':

    pickl = pickle.load(open("BiodictData/biodictdist4.pickle", "rb"))
    construct(pickl, 1, 2, 'c')
    # # "Centroids" , centr
