import numpy as np
from itertools import chain
import pickle
import similarityGraph5 as sg
import matplotlib.pyplot as plt

def agg_dist(clusters, centroids):
    """
    Finds intra cluster agregate distance
    for each cluster and sums them up
    as an overal measure of dispersion 
    in the clustering.
    """
    euc_dists = list()
    for i, clut in enumerate(clusters):

        cent = centroids[i]
        # Matrix mult way:
        z =  clut - cent
        # Take advantage of matrix mult in numpy
        # to avoid slow loops in python
        # when calculating the cluster inter aggregate distances
        dist_all = np.diag(np.dot(z, z.T))
        # this check here checks the method above is correct
        # Although providing proof is easy.
        assert dist_all.all() >= 0
        # sum up and append aggregate distances
        euc_dists.append(np.sum(dist_all))

    return np.sum(euc_dists)


def build_eigen_space_clusters(ypred, V, k):
    """
    Builds the euclidean space for the
    spectral clustering to happen.
    """
    out_clust = [[]]*k

    for i, y in enumerate(ypred):
        # V[i].ravel().tolist()[0][1:k+1] is an eigen vec
        # representation of graph in R^k
        out_clust[y].append(V[i].ravel().tolist()[0][1:k+1])

    return out_clust


def evaluate_kmeans_radial(graphs, k):
    """
    Carries out our cluster metrics for different balls
    of radius tinder generated graphs.
    """
    result = list()
    final_result = list()
    for graph in sorted(graphs.keys()):
        r, e, V = sg.communityGraph(graphs[graph])
        ypred, nodeCluster, centroids = sg.specKMeans(graph, V, k)
        result.append((build_eigen_space_clusters(ypred, V, k), centroids))
    for n, c in result:
        final_result.append(agg_dist(n, c))
    return final_result


def evaluate_kmeans_clust_no(graph,k_min, k_max):
    """
    Carries out our cluster metrics for different clusters
    and a given ball of radius tinder graph. 
    This is used to pick the best cluster number
    based on methods from machine learning.
    """
    result = list()
    final_result = list()
    r, e, V = sg.communityGraph(graph)
    for k in range(k_min, k_max):
        ypred, nodeCluster, centroids = sg.specKMeans(graph, V, k)
        result.append((build_eigen_space_clusters(ypred, V, k), centroids))
    for n, c in result:
        final_result.append(agg_dist(n, c))
    return final_result

if __name__ == "__main__":
        typ = 'c'
        g_dict =  pickle.load(open("giganticData", "rb"))

        # Best fond k was 14 second 3 .
        k = 14

        # for radial cluster evolution.
        if typ == 'r':
            plt.plot(range(4,65,4),
                     evaluate_kmeans_radial(g_dict, k), '-o')
            plt.xlabel("Radius(km)")
            plt.ylabel("$\sum_{c \in centroids} \sum_{\quad x \in c} D(x, c)^{2}$")
            plt.show()
        # for cluster number picking
        else:
            for i in range(len(g_dict) - 6):
                plt.plot(range(2,16),
                         evaluate_kmeans_clust_no(g_dict.values()[i], 2, 16), '-o')
            plt.xlabel("Cluster number")
            plt.ylabel("$\sum_{c \in centroids} \sum_{\quad x \in c} D(x, c)^{2}$")
            plt.show()