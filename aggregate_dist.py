import numpy as np
from itertools import chain
import pickle
import similarityGraph5 as sg
import matplotlib.pyplot as plt

def agg_dist(clusters, centroids):
    # print len(clusters), len(centroids)
    # print centroids
    # centroids = centroids[0
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
    out_clust = [[]]*k

    for i, y in enumerate(ypred):
        # V[i].ravel().tolist()[0][1:k+1] is an eigen vec
        # representation of graph in R^k
        out_clust[y].append(V[i].ravel().tolist()[0][1:k+1])

    return out_clust


def evaluate_kmeans_radial(graphs, k):
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
    result = list()
    final_result = list()
    r, e, V = sg.communityGraph(graph)
    for k in range(k_min, k_max):
        # print k
        ypred, nodeCluster, centroids = sg.specKMeans(graph, V, k)
        result.append((build_eigen_space_clusters(ypred, V, k), centroids))
    for n, c in result:
        final_result.append(agg_dist(n, c))
    return final_result

if __name__ == "__main__":
        typ = 'r'
        g_dict =  pickle.load(open("giganticData", "rb"))
        print len(g_dict)
        # Best fond k was 12
        k = 10

        # for radial cluster
        if typ == 'r':
            plt.plot([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 52, 56, 60, 64],
                     evaluate_kmeans_radial(g_dict, k), '-o')
            plt.show()
        # for cluster number picking
        else:
            for i in range(len(g_dict) - 6):
                print i
                plt.plot(range(2,16),
                         evaluate_kmeans_clust_no(g_dict.values()[i], 2, 16), '-o')
                plt.show()