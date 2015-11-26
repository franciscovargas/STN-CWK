import numpy as np
from itertools import chain
import pickle
import similarityGraph4 as sg
import matplotlib.pyplot as plt

def agg_dist(clusters, centroids):
    # print len(clusters), len(centroids)
    # print centroids
    # centroids = centroids[0
    euc_dists = list()
    for i, clut in enumerate(clusters):
        # # print i
        cent = centroids[i]

        z =  clut - cent

        dist_all = np.diag(np.dot(z, z.T))
        # print dist_all
        euc_dists.append(np.sqrt(dist_all))

    return np.sum(list(chain(*euc_dists)))


def build_eigen_space_clusters(ypred, V, k):
    out_clust = [[]]*k

    for i, y in enumerate(ypred):
        # print V[i].ravel().tolist()[0]
        # exit()
        out_clust[y].append(V[i].ravel().tolist()[0][1:k+1])
    # print out_clust
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


def evaluate_kmeans_clust_no(graph, k_max):
    result = list()
    final_result = list()
    for k in range(2,k_max):
        r, e, V = sg.communityGraph(graph)
        ypred, nodeCluster, centroids = sg.specKMeans(graph, V, k)
        result.append((build_eigen_space_clusters(ypred, V, k), centroids))
    for n, c in result:
        final_result.append(agg_dist(n, c))
    return final_result

if __name__ == "__main__":
        g_dict =  pickle.load(open("massiveData.pickle", "rb"))
        # print evaluate_kmeans_radial(g_dict, 2)
        k = 2
        print len(evaluate_kmeans_clust_no(g_dict.values()[0], k))
        plt.plot([4,8,16,32,64] , evaluate_kmeans_radial(g_dict, k), '-o')
        plt.show()