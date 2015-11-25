import numpy as np


def agg_dist(clusters, centroids):
    centroids = centroids[0]
    euc_dists = list()
    for i, clut in enumerate(clusters):
        cent = centroids[i]
        euc_dists.append(np.dot(clut, cent))
    return np.sum(euc_dists)
