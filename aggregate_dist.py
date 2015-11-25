import numpy as np
from itertools import chain


def agg_dist(clusters, centroids):
    centroids = centroids[0]
    euc_dists = list()
    for i, clut in enumerate(clusters):
        print i
        cent = centroids[i]
        euc_dists.append(np.dot(clut, cent))
    # print euc_dists
    return np.sum(list(chain(*euc_dists)))
