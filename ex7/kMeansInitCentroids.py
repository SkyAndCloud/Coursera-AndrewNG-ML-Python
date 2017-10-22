import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#
    random_index = np.random.permutation(X.shape[0])
    centroids = X[random_index[:K],:]
# =============================================================
    return centroids
