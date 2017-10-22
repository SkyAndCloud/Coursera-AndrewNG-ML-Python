import numpy as np


def computeCentroids(X, idx, K):
    """returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

# Useful variables
    m, n = X.shape

# You need to return the following variables correctly.
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every centroid and compute mean of all points that
#               belong to it. Concretely, the row vector centroids(i, :)
#               should contain the mean of the data points assigned to
#               centroid i.
#
# Note: You can use a for-loop over the centroids to compute this.
# 
    idx_count = np.bincount(idx)
    for k in xrange(K):
        if k < len(idx_count):
            centroids[k] = np.sum(X[idx == k,:], axis=0) / float(idx_count[k])

# =============================================================

    return centroids

# if __name__ == '__main__':
#     print computeCentroids(
#         np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
#         np.array([0, 0]),
#         2
#     )