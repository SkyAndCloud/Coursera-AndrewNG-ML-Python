import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K
    K = len(centroids)

# You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=np.int32)

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the 
#               range 1..K
#
# Note: You can use a for-loop over the examples to compute this.


# =============================================================
    for i, x in enumerate(X):
        min_d = float('inf')
        for index, center in enumerate(centroids):
            d = np.sqrt(np.sum(np.square(x - center)))
            if d < min_d:
                min_d = d
                idx[i] = index
    return idx

