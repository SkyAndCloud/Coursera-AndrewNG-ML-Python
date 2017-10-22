import numpy as np

def polyFeatures(X, p):
    """takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    """
# You need to return the following variables correctly.
    X_poly = np.zeros((X.size, p))

# ====================== YOUR CODE HERE ======================
# Instructions: Given a vector X, return a matrix X_poly where the p-th 
#               column of X contains the values of X to the p-th power.
#
# 

# =========================================================================
    for i in xrange(X.size):
        for j in xrange(1, p + 1):
            X_poly[i,j - 1] = np.power(X[i], j)
    return X_poly