import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#
    J = (np.sum(np.square(X.dot(theta) - y)) + Lambda * np.sum(np.square(theta[1:]))) / float(2 * m)
    grad = (X.T.dot(X.dot(theta) - y) + Lambda * np.concatenate([[0], theta[1:]])) / float(m)
    return J, grad
# =========================================================================