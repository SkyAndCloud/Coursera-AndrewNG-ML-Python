import numpy as np
from scipy.optimize import minimize
from lrCostFunction import lrCostFunction
from ex2.gradientFunctionReg import gradientFunctionReg

def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """
# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))
  
    # This function will return theta and the cost
    for i in xrange(num_labels):
        function_args = (X, (y % 10 == i).astype(int), Lambda)
        theta = minimize(lrCostFunction, x0=initial_theta, args=function_args, options={'disp': True, 'maxiter': 1500}, method=None, jac=gradientFunctionReg)
        all_theta[i,:] = theta.x
# =========================================================================

    return all_theta

