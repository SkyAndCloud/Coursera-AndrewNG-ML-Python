import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]
   
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

# =========================================================================
    X = np.column_stack((np.ones((m, 1)), X))
    a2 = sigmoid(X.dot(Theta1.T))
    a2 = np.column_stack((np.ones((m, 1)), a2))
    a3 = sigmoid(a2.dot(Theta2.T))
    p = np.argmax(a3, axis=1)

    return p + 1        # add 1 to offset index of maximum in A row

