
import numpy as np

class ForwardPropagation(object):
  '''Forward propagate 3 layers neural network.'''

  def predict(self, M, theta1, theta2):
    '''
    Predict the label of an input given a 3 layers trained
    neural network by forwarding.
    Outputs the predicted label of M given the trained weights
    of a neural network (theta1, theta2). 
    
    Predicted labels are digits 0,1,2...9 but remember it is
    according to Python 0-based indexing, you need mapping.

    Remember in assignment labels were according to
    Matlab indexing which is 1-based indexing.

    Hence 1,2..9 are as 1,2..9 but 0 is as 10. For
    details see the description of Assignment No 3.

    Arguments:
      M (m x n float matrix):  Training examples.
      theta1 (m x n float matrix): Pre-calculated layer 1 parameters/weights.
      theta2 (m x n float matrix): Pre-calculated layer 2 parameters/weights.

    Return:
    (1d int array): Predicted label of M, labels are digits 0,1,2...9 but remember
      it is according to Python 0-based indexing, you need mapping.
    '''
    # We add a column i.e x0=1 hence shape 
    # becomes m x n+1.
    X = np.ones(shape=(M.shape[0], M.shape[1] + 1))
    # First column will be 1.
    X[:,1:] = M

    z2 = np.dot(X, theta1)
    a2 = np.ones(shape=(X.shape[0], theta2.shape[0]))
    a2[:,1:] = self.__sigmoid(z2)
    z3 = np.dot(a2, theta2)
    hyp = self.__sigmoid(z3)
    return np.argmax(hyp, axis=1)


  def __sigmoid(self, z):
    '''
    Sigmoid function for scalar, vector or matrix.

    Arguments:
      z (scalar, vector, matrix): Neuron or unit value.
    Return:
      Sigmoid values, same size as 'z'.
    '''
    return 1.0 / (1.0 + np.exp(-z))
pass

