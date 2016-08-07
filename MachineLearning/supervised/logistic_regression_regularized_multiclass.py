
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opt
import copy

class LogisticRegRegularizedMulticlass(object):
  '''
  Logistic regression regularized multi-class classifier,
  implements both vectorized and non vectorized versions.
  '''

  def __init__(self):
    '''
    Initialize attributes.
    '''
    # Number of training samples.
    self.m = 0
    self.classes_min_theta = list()
    # Array to keep J(theta) for each iteration.
    self.classes_costs = list()


  def predict_one_vs_all(self, V):
    '''
    Predict the label for a trained one-vs-all classifier. 
    The labels are in the range 1..K, where K = size(all_theta, 1).

    Arguments:
      V (m x n float matrix): Test examples matrix.

    Return:
      (1d int array): Predicted labels for each test example in the
      matrix V.
    '''
    # We add a column i.e x0=1 hence shape 
    # becomes m x n+1.
    X = np.ones(shape=(V.shape[0], V.shape[1] + 1))
    # First column will be 1.
    X[:,1:] = V
    labels = np.zeros(X.shape[0], dtype=np.int32)

    for i, x in enumerate(X):
      predictions = [np.sum(x * t) for t in self.classes_min_theta]
      labels[i] = np.argmax(predictions)
    return labels


  def one_vs_all_vectorized(self, M, y, num_labels, lmda=0.1):
    '''
    Train multiple logistic regression classifiers using
    vectorized algorithm and saves all the classifiers internally.

    Arguments:
      M (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      num_labels (1d int array): Labels of images such 0...9
      lmda (float): Lambda value for regularization.
    '''
    print('Executing vectorized version of algorithm...')

    # We add a column i.e x0=1 hence shape 
    # becomes m x n+1.
    X = np.ones(shape=(M.shape[0], M.shape[1] + 1))
    # First column will be 1.
    X[:,1:] = M
    self.m = X.shape[0]

    # Labels are nothing but digits 0,1..9.
    for i in num_labels:
      label = copy.deepcopy(y)
      # Set outputs negatives or positives.
      label[label == i] = -1  # Set temp flag.
      label[label != -1] = 0  # Set negative.
      label[label == -1] = 1  # Set negative.

      # Train and get cost min theta and cost for
      # each class.
      cost, min_theta = self._train_vectorized(X, label, lmda)
      self.classes_costs.append(cost)
      self.classes_min_theta.append(min_theta)
      print('{} min cost estimation for label {}'.format(cost, i))


  def _train_vectorized(self, X, y, lmda):
    '''
    Call scipy's fming_cg function for optimization for
    vectorized algorithm.

    Arguments:
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (float): Cost.
      (1d float array): Minimum thetas.
    '''
    theta = np.zeros(X.shape[1])
    args = (X, y, lmda)
    res = opt.fmin_cg(self._cost_vectorized, 
                      x0=theta, 
                      fprime=self._gradient_vectorized,
                      args=args, 
                      maxiter=200, 
                      disp=False, 
                      full_output=True)
    return res[1], res[0] # cost, thetas


  def _gradient_vectorized(self, theta, X, y, lmda):
    '''
    Compute gradient for vectorized algorithm.

    Arguments:
      theta (1d float array): Theta or parameter.
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (1d float array): Gradients.
    '''
    theta = theta.reshape((X.shape[1], 1))

    # Compute gradients.
    Z = X.dot(theta)
    h = 1.0 / (1.0 + np.exp(-Z))
    grads = X.transpose().dot(h - y) / self.m

    # Add regularization.
    t = theta[:,:]
    t[0] = 0
    grads += (lmda / self.m) * t
    return grads.flatten()


  def _cost_vectorized(self, theta, X, y, lmda):
    '''
    Compute cost for vectorized algorithm.

    Arguments:
      theta (1d float array): Theta or parameter.
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (float): The cost J for the given theta values vector.
    '''
    theta = theta.reshape((X.shape[1], 1))

    # Compute cost.
    Z = X.dot(theta)
    h = 1.0 / (1.0 + np.exp(-Z))
    likelihood = np.sum((y * np.log(h)) + ((1.0 - y) * np.log(1.0 - h)))

    # Add regularization.
    Jtheta_reg = (lmda / (2 * self.m)) * np.sum(theta ** 2)
    cost = -(likelihood / self.m) + Jtheta_reg
    return cost


  def one_vs_all_non_vectorized(self, M, y, num_labels, lmda=0.1):
    '''
    Train multiple logistic regression classifiers using
    non vectorized algorithm and saves all the classifiers internally.

    Arguments:
      M (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      num_labels (1d int array): Labels of images such 0...9
      lmda (float): Lambda value for regularization.
    '''
    print('Executing non vectorized version of algorithm...')

    # We add a column i.e x0=1 hence shape 
    # becomes m x n+1.
    X = np.ones(shape=(M.shape[0], M.shape[1] + 1))
    # First column will be 1.
    X[:,1:] = M
    self.m = X.shape[0]
    X = np.transpose(X)

    # Labels are nothing but digits 0,1..9.
    for i in num_labels:
      label = copy.deepcopy(y)
      # Set outputs negatives or positives.
      label[label == i] = -1  # Set temp flag.
      label[label != -1] = 0  # Set negative.
      label[label == -1] = 1  # Set negative.

      # Train and get cost min theta and cost for
      # each class.
      cost, min_theta = self._train_non_vectorized(X, label, lmda)
      self.classes_costs.append(cost)
      self.classes_min_theta.append(min_theta)
      print('{} min cost estimation for label {}'.format(cost, i))


  def _train_non_vectorized(self, X, y, lmda):
    '''
    Call scipy's fming_cg function for optimization for
    non vectorized algorithm.

    Arguments:
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (float): Cost.
      (1d float array): Minimum thetas.
    '''
    theta = np.zeros(X.shape[0])
    args = (X, y, lmda)
    res = opt.fmin_cg(self._cost__non_vectorized, 
                      x0=theta, 
                      fprime=self._gradient__non_vectorized,
                      args=args, 
                      maxiter=200, 
                      disp=False, 
                      full_output=True)
    return res[1], res[0] # cost, thetas


  def _gradient__non_vectorized(self, theta, X, y, lmda):
    '''
    Compute gradient descent to learn theta values for
    non vectorized algorithm.

    Arguments:
      theta (n x 1 float vector): Theta values vector.
      X (n x m float matrix): Feature values vectors, e.g.
        1   1  1  . .
        a1  b1 c1 . .
        a2  b2 c2 . .
        .   .  .  . .
        .   .  .  . .
        Notice dimension of feature vectors are by columns 
        NOT by rows.
      y (1d float array): Output of the features vectors in 
        the training data such as 0s or 1s.
      lmda (float): Lambda value for regularization.

    Return:
      (1d float array): A gradient values vector, size of
        vector will match 'theta' vector.
    '''
    theta = theta.reshape((X.shape[0], 1))
    grad = np.zeros_like(theta)

    # Iterate over columns.
    for i, yi in enumerate(y):
      xi = X[:, [i]]
      h = self._sigmoid__non_vectorized(xi, theta)
      grad += (h - yi) * xi

    # Grads regularization, same shape as theta.
    grads_reg = np.zeros_like(grad)
    grads_reg = (lmda / self.m) * theta
    # Set extra term.
    grads_reg[0] = 0
    grad = (grad / self.m) + grads_reg
    return grad.flatten()


  def _cost__non_vectorized(self, theta, X, y, lmda):
    '''
    Compute cost of the hypothesis for the given
    theta values for non vectorized algorithm.

    Arguments:
      theta (n x 1 float vector): Theta values vector.
      X (n x m float matrix): Feature values vectors, e.g.
        1   1  1  . .
        a1  b1 c1 . .
        a2  b2 c2 . .
        .   .  .  . .
        .   .  .  . .
        Notice dimension of feature vectors are by columns 
        NOT by rows.
      y (1d float array): Output of the features vectors in 
        the training data such as 0s or 1s.
      lmda (float): Lambda value for regularization.


    Return:
      (float): The cost J for the given theta values vector.
    '''
    theta = theta.reshape((X.shape[0], 1))
    likelihood = 0.0

    # Iterate over columns.
    for i in range(X.shape[1]):
      h = self._sigmoid__non_vectorized(X[:, [i]], theta)
      likelihood += ((y[i] * math.log(h)) + ((1.0 - y[i]) * math.log(1.0 - h)))

    # J(theta) is cost of the hypothesis.
    Jtheta_reg = ((lmda / (2 * self.m)) * (np.sum(theta ** 2)))
    cost = -(likelihood / (self.m)) + Jtheta_reg
    return cost[0]


  def _sigmoid__non_vectorized(self, xi, theta):
    '''
    Compute sigmoid function.

    Arguments:
      xi (n x 1 float vector): Feature values vector.
      theta (n x 1 float vector): Theta values vector.

    Return:
      (float): A sigmoid value.
    '''
    z = np.transpose(theta).dot(xi)
    return 1.0 / (1.0 + math.exp(-z))
pass
