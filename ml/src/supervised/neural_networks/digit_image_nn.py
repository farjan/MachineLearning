
import numpy as np
import utils.label_mapping as mapping
import scipy.optimize as opt
import copy

class DigitImageNeuralNetwork(object):
  '''
  Neural Network implementation to predict
  20x20 pixel digit (0,1...9) image.
  '''

  def __init__(self, input_layer_size, hidden_layer_size,
               output_layer_size):
    '''
    Initialize attributes.
    '''
    self.min_cost = None
    self.min_theta1 = None
    self.min_theta2 = None

    self.input_layer_size  = input_layer_size
    self.hidden_layer_size = hidden_layer_size
    self.output_layer_size = output_layer_size


  def train(self, X, y, lmda):
    '''
    Train the neural networks parameters. Call scipy's
    fming_cg function for optimization.

    Arguments:
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.
    '''
    theta1 = self._rand_init_weights(self.input_layer_size,
                                     self.hidden_layer_size)
    theta2 = self._rand_init_weights(self.hidden_layer_size, 
                                     self.output_layer_size)
    theta = self.unroll(theta1, theta2)
    args = (X, y, lmda)

    res = opt.fmin_cg(self._cost, 
                      x0=theta, 
                      fprime=self._gradient_backpropagation,
                      args=args, 
                      maxiter=50, 
                      disp=False, 
                      full_output=True)
    # Save min cost and thetas.
    self.min_theta1, self.min_theta2 = self.roll(res[0])
    self.min_cost = res[1]
    print('cost {}'.format(self.min_cost))


  def predict(self, V):
    '''
    Predict for test examples, remember labels will
    be according to Matlab indexing, there is need
    to convert it accordingly. The labels are in the
    range 1..K, where K = size(all_theta, 1).

    Arguments:
      V (m x n float matrix): Training examples.

    Return:
      (1d int row vector): Predicted labels.
    '''
    m = V.shape[0]
    labels = np.zeros(m, dtype=np.int32)
    theta1 = self.min_theta1.transpose()
    theta2 = self.min_theta2.transpose()

    for i in range(m):
      h = self._forward(V[[i],:], theta1, theta2)
      labels[i] = np.argmax(h)
    return labels


  def roll(self, theta):
    '''
    Roll theta1 and theta2 matrices.

    Arguments:
      theta (unrolled thetas vector): Unrolled theta1 and theta2.

    Return:
      (theta1 matrix): Rolled theta1 matrix.
      (theta2 matrix): Rolled theta2 matrix.
    '''
    boundary = self.hidden_layer_size * (self.input_layer_size + 1)
    theta1_shape = (self.hidden_layer_size, (self.input_layer_size + 1))
    theta2_shape = (self.output_layer_size, (self.hidden_layer_size + 1))

    theta1 = np.reshape(theta[:boundary], theta1_shape)
    theta2 = np.reshape(theta[boundary:], theta2_shape)
    return theta1, theta2


  def unroll(self, theta1, theta2):
    '''
    Unroll theta1 and theta2 into a vector.

    Arguments:
      theta1 (matrix): theta1 matrix.
      theta2 (matrix): theta2 matrix.

    Return:
      (thetas vector): Unrolled theta1 and theta1 into a vector.
    '''
    return np.concatenate((theta1.ravel(), theta2.ravel()))


  def _cost(self, theta, X, y, lmda):
    '''
    Compute cost for given 'theta1' and 'theta2'.
    Assumes 'X' has column of 1s i.e x0=1 added.

    Arguments:
      theta (float vector): Parameters/weights.
      X (m x n float matrix):  Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (float): Cost value.
    '''
    # Form thetas.
    theta1, theta2 = self.roll(theta)
    # We need transposed in computation.
    theta1 = theta1.transpose()
    theta2 = theta2.transpose()

    cost = 0.0
    m = X.shape[0]
    for i in range(m):
      h = self._forward(X[[i],:], theta1, theta2)

      # Create a column vector for y output of the
      # current row.
      y_vect = np.zeros(shape=(self.output_layer_size, 1))
      # Set the cell that has output of the image.
      y_vect[mapping.output_layer_index_matlab(y[i]), 0] = 1

      likelihood = 0.0
      for k in range(h.shape[1]):
        hk = h[0,k]
        yk = y_vect[k,0]
        likelihood += -(yk * np.log(hk)) - ((1.0 - yk) * np.log(1.0 - hk))
      cost += likelihood

    # Regularization theta1
    reg1 = 0.0
    for k in range(1, theta1.shape[0]):
      for j in range(0, theta1.shape[1]):
        reg1 += theta1[k, j] ** 2

    # Regularization theta2
    reg2 = 0.0
    for k in range(1, theta2.shape[0]):
      for j in range(0, theta2.shape[1]):
        reg2 += theta2[k, j] ** 2
    c = (cost / m) + ((reg1 + reg2) * (lmda / (2 * m)))
    return (cost / m) + ((reg1 + reg2) * (lmda / (2 * m)))


  def _gradient_backpropagation(self, theta, X, y, lmda):
    '''
    Compute gradient for given 'theta1' and 'theta2'.
    Assumes 'X' has column of 1s i.e x0=1 added.

    Arguments:
      theta (float vector): Parameters/weights.
      X (m x n float matrix):  Training examples.
      y (m 1d int array): Outputs of training examples.
      lmda (float): Lambda value for regularization.

    Return:
      (float vector): Unrolled theta1 and theta2 vector.
    '''
    # Form thetas.
    theta1, theta2 = self.roll(theta)
    gradient1 = np.zeros_like(theta1)
    gradient2 = np.zeros_like(theta2)

    # We need transposed in computation.
    theta1 = theta1.transpose()
    theta2 = theta2.transpose()

    m = X.shape[0]
    for i in range(m):
      a1 = X[[i],:]
      z2 = np.dot(a1, theta1)
      a2 = np.ones(shape=(a1.shape[0], theta2.shape[0]))
      a2[:,1:] = self.__sigmoid(z2)
      z3 = np.dot(a2, theta2)
      a3 = self.__sigmoid(z3)

      # Create a column vector for y output of the current
      # row. Set the cell that has output of the image.
      yk = np.zeros(shape=(self.output_layer_size, 1))
      yk[mapping.output_layer_index_matlab(y[i]), 0] = 1

      # 'Error term' for output layer.
      delta3 = a3.transpose() - yk

      # 'Error term' for hidden layer.
      gprime = np.ones(shape=(z2.shape[0], z2.shape[1] + 1))
      gprime[:,1:] = self._sigmoid_gradient(z2)
      gprime = gprime.transpose()
      delta2 = theta2.dot(delta3) * gprime
      # deleting bias delta2[0].
      delta2 = delta2[1:,:]

      # Accumulate the gradient from this example.
      # Layer1.
      gradient1 = gradient1 + delta2.dot(a1)
      # Layer2.
      gradient2 = gradient2 + delta3.dot(a2)

    gradient1 /= m
    gradient2 /= m
    theta1 = theta1.transpose()
    theta2 = theta2.transpose()
    # Add regularization.
    gradient1[:,1:] += (theta1[:,1:] * (lmda / m))
    gradient2[:,1:] += (theta2[:,1:] * (lmda / m))

    return self.unroll(gradient1, gradient2)


  def _forward(self, X, theta1, theta2):
    '''
    Compute hypothesis of an input given a 3 layers trained
    neural network by forwarding.

    Arguments:
      X (m x n float matrix):  Training examples.
      theta1 (m x n float matrix): Pre-calculated layer 1 parameters/weights.
      theta2 (m x n float matrix): Pre-calculated layer 2 parameters/weights.

    Return:
      (1d float array): Hypothetical values row vector.
    '''
    z2 = np.dot(X, theta1)
    a2 = np.ones(shape=(X.shape[0], theta2.shape[0]))
    a2[:,1:] = self.__sigmoid(z2)
    z3 = np.dot(a2, theta2)
    hyp = self.__sigmoid(z3)
    return hyp


  def _sigmoid_gradient(self, z):
    '''
    Compute sigmoid gradient.

    Arguments:
      z (scalar, vector, matrix): Neuron or unit value.

    Return:
      Sigmoid gradient values, same size as 'z'.
    '''
    g = self.__sigmoid(z)
    return g * (1.0 - g)


  def __sigmoid(self, z):
    '''
    Sigmoid function for scalar, vector or matrix.

    Arguments:
      z (scalar, vector, matrix): Neuron or unit value.

    Return:
      Sigmoid values, same size as 'z'.
    '''
    return 1.0 / (1.0 + np.exp(-z))


  def _rand_init_weights(self, layer_in, layer_out):
    '''
    Randomly initialize the weights of a layer with 'layer_in'
    incoming connections and 'layer_out' outgoing connections.

    Arguments:
      layer_in (int): Incoming connections.
      layer_out (int): Outgoing connections.

    Return:
      (m * n float matrix): Randomly generated matrix.
    '''
    # Strategy for random initialization is to randomly 
    # select values for 'theta^layer' uniformly in the
    # range [-eps, eps] where eps = 0.12.
    # This range of values ensures that the parameters
    # are kept small and makes the learning more efficient.
    eps = 0.12
    weights = (np.random.rand(layer_out, 1 + layer_in) * 2 * eps) - eps
    return weights
  pass
