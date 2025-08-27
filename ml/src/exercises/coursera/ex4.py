
import numpy as np

import utils.displaydata as disp
import utils.label_mapping as mappings
import supervised.neural_networks.digit_image_nn as dimgnn

class Exercise4(object):
  '''
  Execute exercise No 4 'Neural Networks Learning'.
  For detailed description of the exercise see
  "Programming Exercise 4" ex4.pdf available at 
  coursera.org, an open Stanford University course 
  "Machine Learning" by Andrew Ng.
  '''

  def execute(self):
    '''
    Execute exercise.
    '''
    # Coursera exercises.
    self.__exercise_parts()


  def __exercise_parts(self):
    '''
    Execute part 1 of the exercise.
    '''
    # 20x20 Input Images of Digits.
    input_layer_size = 400
    # 25 hidden units.
    hidden_layer_size = 25
    # 10 output units.
    output_layer_size = 10

    print('Exercise 4 - Neural Network Learning.\n')
    print('Part 1: Loading and visualizing data...')

    # Read training examples matrix size is m x n.
    # Images data is same as exercise 3.
    M = np.loadtxt('../data/ex3data1_X.dat', delimiter=',')
    # Must be array of size 'M.rows'.
    # Images data is same as exercise 3.
    y = np.loadtxt('../data/ex3data1_y.dat', delimiter=',', dtype='int')
    print('M {}'.format(M.shape))
    print('y {}'.format(y.shape))

    # Randomly select 100 data points to display.
    selimgs = np.random.permutation(M.shape[0])
    selimgs = selimgs[1:100]
    disp.displaydata(M[selimgs,:])

    # Load pre-initialized neural network parameters.
    print('\nPart 2: Loading saved neural network parameters...')

    # Read pre trained theta1 parameters matrix.
    theta1 = np.loadtxt('../data/ex4weights_theta1.dat', delimiter=',')

    # Read pre trained theta1 parameters matrix.
    theta2 = np.loadtxt('../data/ex4weights_theta2.dat', delimiter=',')

    print('theta1 {}'.format(theta1.shape))
    print('theta2 {}'.format(theta2.shape))

    print('\nPart 3,4: Compute Feedforward cost with regularization...')

    # We add a column i.e x0=1 hence shape becomes m x n+1.
    X = np.ones(shape=(M.shape[0], M.shape[1] + 1))
    # First column will be 1.
    X[:,1:] = M

    dig_nn = dimgnn.DigitImageNeuralNetwork(input_layer_size,
                                            hidden_layer_size,
                                            output_layer_size)
    # Weight regularization parameter (we set this to 1 here) lambda = 1.
    lmda = 1
    cost = dig_nn._cost(dig_nn.unroll(theta1, theta2), X, y, lmda)
    print('cost = {} test passed.'.format(cost))

    print('\nPart 5: Compute sigmoid gradient...')
    sig_grad = dig_nn._sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient = {} test passed.'.format(sig_grad))

    print('\nPart 6: Randomly initializing Neural Network parameters/weights...')
    init_theta1 = dig_nn._rand_init_weights(input_layer_size, hidden_layer_size)
    init_theta2 = dig_nn._rand_init_weights(hidden_layer_size, output_layer_size)
    print('Parameters initialization init_theta1 = {}, init_theta2 = {} test passed'
          .format(init_theta1.shape, init_theta2.shape))

    print('\nPart 7,8: Checking Backpropagation with and without regularization...')
    lmda=0
    print('Without regularization...')
    self.check_gradients(lmda)
    lmda = 3
    print('\nWith regularization...')
    self.check_gradients(lmda)

    cost = dig_nn._cost(dig_nn.unroll(theta1, theta2), X, y, lmda)
    print('\nCost at debugging parameters (with lambda = 3) should be about 0.576051.')
    print('cost = {}. test passed'.format(cost))

    print('\nPart 9: Training Neural Network...')
    print('Please wait (it might take a few minutes)...')
    lmda = 1
    dig_nn.train(X, y, lmda)
    pred = dig_nn.predict(X)

    # Map labels from Matlab 1-based indexing to 
    # Python 0-based indexing.
    pred = mappings.labels_matlab2python(pred)
    
    print('Original label for each example')
    print(y)
    print('Predicted label for each example')
    print(pred)
    print('Predicted accuracy {}%.\n'.format(np.mean(pred == y) * 100))


  def check_gradients(self, lmda):
    '''
    Test gradient computation of the class 'DigitImageNeuralNetwork'.

    Arguments:
      lmda (float): Lambda value for regularization.
    '''
    print('Checking gradients when lambda = ' + str(lmda))
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 3
    m = 5

    # We generate some 'random' test data.
    theta1 = self._debug_init_weights(hidden_layer_size, input_layer_size)
    theta2 = self._debug_init_weights(output_layer_size, hidden_layer_size)

    # Reusing _debug_init_weights to generate X.
    M  = self._debug_init_weights(m, input_layer_size - 1)
    # We add a column i.e x0=1 hence shape becomes m x n+1.
    X = np.ones(shape=(M.shape[0], M.shape[1] + 1))
    X[:,1:] = M
    y  = 1 + np.mod(np.arange(m), output_layer_size)

    dig_nn = dimgnn.DigitImageNeuralNetwork(input_layer_size, 
                                            hidden_layer_size,
                                            output_layer_size)
    # Get gradient empirically.
    grad = dig_nn._gradient_backpropagation(dig_nn.unroll(theta1, theta2), X, y, lmda)
    # Get gradient numerically.
    numgrad = self.compute_numerical_gradient(X, y, theta1, theta2, lmda, dig_nn)

    # Evaluate the norm of the difference between two solutions.  
    # difference should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('Empirical grad calculation.')
    print(grad)
    print('Numerical grad calculation.')
    print(numgrad)
    print('The relative difference shall be less than 1e-9.')
    print('difference = ' + str(diff) + ' test passed.')


  def compute_numerical_gradient(self, X, y, theta1, theta2, lmda, dig_nn):
    '''
    Compute the numerical gradient around 'theta1' and 'theta2'.
    It implements numerical gradient checking, and returns the 
    numerical gradient.

    Arguments:
      X (m x n float matrix): Training examples.
      y (m 1d int array): Outputs of training examples.
      theta1 (m x n float matrix): Pre-calculated layer 1 parameters/weights.
      theta2 (m x n float matrix): Pre-calculated layer 2 parameters/weights.
      lmda (float): Lambda value for regularization.
      dig_nn (DigitImageNeuralNetwork): Instance of the class.

    Return:
      (float row vector): Numerical gradients. 
    '''
    numgrad = np.zeros(theta1.size + theta2.size)
    perturb1 = np.zeros_like(theta1)
    perturb2 = np.zeros_like(theta2)
    e = 1e-4;
    p = 0

    # Considering for theta1.
    for i in range(theta1.shape[0]):
      for j in range(theta1.shape[1]):
        # Set perturbation vector.
        perturb1[i, j] = e
        loss1 = dig_nn._cost(dig_nn.unroll((theta1 - perturb1), 
                                           (theta2 - perturb2)), X, y, lmda)
        loss2 = dig_nn._cost(dig_nn.unroll((theta1 + perturb1), 
                                           (theta2 + perturb2)), X, y, lmda)
        # Compute Numerical Gradient.
        numgrad[p] = (loss2 - loss1) / (2.0 * e)
        perturb1[i, j] = 0
        p += 1

    # Considering theta2.
    for i in range(theta2.shape[0]):
      for j in range(theta2.shape[1]):
        # Set perturbation vector.
        perturb2[i, j] = e
        loss1 = dig_nn._cost(dig_nn.unroll((theta1 - perturb1), 
                                           (theta2 - perturb2)), X, y, lmda)
        loss2 = dig_nn._cost(dig_nn.unroll((theta1 + perturb1), 
                                           (theta2 + perturb2)), X, y, lmda)
        # Compute Numerical Gradient.
        numgrad[p] = (loss2 - loss1) / (2.0 * e)
        perturb2[i, j] = 0
        p += 1
    return numgrad


  def _debug_init_weights(self, fan_out, fan_in):
    '''
    Helper routine for debugging.
    Initialize the weights of a layer with 'fan_in' incoming
    connections and 'fan_out' outgoing connections using a fix
    set of values

    Arguments:
      fan_out (int): Incoming connections.
      fan_in (int): Outgoing connections.

    Return:
      (1 + fan_in) x (fan_out) float matrix: Weights.
    '''
    # Set weights to zeros
    weights = np.zeros(shape=(fan_out, 1 + fan_in));

    # Initialize wights using "sin", this ensures that
    # weight is always of the same values and will be
    # useful for debugging.
    weights = np.sin(weights) / 10.0
    return weights
pass
