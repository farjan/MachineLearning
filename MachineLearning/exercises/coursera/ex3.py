
import numpy as np

import supervised.neural_networks.forward_propagation as fp
import supervised.logistic_regression_regularized_multiclass as lrrm
import utils.displaydata as disp
import utils.label_mapping as mappings

class Exercise3(object):
  '''
  Execute exercise No 3 
  'Multi-class Classification and Neural Networks'.
  For detailed description of the exercise see
  "Programming Exercise 3" ex3.pdf available at 
  coursera.org, an open Stanford University course 
  "Machine Learning" by Andrew Ng.
  '''

  def execute(self):
    '''
    Execute exercise and return loaded data from the files.

    Return:
      (m x n float matrix): Loaded training examples.
      (m 1d int array): Loaded output of each training example.
    '''
    print('Exercise 3 - Multi-class Classification and Neural Networks.\n')
    # Coursera exercises.
    M, y = self.__exercise_part1()
    self.__exercise_part2(M, y)

  
  def __exercise_part1(self):
    '''
    Execute part 1 of the exercise.
    '''
    print('Exercise 3 - Part 1: One-vs-all.\n')
    print('Part 1: Loading and visualizing data...')

    # Read training examples matrix size is m x n.
    M = np.loadtxt('data/ex3data1_X.dat', delimiter=',')
    # Must be array of size 'M.rows'.
    y = np.loadtxt('data/ex3data1_y.dat', delimiter=',', dtype='int');
    print('M {}'.format(M.shape))
    print('y {}'.format(y.shape))

    # Randomly select 100 data points to display.
    selimgs = np.random.permutation(M.shape[0])
    selimgs = selimgs[1:100]
    disp.displaydata(M[selimgs,:])

    print('Part 2: Training One-vs-All Logistic Regression Regularized...')
    classifiers = lrrm.LogisticRegRegularizedMulticlass()
    # Make a call to vectorized implementation.
    classifiers.one_vs_all_vectorized(M, y.reshape((len(y), 1)),
                                      mappings.labels)
    # Following is non vectorized call, to test uncomment and
    # comment vectorized call above. Non vectorized call is slower.
    #classifiers.one_vs_all_non_vectorized(M, y.reshape((len(y), 1)),
    #                                  mappings.labels)

    print('Part 3: Predicting for One-Vs-All...')
    pred = classifiers.predict_one_vs_all(M)
    print('Original label for each example')
    print(y)
    print('Predicted label for each example')
    print(pred)
    print('Predicted accuracy {}%.\n'.format(np.mean(pred == y) * 100))
    return M, y


  def __exercise_part2(self, M, y):
    '''
    Execute part 2 of the exercise.

    Argument:
      M (m x n float matrix): Training examples.
      y (m 1d int array): Output of each training example.
    '''
    print('\n')
    print('Exercise 3 - Part 2: Neural Networks.\n')
    print('Part 1: Loading and visualizing data, already done in exercise part 1.')
    print('Part 2: Loading Parameters...')

    # 20x20 Input Images of Digits.
    input_layer_size = 400;
    # 25 hidden units.
    hidden_layer_size = 25;
    # 10 output units.
    output_layer_size = 10;

    # Read pre trained theta1 parameters matrix is 25x401 
    # but we transpose hence shape=401x25.
    theta1 = np.transpose(np.loadtxt('data/ex3weights_theta1.dat', delimiter=','))

    # Read pre trained theta1 parameters matrix is 26x10 
    # but we transpose hence shape=26x10.
    theta2 = np.transpose(np.loadtxt('data/ex3weights_theta2.dat', delimiter=','))

    print('Parameters after transpose.')
    print('theta1 {}'.format(theta1.shape))
    print('theta2 {}'.format(theta2.shape))

    # Data read completes now make predictions.
    print('Part 3: Predicting using Forward Propagation.')
    nn_fp = fp.ForwardPropagation()
    pred = nn_fp.predict(M, theta1, theta2)

    # Map labels from Matlab 1-based indexing to 
    # Python 0-based indexing.
    pred = mappings.labels_matlab2python(pred)

    print('Original label for each example')
    print(y)
    print('Predicted label for each example')
    print(pred)
    print('Predicted accuracy {}%.\n'.format(np.mean(pred == y) * 100))
