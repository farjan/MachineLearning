
import numpy as np
import matplotlib.pyplot as plt
import supervised.linear_regression_regularized as lreg_reg

class Exercise5(object):
  '''
  Execute exercise No 5 'Regularized Linear Regression and Bias v.s. Variance'.
  For detailed description of the exercise see "Programming Exercise 5" ex5.pdf
  available at coursera.org, an open Stanford University course "Machine Learning"
  by Andrew Ng.
  '''

  def __init__(self, **kwargs):
    '''
    Initialize instance parameters.
    '''
    self.datafname_train = '../data/ex5data1_train.dat'
    self.datafname_cv = '../data/ex5data1_cv.dat'
    self.datafname_test = '../data/ex5data1_test.dat'


  def execute(self):
    '''
    Execute exercise.
    '''
    print('Exercise 5 - Regularized Linear Regression and Bias v.s. Variance.\n')
    # Coursera exercises.
    self.__exercise_parts()


  def __exercise_parts(self):
    '''
    Execute parts of the exercise.
    '''
    p1 = 0
    print('Part 1: Loading and Visualizing Data...')
    Mtrain = np.loadtxt(self.datafname_train, delimiter=',')
    lrr1 = lreg_reg.LinearRegressionRegularized(Mtrain, False, p1)
    lrr1.tail()

    print('Plot training data...')
    lrr1.plot_single_feature(0, 'Change in water level (x)', 
                             'Water flowing out of the dam (y)')

    print('\nPart 2: Regularized Linear Regression Cost...')
    X, y = self._readmatrix(self.datafname_train)
    lmda1 = 1.0
    cost = lreg_reg.LinearRegressionRegularized._cost(np.array([1.0, 1.0]), X, y, lmda1)
    print('Cost at theta [1, 1] should be about 303.993192.')
    print('Cost {} test passed.'.format(cost))

    print('\nPart 3: Regularized Linear Regression Gradient...')
    grad = lreg_reg.LinearRegressionRegularized._gradient(np.array([1.0, 1.0]), X, y, lmda1)
    print('Gradient at theta [1, 1] should be about [-15.303016, 598.250744].')
    print('Gradient {} test passed.'.format(grad))

    print('\nPart 4 Train Linear Regression...')
    lmda1 = 0.0
    # train() internally has X and y.
    lrr1.train(lmda1)
    print('Trained theta \n {}'.format(lrr1._min_theta))
    print('Plot fit over the data...')
    lrr1.plot_regression_line('Change in water level (x)',
                              'Water flowing out of the dam (y)',
                              'Regression Line Fit (lambda = {})'.format(lmda1))

    print('\nPart 5: Learning Curve for Linear Regression...')
    print('Plotting the learning curve...')
    self._plot_learning_curve(lmda1, False, p1, 'Number of training examples', 
                              'Error', 'Learning curve for linear regression')

    print('\nPart 6: Feature Mapping for Polynomial Regression...')
    p2 = 8
    lrr2 = lreg_reg.LinearRegressionRegularized(Mtrain, True, p2)

    print('Normalized Training Example 1:')
    print('{}'.format(lrr2._data_norm[0,:]))

    print('\nPart 7: Learning Curve for Polynomial Regression...')
    print('Training with normalized examples...')
    lmda2 = 0.0
    lrr2.train(lmda2)
    print('Trained theta \n {}'.format(lrr2._min_theta))
    lrr2.plot_regression_line('Change in water level (x)',
                              'Water flowing out of the dam (y)',
                              'Polynomial Regression Fit (lambda = {})'.format(lmda2))
    self._plot_learning_curve(lmda2, True, p2, 'Number of training examples', 'Error', 
                              'Polynomial Regression Learning Curve (lambda = {})'.format(lmda2))

    print('Part 8: Validation for Selecting Lambda...')
    lambda_vec = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    for l in lambda_vec:
      self._plot_learning_curve(l, True, p2, 'Lambda', 'Error', 
                                'Learning Curve (lambda = {})'.format(l))



  def _plot_learning_curve(self, lmda, poly, p, x_label='', y_label='', y_title=''):
    '''
    Plot learning curves..

    Arguments:
      lmda (float): Lambda, the regularization parameter.
      poly (Boolean): True if polynomial covert.
      p (int): Polynomial degree.
      x_label (str): X axis label on the graph.
      y_label (str): Y axis label on the graph.
      title_label (str): Title label.
    '''
    error_train, error_cv = self._learning_curve(lmda, poly, p)
    print('Training error \n {}'.format(error_train))
    print('Cross validation error \n {}'.format(error_cv))

    plt.title(y_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot regression line.
    plt.xlim(0, 13)
    plt.ylim(1, 150)
    plt.plot(error_train[2:], color='blue', label='Train')
    plt.plot(error_cv[2:], color='green', label='Cross Validation')
    plt.legend()
    plt.show()


  def _learning_curve(self, lmda, poly, p):
    '''
    Compute learning curve errors for training and cross validation.

    Arguments:
      lmda (float): Lambda, the regularization parameter.
      poly (Boolean): True if polynomial covert.
      p (int): Polynomial degree.

    Return:
      (error_train m x 1 float vector): Training errors.
      (error_cv m x 1 float vector): Cross validation errors.
    '''
    Mcv = np.loadtxt(self.datafname_cv, delimiter=',')
    Mtrain = np.loadtxt(self.datafname_train, delimiter=',')
    m = Mtrain.shape[0] 
    error_train = np.zeros(m)
    error_cv = np.zeros(m)

    for i in range(2, m):
      lrr = lreg_reg.LinearRegressionRegularized(Mtrain[:i,:], poly, p)
      lrr.train(lmda)
      theta = lrr._min_theta.reshape((1, lrr._min_theta.shape[0]))[0]

      # Compute train/cross validation errors using training examples.
      error_train[i] = lreg_reg.LinearRegressionRegularized._cost(theta, lrr._X, lrr._y, lmda)
      X_cv, y_cv = self._form_X_and_y(poly, p, Mcv, lrr)
      error_cv[i] = lreg_reg.LinearRegressionRegularized._cost(theta, X_cv, y_cv, lmda)
    return error_train, error_cv


  def _readmatrix(self, fname):
    '''
    Read data file and return feature vectors 'X' and
    output vectors 'y'.

    Arguments:
      fname (str): File name with full path.

    Return:
      X (m x n float matrix): Feature vectors.
      y (m x 1 float vector): Output vector.
    '''
    # Matrix vectors, last column is y.
    M = np.loadtxt(fname, delimiter=',')
    rows = M.shape[0]
    cols = M.shape[1] - 1

    X = np.ones(shape=(cols + 1, rows))
    X[1:,:] = M[:,0:cols].transpose()
    y = M[:,cols]
    return X, y


  def _form_X_and_y(self, poly, p, M, lrr):
    '''
    Separate X and y from 'M' matrix, last column
    is 'y', if 'poly' is true it generates polynomial
    of degree 'p' using object 'lrr' for the feature
    in 'M'.

    Arguments:
      poly (Boolean): True if polynomial covert.
      p (int): Polynomial degree.
      M (m x n): Matrix.
      lrr (LinearRegressionRegularized): Class instance.

    Return:
      X (m x n float matrix): Feature vectors.
      y (m x 1 float vector): Output vector.
    '''
    M_tmp = M
    if poly:
      M_tmp = lrr._feature_polynomial5(M[:,[0]], p)
      M_tmp = lrr._feature_normalize(M_tmp)
      M_tmp = np.append(M_tmp, M[:,[1]], 1)

    rows = M_tmp.shape[0]
    cols = M_tmp.shape[1] - 1
    X = np.ones(shape=(cols + 1, rows))
    X[1:,:] = M_tmp[:,0:cols].transpose()
    y = M_tmp[:,cols]
    return X, y
