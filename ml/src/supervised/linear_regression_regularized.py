
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class LinearRegressionRegularized(object):
  '''
  Linear regression classifier it is regularized version. 
  Hypothesis for the classifier is polynomial 5 based.

  Other than instructor's (Andrew Ng) coursera.org course lectures
  and notes you can get help at instructor's course website for 
  following implementation. Here is the link.
  http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
  '''
  def __init__(self, M, poly, p):
    '''
    Initialize instance parameters, if 'poly' is true
    normalization will also be performed.

    Arguments:
      M (m x n float matrix): Float feature vectors
        where last column is y (output).
      poly (Boolean): Is us polynomial.
      p (int): Polynomial degree.
    '''
    self._poly = poly
    self._p = p
    self._data = M

    self._min_theta = None
    # Array to keep J(theta) for each iteration.
    self._costs = None

    # Stores means and standard deviations used
    # for normalization if data is normalized.
    self._means = None
    self._stds = None

    # Stores normalized data.
    self._data_norm = None
    if self._poly:
      self._data_norm = self._feature_polynomial5(M[:,[0]], self._p)
      self._means = np.mean(self._data_norm, axis=0)
      self._stds = np.std(self._data_norm, axis=0)
      self._data_norm = self._feature_normalize(self._data_norm)
      self._data_norm = np.append(self._data_norm, M[:,[1]], 1)
    else:
      # Feature matrix, last column is y.
      self._data_norm = M

    # Number of training samples
    self._m = self._data_norm.shape[0]
    # Remove count of y from x.
    xcols = self._data_norm.shape[1] - 1

    # Though input feature vectors are row vectors but
    # for the operational ease it is better to keep 
    # them as column vectors, that way it resembles
    # the notation of formula.
    
    # Remember x0 = 1 , is required by the algorithm.
    # So it is two dimensional array, notice the
    # shaped of the array, if you had 80 rows in
    # your input data it will become 80 columns
    # and first row always be 1s since x0 =1 i.e
    # 1   1  1  . .
    # a1  b1 c1 . .
    # a2  b2 c2 . .
    # .   .  .  . .
    # .   .  .  . .
    self._X = np.ones(shape=(xcols + 1, self._m))
    self._X[1:,:] = self._data_norm[:,0:xcols].transpose()

    # y keeps last column of each feature vector
    # of input data, 0 or 1 such as (see last column)
    # a1, a2, a3, 1
    # b1, b2, b3, 0
    # c1, c2, c3, 0
    self._y = self._data_norm[:,xcols]


  def tail(self, no_lines=3):
    '''
    Show last few rows of original data (without transformation).

    Arguments:
      no_lines (int): Number of lines to show.
    '''
    if self._m >= no_lines:
      print(self._data[self._m - no_lines:self._m,], end='\n\n')
    else:
      print(self._data[:,:], end='\n\n')


  def train(self, lmda):
    '''
    Train using training examples, call scipy's fming_cg
    function for optimization.

    Arguments:
      lmda (float): Lambda value for regularization.

    Return:
      (float): Cost.
      (1d float array): Minimum thetas.
    '''
    # Reset before training.
    self._min_theta = None
    self._costs = None

    # Initialize theta and args for fmin_cg.
    theta = np.zeros(self._X.shape[0])
    args = (self._X, self._y, lmda)

    res = opt.fmin_cg(LinearRegressionRegularized._cost, 
                      x0=theta, 
                      fprime=LinearRegressionRegularized._gradient,
                      args=args, 
                      maxiter=200, 
                      disp=False, 
                      full_output=True)
    self._min_theta = res[0]
    self._costs = res[1]


  def predict(self, feature_vect):
    '''
    Predict for the given feature vector.
    This function should be called after "train".

    Arguments:
      feature_vect (n x 1 float vector): Features vector.

    Return:
      (float): A predicted value.
    '''
    # If features were normalized, there is need
    # to normalize vector before predicting.
    if self._poly:
      X_poly = self._feature_polynomial5(feature_vect, self._p)
      feature_vect = self._feature_normalize(X_poly)
      feature_vect = feature_vect.reshape((feature_vect.shape[1], 1))

    x = np.ones(shape=(feature_vect.shape[0] + 1, 1))
    x[1:,] = feature_vect[:,]
    return np.transpose(self._min_theta).dot(x)[0]


  def plot_single_feature(self, f_idx=0, x_label='',  y_label=''):
    '''
    Plot a single feature and the decision line.

    Arguments:
      f_idx (int): Feature index in the data file.
      x_label (string): X axis label on the graph.
      y_label (string): Y axis label on the graph.
    '''
    # Regression line plot, first plot the feature.
    plt.title('Scatter Plot')
    plt.scatter(self._data[:, f_idx], self._data[:, self._data.shape[1] - 1],
                color='red', marker='x')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


  def plot_regression_line(self, x_label='', y_label='', title_label=''):
    '''
    Plot a single feature and the decision line. 'train' function
    must be called before calling this function.

    Arguments:
      x_label (str): X axis label on the graph.
      y_label (str): Y axis label on the graph.
      title_label (str): Title label.
    '''
    # Regression line plot, first plot the feature.
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.scatter(self._data[:, 0], self._data[:, self._data.shape[1] - 1],
                color='red', marker='x')
    x = np.linspace((np.min(self._data[:,0]) - 10),
                    (np.max(self._data[:,0]) + 10), 100)
    y = [self.predict(np.array([[xi]])) for xi in x]
    plt.plot(x, y, color='blue')
    plt.show()


  @staticmethod
  def _gradient(theta, X, y, lmda):
    '''
    Compute gradient descent to learn theta values.

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
        the training data.
      lmda (float): Lambda, the regularization parameter.

    Return:
      (n x 1 float vector): A gradient values vector, size of
        vector will match 'theta' vector.
    '''
    m = X.shape[1]
    theta = theta.reshape((len(theta), 1))
    grad = np.zeros_like(theta)
    reg = np.zeros_like(theta)
    reg[:,:] = theta * (lmda / m)
    reg[0,:] = 0

    # Iterate over columns.
    for i, yi in enumerate(y):
      xi = X[:, [i]]
      h =  np.transpose(theta).dot(xi)
      grad += (h - yi) * xi
    grad = (grad * (1.0 / m)) + reg
    return grad.reshape((1, grad.shape[0]))[0]


  @staticmethod
  def _cost(theta, X, y, lmda):
    '''
    Compute cost of the hypothesis for the given
    theta values.

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
      lmda (float): Lambda, the regularization parameter.

    Return:
      (float): The cost J for the given theta values vector.
    '''
    theta = theta.reshape((len(theta), 1))
    sqerr = 0.0

    # Iterate over columns.
    m = X.shape[1]
    for i in range(X.shape[1]):
      h = np.transpose(theta).dot(X[:, [i]])[0]
      sqerr += ((h - y[i]) ** 2)
    # J(theta) is cost of the hypothesis.
    sqerr = sqerr / (2.0 * m)
    reg = (lmda / (2.0 * m)) * (np.sum(theta ** 2))
    return sqerr + reg


  def _feature_normalize(self, f):
    '''
    Normalize data where the mean value of each feature
    is 0 and the standard deviation is 1. 
    This is often a good preprocessing step to do when 
    working with learning algorithms.

    Arguments:
      f (n x 1 float vector): Feature values vector.

    Return:
      (n x 1 float vector): Normalized '.f'
    '''
    # Scale features and set them to zero mean.
    for j in range(f.shape[1]):
      f[:,j] = (f[:,j] - self._means[j]) / self._stds[j]
    return f


  def _feature_polynomial5(self, f, p):
    '''
    Compute 5th degree polynomial.
    We have only one features f hence 5th degree polynomial.
    x = 1, f1, f2, f1^2, f1f2, f2^2, f1^3...f1f2^5, f2^6

    Arguments:
      f (n x 1 float vector): Feature values vector.
      p (int): Degree.

    Return:
      (n x 5 float matrix): 5-feature vectors.
    '''
    out = np.zeros((f.shape[0], p))

    for i in range(f.shape[0]):
      for j in range(p):
        out[i, j] = np.power(f[i], j+1)
    return out
pass
