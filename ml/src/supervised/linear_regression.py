
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
  '''
  Linear regression classifier it is non regularized
  version, it can handle n feature variables. 
  Hypothesis for the classifier is a linear line.

  Other than instructor's (Andrew Ng) coursera.org course lectures
  and notes you can get help at instructor's course website for 
  following implementation. Here are the links.
  http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
  http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
  '''
  def __init__(self, datafname = ''):
    '''
    Initialize by reading the data file.

    Arguments:
      datafname (string): Full path of data file,
        a csv like file is expected where the last column
        should be output 'y' values.
    '''
    # Feature matrix, last column is y.
    self.data = np.loadtxt(datafname, delimiter=',')

    # Number of training samples
    self.m = self.data.shape[0]
    self.min_theta = None

    # Array to keep J(theta) for each iteration.
    self.costs = None

    # Stores means and standard deviations used
    # for normalization if data is normalized.
    self.means = None
    self.stds = None


  def normalize_feature(self):
    '''
    Normalize data where the mean value of each feature
    is 0 and the standard deviation is 1. 
    This is often a good preprocessing step to do when 
    working with learning algorithms.
    '''
    col = self.data.shape[1]
    # Exclude last column which is y.
    self.means = np.mean(self.data, axis=0)[:col-1]
    self.stds = np.std(self.data, axis=0)[:col-1]

    # Scale features and set them to zero mean.
    # Exclude last column which is y.
    for j in range(col - 1):
      self.data[:,j] = (self.data[:,j] - self.means[j]) / self.stds[j]


  def tail(self, no_lines=3):
    '''
    Show last few rows.

    Arguments:
      no_lines (int): Number of lines to show.
    '''
    if self.m >= no_lines:
      print(self.data[self.m - no_lines:self.m,], end='\n\n')
    else:
      print(self.data[:,:], end='\n\n')


  def train(self, alpha=0.07, iter_min=1500):
    '''
    Train from training data.

    Arguments:
      alpha (float): learning rate default to 0.07.
      iter_min (int): Iterations to converge theta values
      default to 1500.
    '''
    # Reset before training.
    self.min_theta = None
    self.costs = None

    # Remove count of y from x.
    xcols = self.data.shape[1] - 1

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
    X = np.ones(shape=(xcols + 1, self.m))
    X[1:,:] = self.data[:,0:xcols].transpose()

    # y keeps last column of each feature vector
    # of input data, 0 or 1 such as (see last column)
    # a1, a2, a3, 1
    # b1, b2, b3, 0
    # c1, c2, c3, 0
    y = self.data[:,xcols]

    # Minimize theta values using.
    self.min_theta, self.costs = \
      self._min_thetas(X, y, iter_min, alpha)


  def train_normal_eq_method(self):
    '''
    Train from training data using Normal Equation method.
    You need to call either 'train' or 'train_normal_eq_method'
    calling both does not make sense.
    '''
    # Reset before training.
    self.min_theta = None
    self.costs = None

    # Remove count of y from x.
    xcols = self.data.shape[1] - 1
    X = np.ones_like(self.data)
    X[:,1:] = self.data[:,:xcols]
    y = self.data[:,xcols]

    # Compute theta.
    theta = ((np.linalg.pinv(X.transpose().dot(X))).dot(X.transpose())).dot(y)
    # Save it in column vector.
    self.min_theta = np.zeros(shape=(len(theta), 1))
    self.min_theta[:,0] = theta.transpose()


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
    if self.means is not None and self.stds is not None:
      row = feature_vect.shape[0]
      for i in range(row):
        feature_vect[i, 0] = (feature_vect[i, 0] - self.means[i]) / self.stds[i]

    x = np.ones(shape=(feature_vect.shape[0] + 1, 1))
    x[1:,] = feature_vect[:,]
    return np.transpose(self.min_theta).dot(x)[0]


  def min_cost(self):
    '''
    Return min cost. It should be called after 'train' function.

    Return:
      (float): Min cost.
    '''
    return np.min(self.costs)


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
    plt.scatter(self.data[:, f_idx], self.data[:, self.data.shape[1] - 1],
                color='red', marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


  def plot_regression_line(self, f_idx=0, x_label='', y_label=''):
    '''
    Plot a single feature and the decision line. 'train' function
    must be called before calling this function.

    Arguments:
      f_idx (int): Feature index in the data file.
      x_label (string): X axis label on the graph.
      y_label (string): Y axis label on the graph.
    '''

    # Regression line plot, first plot the feature.
    plt.title('Decision Line Plot')
    plt.scatter(self.data[:, f_idx], self.data[:, self.data.shape[1] - 1],
                color='red', marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot regression line.
    x = self.data[:, f_idx]
    y = [self.predict(np.array([[xi],])) for xi in x]
    plt.plot(x, y, color='blue')
    plt.show()


  def plot_cost_curve(self):
    '''
    Plot cost curve. 'train' function must be called before
    calling this function.
    '''
    # Plot iterations and costs.
    plt.title('Cost Curve')
    plt.xlabel('Iterations')
    plt.ylabel(r'J($\theta$)')
    plt.plot([i for i in range(len(self.costs))], self.costs)
    plt.show()


  def plot_contour(self):
    '''
    Plot the contour for the cost. 'train' function must be
    called before calling this function.
    '''
    xcols = self.data.shape[1] - 1
    X = np.ones(shape=(xcols + 1, self.m))
    X[1:,:] = self.data[:,0:xcols].transpose()
    y = self.data[:,xcols]

    # Define the ranges of the grid
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    # Initialize space for the values to be plotted.
    J_vals = np.zeros(shape=(len(theta0_vals), len(theta1_vals)))

    ## Evaluate J_vals = cost(theta) over the grid.
    for i in range(len(theta0_vals)):
      for j in range(len(theta1_vals)):
        theta = np.concatenate((theta0_vals[i].ravel(), theta1_vals[j].ravel()))
        J_vals[i, j] = self._cost(X, y, theta)

    J_vals = J_vals.transpose()

    plt.title('Contour Plot')
    plt.xlabel('$\\theta_0$')
    plt.ylabel('$\\theta_1$')

    # Plot J_vals = 0 by specifying the range.
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(self.min_theta[0], self.min_theta[1], marker='x', color='red')
    plt.show()


  def _min_thetas(self, X, y, iter_min, alpha):
    ''' 
    Minimize theta values.

    Arguments:
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
      iter_min (int): Iterations to converge theta values.
      alpha (float): Learning rate.

    Return:
      (1 x n float vector): The theta values vector.
      (float list): The cost value for each iteration.
    '''
    # Create a 2d array for the thetas.
    # Notice it is 1 column that has (features + 1) rows.
    theta = np.zeros(shape=(X.shape[0], 1))
    Jtheta = list()

    for i in range(iter_min):
      grads = self._gradient(X, y, theta, alpha)
      theta -= grads
      Jtheta.append(self._cost(X, y, theta))
    return theta, Jtheta


  def _gradient(self, X, y, theta, alpha):
    '''
    Compute gradient descent to learn theta values.

    Arguments:
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
      theta (n x 1 float vector): Theta values vector.
      alpha (float): Learning rate.

    Return:
      (n x 1 float vector): A gradient values vector, size of
        vector will match 'theta' vector.
    '''
    grad = np.zeros_like(theta)
    # Iterate over columns.
    for i, yi in enumerate(y):
      xi = X[:, [i]]
      h =  np.transpose(theta).dot(xi)
      grad += (h - yi) * xi
    return (alpha / self.m) * (grad)


  def _cost(self, X, y, theta):
    '''
    Compute cost of the hypothesis for the given
    theta values.

    Arguments:
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
      theta (n x 1 float vector): Theta values vector.

    Return:
      (float): The cost J for the given theta values vector.
    '''
    sqerr = 0.0
    # Iterate over columns.
    for i in range(X.shape[1]):
      h = np.transpose(theta).dot(X[:, [i]])[0]
      sqerr += ((h - y[i]) ** 2)
    # J(theta) is cost of the hypothesis.
    return sqerr / (2.0 * self.m)
pass
