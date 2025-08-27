
import math
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
  '''
  Logistic regression classifier, It is non regularized, 
  non vectorized version uses Newton Method to minimize.
  It can handle n feature variables. Hypothesis for the classifier
  is a linear line.

  Other than instructor's (Andrew Ng) coursera.org course lectures
  and notes you can get help at instructor's course website for 
  following implementation. Here is the link.
  http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html
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


  def train(self, iter_min=7):
    '''
    Train from training data. Uses Newton Method to
    minimize theta values.

    Arguments:
      iter_min (int): Iterations to converge theta values
        by Newton Method default to 7.
    '''
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
    y = self.data[:, xcols]

    # Minimize theta values using Newton's method.
    self.min_theta, self.costs = \
      self._min_thetas_newton_method(X, y, iter_min)


  def predict(self, feature_vect):
    '''
    Predict for the given feature vector.
    This function should be called after "train".

    Arguments:
      feature_vect (n x 1 float vector): Features vector.

    Return:
      (float): A predicted value.
    '''
    x = np.ones(shape=(feature_vect.shape[0] + 1, 1))
    x[1:,] = feature_vect[:,]
    return self._sigmoid(x, self.min_theta)


  def predict_label(self, X):
    '''
    Predict label for each test example in the 
    feature matrix, where 0 means False and 1 means
    True. 
    This function should be called after "train".

    Arguments:
      X (m x n float matrix): Test examples matrix.

    Return:
      (vector with values of 0, 1): Predicted labels
      for each test example in the matrix.
    '''
    labels = np.zeros((X.shape[0], 1), dtype=int)
    # Iterate over rows.
    for i, xi in enumerate(X):
      xi = np.reshape(xi, (len(xi), 1))
      prob = self.predict(xi)
      if prob >= 0.5:
        labels[i] = 1
    return labels
  

  def plot_scatter(self, f1_idx=0, f2_idx=1, x_label='', y_label=''):
    '''
    Plot scatter graph.

    Arguments:
      f1_idx (int): Feature 1 index in the data file.
      f2_idx (int): Feature 2 index in the data file.
      x_label (string): X axis label on the graph.
      y_label (string): Y axis label on the graph.
    '''
    # Read when y = 1.
    pos = np.where(self.data[:,2] == 1)
    neg = np.where(self.data[:,2] == 0)

    # First plot classes.
    plt.title('Scatter Plot')
    plt.scatter(self.data[pos, f1_idx], self.data[pos, f2_idx],
                color='blue', marker='+', label='Admitted')
    plt.scatter(self.data[neg, f1_idx], self.data[neg, f2_idx],
                color='red', marker='o', label='Not Admitted')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


  def plot_decision_line(self, f1_idx=0, f2_idx=1, x_label='', y_label=''):
    '''
    Plot the two features and the decision line.

    Arguments:
      f1_idx (int): Feature 1 index in the data file.
      f2_idx (int): Feature 2 index in the data file.
      x_label (string): X axis label on the graph.
      y_label (string): Y axis label on the graph.
    '''
    # Read when y = 1.
    pos = np.where(self.data[:,2] == 1)
    neg = np.where(self.data[:,2] == 0)

    # First plot classes.
    plt.title('Decision Line Plot')
    plt.scatter(self.data[pos, f1_idx], self.data[pos, f2_idx],
                color='blue', marker='+', label='Admitted')
    plt.scatter(self.data[neg, f1_idx], self.data[neg, f2_idx],
                color='red', marker='o', label='Not Admitted')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Plotting decision boundary is equivalent to plotting
    # transpose(theta)*x = 0  solving this equation gives
    # x2 = -(1/theta2) * (theta0 + (theta1 * x1))
    # Now we can figure out x1 and x2 for decision boundary.
    
    # Only need 2 points for a line, get f1s first.
    f1_points = [min(self.data[:, f1_idx]) - 2, 
                  max(self.data[:, f1_idx]) + 2];

    # Function calculates f2.
    def get_f2(f1): 
      return (-(1.0/self.min_theta[f2_idx + 1]) * 
              (self.min_theta[0] + (self.min_theta[f1_idx + 1] * f1)))
    
    # Get f2s
    f2_points = [get_f2(f1) for f1 in f1_points]
    # Plot decision boundary.
    plt.plot(f1_points, f2_points, label='Decision Line')
    plt.legend(fontsize='x-small')
    plt.show()


  def plot_cost_curve(self):
    '''
    Plot cost curve. 'train' function must be
    called before calling this function.
    '''
    # Plot iterations and costs.
    plt.title('Cost Curve')
    plt.xlabel('Iterations')
    plt.ylabel(r'J($\theta$)')
    plt.plot([i for i in range(len(self.costs))], self.costs, marker='o')
    plt.show()


  def min_cost(self):
    '''
    Return min cost. It should be called after 'train' function.

    Return:
      (float): Min cost.
    '''
    return np.min(self.costs)


  def _sigmoid(self, xi, theta):
    '''
    Sigmoid function.

    Arguments:
      xi (n x 1 float vector): Feature values vector.
      theta (n x 1 float vector): Theta values vector.

    Return:
      (float): A sigmoid value.
    '''
    z = np.transpose(theta).dot(xi)
    return 1.0 / (1.0 + math.exp(-z))


  def _gradient(self, X, y, theta):
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
        the training data such as 0s or 1s.
      theta (n x 1 float vector): Theta values vector.

    Return:
      (n x 1 float vector): A gradient values vector, size of
        vector will match 'theta' vector.
    '''
    grad = np.zeros_like(theta)
    # Iterate over columns.
    for i, yi in enumerate(y):
      xi = X[:, [i]]
      h = self._sigmoid(xi, theta)
      grad += (h - yi) * xi
    return grad / self.m


  def _hessian(self, X, theta):
    '''
    Compute Hessian.

    Arguments:
      X (n x m float matrix): Feature values vectors, e.g.
        1   1  1  . .
        a1  b1 c1 . .
        a2  b2 c2 . .
        .   .  .  . .
        .   .  .  . .
        Notice dimension of feature vectors are by columns 
        NOT by rows.
      theta (n x 1 float vector): Theta values vector.

    Return:
      (n x n float matrix): The Hessian values matrix where
        size is 'X.rows x X.rows'.
    '''
    hessian = np.zeros(shape=(X.shape[0], X.shape[0]))
    # Iterate over columns.
    for i in range(X.shape[1]):
      xi = X[:, [i]]
      h = self._sigmoid(xi, theta)
      hessian += (h * (1 - h)) * (xi.dot(np.transpose(xi)))
    return hessian / self.m


  def _min_thetas_newton_method(self, X, y, iter_min):
    ''' 
    Minimize theta values using Newton Method.

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
      iter_min (int): Iterations to converge theta values
        by Newton Method.

    Return:
      (1 x n float vector): The theta values vector.
      (float list): The cost value for each iteration.
    '''
    # Create a 2d array for the thetas.
    # Notice it is 1 column that has (features + 1) rows.
    theta = np.zeros(shape=(X.shape[0], 1))
    Jtheta = list()

    for i in range(iter_min):
      grads = self._gradient(X, y, theta)
      hessian = self._hessian(X, theta)
      theta -= (np.linalg.pinv(hessian).dot(grads) )
      Jtheta.append(self._cost(X, y, theta))
    return theta, Jtheta


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
    likelihood = 0.0
    # Iterate over columns.
    for i in range(X.shape[1]):
      h = self._sigmoid(X[:, [i]], theta)
      likelihood += ((y[i] * math.log(h)) + ((1.0 - y[i]) * math.log(1.0 - h)))
    # J(theta) is cost of the hypothesis.
    return -(likelihood / (self.m))
pass
