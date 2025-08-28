
import numpy as np
import matplotlib.pyplot as plt
import scipy as si

class AnomalyDetector(object):
  '''
  Anomaly detection implementation.
  '''

  def __init__(self, X):
    '''
    Initialize instance parameters.

    Arguments:
      X (m x n float matrix): Data matrix.
    '''
    self.X = X


  def estimate_gaussian_params(self):
    '''
    Estimates the parameters of a Gaussian distribution 
    using the data in X.

    Return:
      (n x 1): Mean vector.
      (n x 1): Sigma^2 vector.
    '''
    m, n = self.X.shape
    self.mu = np.zeros((n, 1))
    self.sigma2 = np.zeros((n, 1))

    self.mu[:,0] = np.mean(self.X, axis=0)
    self.sigma2[:,0] = np.std(self.X, axis=0) ** 2


  def multivariate_gaussian(self, X):
    '''
    Computes the probability density function of the examples X under the 
    multivariate Gaussian distribution with parameters mu and Sigma2. 
    If Sigma2 is a matrix, it is treated as the covariance matrix. If Sigma2
    is a vector, it is treated as the \sigma^2 values of the variances in
    each dimension (a diagonal covariance matrix)

    Arguments:
      (m x n float matrix): Features samples.

    Return
      p(x) values vector.
    '''
    sigma2 = self.sigma2
    if self.sigma2.shape[1] == 1 or self.sigma2.shape[0] == 1:
      sigma2 = np.diag(self.sigma2.reshape((1, self.sigma2.shape[0]))[0])

    # Size of longest dimension.
    k = np.max(self.mu.shape)
    X = X - self.mu.transpose()
    e =  np.exp(-0.5 * np.sum((X.dot(np.linalg.pinv(sigma2)) * X), 1))
    px = (1.0 / (((2.0 * np.pi) ** (k/2)) * (np.linalg.det(sigma2) ** (1/2)))) * e
    return px


  def select_threshold(self, pval, yval):
    '''
    Find the best threshold (epsilon) to use for selecting
    outliers

    Arguments:
      pval (1d vector): Results from a validation set.
      yval (1d vector): The ground truth.

    Return:
      (float value): Best F1.
      (float value): Best epsilon.
    '''
    best_epsilon = 0.0
    best_F1 = 0.0
    F1 = 0.0

    step_size = (np.max(pval) - np.min(pval)) / 1000
    epsilons = np.arange(min(pval), max(pval), step_size)
    yval = (yval == 1)

    for epsilon in epsilons:
      pred = pval < epsilon
      tp = self._tp(pred, yval)
      fp = self._fp(pred, yval)
      fn = self._fn(pred, yval)

      prec = 0.0
      if tp + fp > 0:
        prec = tp / (tp + fp)

      rec = 0.0
      if tp + fn > 0:
        rec = tp / (tp + fn)

      F1 = 0.0
      if prec + rec > 0:
        F1 = (2.0 * prec * rec) / (prec + rec)

      if F1 > best_F1:
         best_F1 = F1
         best_epsilon = epsilon
    return best_epsilon, best_F1


  def plot_data_set(self, X, xlim, ylim, title_label=''
                    , x_label='', y_label=''):
    '''
    Plot data points.

    Arguments:
      X (m x n float matrix): Data points.
      xlim (float pair): x-axis limit.
      ylim (float pair): y-axis limit.
      title_label (str): Title label.
      x_label (str): X-axis label.
      y_label (str): Y-axis label.
    '''
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(X[:,0], X[:,1], 'bx')
    plt.show()


  def plot_fit(self, xlim, ylim, title_label='',
               x_label='', y_label='', outliers=None):
    '''
    Visualize the dataset and its estimated distribution.
    '''
    step_size = 0.5
    u = np.arange(0, 30, step_size)
    v = np.arange(0, 30, step_size)
    xx, yy = np.meshgrid(u, v)
    z = self.multivariate_gaussian(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if outliers:
      plt.plot(self.X[outliers, 0], self.X[outliers, 1],
               marker='o', markersize=10, mfc='none')

    plt.plot(self.X[:, 0], self.X[:, 1], 'bx');
    plt.contour(xx, yy, z, 10, colors='green')
    plt.show()


  def _tp(self, pred, yval):
    '''
    tp is the number of true positives: The ground truth
    label says it's an anomaly and our algorithm correctly 
    classified it as an anomaly.

    Arguments:
      pred (1d vector): Predicted labels.
      yval (1d vector): Orginal labels.

    Return:
      (int): The number of true positives.
    '''
    c = 0
    for p, o in zip(pred, yval):
      if p == True and o == True:
        c += 1
    return c


  def _fp(self, pred, yval):
    '''
    fp is the number of false positives: The ground
    truth label says it's not an anomaly, but our algorithm
    incorrectly classified it as an anomaly.

    Arguments:
      pred (1d vector): Predicted labels.
      yval (1d vector): Orginal labels.

    Return:
      (int): The number of false positives.
    '''
    c = 0
    for p, o in zip(pred, yval):
      if p == False and o == True:
        c += 1
    return c


  def _fn(self, pred, yval):
    '''
    fn is the number of false negatives: The ground 
    truth label says it's an anomaly, but our algorithm 
    incorrectly classified it as not being anomalous.

    Arguments:
      pred (1d vector): Predicted labels.
      yval (1d vector): Orginal labels.

    Return:
      (int): The number of false negatives.
    '''
    c = 0
    for p, o in zip(pred, yval):
      if p == True and o == False:
        c += 1
    return c

