

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class Svm(object):
  '''
  Support vector machine.
  '''
  def __init__(self, X, y):
    '''
    Initialize instance parameters.
    '''
    self.X = X
    self.y = y

    # Instance of SVC() of scipy, it is set
    # after 'train()' is called.
    self.svc = None


  def plot_data(self, x_label='',  y_label='', title_label='',
                xlim=(), ylim=()):
    '''
    Plot data.

    Arguments:
      x_label (str): X axis label on the graph.
      y_label (str): Y axis label on the graph.
      title_label (str): Title label.
      xlim (float): X-axis limit.
      ylim (float): Y-axis limit.
    '''
    # Plot examples.
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Find Indices of Positive and Negative Examples
    pos = np.where(self.y == 1)
    neg = np.where(self.y == 0)
    plt.scatter(self.X[pos, 0], self.X[pos, 1],
                color='blue', marker='+', label='y=1')
    plt.scatter(self.X[neg, 0], self.X[neg, 1],
                color='red', marker='o', label='y=0')
    plt.show()


  def plot_decision_boundary_linear(self, x_label='',  y_label='',
                                    title_label='', xlim=(), ylim=()):
    '''
    Plot data with linear decision boundary.

    Arguments:
      x_label (str): X axis label on the graph.
      y_label (str): Y axis label on the graph.
      title_label (str): Title label.
      xlim (float): X-axis limit.
      ylim (float): Y-axis limit.
    '''
    # Find Indices of Positive and Negative Examples
    pos = np.where(self.y == 1)
    neg = np.where(self.y == 0)
    plt.scatter(self.X[pos, 0], self.X[pos, 1],
                color='blue', marker='+', label='y=1')
    plt.scatter(self.X[neg, 0], self.X[neg, 1],
                color='red', marker='o', label='y=0')

    # create a mesh to plot in
    step_size = 0.02
    x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
    y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    z = self.svc.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot examples.
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


  def plot_decision_boundary(self, x_label='',  y_label='',
                             title_label='', xlim=(), ylim=()):
    '''
    Plot data with non linear decision boundary.

    Arguments:
      x_label (str): X axis label on the graph.
      y_label (str): Y axis label on the graph.
      title_label (str): Title label.
      xlim (float): X-axis limit.
      ylim (float): Y-axis limit.
    '''
    # Find Indices of Positive and Negative Examples
    pos = np.where(self.y == 1)
    neg = np.where(self.y == 0)
    plt.scatter(self.X[pos, 0], self.X[pos, 1],
                color='blue', marker='+', label='y=1')
    plt.scatter(self.X[neg, 0], self.X[neg, 1],
                color='red', marker='o', label='y=0')

    u = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
    v = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), 100)
    # Initialize space for the values to be plotted.
    z = np.zeros(shape=(len(u), len(v)))

    for i in range(len(u)):
      for j in range(len(v)):
        # Notice the order of j, i here.
        z[j, i] = self.svc.predict(np.c_[u[i].ravel(), v[j].ravel()])
    plt.contour(u, v, z, 3, colors='green')

    # Plot examples.
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


  def train(self, X, y, C, tol, kernel='linear', gamma='auto'):
    '''
    Train using scipy's sklearn.svm.SVC which is libsvm
    based.

    Arguments
      X (float matrix): Training examples by rows.
      y (n x 1 float vector): Corresponding output of each
        training example.
      C (float): Regularization parameter.
      tol (float): Tolerance.
      kernel (float): Kernel function type.
      gamma (float): Parameter for kernel used for RBF here.
    '''
    self.svc = svm.SVC(C=C, kernel=kernel, tol=tol, gamma=gamma)
    self.svc.fit(X, y)


  def predict(self, V):
    '''
    Predict for the given test samples.

    Arguments:
      V (Test sample matrix): Test samples

    Return
      (1d vector): Predicted values vector i.e 0 or 1
        for each test sample.
    '''
    return self.svc.predict(V)


  def gaussian_kernel(self, x0, x1, sigma):
    '''
    Compute and return a radial basis function kernel
    between 'x0' and 'x1'.

    Arguments
      x0 (float vector): Feature vector.
      x1 (float vector): Feature vector.
      sigma (float): Bandwidth parameter.

    Return:
      (float value): Computed value.
    '''
    return np.exp(-np.sum((x0 - x1) ** 2) / (2.0 * (sigma ** 2)))
