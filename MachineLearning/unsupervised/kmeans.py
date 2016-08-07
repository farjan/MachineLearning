
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Kmeans(object):
  '''
  K-means, PCA implementation.
  '''
  def __init__(self, X):
    '''
    Initialize instance parameters.

    Arguments:
      X (m x n float matrix): Data matrix.
    '''
    self.X = X
    self.mu = None
    self.sigma = None
    self._colors = None


  def kmeans_init_centroids(self, K):
    '''
    Initialize and return K centroids that are to
    be used in K-Means on the dataset X.

    Arguments:
      K (int): Number of centroids.

    Return:
      (K x X.shape[1]) Centroids values.
    '''
    # Initialize the centroids to be random examples.
    # Randomly reorder the indices of examples.
    rand_idx = np.random.permutation(self.X.shape[0])
    # Take the first K examples as centroids
    centroids = self.X[rand_idx[:K],:]
    return centroids


  def kmeans_showprogress(self, centroids,  iter, 
                              plt_progress=False):
    '''
    Compute K-Means algorithm and plot progress.

    Arguments:
      centroids (m x n float matrix): Centroids values.
      iter (int): Max number of iterations.
      plt_progress (Boolean): 'True' if show progress on the graph.

    Returns:
      ci (m x n float matrix): Centroid values.
      cc (int vector): The indices of the closest centroids.
    '''
    K = centroids.shape[0]
    self._colors = cm.rainbow(np.linspace(0, 1, K))
    cc = None

    for i in range(iter):
      # For each example in X, assign it to the
      # closest centroid.
      cc = self.closest_centroid(centroids);

      if plt_progress:
        self._plot_progress_kmeans(centroids, cc, K, i)
      centroids = self.centroids_means(cc, K)
    return centroids, cc.astype(int)


  def closest_centroid(self, centroids):
    '''
    Compute the closest centroid memberships for every example.

    Arguments:
      centroids (m x n float matrix): Centroid values.

    Return:
      cc (int vector): The indices of the closest centroids.
    '''
    ci = np.zeros((self.X.shape[0], 1))

    for i, xi in enumerate(self.X):
      mod_list = []
      for c in centroids:
        mod_list.append(np.sqrt(np.sum((xi - c) ** 2)))
      ci[i] = np.argmin(mod_list)
    return ci.astype(int)


  def centroids_means(self, ci, K):
    '''
    Compute means based on the closest centroids.

    Computes the new centroids by computing the means
    of the data points assigned to each centroid.

    Arguments:
      ci (int vector): A vector of centroid indices assignments.
      K (int): Number of dimensions.

    Return:
      (m x n float matrix): Mean calculated centroids.
    '''
    m = self.X.shape[0]
    n = self.X.shape[1]
    centroids = np.zeros((K, n))

    for k in range(K):
      xi_idx = np.where(ci[:,0] == k)[0]
      centroids[k,:] = np.sum(self.X[xi_idx,:], axis=0) / len(xi_idx)
    return centroids
  

  def pca(self, norm_first=True):
    '''
    Compute Principal Component Analysis on the dataset X.

    Computes eigenvectors of the covariance matrix of X 
    returns the eigenvectors U, the eigenvalues (on diagonal)
    in s.

    Arguments:
      norm_first (Boolean): 'True' if normalize data before processing.

    Return:
      U, Unitary matrices.
      s, The singular values for every matrix, sorted in descending order.
    '''
    m = self.X.shape[0]
    X_norm = self.X

    # First compute the covariance matrix.
    if norm_first:
      X_norm = self._normalize()
    cov_mat = X_norm.transpose().dot(X_norm) / m

    # Compute the eigenvectors and eigenvalues
    # of the covariance matrix.
    U, s, V  = np.linalg.svd(cov_mat, full_matrices=True)
    return U, s


  def project_data(self, X_norm, U, K):
    '''
    Project data to 'K' dimensions.

    Arguments:
      U (m x n float matrix): Unitary matrices.
      K (int): Number of dimensions.

    Return:
      Projected matrix to the 'K' dimensions.
    '''
    Z = np.zeros((X_norm.shape[0], K))
    for i in range(X_norm.shape[0]):
      xi = X_norm[[i],:]
      Z[i,:] = xi.dot(U[:,:K])
    return Z


  def recover_data(self, Z, U, K):
    '''
    Recover data from 'K' dimension back to
    original dimensions.

    Arguments:
      Z (m x n float matrix): Projected dimensions.
      U (m x n float matrix): Unitary matrices.
      K (int): Number of dimensions.

    Return:
      Recovered matrix to the original dimensions.
    '''
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    for i in range(Z.shape[0]):
      zi = Z[[i],:].transpose()
      for j in range(U.shape[0]):
        X_rec[i,j] = zi.transpose().dot((U[[j],:K]).transpose())[0][0]
    return X_rec


  def plot_data_set(self, X, title_label='', x_label='', y_label=''):
    '''
    Plot data points.

    Arguments:
      X (m x n float matrix): Data points.
      title_label (str): Title label.
      x_label (str): X-axis label.
      y_label (str): Y-axis label.
    '''
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.show()


  def plot_pca_vector(self, p1, p2, xlim, ylim):
    '''
    Plot PCA vector.

    Arguments:
      p1 (float pair): Point 1.
      p2 (float pair): Point 2.
      xlim (float pair): x-axis limit.
      ylim (float pair): y-axis limit.
    '''
    plt.title('Computed eigenvectors of the dataset')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(self.X[:,0], self.X[:,1], 'bo')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
    plt.show()


  def _normalize(self):
    '''
    Normalized version of X where the mean value
    of each feature is 0 and the standard deviation is 1. 
    This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    #self.mu = np.mean(self.X)
    #self.sigma = np.std(self.X)
    #X_norm = (self.X - self.mu) / self.sigma

    self.mu = np.mean(self.X, axis=0)
    self.sigma = np.std(self.X, axis=0)

    X_norm = np.zeros_like(self.X)
    for j in range(self.X.shape[1]):
      X_norm[:,j] = (self.X[:,j] - self.mu[j]) / self.sigma[j]
    return X_norm


  def _plot_progress_kmeans(self, centroids, closes_centroids, K, i):
    '''
    Plots kMeans data, 2d only.

    Arguments:
      centroids (m x n): Centroids matrix.
      closest_centroids (vector): Closest centroids.
      K (int): Dimensions.
      i (int): Index of iteration.
    '''
    for k, c in zip(list(range(K)), self._colors):
      plt.scatter(self.X[np.where(closes_centroids == k),0], 
                  self.X[np.where(closes_centroids == k),1],
                  15, color=c)

    # Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1],
             marker='x', color='black')
    plt.title('Iteration number {}'.format(i))
    plt.show()

