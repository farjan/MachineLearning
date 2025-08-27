
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class CollaborativeFilteringLearning(object):
  '''
  Implementation of Collaborative Filtering Learning Algorithm.
  '''
  def __init__(self, num_users, num_movies, num_features):
    '''
    Initialize instance parameters.

    Arguments:
      num_users (int): Number of users.
      num_movies (int): Number of movies.
      num_features (int): Number of features.
    '''
    self.num_users = num_users
    self.num_movies = num_movies
    self.num_features = num_features

    # Get set after training.
    self.min_X = None
    self.min_Theta = None
    self.min_cost = None


  def train(self, X, Theta, Y, R, lmda):
    '''
    Train for the collaborative filtering and keep
    learned parameters in instance variable.

    Arguments:
      X (num_movies  x num_features float): Matrix of movies features 
         samples where each row of X corresponds to the feature
         vector x[i] for the i-th movie.
      Theta (num_users  x num_features float): Matrix of user features
        the j-th row of Theta corresponds to one parameter vector theta[j], 
        for the jth user.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.
    '''
    self.min_X = None
    self.min_Theta = None
    self.min_cost = None

    # Initialize theta and args for fmin_cg.
    params = self.unroll(X, Theta)
    args = (Y, R, lmda)

    res = opt.fmin_cg(self.cost, 
                      x0=params, 
                      fprime=self.gradient,
                      args=args, 
                      maxiter=100,
                      disp=False, 
                      full_output=True)
    self.min_X, self.min_Theta = self.roll(res[0])
    self.min_cost = res[1]


  def cost(self, params, Y, R, lmda):
    '''
    Compute the cost for the collaborative filtering.

    Arguments:
      params (vector float): Combined vector of X and Theta matrix,
        X and Theta flattened and concatenated.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.

    Return:
      (float): Cost value.
    '''
    X, Theta = self.roll(params)

    # Compute cost.
    J = 0.0
    for i in range(self.num_movies):
      xi = X[[i],:]
      for j in range(self.num_users):
        tj = Theta[[j],:]
        if R[i,j] == 1:
          e = (tj.dot(xi.transpose()) - Y[i,j])[0,0]
          J += (e ** 2)
    J *= 0.5

    # Compute regularization term 1.
    reg1 = 0.0
    for j in range(self.num_users):
      for k in range(self.num_features):
        reg1 += (Theta[j,k] ** 2)
    reg1 *= (lmda / 2.0)

    # Compute regularization term 2.
    reg2 = 0.0
    for i in range(self.num_movies):
      for k in range(self.num_features):
        reg2 += (X[i,k] ** 2)
    reg2 *= (lmda / 2.0)

    # Add regularization.
    J += (reg1 + reg2)
    return J


  def gradient(self, params, Y, R, lmda):
    '''
    Compute x gradient for the given 'X' and theta 
    gradient for given 'Theta' for the collaborative filtering.

    Arguments:
      params (1d vector float): Combined vector of X and Theta matrix,
        X and Theta flattened and concatenated.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.

    Returned:
      (1d vector float): X and Theta flattened and concatenated.
    '''
    X, Theta = self.roll(params)

    X_grad = self.__X_grad(X, Theta, Y, R, lmda)
    Theta_grad = self.__Theta_grad(X, Theta, Y, R, lmda)
    return self.unroll(X_grad, Theta_grad)


  def __X_grad(self, X, Theta, Y, R, lmda):
    '''
    Compute x gradient for the given 'X'.

    Arguments:
      X (num_movies x num_features float): Matrix of movies features 
         samples where each row of X corresponds to the feature
         vector x[i] for the i-th movie.
      Theta (num_users  x num_features float): Matrix of user features
        the j-th row of Theta corresponds to one parameter vector theta[j], 
        for the jth user.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.

    Return:
      (num_movies x num_features float): Gradient matrix for 'X'
    '''
    X_grad = np.zeros_like(X)

    # Compute X gradient.
    for i in range(self.num_movies):
      xi = X[[i],:]
      for j in range(self.num_users):
        tj = Theta[[j],:]
        if R[i,j] == 1:
          e = (tj.dot(xi.transpose()) - Y[i,j])[0,0]
          for k in range(self.num_features):
            X_grad[i,k] += (e * tj[0,k])
      # Add regularization.
      X_grad[[i],:] += (lmda * xi)
    return X_grad


  def __Theta_grad(self, X, Theta, Y, R, lmda):
    '''
    Compute theta gradient for the given 'Theta'.

    Arguments:
      X (num_movies x num_features float): Matrix of movies features 
         samples where each row of X corresponds to the feature
         vector x[i] for the i-th movie.
      Theta (num_users  x num_features float): Matrix of user features
        the j-th row of Theta corresponds to one parameter vector theta[j], 
        for the jth user.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.

    Return:
      (num_users x num_features float): Gradient matrix for 'Theta'
    '''
    Theta_grad = np.zeros_like(Theta)

    # Compute theta gradient.
    for i in range(self.num_movies):
      xi = X[[i],:]
      for j in range(self.num_users):
        tj = Theta[[j],:]
        if R[i,j] == 1:
          e = (tj.dot(xi.transpose()) - Y[i,j])[0,0]
          for k in range(self.num_features):
            Theta_grad[j,k] += (e * xi[0,k])

    # Add regularization.
    for j in range(self.num_users):
      tj = Theta[[j],:]
      Theta_grad[[j],:] += (lmda * tj)
    return Theta_grad


  def roll(self, vect):
    '''
    Roll X and Theta matrices.

    Arguments:
      vect (vector): Flattened and concatenated X and Theta.

    Return:
      (X matrix): Rolled X matrix.
      (Theta matrix): Rolled Theta matrix.
    '''
    X = vect[:self.num_movies * self.num_features].reshape((
                self.num_movies, self.num_features))
    Theta = vect[self.num_movies * self.num_features:].reshape((
                  self.num_users, self.num_features))
    return X, Theta


  def unroll(self, X, Theta):
    '''
    Unroll X and Theta into a vector.

    Arguments:
      X (matrix): theta1 matrix.
      Theta (matrix): theta2 matrix.

    Return:
      (vector): Flattened and concatenated X and Theta.
    '''
    return np.concatenate((X.ravel(), Theta.ravel()))
