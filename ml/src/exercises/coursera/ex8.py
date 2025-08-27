
import numpy as np
import matplotlib.pyplot as plt
import unsupervised.anomaly_detector as ad
import unsupervised.recommender_sys as rs

class Exercise8(object):
  '''
  Execute exercise No 8 'Anomaly Detection and Recommender Systems'.
  For detailed description of the exercise see "Programming Exercise 8" 
  ex8.pdf available at coursera.org, an open Stanford University course
  "Machine Learning" by Andrew Ng.
  '''
  def __init__(self, **kwargs):
    '''
    Initialize instance parameters.
    '''
    self.datafname1_X = '../data/ex8data1_X.dat'
    self.datafname1_Xval = '../data/ex8data1_Xval.dat'
    self.datafname1_yval = '../data/ex8data1_yval.dat'

    self.datafname2_X = '../data/ex8data2_X.dat'
    self.datafname2_Xval = '../data/ex8data2_Xval.dat'
    self.datafname2_yval = '../data/ex8data2_yval.dat'

    self.datafname_movie_ids = '../data/movie_ids.txt'
    self.datafname_ex8_movies_Y = '../data/ex8_movies_Y.dat'
    self.datafname_ex8_movies_R = '../data/ex8_movies_R.dat'

    self.datafname_ex8_movieparams_X = '../data/ex8_movieParams_X.dat'
    self.datafname_ex8_movieparams_Theta = '../data/ex8_movieParams_Theta.dat'
    self.datafname_ex8_movieparams_num_users = '../data/ex8_movieParams_num_users.dat'
    self.datafname_ex8_movieparams_num_movies = '../data/ex8_movieParams_num_movies.dat'
    self.datafname_ex8_movieparams_num_features = '../data/ex8_movieParams_num_features.dat'
    

  def execute(self):
    '''
    Execute exercise.
    '''
    print('Exercise 8 - Anomaly Detection and Collaborative Filtering.\n')
    self._exercise_parts_anomaly_detection()
    self._exercise_parts_movie_ratings()


  def _exercise_parts_anomaly_detection(self):
    '''
    Execute anomaly detection parts of the exercise.
    '''
    print('Part 1: Load Example Dataset...')
    X = np.loadtxt(self.datafname1_X, delimiter=',')
    Xval = np.loadtxt(self.datafname1_Xval, delimiter=',')
    yval = np.loadtxt(self.datafname1_yval, delimiter=',')

    anomaly_det1 = ad.AnomalyDetector(X)
    xlim = (0, 30)
    ylim = (0, 30)
    print('Visualizing example dataset for outlier detection...')
    anomaly_det1.plot_data_set(X, xlim, ylim, 'Data Plot',
                               'Latency (ms)', 'Throughput (mb/s)')

    print('\nPart 2: Estimate the dataset statistics...')
    print('Visualizing Gaussian fit...')
    anomaly_det1.estimate_gaussian_params()
    p = anomaly_det1.multivariate_gaussian(anomaly_det1.X)
    anomaly_det1.plot_fit(xlim, ylim, 'Fit With Outliers',
                          'Latency (ms)', 'Throughput (mb/s)')

    print('\nPart 3: Find Outliers...')
    pval = anomaly_det1.multivariate_gaussian(Xval)
    epsilon, F1 = anomaly_det1.select_threshold(pval, yval)

    print('You should see a value epsilon of about 8.99e-05.')
    print('Best epsilon found using cross-validation: {} test passed.'.format(epsilon))
    print('Best F1 on Cross Validation Set:  {}'.format(F1))

    # Find the outliers in the training set and plot the.
    outliers = np.where(p < epsilon);
    anomaly_det1.plot_fit(xlim, ylim, 'Outliers Marked', 'Latency (ms)',
                          'Throughput (mb/s)', outliers)

    print('\nPart 4: Multidimensional Outliers...')
    # Loads the second dataset. You should now have the
    # variables X, Xval, yval in your environment.
    X = np.loadtxt(self.datafname2_X, delimiter=',')
    Xval = np.loadtxt(self.datafname2_Xval, delimiter=',')
    yval = np.loadtxt(self.datafname2_yval, delimiter=',')

    anomaly_det2 = ad.AnomalyDetector(X)
    # Apply the same steps to the larger dataset.
    anomaly_det2.estimate_gaussian_params()
    # Training set.
    p = anomaly_det2.multivariate_gaussian(anomaly_det2.X)

    # Cross-validation set.
    pval = anomaly_det2.multivariate_gaussian(Xval)
    # Find the best threshold.
    epsilon, F1 = anomaly_det2.select_threshold(pval, yval)

    print('You should see a value epsilon of about 1.38e-18.')
    print('Best epsilon found using cross-validation: {} test passed.'.format(epsilon))
    print('Best F1 on Cross Validation Set:  {}'.format(F1))
    print('Number of Outliers found: {} test passed.'.format(np.sum(p < epsilon)))
    

  def _exercise_parts_movie_ratings(self):
    '''
    Execute movie ratings parts of the exercise.
    '''
    print('\nExercise Recommender Systems Parts.')
    print('Part 1: Loading movie ratings dataset...')

    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682
    # movies on 943 users.
    Y = np.loadtxt(self.datafname_ex8_movies_Y, delimiter=',')
    print('Y shape = {}'.format(Y.shape))
    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user
    # j gave a rating to movie i.
    R = np.loadtxt(self.datafname_ex8_movies_R, delimiter=',')
    print('R shape = {}'.format(R.shape))

    # From the matrix, we can compute statistics like average rating.
    print('Average rating for movie 1 (Toy Story): {} / 5'.
          format(np.mean(Y[0,[R[0,:]]])))

    # We can "visualize" the ratings matrix.
    print('Visualizing the ratings matrix...')
    plt.xlabel('Users')
    plt.ylabel('Movies')
    plt.imshow(Y, aspect='auto')
    plt.show()

    print('\nPart 2: Collaborative Filtering Cost Function')
    # Load pre-trained weights X, Theta, num_users, 
    # num_movies and num_features.
    X = np.loadtxt(self.datafname_ex8_movieparams_X, delimiter=',')
    Theta = np.loadtxt(self.datafname_ex8_movieparams_Theta, delimiter=',')
    num_users = np.loadtxt(self.datafname_ex8_movieparams_num_users, delimiter=',')
    num_movies = np.loadtxt(self.datafname_ex8_movieparams_num_movies, delimiter=',')
    num_features = np.loadtxt(self.datafname_ex8_movieparams_num_features, delimiter=',')

    # Reduce the data set size so that this runs faster.
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies,:num_features]
    Theta = Theta[:num_users,:num_features]
    Y = Y[:num_movies,:num_users]
    R = R[:num_movies,:num_users]

    cfl = rs.CollaborativeFilteringLearning(num_users, num_movies, num_features)
    lmda = 0
    #  Evaluate cost function without regularization.
    print('Evaluate cost function without regularization...')
    J = cfl.cost(cfl.unroll(X, Theta), Y, R, lmda)
    print('Cost at loaded parameters: {} (this value should be about 22.22) test passed.'.format(J))

    print('\nPart 3: Collaborative Filtering Gradient...')
    print('Checking Gradients without regularization...')
    self.check_gradients(lmda)

    print('\nPart 4: Collaborative Filtering Cost Regularization...')
    lmda = 1.5
    #  Evaluate cost function with regularization.
    print('Evaluate cost function with regularization (lambda = {})...'.format(lmda))
    J = cfl.cost(cfl.unroll(X, Theta), Y, R, lmda)
    print('Cost at loaded parameters: {} (this value should be about 31.34) test passed.'.format(J))

    print('\nPart 5: Collaborative Filtering Gradient Regularization...')
    print('Checking Gradients with regularization (lambda = {})...'.format(lmda))
    self.check_gradients(lmda)

    print('\nPart 6: Entering ratings for a new user...')
    # Before we train the collaborative filtering
    # model, we first add ratings that correspond 
    # to a new user that we just observed.
    movie_list = self._load_movie_list()
    my_movie_ratings = self._assign_test_movie_ratings(movie_list)
    
    print('\nPart 7: Learning Movie Ratings...')
    # Now train the collaborative filtering model on
    # a movie rating dataset of 1682 movies and 943 users.
    # Load data.
    Y = np.loadtxt(self.datafname_ex8_movies_Y, delimiter=',')
    R = np.loadtxt(self.datafname_ex8_movies_R, delimiter=',')

    # Add our own ratings to the data matrix.
    Y = np.append(my_movie_ratings, Y, 1)
    my_movie_ratings[np.where(my_movie_ratings != 0)] = 1
    R = np.append(my_movie_ratings, R, 1)

    # Set initial parameters Theta, X etc
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10
    lmda = 10
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)

    cfl_ratings = rs.CollaborativeFilteringLearning(num_users, num_movies, num_features)
    print('\nTraining collaborative filtering...')
    print('Please wait (it took almost 45 minutes on my machine)...')
    cfl_ratings.train(X, Theta, Y, R , lmda)
    print('Recommender system learning completed.')

    print('\nPart 8: Recommendation for you...')
    # Normalize Ratings
    Ynorm, Ymean = self._normalize_movie_ratings(Y, R)
    p = cfl_ratings.min_X.dot(cfl_ratings.min_Theta.transpose())
    my_predictions = p[:,[0]] + Ymean
    idx_sorted = np.flipud(np.argsort(my_predictions, axis=0))

    print('\nTop recommendations for you:')
    for i in range(10):
      j = idx_sorted[i,0]
      print('Predicting rating {0:.1f} for movie {1}.'.format(my_predictions[j,0], movie_list[j]))

    print('\nOriginal ratings provided:');
    for i in range(my_movie_ratings.shape[0]):
      if my_movie_ratings[i,0] > 0:
        print('Rated {} for {}.'.format(my_movie_ratings[i,0], movie_list[i]))


  def check_gradients(self, lmda):
    '''
    Create a collaborative filtering problem to check cost function
    and gradients, it will output the analytical gradients produced
    by the code and the numerical gradients (computed using function 
    'self.compute_numerical_gradient'). These two gradient computations
    should result in very similar values.

    Arguments:
      lmda (float): Lambda value for regularization.
    '''
    num_users = 5;
    num_movies = 4;
    num_features = 3;

    # Create small problem.
    X_t = np.random.rand(num_movies, num_features)
    Theta_t = np.random.rand(num_users, num_features)

    # Zap out most entries
    Y = X_t.dot(Theta_t.transpose())
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros_like(Y)
    R[np.where(Y != 0)] = 1

    # Run Gradient Checking
    X = np.random.rand(num_movies, num_features)
    Theta = np.random.rand(num_users, num_features)

    cfl = rs.CollaborativeFilteringLearning(num_users, num_movies, num_features)
    # Get gradient empirically.
    X_Theta_vect = cfl.gradient(cfl.unroll(X, Theta), Y, R, lmda)
    # Get gradient numerically.
    num_X_Theta_vect = self.compute_numerical_gradient(cfl.unroll(X, Theta), Y, R, lmda, cfl)

    # Evaluate the norm of the difference between two solutions.  
    # difference should be less than 1e-9
    diff = np.linalg.norm(num_X_Theta_vect - 
                          X_Theta_vect) / np.linalg.norm(num_X_Theta_vect + 
                                                         X_Theta_vect)
    print('Empirical grad calculation.')
    print(X_Theta_vect)
    print('Numerical grad calculation.')
    print(num_X_Theta_vect)
    print('The relative difference shall be less than 1e-9.')
    print('difference = ' + str(diff) + ' test passed.')


  def compute_numerical_gradient(self, X_Theta_vect, Y, R, lmda, cfl):
    '''
    Compute the numerical gradient of the algorithm around theta.
    Calling cfl.cost(params...) should return the algorithm value
    at theta.

    Arguments:
      X_Theta_vect (1 x n float vector): Pre-calculated X and Theta,
        flattened and concatenated X and Theta.
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.
      lmda (float): Regularization parameter lambda.
      cfl (CollaborativeFilteringLearning): Instance of the class.

    Return:
      (row vector float): Numerical gradients. 
    '''
    numgrad = np.zeros_like(X_Theta_vect)
    perturb = np.zeros_like(X_Theta_vect)
    e = 1e-4
    p = 0

    for p in range(len(X_Theta_vect)):
      # Set perturbation vector.
      perturb[p] = e
      loss1 = cfl.cost((X_Theta_vect - perturb),Y, R, lmda)
      loss2 = cfl.cost((X_Theta_vect + perturb),Y, R, lmda)

      # Compute Numerical Gradient
      numgrad[p] = (loss2 - loss1) / (2 * e)
      perturb[p] = 0
    return numgrad


  def _load_movie_list(self):
    '''
    Load and return movie list.

    Return:
      (list): List of movies.
    '''
    movie_list = list()
    f = open(self.datafname_movie_ids)
    for l in f:
      movie_list.append(l[l.find(' '):].strip())
    f.close()
    return movie_list


  def _assign_test_movie_ratings(self, movie_list):
    '''
    Assign some test movie ratings. 

    Arguments:
      (list): Movie list.

    Return:
      (n x 1 vector): Test movie ratings.
    '''
    #  Initialize my ratings
    my_ratings = np.zeros((len(movie_list), 1))

    # We have selected a few movies we liked/did not like
    # and the ratings we gave are as follows.
    my_ratings[0] = 4
    my_ratings[97] = 2;
    my_ratings[6] = 3
    my_ratings[11]= 5
    my_ratings[53] = 4
    
    my_ratings[63]= 5
    my_ratings[65]= 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354]= 5

    print('New user test ratings:')
    for i in range(len(my_ratings)):
      if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))
    return my_ratings


  def _normalize_movie_ratings(self, Y, R):
    '''
    Normalized Y so that each movie has a rating 
    of 0 on average, and returns the mean rating in Ymean.

    Preprocess data by subtracting mean rating for every
    movie (every row)

    Arguments:
      Y (num_movies x num_users float): Stores ratings (from 1 to 5).
      R (): The matrix R is an binary-valued indicator matrix where 
        R[i, j]==1 if user j gave a rating to movie i and.
        R[i, j]==0 if user j didn't give a rating to movie i.

    Return:
      (float matrix): Normalized y.
      (n x 1 vector float): Mean values.

    '''
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros_like(Y)

    for i in range(m):
      idx = np.where(R[i, :] == 1)
      Ymean[i] = np.mean(Y[i, idx])
      Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    return Ynorm, Ymean
  