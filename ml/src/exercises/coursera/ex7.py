
import numpy as np
import matplotlib.image as im
import matplotlib.pyplot as plt
import scipy.io as sio

import utils.displaydata as d
import unsupervised.kmeans as kmeans

class Exercise7(object):
  '''
  Execute exercise No 7 'K-means Clustering and Principal Component Analysis'.
  For detailed description of the exercise see "Programming Exercise 7" 
  ex7.pdf available at coursera.org, an open Stanford University course
  "Machine Learning" by Andrew Ng.
  '''
  def __init__(self, **kwargs):
    '''
    Initialize instance parameters.
    '''
    self.datafname1 = '../data/ex7data1.dat'
    self.datafname2 = '../data/ex7data2.dat'
    self.birdimg_fname = '../data/bird_small.png'
    self.ex7faces = '../data/ex7faces.dat'
    

  def execute(self):
    '''
    Execute exercise.
    '''
    print('Exercise 7 - K-means Clustering and Principal Component Analysis.')
    # Coursera exercises.
    self.__exercise_parts_kmean()
    self.__exercise_parts_kmeans_clustering_pixels()
    self.__exercise_parts_pca()
    self.__exercise_parts_pca_faces()


  def __exercise_parts_kmean(self):
    '''
    Execute parts of the exercise related to KMeans.
    '''
    print('\nPart 1: Find Closest Centroids...')
    print('Loading data...')
    X = np.loadtxt(self.datafname2, delimiter=',')

    # Select an initial set of centroids
    K = 3; # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    print(initial_centroids)

    # Find the closest centroids for the examples using the
    # initial_centroids
    km = kmeans.Kmeans(X)
    cc = km.closest_centroid(initial_centroids)
    print('The closest centroids should be 0, 2, 1 respectively.')
    print('Closest centroids for the first 3 examples. {}, test passed.'.format(cc[:3,0]))

    print('\nPart 2: Compute centroids means...')
    print('The centroids should be about.');
    print('[ 2.428301 3.157924 ]');
    print('[ 5.813503 2.633656 ]');
    print('[ 7.119387 3.616684 ]');
    centroids = km.centroids_means(cc, K)
    print('Computed centroids, test passed.')
    print(centroids)

    print('\nPart 3: K-Means Clustering...')
    print('Running K-Means clustering on example dataset.')
    K = 3;
    iter = 10;
    km.kmeans_showprogress(initial_centroids, iter, True)
    print('K-Means Done.')


  def __exercise_parts_kmeans_clustering_pixels(self):
    '''
    Execute parts of the exercise related to KMeans clustering pixels.
    '''
    print('\nPart 4: K-Means Clustering on Pixels...')
    print('Running K-Means clustering on pixels from an image (small bird image) please wait...')

    #  Load an image of a bird
    A = im.imread(self.birdimg_fname)

    # Divide by 255 so that all values are in the range 0 - 1.
    A = A / 255

    # Size of the image
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = np.reshape(A, newshape=(img_size[0] * img_size[1], 3))
    K = 16
    iter = 10
    print('Processing when K={} and max_iter={}.'.format(K, iter))

    km_bird = kmeans.Kmeans(X)
    initial_centroids = km_bird.kmeans_init_centroids(K)
    centroids = km_bird.kmeans_showprogress(initial_centroids, iter)[0]
    print('Final centroids after processing.')
    print(centroids)

    print('\nPart 5: Image Compression...')
    print('Applying K-Means to compress an image...')

    # Find closest cluster members
    closest_centroids = km_bird.closest_centroid(centroids)

    # Essentially, now we have represented the image X as in
    # terms of the indices of closest centroids.
    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by it's index in idx) to the centroid value
    X_recovered = centroids[closest_centroids[:,0],:]

    # Reshape the recovered image into proper dimensions.
    X_recovered = np.reshape(X_recovered, newshape=(img_size[0], img_size[1], 3))

    # Display the original image .
    plt.title('Original')
    plt.imshow(A * 255)
    plt.show()

    # Display compressed image side by side
    plt.title('Compressed, with {} colors.'.format(K))
    plt.imshow(X_recovered * 255)
    plt.show()


  def __exercise_parts_pca(self):
    '''
    Execute parts of the exercise related Principal
    Component Analysis (PCA).
    '''
    print('\n\nPrincipal Component Analysis (PCA).')
    print('\nPart 1: Load Example Dataset...')
    X = np.loadtxt(self.datafname1, delimiter=',')
    print('Visualizing example dataset for PCA...')
    km = kmeans.Kmeans(X)
    km.plot_data_set(km.X, 'Data Plot')

    print('\nPart 2: Principal Component Analysis...')
    print('Running PCA on example dataset...')
    # Run PCA and before PCA normalize.
    U, S = km.pca(True)

    xlim = (0.5, 6.5)
    ylim = (2, 8)
    p1 = km.mu
    p2 = km.mu * 1.5 * S[0] * U[:,0]
    km.plot_pca_vector(p1, p2, xlim, ylim)

    print('Top eigenvector:')
    print('You should expect to see -0.707107 -0.707107.')
    print('U[:, 0] = {} {} test passed.'.format(U[0,0], U[1,0]))


    print('\nPart 3: Dimension Reduction...')
    print('Dimension reduction on example dataset...')
    xlim = (-4, 3)
    ylim = (-4, 3)
    X_norm = km._normalize()
    km.plot_data_set(X_norm, 'Normalize Data Plot')

    # Project the data onto K = 1 dimension.
    K = 1
    Z = km.project_data(X_norm, U, K)
    print('Projecting the data onto K = 1 dimension...')
    print('This value should be about 1.481274.')
    print('Projection of the first example: {} test passed.'.format(Z[0,0]))

    X_rec = km.recover_data(Z, U, K)
    print('\nRecovering data from the projection...')
    print('This value should be about  -1.047419 -1.047419.')
    print('Approximation of the first example: {} {} test passed.'.format(X_rec[0,0], X_rec[0,1]))


  def __exercise_parts_pca_faces(self):
    '''
    Execute parts of the exercise related Principal Component
    Analysis (PCA) related to faces.
    '''
    print('\nPart 4: Loading and Visualizing Face Data...')
    print('Loading face dataset...')
    X = np.loadtxt(self.ex7faces, delimiter=',')
    km_faces = kmeans.Kmeans(X)
    # Display the first 100 faces in the dataset.
    d.displaydata(X[:100,:], 'First 100 faces in the dataset')

    print('\nPart 5: PCA on Face Data: Eigenfaces...')
    print('Running PCA on face dataset, please wait...')
    # Run PCA and before PCA normalize.
    U, S = km_faces.pca(True)

    #  Visualize the top 36 eigenvectors found
    d.displaydata(U[:,:36].transpose(), 'Top 36 eigenvectors faces')

    print('\nPart 6: Dimension Reduction for Faces...')
    print('Dimension reduction for face dataset...')
    X_norm = km_faces._normalize()
    # Project the data onto K = 100 dimension.
    K = 100
    Z = km_faces.project_data(X_norm, U, K)
    print('The projected data Z has a size of: {}'.format(Z.shape))

    print('\nPart 7: Visualization of Faces after PCA Dimension Reduction...')
    print('Original faces...')
    d.displaydata(X_norm[:100,:], 'Original faces (normalized data though)')

    print('Recovering original dimensions, please wait...')
    X_rec = km_faces.recover_data(Z, U, K)
    print('Visualizing the projected (reduced dimension) faces...')
    print('Recovered faces...')
    d.displaydata(X_rec[:100,:], 'Recovered faces')

