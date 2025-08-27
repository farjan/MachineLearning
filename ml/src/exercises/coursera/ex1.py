
import numpy as np
import supervised.linear_regression as lnr_reg

class Exercise1(object):
  '''
  Execute exercise No 1 'Linear Regression'.
  For detailed description of the exercise see
  "Programming Exercise 1" ex1.pdf available at 
  coursera.org, an open Stanford University course 
  "Machine Learning" by Andrew Ng.
  '''

  def execute(self):
    '''
    Execute exercise.
    '''
    # Coursera exercises.
    print('Exercise 1 - Linear Regression.\n')
    self.__exercise_parts()
    self.__exercise_optional_parts()

    # Instructor's course website, uncomment following if interested.
    #self.__exercise_website()


  def __exercise_parts(self):
    '''
    Execute parts of the exercise.
    '''
    print('Part 1: Basic Function...')
    print('Running warm up exercise 5x5 identity matrix...')
    print(np.eye(5))

    print('\nPart 2: Plotting...')
    print('Loading data...')
    lr = lnr_reg.LinearRegression('data/ex1data1.dat')
    print('Showing data tail...')
    lr.tail()
    lr.plot_single_feature(0, 'Population of City in 10,000s', 'Profit in $10,000s')

    print('Part 3: Training using gradient descent...')
    # Some gradient descent settings.
    iterations = 1500;
    alpha = 0.01;
    lr.train(alpha, iterations)

    # Display cost and theta.
    print('Cost {}'.format(lr.min_cost()))
    print('Theta found by gradient descent\n {}'.format(lr.min_theta))

    # Plot the linear fit.
    lr.plot_regression_line(0, 'Population of City in 10,000s', 'Profit in $10,000s')
    # Make some predictions.
    predict1 = lr.predict(np.array([[3.5],]))
    print('For population = 35,000, we predict a profit of {}'.format(predict1 * 10000))
    predict2 = lr.predict(np.array([[7.0],]))
    print('For population = 70,000, we predict a profit of {}'.format(predict2 * 10000))

    print('\nPart 4: Visualizing J(theta)...')
    print('Cost curve plot...')
    lr.plot_cost_curve()
    print('Contour plot...')
    lr.plot_contour()


  def __exercise_optional_parts(self):
    '''
    Execute OPTIONAL parts of the exercise.
    '''
    print('\nExecuting batch of optional exercises...')
    print('Part 1: Feature Normalization...')
    print('Loading data...')
    lr = lnr_reg.LinearRegression('data/ex1data2.dat')
    print('Showing data tail...')
    lr.tail(10)

    # Scale features and set them to zero mean
    print('Normalizing Features ...');
    lr.normalize_feature()
    print('Showing data tail of normalized data...')
    lr.tail(10)

    print('Part 2: Training using gradient descent...')
    # Some gradient descent settings.
    iterations = 100
    alpha = 1.0
    lr.train(alpha, iterations)

    # Display cost and theta.
    print('Cost {}'.format(lr.min_cost()))
    print('Theta found by gradient descent\n {}'.format(lr.min_theta))
    print('Cost curve plot...')
    lr.plot_cost_curve()

    # Make prediction.
    predict = lr.predict(np.array([[1650.0], [3.0]]))
    print('Predicted price of a 1650 sq-ft, 3 bedrooms house (using gradient descent) {}'.format(predict))

    print('\nPart 3: Normal Equations...')
    print('Solving with normal equations...')
    # Create new object this time train using
    # normal equations method.
    lr_normal = lnr_reg.LinearRegression('data/ex1data2.dat')
    lr_normal.train_normal_eq_method()

    print('Theta computed from the normal equations\n {}'.format(lr_normal.min_theta))
    # Make prediction.
    predict = lr_normal.predict(np.array([[1650.0], [3.0]]))
    print('Predicted price of a 1650 sq-ft, 3 bedrooms house (using normal equations) {}'.format(predict))


  def __exercise_website(self):
    '''
    Instructor's (Andrew Ng) course website exercise.
    http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
    '''
    print("\nExercise - Linear Regression - Available at instructor's course website...")
    print("http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html\n")
    lr = lnr_reg.LinearRegression('data/age_height.dat')

    print('Input data file tail')
    lr.tail()

    print('Minimized theta values after training with alpha=0.07 and iteration=1500')
    lr.train()
    print(lr.min_theta)

    print('The predicted height for age 3.5 is:')
    print(lr.predict(np.array([[3.5],])))

    lr.plot_regression_line(0, 'Age in years', 'Height in meters')
    lr.plot_cost_curve()
pass
