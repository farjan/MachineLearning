

import numpy as np
import supervised.logistic_regression_non_vectorized_newton as lg_reg_nreg
import supervised.logistic_regression_non_vectorized_newton_regularized_poly6 as lg_reg_reg

class Exercise2(object):
  '''
  Execute exercise No 2 'Logistic Regression'.
  For detailed description of the exercise see
  "Programming Exercise 2" ex2.pdf available at 
  coursera.org, an open Stanford University course 
  "Machine Learning" by Andrew Ng.
  '''

  def execute(self):
    '''
    Execute exercise.
    '''
    # Coursera exercises.
    print('Exercise 2: Logistic Regression\n')
    self.__exercise_parts_non_reg()
    self.__exercise_parts_reg()

    # Instructor's course website, uncomment following if interested.
    #self.__exercise_website_non_reg()
    #self.__exercise_website_reg()


  def __exercise_parts_non_reg(self):
    '''
    Execute parts of the exercise of non regularized algorithm.
    '''
    print('Part 1: Plotting...')
    print('Loading data...')
    datafname = '../data/ex2data1.dat'
    lgr = lg_reg_nreg.LogisticRegression(datafname)
    print('Plotting data with + indicating (y = 1) examples and o ' 'indicating (y = 0) examples.')
    lgr.plot_scatter(0, 1, 'Exam 1 score', y_label='Exam 2 score')

    print('\nPart 2, 3: Compute Cost, Gradient Descent and Optimize...')
    print('Important Note:- Please note fminunc is not used for optimization as suggested in ' + 
          'exercise rather algorithm uses implementation of gradient descent (see lectures).' +
          ' If you are interested how to use fmin_XX functions for optimization' +
          ' please see class LogisticRegRegularizedMulticlass.\n')
    lgr.train()
    print('Cost {} test passed'.format(lgr.min_cost()))
    print('Thetas\n {}'.format(lgr.min_theta))
    lgr.plot_decision_line(0, 1, 'Exam 1 score', y_label='Exam 2 score')
    lgr.plot_cost_curve()

    print('\nPart 4: Predict and Accuracies...')
    predict = lgr.predict(np.array([[45.0], [85.0]]))
    print('For a student with scores 45 and 85, ' + 
          'we predict an admission probability of {} test passed.'.format(predict))

    print('Computing accuracy...')
    data = np.loadtxt(datafname, delimiter=',')
    cols = data.shape[1] - 1
    X = data[:,:cols]
    y = data[:,cols:]
    
    pred = lgr.predict_label(X)
    print('Predicted accuracy {}%.'.format(np.mean(pred == y) * 100))


  def __exercise_parts_reg(self):
    '''
    Execute parts of the exercise of regularized algorithm.
    '''
    print('\nRegularized Logistic Regression...')
    print('Important Note:- Please note fminunc is not used for optimization as suggested in ' + 
          'exercise rather algorithm uses implementation of gradient descent (see lectures).' +
          ' If you are interested how to use fmin_XX functions for optimization' +
          ' please see class LogisticRegRegularizedMulticlass.\n')

    datafname = '../data/ex2data2.dat'
    plot_scatter = True

    print('Part 1, 2: Load data and map features to polynomial 6...')
    lmda = [0, 1, 10, 100]
    for l in lmda:
      lgrr = lg_reg_reg.LogisticRegRegularizedPoly6(datafname)

      if plot_scatter:
        print('Plotting data scatter...')
        lgrr.plot_scatter(0, 1, 'Microchip Test 1', 'Microchip Test 2')
        plot_scatter = False

      print('\nTraining with lambda {}...'.format(l))
      lgrr.train(15, l)

      print('Plotting decision line...')
      lgrr.plot_decision_line(str(l), 0, 1, 'Microchip Test 1', 'Microchip Test 2')

      print('Computing accuracy...')
      data = np.loadtxt(datafname, delimiter=',')
      cols = data.shape[1] - 1
      X = data[:,:cols]
      y = data[:,cols:]
      pred = lgrr.predict_label(X)
      print('Predicted accuracy {}%.'.format(np.mean(pred == y) * 100))


  def __exercise_website_non_reg(self):
    '''
    Instructor's (Andrew Ng) course website exercise.
    http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html
    '''
    print("\nExercise - Logistic Regression and Newton's Method." + 
          " - Available at instructor's course website...")
    print("http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html\n")
    logr = lg_reg_nreg.LogisticRegression('../data/student_score_admission_chance.dat')
    print('Minimized Theta values after training with iteration=7.')
    logr.train()

    print('The probability that a student with a score of' + 
          ' 20 on Exam 1 and 80 on Exam 2 will not be admitted to college is:')
    print(1.0 - logr.predict(np.array([[20.0], [80.0]])))

    logr.plot_scatter(0, 1, 'Exam Score 1', 'Exam Score 2')
    logr.plot_decision_line(0, 1, 'Exam Score 1', 'Exam Score 2')
    logr.plot_cost_curve()


  def __exercise_website_reg(self):
    '''
    Instructor's (Andrew Ng) course website exercise.
    http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
    '''

    print("\nExercise - Logistic Regression Regularize and Newton's Method." + 
          " - Available at instructor's course website...")
    print("http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html\n")
    print("Newton's Method Regularization.")

    datafname = '../data/ex5Log.dat'
    lmda = [0, 1, 10]
    plot_scatter = True
    for l in lmda:
      logrr = lg_reg_reg.LogisticRegRegularizedPoly6(datafname)
      logrr.train(15, l)
  
      norm = np.linalg.norm(logrr.min_theta)
      print('\nNorms of theta {} when lambda {}'.format(norm, str(l)))
      pred = logrr.predict(-0.046659, 0.81652)
      print('The predicted value {} for f1=-0.046659 and f2=0.81652: '.format(pred))

      if plot_scatter:
        logrr.plot_scatter(0, 1, 'u', 'v')
        plot_scatter = False
      logrr.plot_decision_line(str(l), 0, 1, '', '')
      logrr.plot_cost_curve()
pass