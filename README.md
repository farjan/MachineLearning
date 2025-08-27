### Stanford University Machine Learning Course Exercises and Algorithms - By Andrew Ng 

Python implementation of the Machine Learning exercises of an open course by
Standford University by Andrew Ng available at coursera.org.
https://www.coursera.org/learn/machine-learning

To get most out of this Python exercises implementation,
it is recommended to first take course video lectures on coursra.org.
Lectures and exercises PDFs are available under ml/docs directory.
  
#### How to run
  * $> python -m venv ml_course_env
  * $> source ml_course_env/bin/activate
  * $> pip install -r requirements_ml.txt
  * $> python main.py  which is under ml/src
  * or run Jupyter notebook for each exercise indivitually, each notebook
    contains all the related exercise code
  
#### Code organization
  * Course test data files already included under **ml/data** directory.
  * Course lectures and exercises included under **ml/docs** directory.
  * Usually each algorithm is in its own Python class.
  * Supervised learning algorithms are under **ml/src/supervised** package.
  * Unsupervised learning algorithm are under **ml/src/unsupervised** package.
  * Neural Networks algorithms are under **ml/src/supervised/neural_networks** package.
  * Some utility routines are under **ml/src/utils** package.
  * Exercises solutions are under **src/exercises/coursera** package.
  * main.py scripts starts the execution of all exercises.
  * Although **notebooks also exist** for each exercise under **ml/** directory

#### List of exercises
  * [Programming Exercise 1: Linear Regression] (https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex1.pdf)
