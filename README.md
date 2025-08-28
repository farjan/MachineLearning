### Stanford University Machine Learning Course Exercises and Algorithms - By Andrew Ng 

Python implementation of the Machine Learning exercises of an open course by
Standford University by Andrew Ng available at coursera.org.
https://www.coursera.org/learn/machine-learning

To get most out of this Python exercises implementation,
it is recommended to first take course video lectures on coursra.org.
Lectures and exercises PDFs are available under ml/docs directory.

#### List of exercises
  * Programming Exercise 1: **Linear Regression**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex1.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/600104cc1c44a47f6c2a1f0209e6ee0af583280c/ml/Exercise%201.%20Linear%20Regression%20with%20multiple%20variables%20-%20non%20regularized.ipynb)
  * Programming Exercise 2: **Logistic Regression**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex2.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/afd6282de627242435a81aa14a716a5d2595dd5b/ml/Exercise%202%20Logistic%20Regression.ipynb)
  * Programming Exercise 3: **Multiclass Classification and Neural Networks**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex3.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/afd6282de627242435a81aa14a716a5d2595dd5b/ml/Exercise%203%20Multi-class%20Classification%20and%20Neural%20Networks.ipynb)
  * Programming Exercise 4: **Neural Networks Learning**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex4.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/afd6282de627242435a81aa14a716a5d2595dd5b/ml/Exercise%204%20Neural%20Networks%20Learning.ipynb)
  * Programming Exercise 5: **Regularized Linear Regression and Bias v.s Variance**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex5.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/afd6282de627242435a81aa14a716a5d2595dd5b/ml/Exercise%205%20Regularized%20Linear%20Regression%20and%20Bias%20v.s.%20Variance.ipynb)
  * Programming Exercise 6: **Support Vector Machine**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex6.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/afd6282de627242435a81aa14a716a5d2595dd5b/ml/Exercise%206%20Support%20Vector%20Machines.ipynb)
  * Programming Exercise 7: **K-means Clustering and Principal Component Analysis**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex7.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/3d961fe1d41250ee5eaa3dc08a141c7ede06364a/ml/Exercise%207%20K-means%20Clustering%20and%20Principal%20Component%20Analysis.ipynb)
  * Programming Exercise 8: **Anamoly Detection Recommender Systems**
      [PDF](https://raw.githubusercontent.com/farjan/MachineLearning/master/ml/docs/exercises/ex8.pdf)
      [Notebook](https://github.com/farjan/MachineLearning/blob/750c4af64268112e22e6b8e9ba2443463d2b818a/ml/Exercise%208%20Anomaly%20Detection%20and%20Recommender%20Systems.ipynb)

#### Course lectures
  * [Lectures PDFs](https://github.com/farjan/MachineLearning/tree/35084a05198e7130f96cd2fd5f909cd497cd33e5/ml/docs/lectures)
  
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

