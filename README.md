
OBJECTIVE
  Implementation of the Machine Learning exercises of an open course by
  Standford University by Andrew Ng available at coursera.org.
  
SUGGESTION
  In order to get most out of this Python exercises implementation,
  it is recommended to first take course video lectures each lecture
  comes with an exercise description PDF and Octave skeleton code
  please go through that. Specially go through the Octave scripts
  such as ex1.m, ex2.m...,ex8_cofi.m, ex8.m etc, starting
  with exX_ where X is a digit.
  After each class session there... you can run and study related
  Python code implementation included in this project.
  
DEPENDENCIES
  -> Tested on Python 64 bit 3.5
  -> numpy for Python 64 bit 3.5
  -> scipy Python 64 bit 3.5
  -> It is suggested to install "Anaconda 64 bit Python 3.5" to get Python environment.
  
HOW TO RUN
  $> python main.py
  
CODE ORGANIZATION
  -> Course test data files already included under "./data" directory.
  -> Usually each algorithm is in its own Python class.
  -> Supervised learning algorithms are under "./supervised" package.
  -> Unsupervised learning algorithm are under "./unsupervised" package.
  -> Neural Networks algorithms are under "./supervised/neural_networks" package.
  -> Some utility routines are under "./utils" package.
  -> Scripts that execute and test the algorithms are under "./exercises/coursera"
     package.
  -> main.py scripts starts the execution.
  
NOTES
  -> "Anaconda 64 bit Python 3.5" was installed to get Python environment.
  -> Testing was done on Windows 7 environment but since it is Python other
      platforms should not be a problem.
  -> Development was done in Visual Studio 2013 hence a workspace "MachineLearning.sln"
      file is included but other platform users can ignore "MachineLearning.sln"
      and "MachineLearning.pyproj" files.
