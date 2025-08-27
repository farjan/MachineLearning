
import re
import numpy as np
import matplotlib.pyplot as plt
import supervised.svm as s
from nltk import PorterStemmer

class Exercise6(object):
  '''
  Execute exercise No 6 'Support Vector Machines'.
  For detailed description of the exercise see "Programming Exercise 6" 
  ex6.pdf available at coursera.org, an open Stanford University course
  "Machine Learning" by Andrew Ng.
  '''

  def __init__(self, **kwargs):
    '''
    Initialize instance parameters.
    '''
    self.datafname1_X = '../data/ex6data1_X.dat'
    self.datafname1_y = '../data/ex6data1_y.dat'
    self.datafname2_X = '../data/ex6data2_X.dat'
    self.datafname2_y = '../data/ex6data2_y.dat'
    self.datafname3_X = '../data/ex6data3_X.dat'
    self.datafname3_y = '../data/ex6data3_y.dat'

    self.fname_email_sample1 = '../data/emailSample1.txt'
    self.fname_spam_train_X  = '../data/ex6data_spam_train_X.dat'
    self.fname_spam_train_y = '../data/ex6data_spam_train_y.dat'
    

  def execute(self):
    '''
    Execute exercise.
    '''
    print('Exercise 6 - Support Vector Machines.\n')
    # Coursera exercises.
    self.__exercise_svm()
    self.__exercise_svm_spam()


  def __exercise_svm(self):
    '''
    Execute plain SVM parts of the exercise.
    '''
    print('Part 1: Loading and Visualizing Dataset 1...')
    print('Loading and visualizing data...')
    X = np.loadtxt(self.datafname1_X, delimiter=',')
    y = np.loadtxt(self.datafname1_y, delimiter=',')
    xlim = (0, 4.5)
    ylim = (1.5, 5)
    svm1 = s.Svm(X, y)
    svm1.plot_data('', '', 'Scatter Data Plot', xlim, ylim)

    print('\nPart 2: Training Linear SVM and Plotting...')
    svm1.train(X, y, 1.0, 1e-3)
    svm1.plot_decision_boundary_linear('', '', 
                                'SVM Decision Boundary with C = 1 (Example Dataset 1)',
                                xlim, ylim)

    svm2 = s.Svm(X, y)
    svm2.train(X, y, 1000.0, 1e-3)
    svm2.plot_decision_boundary_linear('', '', 
                                'SVM Decision Boundary with C = 1000 (Example Dataset 1)',
                                xlim, ylim)

    print('\n\nPart 3: Implementing Gaussian Kernel...')
    x0 = np.array([[1], [2], [1]])
    x1 = np.array([[0], [4], [-1]])
    print('Evaluating the Gaussian Kernel...')
    svm_test = s.Svm(None, None)
    sim = svm_test.gaussian_kernel(x0, x1, 2)
    print('Gaussian Kernel between\nx0 = [1, 2, 1], x1 = [0, 4, -1] and \nsigma = 2.0 is {} ' 
          '(should be about 0.324652) test passed.'.format(sim))

    print('\nPart 4: Loading Visualizing Dataset 2...')
    print('Loading and visualizing data...')
    X = np.loadtxt(self.datafname2_X, delimiter=',')
    y = np.loadtxt(self.datafname2_y, delimiter=',')
    xlim = (0, 1) 
    ylim = (0.4, 1)

    svm3 = s.Svm(X, y)
    svm3.plot_data('', '', 'Scatter Data Plot', xlim, ylim)

    # Please note rather then exercise's SVM.gaussian_kernel()
    # scipy RBF was used.

    print('\nPart 5: Training SVM with RBF Kernel (Dataset 2)...')
    print('Training SVM with RBF Kernel...')
    svm3.train(X, y, 1.0, 1e-3, 'rbf', 100)
    svm3.plot_decision_boundary('', '', 
                                'SVM Decision Boundary with C = 1 and gamma = 100 (Example Dataset 2)',
                                xlim, ylim)

    print('\nPart 6: Loading Visualizing Dataset 3...')
    print('Loading and visualizing data...')
    X = np.loadtxt(self.datafname3_X, delimiter=',')
    y = np.loadtxt(self.datafname3_y, delimiter=',')
    xlim = (-0.6, 0.3) 
    ylim = (-0.8, 0.6)

    svm4 = s.Svm(X, y)
    svm4.plot_data('', '', 'Scatter Data Plot', xlim, ylim)

    # Please note rather then exercise's SVM.gaussian_kernel()
    # scipy RBF was used.

    print('\nPart 7: Training SVM with RBF Kernel (Dataset 3)...')
    print('Training SVM with RBF Kernel...')
    svm4.train(X, y, 3, 1e-3, 'rbf', 30.0)
    svm4.plot_decision_boundary('', '', 
                                'SVM Decision Boundary with C = 3 and gamma = 30.0 (Example Dataset 3)',
                                xlim, ylim)


  def __exercise_svm_spam(self):
    '''
    Execute spam SVM parts of the exercise.
    '''
    print('\nSpam Classification with SVMs\n')
    print('Part 1: Email Preprocessing...')
    f = open(self.fname_email_sample1)
    vocab = self.__read_vocab()
    word_indices = self.__process_email(f.read(), vocab)
    print('Preprocessing sample email ({})'.format(self.fname_email_sample1))
    print(word_indices)

    print('\nPart 2: Feature Extraction...')
    features = self.__email_features(word_indices, vocab)
    
    # Print Stats
    print('Length of feature vector: {}'.format(len(features)))
    print('Number of non-zero entries: {}'.format(np.sum(features > 0)))

    print('\nPart 3: Train Linear SVM for Spam Classification...')
    X = np.loadtxt(self.fname_spam_train_X, delimiter=',')
    y = np.loadtxt(self.fname_spam_train_y, delimiter=',')

    svm1 = s.Svm(X, y)
    print('Training...')
    svm1.train(X, y, 0.1, 1e-3)
    p = svm1.predict(X)
    accuracy = np.mean(p == y) * 100
    print('Original labels \n{}'.format(y))
    print('Predicted labels \n{}'.format(p))
    print('Training Accuracy: {}%'.format(accuracy))

    #print('\nPart 5: Top Predictors of Spam...')
    # Sort the weights.
    #weights = svm1.svc._get_coef() Are these wights I don't know?
    #print('Top predictors of spam:')
    #print('{}'.format(np.argsort(weights)[0][0:15]))


  def __process_email(self, email_contents, vocab):
    '''
    Preprocess a the body of an email and returns a
    list of word_indices.

    Arguments:
      email_contents (str): Email body.
      vocab (dict): Words dictionary.

    Return:
      (str list): Tokenized email body after processing.
    '''
    # Lower case.
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # Tokenize and also get rid of any punctuation
    word_list = re.split(' |@|$|/|#|\.|-|:|&|\*|\+|=|[|]|\?|!|(|)|{|}|,|''|"|>|_|<|;|%',
                        email_contents)

    # Remove empty string and skip the word if it is too short.
    word_list = [s for s in word_list if s and len(s) > 1]

    # Remove any non alphanumeric characters
    word_list = [re.sub('[^a-zA-Z0-9]', '', s) for s in word_list]

    # Remove empty string and skip the word if it is too short.
    word_list = [s for s in word_list if s and len(s) > 1]

    # Stem the word
    ps = PorterStemmer() 
    word_list = [ps.stem_word(s) for s in word_list]
    word_indices = []

    # Find index in vocab list.
    for w in word_list:
      if w in vocab:
        word_indices.append(vocab[w])
    return word_indices


  def __email_features(self, word_indices, vocab):
    '''
    Take in a word_indices vector and produces a
    feature vector from the word indices. 

    Arguments:
      word_indices (list): Word indices.
      vocab (dict): Words dictionary.

    Return:
      (1d column vector): 0, 1 feature vector.
    '''
    x = np.zeros((len(vocab), 1))
    
    for i in word_indices:
      if i in vocab.values():
        x[i, 0] = 1
    return x


  def __read_vocab(self):
    '''
    Read vocabulary file and return dictionary.

    Return:
      (dict): Dictionary of the words.
    '''
    vocab = {}
    f = open('../data/vocab.txt')

    for l in f:
      (idx, word) = l.split()
      vocab[word] = idx
    return vocab
