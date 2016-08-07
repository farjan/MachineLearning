
import os

import exercises.coursera.ex1
import exercises.coursera.ex2
import exercises.coursera.ex3
import exercises.coursera.ex4
import exercises.coursera.ex5
import exercises.coursera.ex6
import exercises.coursera.ex7
import exercises.coursera.ex8

def help():
  print()
  print('1. Linear Regression')
  print('2. Logistic Regression')
  print('3. Multi-class Classification and Neural Networks')
  print('4. Neural Networks Learning')

  print('5. Regularized Linear Regression and Bias v.s. Variance')
  print('6. Support Vector Machines')
  print('7. K-means Clustering and Principal Component Analysis')
  print('8. Anomaly Detection and Recommender Systems')
  print('9. To exit')
  print('-----------------------------------')


if __name__ == '__main__':
  ml_exercises = {}
  ml_exercises[1] = exercises.coursera.ex1.Exercise1
  ml_exercises[2] = exercises.coursera.ex2.Exercise2
  ml_exercises[3] = exercises.coursera.ex3.Exercise3
  ml_exercises[4] = exercises.coursera.ex4.Exercise4

  ml_exercises[5] = exercises.coursera.ex5.Exercise5
  ml_exercises[6] = exercises.coursera.ex6.Exercise6
  ml_exercises[7] = exercises.coursera.ex7.Exercise7
  ml_exercises[8] = exercises.coursera.ex8.Exercise8

  help()
  while True:
    key = input("Please enter exercise number:")
    print('Number selected {}.'.format(key))
    if key.isdigit():
      key = int(key)
      if 1 <= key <= 8:
        # Comment this line if does not work on your OS/shell.
        os.system('cls' if os.name == 'nt' or os.name == 'dos' else 'clear')

        # Execute the exercise.
        ex = ml_exercises[key]()
        ex.execute()
      elif key == 9:
        break
      else:
        help()
        continue
    else:
      help()
      continue
    print('\n----------Finished executing exercise {}----------\n'.format(key))
  print('Exited')
pass
