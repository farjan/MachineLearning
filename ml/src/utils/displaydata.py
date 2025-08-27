'''
 Ported from Matlab routine included in the
 "Programming Exercise 3" ex3.zip package.
'''

import numpy as np
import matplotlib.pyplot as plt

def displaydata(X, title_label=''):
  '''
  Display 2D data stored in X in a nice grid.

  Arguments:
    X (m x n float matrix): Stores image data.
  '''

  # Compute rows, cols.
  m = X.shape[0]
  n = X.shape[1]
  example_width = int(np.round(np.sqrt(n)))
  example_height = int((n / example_width))

  # Compute number of items to display.
  display_rows = int(np.floor(np.sqrt(m)))
  display_cols = int(np.ceil(m / display_rows))

  # Between images padding.
  pad = 1
  # Setup blank display.
  display_array = -np.ones(shape=(
    pad + display_rows * (example_height + pad),
    pad + display_cols * (example_width + pad)))

  # Copy each example into a patch on the display array.
  curr_ex = 0
  for j in range(display_rows):
    for i in range(display_cols):
      if curr_ex >= m: 
        break; 
		  # Copy the patch
      rs = pad + j * (example_height + pad) 
      re = rs + example_height
      cs = pad + i * (example_width  + pad)
      ce = cs + example_width
      display_array[rs:re, cs:ce] = \
        (np.reshape(X[curr_ex, :], (example_height, example_width))).transpose()
      curr_ex = curr_ex + 1
    if curr_ex > m:
      break 
  pass

  plt.axis('off')
  plt.imshow(display_array, cmap='gray', aspect='auto')
  plt.title(title_label)
  plt.show()
