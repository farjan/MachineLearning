
# Labels of the images in the data file.
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def labels_matlab2python(l):
  '''
  Map labels 0,1..9 according to 0-base indexing.

  Remember in exercise labels were according to
  Matlab indexing which is 1-based indexing.

  Hence 1,2..9 are as 1,2..9 but 0 is as 10. For
  details see the description of Exercise No 3.
  '''
  # First add 1 since python has 0-based indexing.
  l = l + 1
  # Remember 10 means 0.
  l[l == 10] = 0
  return l


def output_layer_index_matlab(img_digit):
  '''
  Return index in output layer for the image
  digit.

  Remember in exercise labels were according to
  Matlab indexing which is 1-based indexing and
  we are dealing with Python 0 based indexing.

  Hence 1,2..9 are as 1,2..9 but 0 is as 10. For
  details see the description of Exercise No 3.
  '''
  # in Matlab 10 means 0.
  if img_digit == 0:
    # Which is 10 in Matlab but in Python
    # it will fall on index 9.
    return 9
  else:
    # In Python.
    # 1 will fall on index 0, 2 will fall
    # on index 1 ... 9 will fall on index 8.
    return img_digit - 1

