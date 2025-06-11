# to install numpy: pip install numpy
# to import numpy: import numpy as np

import numpy as np

# Arrays in NumPy

'''
arr = np.array([1, 2, 3, 4]) #1D array
arr2d = np.array([[1, 2], [2, 3], [3, 4]]) #2D array

print(arr[1]) # 1D array indexing
print(arr2d[0][1]) # 2D array indexing

print(arr) # print the 1D array
print(arr2d) # print the 2D array with \n

# Slicing in Arrays

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(a[1:4]) # Slicing from index 1 to 3 (4 is excluded)
print(a[:3]) # Slicing from start to index 2 (3 is excluded)
print(a[::3]) # Slicing with step 3, starting from index 0
print(a[1:10:2]) # Slicing from index 1 to 9 with step 2
print(a[-3:]) # Slicing from index -3 to the end of the array

b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0, :]) #first row
print(b[:, 1]) #second row
print(b[1:3, 0:2]) #submatrix 
'''

# Commonly Used Functions In NumPy

## reshape() // Used to change the shape of an array without changing its data.
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(3, 2)  # Reshape to 3 rows and 2 columns
print(b)

## arange() // Used to create an array with evenly spaced values within a given interval.

arr = np.arange(0, 10, 3) # Creates an array with values from 0 to 10 with a step of 3, excluding 10 if it were to be included.
print(arr)

## eye() // Creates an identity matrix (square matrix with 1s on the diagonal and 0s elsewhere).

e = np.eye(3)
print(e)

## ndim // Returns the number of dimensions of the array.

x = np.array([[1, 2, 3], [3, 4, 5]])
print(x.ndim)



