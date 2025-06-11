// to install numpy: pip install numpy  
// to import numpy: import numpy as np

# Arrays in NumPy

        arr = np.array([1, 2, 3, 4]) #1D array
        arr2d = np.array([[1, 2], [2, 3], [3, 4]]) #2D array

        print(arr[1]) # 1D array indexing
        print(arr2d[0][1]) # 2D array indexing

        print(arr) # print the 1D array
        print(arr2d) # print the 2D array with rows and columns

## Slicing in Arrays

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

## ndim // Returns the number of dimensions of the array. Note that the number of nested lists in the array determines its dimensionality.

        x = np.array([[1, 2, 3], [3, 4, 5]])
        print(x.ndim) 

## size // Returns the total number of elements in the array.

        print(x.size) 

## dtype // Returns the data type of the elements in the array.

        print(x.dtype) # Output: int64 (or int32 depending on the system architecture)

## itemsize // Returns the size in bytes of each element in the array.

        print(x.itemsize) # Output: 8 (for int64, or 4 for int32), why?  # Because int64 uses 8 bytes and int32 uses 4 bytes to store integer values.

## flatten() // Used to convert a multi-dimensional array into a one-dimensional array.

        a = np.array([[1, 2], [3, 4]])
        b = a.flatten()
        print(b)

# NumPy Vectorised Functions

## Vectorised Arthematic Operations

        a = np.array([1, 2, 3])
        b = np.array([10, 20, 30])
        print(a+b)
        print(a-b)
        print(a/b)
        print(a*b)
        print(a**2)  # Element-wise squaring of the array
        print(a%2)  # Element-wise modulus operation
        print(a//2)  # Element-wise floor division
        print(a**b)  # Element-wise exponentiation (a raised to the power of b)

## Universal Functions (ufuncs)

        angles = np.array([0, np.pi/2, np.pi])
        sines = np.sin(angles)
        print(sines)

# note: can use pi as np.pi, and e as np.e

## Common Vectorized (ufunc) Functions in NumPy:

- `np.add(x, y)` – Element-wise addition
- `np.subtract(x, y)` – Element-wise subtraction
- `np.multiply(x, y)` – Element-wise multiplication
- `np.divide(x, y)` – Element-wise division
- `np.sqrt(x)` – Square root
- `np.exp(x)` – Exponential (e^x)
- `np.log(x)` – Natural logarithm
- `np.sin(x)` – Sine
- `np.cos(x)` – Cosine
- `np.abs(x)` – Absolute value

# Statistical Functions in NumPy

## mean() // Returns the average of the elements in the array.

        a = np.array([1, 2, 3, 4, 5])
        print(np.mean(a))  # Output: 3.0

## median() // Returns the median of the elements in the array.

        print(np.median(a))  # Output: 3.0

## More Statistical Functions

- `np.std(a)` – Standard deviation
- `np.var(a)` – Variance
- `np.max(a)` – Maximum value
- `np.min(a)` – Minimum value
- `np.sum(a)` – Sum of all elements
- `np.prod(a)` – Product of all elements
- `np.percentile(a, 50)` – Percentile value (the value below which a given percentage of observations fall)
- `np.cumsum(a)` – Cumulative sum (running total of the elements in the array; e.g., for [1, 2, 3, 4], the running total is [1, 3, 6, 10])
- `np.cumprod(a)` – Cumulative product (running product of the elements in the array; e.g., for [1, 2, 3, 4], the running product is [1, 2, 6, 24])
- `np.argmax(a)` – Index of the maximum value
- `np.argmin(a)` – Index of the minimum value
- `np.quantile(a, 0.6)` – Quantile value (the value below which 60% of the observations fall)  

# Logical Operations

- `==` → Equal to
- `!=` → Not equal to
- `>` → Greater than
- `<` → Less than
- `>=` → Greater than or equal to
- `<=` → Less than or equal to
- `&` → Logical AND (use with parentheses around comparisons)
- `|` → Logical OR
- `~` → Logical NOT

Applying logical operations to arrays returns a boolean array indicating the result of the operation for each element.

# Broadcasting

## add scalar to array

        a = np.array([1, 2, 3])
        b = 5
        c = a + b  # Adds 5 to each element of the array
        print(c)  # Output: [6, 7, 8]

## Add 1D to 2D Array

        A = np.array([[1, 2, 3],
              [4, 5, 6]])

        B = np.array([10, 20, 30])

        C = A + B
        print(C) # Output: [[11, 22, 33], [14, 25, 36]]

## Column-wise Broadcasting

        A = np.array([[1, 2, 3],
              [4, 5, 6]])

        B = np.array([[10],
              [20]])

        C = A + B
        print(C) # Output: [[11, 12, 13], [24, 25, 26]]

## Broadcasting Error Example

        a = np.array([1, 2, 3])
        b = np.array([[1], [2]])

        a + b  # This will raise an error due to incompatible shapes