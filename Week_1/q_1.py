import numpy as np

arr = np.random.randint(1, 51, (5,4)) # np.random.randint(low, high, size) generates random integers from low (inclusive) to high (exclusive)
print(arr)

j = 3

for row in arr:
    print(row[j])
    j-= 1
    if j < 0:
        break


max_per_row = np.max(arr, axis=1) # axis=1 means we are looking for the maximum value across each row
# If axis=0, we would be looking for the maximum value across each column
print(max_per_row)

mean_val = np.mean(arr)
filtered_elements = []

for row in arr:
    for num in row:
        if num <= mean_val:
            filtered_elements.append(int(num)) # Add to list if ≤ mean

print(filtered_elements)

def numpy_boundary_traversal(matrix):
    result = []

    rows = len(matrix)
    cols = len(matrix[0])

    for j in range(cols):
        result.append(int(matrix[0][j]))

    for i in range(1, rows - 1):
        result.append(int(matrix[i][cols - 1]))

    for j in range(cols - 1, -1, -1):
        result.append(int(matrix[rows - 1][j]))

    for i in range(rows - 2, 0, -1):
        result.append(int(matrix[i][0]))

    return result

print(numpy_boundary_traversal(arr))