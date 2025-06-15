import numpy as np

arr = np.random.randint(1, 50, (5,4))
print(arr)

j = 3

for row in arr:
    print(row[j])
    j-= 1
    if j < 0:
        break

