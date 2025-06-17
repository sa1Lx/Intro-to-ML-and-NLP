import numpy as np

arr = np.random.uniform(0, 10, 20)
print(arr)

rounded_arr = np.round(arr, 2)

# print(rounded_arr)

print(np.min(rounded_arr))
print(np.max(rounded_arr))
print(np.median(rounded_arr))

for i in range(len(rounded_arr)):
    if rounded_arr[i] < 5:
        rounded_arr[i] = rounded_arr[i]**2

# print(rounded_arr)

def numpy_alternate_sort(array):
    sorted_list = sorted(array)
    result = []

    i = 0
    j = len(sorted_list) - 1

    while i <= j:
        result.append(float(sorted_list[i]))
        i += 1
        if i <= j:
            result.append(float(sorted_list[j]))
            j -= 1

    return result


# print(numpy_alternate_sort(rounded_arr))
