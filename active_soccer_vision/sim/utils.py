import numpy as np
from functools import reduce

def multiply_list(*list_of_arrays):
    mat = reduce(np.matmul, list(list_of_arrays))
    return mat
