import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    nrow = len(A)
    ncol = len(A[0]) if nrow > 0 else 0
    At = np.zeros((ncol, nrow))
    for i in range (0, nrow):
        for j in range(0, ncol):
            At[j, i]=A[i][j]
    return At
    pass
