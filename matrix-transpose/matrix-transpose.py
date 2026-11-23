import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    m=len(A)
    n=len(A[0])
    ans=np.zeros((n,m))
    for i in range (n):
        for j in range(m):
            ans[i][j]=A[j][i]

    return ans
