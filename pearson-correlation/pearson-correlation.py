# import numpy as np

# def pearson_correlation(X: list) -> np.ndarray:
#     m = len(X[0])
#     n = len(X)

#     matrix = np.zeros((m,m))

#     for i in range(m):
#         for j in range(m):
#             covxy = np.corrcoef(X[0:n,i], X[0:n,j])
#             matrix[i,j] = covxy[0,1]

#     return matrix


import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    X=np.array(X)
    m = len(X[0])
    n = len(X)

    matrix = np.zeros((m,m))

    for i in range(m):
        stdx = np.std(X[0:n,i],ddof=1)

        for j in range(m):
            stdy = np.std(X[0:n,j],ddof=1)

            covxy = np.cov(X[0:n,i], X[0:n,j])[0,1]

            matrix[i,j] = covxy / (stdx * stdy)

    return matrix