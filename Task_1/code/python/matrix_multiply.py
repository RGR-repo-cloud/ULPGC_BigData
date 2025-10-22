"""
Matrix multiplication implementation in Python
"""

def matrix_multiply(A, B):
    """
    Basic matrix multiplication algorithm O(n^3)
    """
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C