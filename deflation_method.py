#! free_hw_env\Scripts\python.exe

from inverse_power_method import inverse_power_method

import numpy as np
from scipy import sparse

np.random.seed(0)

def deflation_method(A, M, niterations_max=50, tol=1e-15):
    """
    Computes the M smallest eigenvalues and corresponding eigenvectors of a 
    sparse matrix using the deflation method.

    Inputs:
    - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, 
    or compatible sparse format)
    - M: Number of smallest eigenvalues to compute
    - niterations_max: Maximum number of iterations for the inverse power 
    method (default: 50)
    - tol: Tolerance for convergence of the inverse power method 
    (default: 1e-15)

    Outputs:
    - eigenvalues: Array of the M smallest eigenvalues
    - eigenvectors: Array of the M corresponding eigenvectors
    """
    eigenvalues = np.zeros(M)
    eigenvectors = np.zeros((A.shape[0], M))

    for i in range(M):
        eigenvalue, eigenvector = inverse_power_method(A, niterations_max, tol)  # Compute smallest eigenvalue and eigenvector
        
        eigenvalues[i] = eigenvalue  # Store eigenvalue
        eigenvectors[:, i] = eigenvector  # Store eigenvector
        
        # Deflation step: Update A by removing the contribution of the computed eigenvector
        A = A - eigenvalue / eigenvalues[0] * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors

def deflation_method_test(A, M, niterations_max=50, tol=1e-15):
    """
    Computes the M smallest eigenvalues and corresponding eigenvectors of a 
    sparse matrix using the deflation method.

    Inputs:
    - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, 
    or compatible sparse format)
    - M: Number of smallest eigenvalues to compute
    - niterations_max: Maximum number of iterations for the inverse power 
    method (default: 50)
    - tol: Tolerance for convergence of the inverse power method 
    (default: 1e-15)

    Outputs:
    - eigenvalues: Array of the M smallest eigenvalues
    - eigenvectors: Array of the M corresponding eigenvectors
    """
    eigenvalues = np.zeros(M)
    eigenvectors = np.zeros((A.shape[0], M))

    for i in range(M):
        eigenvalue, eigenvector = inverse_power_method(A, niterations_max, tol)  # Compute smallest eigenvalue and eigenvector
        
        eigenvalues[i] = eigenvalue  # Store eigenvalue
        eigenvectors[:, i] = eigenvector  # Store eigenvector
        
        # Deflation step: Update A by removing the contribution of the computed eigenvector
        A = A - eigenvalue/eigenvalues[0] * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors