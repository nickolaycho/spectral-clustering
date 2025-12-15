#! free_hw_env\Scripts\python.exe

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

np.random.seed(0)

def inverse_power_method(A, niterations_max=50, tol=1e-15):
    """
    Computes the smallest eigenvalue and corresponding eigenvector of a 
    sparse matrix using the inverse power method.

    Inputs:
    - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, 
    or compatible sparse format)
    - niterations_max: Maximum number of iterations (default: 50)
    - tol: Tolerance for convergence (default: 1e-15)

    Outputs:
    - smallest_eigenvalue: Computed smallest eigenvalue
    - eigenvector: Normalized eigenvector corresponding 
        to the smallest eigenvalue
    """       
    v_old = np.ones((A.shape[0],)) + \
        1e-7 * np.random.rand(A.shape[0])  # Initial guess for eigenvector (!=0)
    
    v_old =  v_old / np.linalg.norm(v_old)

    lambda_1_inv_new = 1

    LU = spla.splu(A)  # LU factorization of the sparse matrix A
  
    for k in range(niterations_max):

        # Update the eigenvalue
        lambda_1_inv_old = lambda_1_inv_new

        # Solve the linear system v_new = A^(-1) v_old using LU factorization
        v_new = LU.solve(v_old)
        
        # Compute the eigenvalue
        lambda_1_inv_new = v_old.T.dot(v_new)

        # Normalize the current eigenvector approximation
        v_old = v_new / np.linalg.norm(v_new)
        
        if abs(lambda_1_inv_new - lambda_1_inv_old) < tol:  # Check for convergence
            break

    return(1/lambda_1_inv_new, v_old)

def inverse_power_method_test(A, niterations_max=50, tol=1e-15):
    """
    Computes the smallest eigenvalue and corresponding eigenvector of a 
    sparse matrix using the inverse power method.

    Inputs:
    - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, 
    or compatible sparse format)
    - niterations_max: Maximum number of iterations (default: 50)
    - tol: Tolerance for convergence (default: 1e-15)

    Outputs:
    - smallest_eigenvalue: Computed smallest eigenvalue
    - eigenvector: Normalized eigenvector corresponding 
        to the smallest eigenvalue
    - eigenvector_iterations: List of eigenvector 
        approximations at each iteration
    - eigenvalue_iterations: List of eigenvalues computed at each iteration
    """        
    v_old = np.zeros(A.shape[0]) + 1e-7 * np.random.rand(A.shape[0])
    v_old = v_old / np.linalg.norm(v_old)

    lambda_1_inv_new = 1

    LU = spla.splu(A)  # LU factorization of the sparse matrix A
    
    eigenvector_iterations = [v_old.copy()]  # List to store eigenvector 
                                        # approximations at each iteration
    eigenvalue_iterations = []  # List to store 
                                # eigenvalues computed at each iteration

    for k in range(niterations_max):

        # Update the eigenvalue
        lambda_1_inv_old = lambda_1_inv_new

        # Solve the linear system using LU factorization
        v_new = LU.solve(v_old)

        # Compute the eigenvalue
        lambda_1_inv_new = v_old.T.dot(v_new)

        # Normalize the current eigenvector approximation
        v_old = v_new / np.linalg.norm(v_new)
        
        # Store the current eigenvector approximation and eigenvalue
        eigenvector_iterations.append(v_old)
        eigenvalue_iterations.append(1 / lambda_1_inv_new)

        if abs(1/lambda_1_inv_new - 1/lambda_1_inv_old) < tol:  # Check for convergence
            break

    return(1/lambda_1_inv_new, v_old,
            eigenvector_iterations, eigenvalue_iterations)
