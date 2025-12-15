#! free_hw_env\Scripts\python.exe

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans

np.random.seed(0)

def similarity_function(Xi, Xj, sigma=1):
    '''
    Calculates the similarity between two vectors using the formula:
    exp( - || Xi - Xj ||²/ 𝟐𝜎²)

    Args:
        Xi (numpy.ndarray): First vector.
        Xj (numpy.ndarray): Second vector.
        sigma (float): Value of sigma.

    Returns:
        float: Calculated distance between the vectors Xi and Xj.
    '''
    squared_distance = np.linalg.norm(Xi - Xj) ** 2
    distance = np.exp(-squared_distance / (2 * sigma ** 2))
    return distance

def calculate_pairwise_distances(X, sigma=1):
    """
    Calculates pairwise distances between a set of points 
    using the given formula.

    Args:
        X (numpy.ndarray): Array representing a set of points. Shape: (N, D), 
        where N is the number of points and D is the 
        dimensionality of each point.
        sigma (float): Value of sigma.

    Returns:
        numpy.ndarray: Pairwise distances between the points. Shape: (N, N), 
        where N is the number of points in X. The (i, j)-th element represents 
        the pairwise distance between the i-th and j-th points.
    """
    # Calculate squared pairwise distances
    pairwise_sq_dists = np.square(np.linalg.norm(X[:, None] - X, axis=2))

    # Calculate exponential of the distances
    pairwise_distances = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    return pairwise_distances

def calculate_adjacency_matrix(df, sigma, k):
    """
    Calculates the adjacency matrix based on pairwise distances 
    between points in a DataFrame,
    considering only the k-nearest neighbors for each point.

    Args:
        df (pandas.DataFrame): DataFrame containing points' coordinates. 
        Rows represent points, and columns represent coordinates.
        sigma (float): Value of sigma.
        k (int): Number of nearest neighbors to consider.

    Returns:
        scipy.sparse.lil_matrix: Adjacency matrix. Shape: (N, N), 
        where N is the number of points. The (i, j)-th element represents 
        the distance between the i-th and j-th points if j is one of
        the k-nearest neighbors of i; otherwise, it is set to 0.
    """
    # Convert DataFrame to numpy array
    X = df.values

    # Calculate pairwise similarities using the provided similarity function
    pairwise_similarities = calculate_pairwise_distances(X, sigma)

    # Find the k points with the highest similarity for each point
    k_highest_indices = np.argsort(pairwise_similarities, axis=1)[:, -k-1:-1]

    # Construct adjacency matrix as sparse lil_matrix
    adjacency_matrix = lil_matrix(pairwise_similarities.shape, dtype=np.float64)
    for i in range(len(pairwise_similarities)):
        adjacency_matrix[i, k_highest_indices[i]] = pairwise_similarities[
            i, k_highest_indices[i]]

    return adjacency_matrix

def calculate_diagonal_matrix(A):
    """
    Calculates the diagonal matrix D using the adjacency matrix A.

    Args:
        A (scipy.sparse.lil_matrix): Adjacency matrix.

    Returns:
        scipy.sparse.lil_matrix: Diagonal matrix D. Shape: (N, N), 
        where N is the number of points. The (i, i)-th element represents 
        the sum of weights (or similarities) for the i-th point.
    """
    # Calculate the sum of weights (or similarities) for each point
    row_sums = np.array(A.sum(axis=1)).flatten()

    # Construct the diagonal matrix D as sparse lil_matrix
    D = lil_matrix(A.shape, dtype=np.float64)
    D.setdiag(row_sums)

    return D

def calculate_laplacian_matrix(A, D):
    """
    Calculates the Laplacian matrix L by subtracting the adjacency matrix A 
    from the diagonal matrix D.

    Args:
        A (scipy.sparse.lil_matrix): Adjacency matrix.
        D (scipy.sparse.lil_matrix): Diagonal matrix.

    Returns:
        scipy.sparse.lil_matrix: Laplacian matrix L. Shape: (N, N),
          where N is the number of points.
    """
    # Calculate the Laplacian matrix L by subtracting A from D
    L = D - A

    return L

def find_best_clusters(df, maximum_K):
    """
    This function finds the best number of clusters for a given dataset using the K-means algorithm.
    
    Inputs:
        - df: A DataFrame or array-like object representing the dataset.
        - maximum_K: An integer specifying the maximum number of clusters to consider.
        
    Outputs:
        - clusters_centers: A list of inertia values representing the sum of squared distances from each sample to its
          nearest cluster center for each number of clusters.
        - k_values: A list of integers from 1 to maximum_K (exclusive) representing the number of clusters considered.
    """
    
    clusters_centers = []  # Initialize an empty list to store the inertia values
    k_values = []  # Initialize an empty list to store the number of clusters
    
    # Iterate over each number of clusters from 1 up to maximum_K (exclusive)
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters=k)  # Create a K-means model with the current number of clusters
        kmeans_model.fit(df)  # Fit the model to the dataset
        
        clusters_centers.append(kmeans_model.inertia_)  # Append the inertia value to the clusters_centers list
        k_values.append(k)  # Append the number of clusters to the k_values list
        
    # Return the list of inertia values and the list of number of clusters
    return clusters_centers, k_values