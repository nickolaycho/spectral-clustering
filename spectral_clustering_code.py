#! env_for_sc\Scripts\python.exe

import time
import os

from deflation_method import deflation_method
from utils import *

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import linalg
import networkx as nx
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
font = {'weight' : 'bold',
        'size'   : 15}
fontsize = 18
matplotlib.rc('font', **font)


np.random.seed(0) # if used

circle_path = os.path.join('data', 'Circle.csv')
spiral_path = os.path.join('data', 'Spiral.csv')

circle_df = pd.read_csv(circle_path, header=None, names=['x','y'])
spiral_df = pd.read_csv(spiral_path, header=None, 
                        names=['x','y', 'cluster'])


# 0. Exploratory Dataset Analysis
fig, ax = plt.subplots(1,2, figsize=(1720/134,800/134), dpi=134)

sns.scatterplot(data=circle_df, x='x', y='y', ax=ax[0])
ax[0].set_title('Circle Dataset', fontweight="bold", fontsize=18)

sns.scatterplot(data=spiral_df, x='x', y='y', hue='cluster', ax=ax[1])
ax[1].set_title('Spiral Dataset', fontweight="bold", fontsize=18)

fig.suptitle('Datasets visualization', fontweight="bold")

path_to_fig = os.path.join('figs', r'01_dataset_visualization_' + str(time.time()) + '.png')
plt.savefig(path_to_fig, dpi=300)

# Starting Analysis
spiral_df = spiral_df[["x","y"]]

sigma = 1.0
k_values = [10, 20, 40] #[5,10,20]

# change here the dataset where to perform the analysis
dataset = spiral_df # can be one of [circle_df, spiral_df]
                

# Function to find threshold for considering the eigenvalues=0
threshold_for_considering_eig_zero = lambda eigs: np.mean(eigs)/2.5 # 2.5 for spiral


fig, ax = plt.subplots(len(k_values), 4, figsize=(16, 9), dpi=600)
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, 
                    top=0.82, wspace=0.3, hspace=0.25)
#fig.suptitle('Datasets visualization', fontweight="bold", size='large')
ax[0,0].set_title(f'k-nearest neighborhood\n similarity graph\nfor the dataset', 
                  fontweight="bold", fontsize=fontsize)
ax[0,1].set_title('eigenvalues of the \ncorresponding\n Laplacian matrix', 
                  fontweight="bold", fontsize=fontsize)
ax[0,2].set_title('Spectral Clustering \nusing our Deflation Method \nto compute eigenvalues', 
                  fontweight="bold", fontsize=fontsize)
ax[0,3].set_title('Spectral Clustering \nusing Scipy functions \nto compute eigenvalues', 
                  fontweight="bold", fontsize=fontsize)

# sns.scatterplot(data=circle_df, x='x', y='y', ax=ax[0])

for j,k in enumerate(k_values):

    # 1. Adjacency matrix W construction
    adjacency_matrix = calculate_adjacency_matrix(dataset, sigma, k)

    # Plot the corresponding graph
    G = nx.Graph(adjacency_matrix)
    pos = dataset.to_dict('split')['data']
    pos = {i: coord for (i,coord) in enumerate(pos)}
    options = {"node_size": 0.5, "alpha": 0.7}
    nx.draw(G, pos=pos, ax=ax[j,0], width=0.1, node_color="tab:blue", **options)

    ax[j,0].grid('on')
    ax[j,0].axis('on')
    ax[j,0].set_ylabel("k= "+str(k), 
        fontweight='bold', fontsize=fontsize)
    
    # 2 Degree matrix D and Laplacian matrix L construction
    degree_matrix = calculate_diagonal_matrix(adjacency_matrix)
    laplacian_matrix = calculate_laplacian_matrix(adjacency_matrix, degree_matrix)

    # 3-4. Calculate number of connected components of the graph 
    # by calculating some small eigenvalues of the Laplacian matrix
    
    eigenvalues, eigenvectors = deflation_method(A=laplacian_matrix, 
                        M=10, niterations_max=100, tol=1e-25)
    
    # Calculate threshold for considering zero the eigenvalue
    ths = threshold_for_considering_eig_zero(eigenvalues)

    indices_of_smallest_eigs = np.argwhere(np.abs(eigenvalues)<ths).flatten()
    
    n_experimental_clusters = len(indices_of_smallest_eigs)
    rise_index = n_experimental_clusters - 1

    # Plot eigenvalues and number of connected components
    ax[j,1].axvline(x=rise_index, color='tab:blue', linestyle='--', 
                    label='First large gap deflation')

    ax[j,1].plot(np.abs(eigenvalues), '-o', label='deflation power method', 
                 color='tab:blue')

    ax[j,1].set_xticks([s for s in range(10) ]) 
    ax[j,1].set_xticklabels([f'$\lambda_{str(s+1)}$' for s in range(10)], fontsize=12)
    ax[j,1].set_ylabel(r'|$\lambda_i$|', fontweight='bold', fontsize=14)

    # 3/4 Check: calculate the smallest eigenvalues
    # using a built method of scipy.sparse.linalg
    theoric_eigenvalues, theoric_eigenvectors = linalg.eigsh(
        laplacian_matrix, which='SM', k=10)
    
    theoric_eigenvectors = theoric_eigenvectors[:,np.argsort(theoric_eigenvalues)]
    theoric_eigenvalues = np.abs(theoric_eigenvalues)
    theoric_eigenvalues = theoric_eigenvalues[np.argsort(theoric_eigenvalues)]
    
    ax[j,1].plot(theoric_eigenvalues, '-o', label="theoretical", color='tab:red')
    
    # Calculate threshold for considering zero the eigenvalue
    ths = threshold_for_considering_eig_zero(theoric_eigenvalues)
    indices_of_smallest_eigs_theo = np.argwhere(
        np.abs(theoric_eigenvalues)<ths).flatten()
    
    n_theoric_clusters = len(indices_of_smallest_eigs_theo)
    rise_index_theo = n_theoric_clusters - 1

    ax[j,1].axvline(x=rise_index_theo, color='tab:red', linestyle='--', 
                    label='First large gap theoric')
    if j==0:
        ax[j,1].legend()

    # 5 Define matrix U [M x n], whose M columns are the M eigenvectors 
    # of size N (with N number of points in the dataset) corresponding
    # to the M smallest eigenvalues of L.
    U_exp = eigenvectors[:, :n_experimental_clusters]
    U_theo = theoric_eigenvectors[:, :n_theoric_clusters]

    # 6. Cluster the rows of U by k-means, into M clusters
    kmeans_exp = KMeans(n_clusters=n_experimental_clusters, max_iter=1000)
    kmeans_exp.fit(U_exp)
    labels_exp = kmeans_exp.labels_

    kmeans_theo = KMeans(n_clusters=n_theoric_clusters, max_iter=1000)
    kmeans_theo.fit(U_theo)
    labels_theo = kmeans_theo.labels_

    # 7.  Assign the original points in X 
    # to the same clusters as their corresponding rows in U
    dataset_to_plot_exp = dataset.copy()
    dataset_to_plot_exp["cluster"] = labels_exp

    dataset_to_plot_theo = dataset.copy()
    dataset_to_plot_theo["cluster"] = labels_theo

    # 8. Plot clusters obtained by Spectral Clustering
    sns.scatterplot(data=dataset_to_plot_exp, x='x', y='y', 
                    hue="cluster", ax=ax[j,2], palette='colorblind')
    sns.scatterplot(data=dataset_to_plot_theo, x='x', y='y', 
                    hue="cluster", ax=ax[j,3], palette='colorblind')
    
path_to_fig = os.path.join('figs', r'02_spectral_clustering_' + str(time.time()) + '.png')
fig.savefig(path_to_fig, dpi=300)


# 9. Compute and plot clusters for the same set of points
# using k-means directly on the initial points
# (i.e., without calculating eigenvectors of the Laplacian matrix
# associated with them)

fig, ax = plt.subplots(2,2, figsize=(1280/134,1000/134), dpi=134)
fig.subplots_adjust(left=None, bottom=None, right=None, 
                    top=None, wspace=0.30, hspace=0.55)

# Circle dataset
clusters_centers, k_values = find_best_clusters(circle_df[["x", "y"]], 10)

ax[0,0].plot(k_values, clusters_centers, 'o-', color = 'orange')
ax[0,0].set_xlabel("Number of Clusters (K)", fontweight="bold", fontsize=fontsize)
ax[0,0].set_ylabel("Cluster Inertia", fontweight="bold", fontsize=fontsize)
ax[0,0].set_title("Elbow Plot of KMeans", fontweight="bold", fontsize=fontsize)

kmeans_check = KMeans(n_clusters=3)
kmeans_check.fit(circle_df[["x", "y"]])
labels_check = kmeans_check.labels_

circle_df["cluster"] = labels_check

sns.scatterplot(data=circle_df, x='x', y='y', 
                hue="cluster", ax=ax[0,1], palette='colorblind')

ax[0,1].set_title("k-Means Clustering", fontweight="bold", fontsize=fontsize)

# Spiral dataset
clusters_centers, k_values = find_best_clusters(spiral_df[["x", "y"]], 10)

ax[1,0].plot(k_values, clusters_centers, 'o-', color = 'orange')
ax[1,0].set_xlabel("Number of Clusters (K)", fontweight="bold", fontsize=fontsize)
ax[1,0].set_ylabel("Cluster Inertia", fontweight="bold", fontsize=fontsize)
ax[1,0].set_title("Elbow Plot of KMeans", fontweight="bold", fontsize=fontsize)

kmeans_check = KMeans(n_clusters=3)
kmeans_check.fit(spiral_df[["x", "y"]])
labels_check = kmeans_check.labels_

spiral_df["cluster"] = labels_check

sns.scatterplot(data=spiral_df, x='x', y='y', 
                hue="cluster", ax=ax[1,1], palette='colorblind')

ax[1,1].set_title("k-Means Clustering", fontweight="bold", fontsize=fontsize)

path_to_fig = os.path.join('figs', r'03_k_means_comparison' + str(time.time()) + '.png')
plt.savefig(path_to_fig, dpi=300)




