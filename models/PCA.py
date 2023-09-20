
# Importing required libraries for the PCA implementation
import numpy as np

# Define the PCA class
class PrincipalComponentAnalysis:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Calculate the mean and subtract it from the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(X.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]
        
    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components.T)
