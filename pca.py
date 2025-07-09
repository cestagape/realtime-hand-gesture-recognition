import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ml_framework.feature_scaling import StandardScaler

class PCA:
    """Principal Component Analysis (PCA) implementation using NumPy."""
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X):
        """Compute the principal components of the dataset."""
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_indices]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
    
    def transform(self, X):
        """Project data onto the principal components."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def explained_variance_ratio(self):
        """Compute the explained variance ratio for each component."""
        return self.explained_variance[:self.n_components] / np.sum(self.explained_variance)
    
    def plot_explained_variance(self):
        """Visualize the explained variance ratio of principal components."""
        explained_var_ratio = self.explained_variance_ratio()
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(explained_var_ratio), marker='o', linestyle='--')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid()
        plt.show()

# Apply PCA to gesture data
if __name__ == "__main__":
    df = pd.read_csv("gesture_data/30_3_M1_gt.csv")
    df = df.drop(columns=['ground_truth'], errors='ignore')
    
    scaler = StandardScaler()
    scaler.load("project/scaler.npz")
    X_scaled = scaler.transform(df.values)
    
    pca = PCA(n_components=10)
    pca.fit(X_scaled)
    transformed_data = pca.transform(X_scaled)
    
    np.save("project/pca_transformed.npy", transformed_data)
    pca.plot_explained_variance()
