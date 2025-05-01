import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA).

    Attributes:
        n_components (int): Number of principal components to keep.
    """
    def __init__(self, n_components=None):
        """
        Initializes the PCA object.

        Args:
            n_components (int, optional): Number of principal components to keep.
                If None, keep all components. Defaults to None.
        """
        self.n_components = n_components
        self.components = None  # Principal components; will be set during `fit`
        self.mean = None  # Mean of the data; will be set during `fit`

    def fit(self, X):
        """
        Fits the PCA model to the data.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).
        """
        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Select the top n_components eigenvectors
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors

    def transform(self, X):
        """
        Applies dimensionality reduction to the data.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).

        Returns:
            ndarray: The transformed data, shape (n_samples, n_components).
        """
        # Center the data using the mean computed during fit
        X_centered = X - self.mean
        # Project the data onto the principal components
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """
        Fits the model to the data and then performs dimensionality reduction on it.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).

        Returns:
            ndarray: The transformed data, shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transforms the data back to its original space.  This is possible because PCA
        is an orthogonal transformation.

        Args:
            X_transformed (ndarray): The transformed data, shape (n_samples, n_components).

        Returns:
            ndarray: The data in the original space, shape (n_samples, n_features).
        """
        return np.dot(X_transformed, self.components.T) + self.mean

    def explained_variance_ratio(self):
        """
        Returns the fraction of the total variance that is explained by each principal component.

        Returns:
            ndarray: The explained variance ratio, shape (n_components,).
        """
        if self.components is None:
            raise ValueError("PCA must be fit before calling explained_variance_ratio()")

        # The eigenvalues are a measure of the variance explained by each component.
        cov_matrix = np.cov((X - self.mean), rowvar=False) # X is the original data
        eigenvalues = np.linalg.eig(cov_matrix)[0]
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order

        total_variance = np.sum(sorted_eigenvalues)
        return sorted_eigenvalues[:self.n_components] / total_variance


if __name__ == '__main__':
    # Simple test case
    X = np.array([[1, 2, 3], [2, 4, 1], [3, 8, 0], [4, 6, 5], [5, 10, 2]])

    # Create and apply PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print("Original Data:\n", X)
    print("Transformed Data:\n", X_transformed)

    # Demonstrate inverse transform
    X_original = pca.inverse_transform(X_transformed)
    print("Reconstructed Data:\n", X_original)

    # Explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio())
