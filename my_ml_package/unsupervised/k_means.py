import numpy as np


class KMeans:
    """
    K-Means clustering.

    Attributes:
        n_clusters (int): The number of clusters to form.
        max_iters (int): Maximum number of iterations of the k-means algorithm.
        random_state (int): Determines random number generation for centroid initialization.
    """
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        Initializes the KMeans object.

        Args:
            n_clusters (int, optional): The number of clusters to form. Defaults to 3.
            max_iters (int, optional): Maximum number of iterations. Defaults to 100.
            random_state (int, optional): Random state for initialization. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None # Centroid positions will be learned
        self.labels = None # Centroid positions will be learned

    def fit(self, X):
        """
        Computes k-means clustering.

        Args:
            X (ndarray): Training data, shape (n_samples, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        # Iterate until convergence or max iterations
        for _ in range(self.max_iters):
            # Assign each sample to the nearest centroid
            self.labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = self._calculate_centroids(X)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Args:
            X (ndarray): New data to predict, shape (n_samples, n_features).

        Returns:
            ndarray: Index of the cluster each sample belongs to, shape (n_samples,).
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """
        Assigns each sample to the nearest centroid.

        Args:
            X (ndarray): Data, shape (n_samples, n_features).

        Returns:
            ndarray: Cluster labels for each sample, shape (n_samples,).
        """
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _calculate_distances(self, X, centroids):
        """
        Calculates the Euclidean distances between each sample in X and each centroid.

        Args:
            X (ndarray): Data, shape (n_samples, n_features).
            centroids (ndarray): Centroids, shape (n_clusters, n_features).

        Returns:
            ndarray: Distances, shape (n_samples, n_clusters).
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                distances[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
        return distances

    def _calculate_centroids(self, X):
        """
        Calculates the new centroids by taking the mean of the samples in each cluster.

        Args:
            X (ndarray): Data, shape (n_samples, n_features).

        Returns:
            ndarray: New centroids, shape (n_clusters, n_features).
        """
        n_clusters = self.n_clusters
        n_features = X.shape[1]
        new_centroids = np.zeros((n_clusters, n_features))
        for cluster_idx in range(n_clusters):
            cluster_points = X[self.labels == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                # Keep the old centroid if the cluster is empty
                new_centroids[cluster_idx] = self.centroids[cluster_idx]
        return new_centroids

    def fit_predict(self, X):
        """
        Computes cluster centers and predicts cluster index for each sample.
        Convenience method; performs fit() and returns the resulting labels.

        Args:
            X (ndarray): Training data, shape (n_samples, n_features).

        Returns:
            ndarray: Cluster labels for each sample, shape (n_samples,).
        """
        self.fit(X)
        return self.labels
    
    def score(self, X):
        """
        Calculates the Silhouette Coefficient for the given data.

        Args:
            X (ndarray): Input data, shape (n_samples, n_features).

        Returns:
            float: Silhouette Coefficient. Returns None if number of clusters is 1.
        """
        from sklearn.metrics import silhouette_score

        if self.n_clusters <= 1:
            return None  # Silhouette score is not defined for a single cluster

        labels = self.predict(X)
        return silhouette_score(X, labels)


if __name__ == '__main__':
    # Simple test case
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    # Create and apply KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels = kmeans.fit_predict(X)

    print("Cluster Labels:", cluster_labels)  # Expected: [0 0 1 1 0 1]
    print("Centroids:", kmeans.centroids)
