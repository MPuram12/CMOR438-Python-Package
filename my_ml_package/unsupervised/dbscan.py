import numpy as np

class DBSCAN:
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise.

    Attributes:
        eps (float): The maximum distance between two samples for them to be considered
            neighbors.
        min_samples (int): The number of samples in a neighborhood for a point to be
            considered a core point.
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initializes the DBSCAN object.

        Args:
            eps (float, optional): The maximum distance between two samples. Defaults to 0.5.
            min_samples (int, optional): The number of samples in a neighborhood. Defaults to 5.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # Cluster labels for each point; will be assigned during `fit`

    def fit(self, X):
        """
        Performs DBSCAN clustering on the data.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).
        """
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1, dtype=int)  # Initialize all labels as -1 (noise)
        cluster_label = 0  # Start with cluster label 0

        for i in range(n_samples):
            if self.labels[i] != -1:  # If the point has already been processed, skip it
                continue

            # Find the neighbors of point i
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Point i is not a core point; mark it as noise
                self.labels[i] = -1
                continue

            # Point i is a core point; start a new cluster
            self.labels[i] = cluster_label
            self._expand_cluster(X, neighbors, cluster_label)
            cluster_label += 1  # Move to the next cluster label

    def _expand_cluster(self, X, neighbors, cluster_label):
        """
        Expands the cluster starting from a core point.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).
            neighbors (list): A list of indices of the neighbors of the core point.
            cluster_label (int): The label of the current cluster.
        """
        i = 0
        while i < len(neighbors):
            j = neighbors[i]
            if self.labels[j] == -1:
                # Point j is not yet visited; assign it to the current cluster
                self.labels[j] = cluster_label
                new_neighbors = self._get_neighbors(X, j)
                if len(new_neighbors) >= self.min_samples:
                    # Point j is also a core point; add its neighbors to the queue
                    neighbors.extend(new_neighbors)
            elif self.labels[j] == 0:
                #if a neighbor is noise, change it to a border point
                self.labels[j] = cluster_label
            i += 1

    def _get_neighbors(self, X, i):
        """
        Finds the neighbors of a data point within a given radius (eps).

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).
            i (int): The index of the data point.

        Returns:
            list: A list of indices of the neighbors of point i.
        """
        neighbors = []
        for j in range(X.shape[0]):
            if i != j and self._euclidean_distance(X[i], X[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    def _euclidean_distance(self, x1, x2):
        """
        Computes the Euclidean distance between two data points.

        Args:
            x1 (ndarray): The first data point, shape (n_features,).
            x2 (ndarray): The second data point, shape (n_features,).

        Returns:
            float: The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit_predict(self, X):
        """
        Performs DBSCAN clustering on the data and returns cluster labels.
        This is a convenience method that combines fit() and a return of the labels_ attribute.

        Args:
            X (ndarray): The input data, shape (n_samples, n_features).

        Returns:
            ndarray: Cluster labels for each point, shape (n_samples,).
                   Label -1 indicates noise.
        """
        self.fit(X)
        return self.labels


if __name__ == '__main__':
    # Simple test case
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10,2], [0, 2], [4, 10], [7, 9], [12,10]])
    

    # Create and apply DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=3)
    cluster_labels = dbscan.fit_predict(X)  # Use fit_predict()

    print("Cluster Labels:", cluster_labels)  #  Output: [ 0  0 -1 -1  0 -1 -1 -1  0 -1 -1 -1]
