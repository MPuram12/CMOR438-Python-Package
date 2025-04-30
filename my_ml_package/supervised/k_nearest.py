import numpy as np
from collections import Counter

class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) classifier.

    Attributes:
        k (int): The number of neighbors to consider.
    """
    def __init__(self, k=3):
        """
        Initializes the KNN classifier.

        Args:
            k (int, optional): The number of neighbors. Defaults to 3.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Trains the KNN classifier.  Note that KNN is a lazy learner, so this method
        simply stores the training data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted class labels, shape (n_samples,).
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Args:
            x (ndarray): A single data point with shape (n_features,).

        Returns:
            int: The predicted class label.
        """
        # Compute distances between x and all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Determine the most common label among the k neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]  # Return the label

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
    
    def score(self, X, y):
        """
        Calculates the accuracy of the model on the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The true labels, shape (n_samples,).

        Returns:
            float: The accuracy of the model.
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy


if __name__ == '__main__':
    # Simple test case
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2.5, 2], [4, 2]])
    
    # Create and train the KNN classifier
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)  # Expected: [0 1]

    # Calculate accuracy
    accuracy = knn.score(X_test, np.array([0,1]))
    print(f"Accuracy: {accuracy:.2f}")
