import numpy as np

class Perceptron:
    """
    A simple Perceptron class for binary classification.

    Attributes:
        learning_rate (float): The learning rate for weight updates.
        n_iters (int): The number of iterations for training.
        weights (ndarray): The weights of the perceptron.
        bias (float): The bias term of the perceptron.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the Perceptron.

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
            n_iters (int, optional): The number of iterations. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the perceptron on the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).  Must be 0 or 1.

        Raises:
            ValueError: If the target values are not 0 or 1.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Validate target values
        if not all(label in [0, 1] for label in y):
            raise ValueError("Target values must be 0 or 1 for Perceptron.")
        
        # Convert y to -1 and 1 for the update rule
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0 #keep as 0 and 1 for consistency with input

                # Update weights if misclassified
                if y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted class labels, shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.where(linear_output >= 0, 1, 0)
        return y_predicted

    def score(self, X, y):
        """
        Calculates the accuracy of the perceptron on the given data.

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
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Instantiate and train the perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iters=10)
    perceptron.fit(X, y)

    # Make predictions
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)  # Expected: [0 0 0 1]

    # Calculate and print accuracy
    accuracy = perceptron.score(X, y)
    print(f"Accuracy: {accuracy:.2f}")  # Expected: 1.0
