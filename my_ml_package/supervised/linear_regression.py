import numpy as np

class LinearRegressionNeuron: # Changed class name here
    """
    A single neuron model for linear regression.  This is a simplified neural network.

    Attributes:
        learning_rate (float): The learning rate for weight updates.
        n_iters (int): The number of iterations for training.
        weights (ndarray): The weights of the neuron.
        bias (float): The bias term of the neuron.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the LinearRegressionNeuron.

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
        Trains the neuron on the given data using gradient descent.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zeros
        self.bias = 0  # Initialize bias to zero

        # Gradient Descent
        for _ in range(self.n_iters):
            # Calculate predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate the error
            error = y_predicted - y

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, error)  # Gradient of weights
            db = (1 / n_samples) * np.sum(error)      # Gradient of bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts the output for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted values, shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculates the coefficient of determination (R^2) of the prediction.

        Args:
            X (ndarray): Input features, shape (n_samples, n_features).
            y (ndarray): True target values, shape (n_samples,).

        Returns:
            float: R^2 score.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


if __name__ == '__main__':
    # Test case with a simple linear relationship
    X = np.array([[1], [2], [3], [4], [5]])  # Example: Single feature
    y = np.array([2, 4, 5, 4, 5])  # Example: y = 2x + 1 with some noise

    # Create and train the neuron
    neuron = LinearRegressionNeuron(learning_rate=0.01, n_iters=1000)
    neuron.fit(X, y)

    # Make predictions
    X_test = np.array([[6], [7], [8]])
    predictions = neuron.predict(X_test)
    print("Predictions for X_test:", predictions)

    # Evaluate the model
    r_squared = neuron.score(X, y)
    print(f"R^2 Score: {r_squared:.2f}")
