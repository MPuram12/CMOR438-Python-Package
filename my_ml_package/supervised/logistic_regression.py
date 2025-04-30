import numpy as np

class LogisticRegressionNeuron:
    """
    A single neuron model for logistic regression.

    Attributes:
        learning_rate (float): The learning rate for weight updates.
        n_iters (int): The number of iterations for training.
        weights (ndarray): The weights of the neuron.
        bias (float): The bias term of the neuron.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the LogisticRegressionNeuron.

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
            y (ndarray): The target values, shape (n_samples,).  Must be 0 or 1.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Validate target values
        if not all(label in [0, 1] for label in y):
            raise ValueError("Target values must be 0 or 1 for Logistic Regression.")

        # Gradient Descent
        for _ in range(self.n_iters):
            # Calculate predictions (probabilities)
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_output)

            # Calculate the error
            error = y_predicted - y

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.5):
        """
        Predicts the class labels for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            threshold (float, optional): The probability threshold for classification. Defaults to 0.5.

        Returns:
            ndarray: The predicted class labels, shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_output)
        y_predicted = np.where(y_predicted_proba >= threshold, 1, 0)
        return y_predicted
    
    def predict_proba(self, X):
        """
        Predicts the probabilities of the class labels for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted probabilities, shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_output)
        return y_predicted_proba

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

    def _sigmoid(self, x):
        """
        Computes the sigmoid function.

        Args:
            x (ndarray): The input values.

        Returns:
            ndarray: The sigmoid function applied to the input values.
        """
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # Test case with a simple example
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 0, 1])

    # Create and train the logistic regression neuron
    neuron = LogisticRegressionNeuron(learning_rate=0.1, n_iters=1000)
    neuron.fit(X, y)

    # Make predictions
    predictions = neuron.predict(X)
    print("Predictions:", predictions)

    # Evaluate the model
    accuracy = neuron.score(X, y)
    print(f"Accuracy: {accuracy:.2f}")
