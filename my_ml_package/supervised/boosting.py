import numpy as np
from .decision_trees import DecisionTreeRegressor  # Import your DecisionTreeRegressor

class BoostingRegressor:
    """
    Boosting Regressor (Gradient Boosting for Regression).

    Attributes:
        n_estimators (int): The number of trees in the ensemble.
        learning_rate (float): The learning rate for boosting.
        max_depth (int): The maximum depth of the trees.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initializes the BoostingRegressor.

        Args:
            n_estimators (int, optional): The number of trees in the ensemble. Defaults to 100.
            learning_rate (float, optional): The learning rate. Defaults to 0.1.
            max_depth (int, optional): The maximum depth of the trees. Defaults to 3.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None  # Store the initial prediction

    def fit(self, X, y):
        """
        Builds the boosting ensemble from the training data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
        """
        n_samples = X.shape[0]
        self.initial_prediction = np.mean(y)  # Initialize prediction with the mean of the target
        self.trees.append(self.initial_prediction) #store the initial prediction as the first element in the trees list

        # Initialize the residuals
        residuals = y - self.initial_prediction

        # Build each tree
        for _ in range(self.n_estimators):
            # 1. Fit a decision tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)  # Store the tree

            # 2. Update the predictions (in place)
            new_predictions = tree.predict(X)
            residuals = residuals - self.learning_rate * new_predictions

    def predict(self, X):
        """
        Predicts the output for the given data by summing the predictions of all trees.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted values, shape (n_samples,).
        """
        predictions = np.full(X.shape[0], self.initial_prediction)  # Initialize with the initial prediction
        for tree in self.trees[1:]: #iterate from the second tree onwards
            predictions += self.learning_rate * tree.predict(X)  # Add the weighted predictions of each tree
        return predictions

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
    # Simple test case
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
    y_train = np.array([2.5, 3.5, 2, 5, 4.5, 6])
    X_test = np.array([[2.5, 2], [4, 2], [7,7]])

    # Create and train the boosting regressor
    boosting = BoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=1)  # You can adjust the hyperparameters
    boosting.fit(X_train, y_train)

    # Make predictions
    predictions = boosting.predict(X_test)
    print("Predictions:", predictions)

    # Evaluate
    r_squared = boosting.score(X_test, np.array([3, 4, 8]))
    print(f"R^2: {r_squared:.2f}")
