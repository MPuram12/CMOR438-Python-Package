import numpy as np
from .decision_trees import DecisionTreeRegressor  # Import your DecisionTreeRegressor

class RandomForestRegressor:
    """
    Random Forest Regressor.

    Attributes:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the trees.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        random_state (int):  Controls the randomness of the bootstrapping and feature selection.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        """
        Initializes the RandomForestRegressor.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
            max_depth (int, optional): The maximum depth of the trees. Defaults to None (unlimited depth).
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
            random_state (int, optional): Controls the randomness. Defaults to None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []  # List to store the individual decision trees
        self.feature_indices = [] # List to store the indices of the features used for each tree

    def fit(self, X, y):
        """
        Builds the random forest from the training data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        if self.random_state:
            np.random.seed(self.random_state)

        # Build each tree
        for _ in range(self.n_estimators):
            # 1. Bootstrap the data
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrapped = X[bootstrap_indices]
            y_bootstrapped = y[bootstrap_indices]

            # 2. Select a random subset of features
            n_features_subset = int(np.sqrt(n_features))  # Typically sqrt(n_features) for regression
            feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
            self.feature_indices.append(feature_indices) #store

            X_bootstrapped_subset = X_bootstrapped[:, feature_indices]

            # 3. Build a decision tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         )  # Pass other hyperparameters as needed
            tree.fit(X_bootstrapped_subset, y_bootstrapped)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts the output for the given data by averaging the predictions of all trees.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted values, shape (n_samples,).
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i, tree in enumerate(self.trees):
            # Get the features that this tree used
            feature_indices = self.feature_indices[i]
            # Select only those features from the input
            X_subset = X[:, feature_indices]
            tree_predictions = tree.predict(X_subset) # Make prediction
            predictions += tree_predictions # Accumulate predictions

        return predictions / self.n_estimators  # Average the predictions

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

    # Create and train the random forest regressor
    rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)  # You can adjust the hyperparameters
    rf.fit(X_train, y_train)

    # Make predictions
    predictions = rf.predict(X_test)
    print("Predictions:", predictions)

    # Evaluate
    r_squared = rf.score(X_test, np.array([3, 4, 8]))
    print(f"R^2: {r_squared:.2f}")
