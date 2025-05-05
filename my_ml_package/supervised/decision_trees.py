import numpy as np

class DecisionTreeRegressor:
    """
    Decision Tree Regressor.

    Attributes:
        max_depth (int): The maximum depth of the tree.
    """
    def __init__(self, max_depth=None):
        """
        Initializes the DecisionTreeRegressor.

        Args:
            max_depth (int, optional): The maximum depth of the tree. Defaults to None (unlimited depth).
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Builds the decision tree from the training data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predicts the output for the given data.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).

        Returns:
            ndarray: The predicted values, shape (n_samples,).
        """
        predictions = [self._predict_one(x, self.tree) for x in X]
        return np.array(predictions)

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).
            depth (int): The current depth of the tree.

        Returns:
            dict: A dictionary representing the decision tree node.
        """
        n_samples, n_features = X.shape

        # Base cases:
        # 1. If there are no more samples, return the mean of the target values.
        if n_samples == 0:
            return {'value': np.mean(y)}
        # 2. If all target values are the same, return that value.
        if np.all(y == y[0]):
            return {'value': y[0]}
        # 3. If the maximum depth is reached, return the mean of the target values.
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': np.mean(y)}
        # 4. If there are no more features to split on
        if n_features == 0:
            return {'value': np.mean(y)}

        # Try to find the best possible feature and threshold to split on
        best_split = self._get_best_split(X, y)

        # If no good split is found, return the mean of the target values
        if best_split['feature_index'] is None:
            return {'value': np.mean(y)}

        # Split data and recursively build left and right subtrees
        left_X, left_y = best_split['left_X'], best_split['left_y']
        right_X, right_y = best_split['right_X'], best_split['right_y']

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        # Return the node
        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left_child': left_child,
            'right_child': right_child,
        }

    def _get_best_split(self, X, y):
        """
        Finds the best split for the data based on the variance reduction.

        Args:
            X (ndarray): The input features, shape (n_samples, n_features).
            y (ndarray): The target values, shape (n_samples,).

        Returns:
            dict: A dictionary containing the best split information.
                {'feature_index': int, 'threshold': float, 'left_X': ndarray, 'left_y': ndarray,
                 'right_X': ndarray, 'right_y': ndarray}
        """
        n_samples, n_features = X.shape
        if n_features == 0:
            return {}

        # Calculate the initial variance
        initial_variance = np.var(y)
        best_variance_reduction = 0
        best_split = {}

        for feature_index in range(n_features):
            # Get the unique values of the feature
            feature_values = np.unique(X[:, feature_index])
            #consider all values as thresholds
            for threshold in feature_values:
                # Split the data based on the threshold
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                left_X, left_y = X[left_mask], y[left_mask]
                right_X, right_y = X[right_mask], y[right_mask]

                # Calculate the variance reduction
                if len(left_y) > 0 and len(right_y) > 0:
                    left_variance = np.var(left_y)
                    right_variance = np.var(right_y)
                    variance_reduction = initial_variance - (len(left_y) / n_samples) * left_variance - (
                        len(right_y) / n_samples
                    ) * right_variance

                    # Update the best split if it's better than the current best
                    if variance_reduction > best_variance_reduction:
                        best_variance_reduction = variance_reduction
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'left_X': left_X,
                            'left_y': left_y,
                            'right_X': right_X,
                            'right_y': right_y,
                        }

        return best_split

    def _predict_one(self, x, node):
        """
        Predicts the output for a single data point using the decision tree.

        Args:
            x (ndarray): A single data point with shape (n_features,).
            node (dict): A dictionary representing a decision tree node.

        Returns:
            float: The predicted value.
        """
        # If the node is a leaf node, return the value
        if 'value' in node:
            return node['value']

        # Otherwise, go to the left or right child based on the feature value
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_one(x, node['left_child'])
        else:
            return self._predict_one(x, node['right_child'])
        
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
    X_test = np.array([[2.5, 2], [4, 2]])

    # Create and train the decision tree regressor
    dt = DecisionTreeRegressor(max_depth=3)  # You can adjust the max_depth
    dt.fit(X_train, y_train)

    # Make predictions
    predictions = dt.predict(X_test)
    print("Predictions:", predictions)

    # Evaluate with R^2
    r_squared = dt.score(X_test, np.array([3,4]))
    print(f"R^2: {r_squared:.2f}")
