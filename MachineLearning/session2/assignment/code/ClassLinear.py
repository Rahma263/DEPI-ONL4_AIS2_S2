import numpy as np
import matplotlib.pyplot as plt


"""
LinearRegressionGD
------------------

This class implements Linear Regression using Gradient Descent.

The model learns a linear equation of the form:

    y = theta_1 * X + theta_0

Where:
    theta_1 → weight (slope or weights vector for multiple features)
    theta_0 → bias (intercept)

The goal is to minimize the Sum of Squared Errors (SSE)
between predicted and actual values by iteratively updating
the parameters using Gradient Descent.
"""


class LinearRegressionGD():

    def __init__(self, learning_rate=0.001, n_iters=100, normalize=False):
        """
        Initialize the model.

        Parameters:
        -----------
        learning_rate : float
            Controls the step size during parameter updates.
            Large values may cause divergence.
            Small values may cause slow convergence.

        n_iters : int
            Number of iterations for Gradient Descent.

        normalize : bool
            If True, features will be normalized before training.
        """

        self.alpha = learning_rate
        self.n_iters = n_iters
        self.normalize = normalize

        # Model parameters
        self.weight = None
        self.bias = None

        # To store SSE values during training
        self.sse_history = []

        # For normalization
        self.mean = None
        self.std = None

    def _normalize(self, X):
        """
        Normalize features using Z-score normalization.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Steps:
        1) Initialize weight and bias.
        2) Compute predictions.
        3) Compute gradients.
        4) Update parameters.
        5) Store SSE to track convergence.
        """

        X = np.array(X)
        y = np.array(y).reshape(-1)

        # Ensure X is 2D (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Normalize features if enabled
        if self.normalize:
            X = self._normalize(X)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            # Compute predictions
            y_pred = np.dot(X, self.weight) + self.bias

            # Compute gradients (CORRECTED)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weight -= self.alpha * dw
            self.bias -= self.alpha * db

            # Compute SSE
            sse = np.sum((y_pred - y) ** 2)
            self.sse_history.append(sse)

        return self

    def predict(self, X):
        """
        Predict target values using trained parameters.

        Accepts:
            - Single number
            - List
            - 1D numpy array
            - 2D numpy array

        Returns:
            Predicted value(s)
        """

        X = np.array(X)

        if X.ndim == 0:
            X = np.array([[X]])
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.normalize:
            X = (X - self.mean) / self.std

        return np.dot(X, self.weight) + self.bias

    def mse(self, y, y_pred):
        """
        Compute Mean Squared Error (MSE).

        Measures average squared difference
        between actual and predicted values.
        """
        return np.mean((y - y_pred) ** 2)

    def r2score(self, X, y):
        """
        Compute R-squared score.

        R² measures how well the regression line
        explains the variance in the data.

        R² = 1 → perfect fit
        R² = 0 → no explanatory power
        R² < 0 → worse than baseline
        """

        y_pred = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)

    def plot_training(self, X, y):
        """
        Visualize training process:

        1) SSE over iterations → shows convergence
        2) Regression line with data points (only for single feature)
        """

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        plt.figure(figsize=(12, 5))

        # Plot SSE history
        plt.subplot(1, 2, 1)
        plt.plot(range(self.n_iters), self.sse_history)
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.title("SSE over Iterations")

        # Plot regression line (only works clearly for 1 feature)
        if X.shape[1] == 1:
            plt.subplot(1, 2, 2)
            plt.scatter(X, y)
            y_line = self.predict(X)
            plt.plot(X, y_line)
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("Regression Line")

        plt.tight_layout()
        plt.show()
