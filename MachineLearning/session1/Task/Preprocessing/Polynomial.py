import numpy as np
from Model.LinearRegression import LinearRegression_r

"""
This class implements Polynomial Regression using Gradient Descent.

Polynomial Regression is an extension of Linear Regression that allows
the model to learn non-linear relationships between input (X) and output (y)
by transforming the original feature into higher-degree polynomial features.

This allows the model to fit curves instead of just straight lines.
"""

class PolynomialRegression(LinearRegression_r):
    
    def __init__(self, degree=2, alpha=0.01, iterations=100):
        """
        Initialize the Polynomial Regression model.

        Parameters
        
        degree : int
            The maximum polynomial degree to generate.
            Example: degree=3 → creates x, x², x³
        
        alpha : float
            Learning rate used in Gradient Descent.
            Controls how fast weights are updated.
        
        iterations : int
            Number of Gradient Descent iterations.
            More iterations allow better convergence.
        """
        
       
        super().__init__(alpha, iterations)
        
        
        self.degree = degree
    
    
    def transform(self, X):
        """
        Transform original input features into polynomial features.

        Parameters
        
        X_type : numpy array
           x_parameter: Original input feature vector

        Returns
        
        X_poly : numpy array
            Rtype:Transformed feature matrix with polynomial features
        """
        
        
        X_poly = X
        
        
        for d in range(2, self.degree + 1):
            
            
            X_poly = np.column_stack((X_poly, X**d))   # Add a new column with X raised to power d.
            
        return X_poly
    
    
    def fit(self, X, y):
        """
        Train the Polynomial Regression model using Gradient Descent.

        Steps performed:
        1. Transform input features into polynomial features
        2. Initialize weights and bias
        3. Perform gradient descent to minimize error
        4. Store loss values for analysis

        Parameters
        
        X_type : numpy array
           :x_parameter: Input feature vector
        
        y_type : numpy array
            :y_parameter:True target values
        """
        
        X_poly = self.transform(X)
    
        self.w = np.zeros(X_poly.shape[1])
        
        self.b = 0
        
        n = len(X)
        
        for i in range(self.iteration):
            
            y_hat = np.dot(X_poly, self.w) + self.b
            
            D_w = (2/n) * np.dot(X_poly.T, (y_hat - y))
            
            D_b = (2/n) * np.sum(y_hat - y)
            
            self.w -= self.alpha * D_w
            self.b -= self.alpha * D_b
            
            loss = np.sum((y_hat - y)**2)
            
            self.loss_history.append(loss)
