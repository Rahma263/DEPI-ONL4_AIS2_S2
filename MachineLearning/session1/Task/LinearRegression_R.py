"""
This class implements Linear Regression using Gradient Descent.
It learns the best line by minimizing the Sum of Squared Errors (SSE).
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression_Rahma():
    
    def __init__(self, alpha:float, iterations:int =20):
        """
        Initialize learning rate and number of training iterations
        alpha = learning rate
        iterations = number of gradient descent steps
        """
        self.alpha = alpha
        self.iterations = iterations
        self.w = 0      
        self.b = 0      
        self.sse_values = []  # store error values over time
    
    def fit_rahma(self, x, y):
        """
        Train the model using Gradient Descent
        x = input features
        y = true target values
        """
        n = len(x)
        
        for i in range(self.iterations):
            # Predict values using current parameters
            y_hat = self.w * x + self.b
            
            # Compute gradients (partial derivatives)
            D_w = (2/n) * np.sum((y_hat - y) * x)
            D_b = (2/n) * np.sum((y_hat - y))
            
            # Update weights using learning rate
            self.w -= self.alpha * D_w
            self.b -= self.alpha * D_b
            
            # Compute Sum of Squared Errors (SSE)
            sse = np.sum((y_hat - y) ** 2)
            self.sse_values.append(sse)
            
            # Print progress every 20 iterations
            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}, SSE = {sse}")
    
    def pred_Y(self, x_new):
        """
        Predict output for a new input value
        """
        y_pred = self.w * x_new + self.b
        return y_pred
    
    def visualize(self, x, y):
        """
        Plot error curve and regression line
        """
        plt.figure(figsize=(12,5))
        
        # Plot SSE vs Iterations
        plt.subplot(1,2,1)
        plt.plot(range(self.iterations), self.sse_values, label='SSE')
        plt.xlabel("Iteration")
        plt.ylabel("Sum Squared Error")
        plt.title("SSE Over Training")
        plt.legend()
        
        # Plot regression line vs real data
        plt.subplot(1,2,2)
        plt.scatter(x, y, color="blue", label="Data points")
        plt.plot(x, self.w * x + self.b, color='red', label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression Fit")
        plt.legend()
        
        plt.show()
