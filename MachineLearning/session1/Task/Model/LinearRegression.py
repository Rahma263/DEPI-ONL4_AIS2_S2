"""
This class implements Linear Regression using Gradient Descent.
It inherits from BaseLinearModel and learns the best line by minimizing SSE.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression_Rahma(BaseLinearModel):
    """
    Linear Regression model using Gradient Descent.

    Inherits from BaseLinearModel to reuse:
    - learning rate (alpha)
    - number of iterations
    - weight (w) and bias (b)
    - predict() function
    - loss_history storage
    """

    def __init__(self, alpha: float, iteration: int = 20):
        """
        Initialize Linear Regression model.

        Parameters:
        alpha (float): learning rate
        iteration (int): number of training iterations
        """

        
        super().__init__(alpha=alpha, iteration=iteration, lambda_=0)

        
        self.sse_values = []


    def fit_rahma(self, X, y):
        """
        Train Linear Regression using Gradient Descent.

        Parameters:
        X (array-like): input feature values
        y (array-like): true target values
        """

        n = len(X)

        for i in range(self.iteration):

            
            y_hat = self.predict(X)

            
            D_w = (2/n) * np.sum((y_hat - y) * X)
            D_b = (2/n) * np.sum(y_hat - y)

            
            self.w -= self.alpha * D_w
            self.b -= self.alpha * D_b

            
            sse = np.sum((y_hat - y) ** 2)

        
            self.loss_history.append(sse)
            self.sse_values.append(sse)

            
            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}, SSE = {sse}")


    def pred_Y(self, X_new):
        """
        Predict output for new input values.

        Uses inherited predict() method.
        """

        return self.predict(X_new)


    def visualize(self, X, y):
        """
        Visualize training loss and regression line.
        """

        plt.figure(figsize=(12,5))

        
        plt.subplot(1,2,1)
        plt.plot(range(self.iteration), self.loss_history, label="SSE")
        plt.xlabel("Iteration")
        plt.ylabel("Sum Squared Error")
        plt.title("Training Error Curve")
        plt.legend()

        
        plt.subplot(1,2,2)
        plt.scatter(X, y, label="Data points")
        plt.plot(X, self.predict(X), color="red", label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression Fit")
        plt.legend()

        plt.show()
