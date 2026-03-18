import numpy as np
from Model.BaseLinear import BaseLinearModel

"""
This class implements Ridge Regression using Gradient Descent.

Ridge Regression is similar to Linear Regression, but it adds a penalty
term to the loss function. This penalty helps reduce overfitting by
preventing the weights from becoming too large.

The penalty used here is L2 regularization, which adds lambda * w²
to the loss function.
"""

class RidgeRegression(BaseLinearModel):
    
    def fit(self, X, y):
        """
        Train the Ridge Regression model using Gradient Descent.

        Parameters
        
        X_type : numpy array
            Input feature values
        
        y_type : numpy array
            Actual target values
        """
        
        n = len(X)
        
        for i in range(self.iteration):
            
            y_hat = self.predict(X)
            
            D_w = (2/n) * np.sum((y_hat - y) * X) + 2 * self.lambda_ * self.w
            
            D_b = (2/n) * np.sum(y_hat - y)
            
            self.w -= self.alpha * D_w
            
            self.b -= self.alpha * D_b
            
            loss = np.sum((y_hat - y)**2) + self.lambda_ * np.sum(self.w**2)
             
            self.loss_history.append(loss)
