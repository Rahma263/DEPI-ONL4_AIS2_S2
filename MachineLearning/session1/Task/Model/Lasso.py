import numpy as np
from Model.BaseLinear import BaseLinearModel

"""
This class implements Lasso Regression using Gradient Descent.

Lasso Regression is similar to Linear Regression, but it adds
an L1 regularization term to the loss function. This helps:

- Reduce overfitting
- Shrink weights toward zero
- Perform feature selection automatically

The objective function minimized is:

Loss = Sum of Squared Errors (SSE) + lambda * |w|
"""
class LassoRegression(BaseLinearModel):
    
    def fit(self, X, y):
        """
        Train the Lasso Regression model using Gradient Descent.
        
        :parameter_Type_X :Input feature values
        :parameter_X : numpy array
        
        :parameter_y: True target values
        :parameter_Type_y : numpy array
            

        This function updates:
        - weight (self.w)
        - bias (self.b)

        and stores loss values in loss_history.
        """
        
       
        n = len(X)
        
        for i in range(self.iteration):
            
            y_hat = self.predict(X)
            
            D_w = (2/n) * np.sum((y_hat - y) * X) + self.lambda_ * np.sign(self.w)   #We use `np.sign(self.w)` because it represents the derivative of the L1 regularization term ( |w| ), which pushes weights toward zero during gradient descent.

            D_b = (2/n) * np.sum(y_hat - y)
            
            self.w -= self.alpha * D_w
            self.b -= self.alpha * D_b
            
            loss = np.sum((y_hat - y)**2) + self.lambda_ * np.abs(self.w)
        
            self.loss_history.append(loss)
