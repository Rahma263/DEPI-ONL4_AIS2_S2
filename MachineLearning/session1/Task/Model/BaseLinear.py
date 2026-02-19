import numpy as np

"""
BaseLinearModel class

This class represents the base model for all linear-based regression algorithms.
It contains the common attributes and methods shared between Linear, Ridge, and Lasso regression.

The model follows the equation:
y = wX + b

where:
w = weight (slope)
b = bias (intercept)

Other regression models can inherit from this class to reuse the predict function
and basic parameter initialization.
"""
class BaseLinearModel:
    
    def __init__(self, alpha: float, iteration: int, lambda_: int):
        """
        Initialize the model parameters.

        Parameters:
        alpha (float): Learning rate used in gradient descent to control update size
        iteration (int): Number of iterations for training the model
        lambda_ (int): Regularization parameter used in Ridge and Lasso regression
        
        Attributes:
        w (float): Weight parameter initialized later during training
        b (float): Bias parameter initialized later during training
        loss_history (list): Stores loss values at each iteration to track training performance
        """
        
        self.alpha = alpha
    
        self.iteration = iteration
        
        self.lambda_ = lambda_
        
        self.w = 0
        self.b = 0
        
        self.loss_history = []
        
    
    def predict(self, X):
        """
        Predict the output using the current model parameters.

        This function applies the linear equation:
        y = wX + b

        Parameters:
        X (array-like): Input feature values

        Returns:
        array-like: Predicted output values
        """
        
        return self.w * X + self.b
