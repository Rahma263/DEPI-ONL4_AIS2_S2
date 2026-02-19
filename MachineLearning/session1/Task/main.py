"""
main.py

This file tests all regression models:
Linear Regression, Ridge Regression, Lasso Regression, and Polynomial Regression
"""

import numpy as np

from Model.LinearRegression import LinearRegression_r
from Model.Ridge import RidgeRegression
from Model.Lasso import LassoRegression
from Preprocessing.Polynomial import PolynomialRegression


X = np.array([1, 2, 3, 4, 5])
y = np.array([7, 9, 11, 13, 15])


print("Training Linear Regression")
linear = LinearRegression_r(alpha=0.01, iteration=100)
linear.fit_r(X, y)
print("Linear prediction for x=6:", linear.pred_Y(6))


print("\nTraining Ridge Regression")
ridge = RidgeRegression(alpha=0.01, iteration=100, lambda_=0.1)
ridge.fit(X, y)
print("Ridge prediction for x=6:", ridge.predict(6))


print("\nTraining Lasso Regression")
lasso = LassoRegression(alpha=0.01, iteration=100, lambda_=0.1)
lasso.fit(X, y)
print("Lasso prediction for x=6:", lasso.predict(6))


print("\nTraining Polynomial Regression")
poly = PolynomialRegression(degree=2, alpha=0.00001, iterations=100)
poly.fit(X, y)

X_poly = poly.transform(np.array([6]))
prediction = np.dot(X_poly, poly.w) + poly.b

print("Polynomial prediction for x=6:", prediction)

