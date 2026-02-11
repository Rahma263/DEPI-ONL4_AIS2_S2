import numpy as np
from LinearRegression_R import LinearRegression_Rahma

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3, 5, 7, 9, 11, 13, 15, 17])

model = LinearRegression_Rahma(alpha=0.01, iterations=20)

model.fit_rahma(x, y)

x_new = 10
prediction = model.pred_Y(x_new)
print(f"Prediction for x = {x_new}: y = {prediction}")

model.visualize(x, y)
