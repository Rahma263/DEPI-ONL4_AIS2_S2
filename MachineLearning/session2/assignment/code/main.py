import numpy as np
from ClassLinear import LinearRegressionGD


def main():

    # ---------------------------------
    # 1) Load & Understand Data
    # ---------------------------------
    # X → house size (m²)
    # y → house price (in thousands)

    X = np.array([50, 60, 70, 80, 90]).reshape(-1, 1)
    y = np.array([150, 180, 210, 240, 270])

    print("Dataset Loaded Successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ---------------------------------
    # 2) Create Model
    # ---------------------------------

    model = LinearRegressionGD(
        learning_rate=0.000001,
        n_iters=10000,
        normalize=False
    )

    # ---------------------------------
    # 3) Train Model
    # ---------------------------------

    model.fit(X, y)

    print("\nModel Trained Successfully")

    # ---------------------------------
    # 4) Print Learned Parameters
    # ---------------------------------

    print("\nLearned Parameters:")
    print("Theta_0 (bias):", model.bias)
    print("Theta_1 (weight):", model.weight)

    # ---------------------------------
    # 5) Prediction
    # ---------------------------------

    test_size = 70
    prediction = model.predict([[test_size]])

    print(f"\nPrediction for {test_size} m²:", prediction[0])

    # ---------------------------------
    # 6) Evaluation
    # ---------------------------------

    r2 = model.r2score(X, y)
    print("\nR2 score:", r2)

    # ---------------------------------
    # 7) MSE
    # ---------------------------------

    y_pred = model.predict(X)
    mse_value = model.mse(y, y_pred)
    print("MSE:", mse_value)

    # ---------------------------------
    # 8) Visualization
    # ---------------------------------

    model.plot_training(X, y)


if __name__ == "__main__":
    main()
